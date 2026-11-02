from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from spatial_support_contract import (
    SupportTokenInputs,
    SupportTokenTargets,
    SupportTokenValueNormalization,
    aggregate_area_weighted_responses,
    aggregate_support_observations,
    soft_support_memberships,
)

from .hdf5_source import HDF5StateSource
from .normalization import NormalizationStats
from .observation import ObservationConfig, sample_virtual_observation
from .windows import WindowRef

_REALIZATION_STRIDE = 10**6


@dataclass(frozen=True)
class SpatialSupportLayout:
    """A fixed fuzzy partition over a bridge mesh, expressed in normalized space."""

    bounds_xy: tuple[float, float, float, float]
    centers_unit: np.ndarray
    bandwidths_unit: np.ndarray
    support_features: np.ndarray
    nodal_area_weights: np.ndarray

    @property
    def support_count(self) -> int:
        return int(self.centers_unit.shape[0])


@dataclass(frozen=True)
class SupportTokenSample:
    """Model inputs and FEM-only targets for one temporal window."""

    values: np.ndarray
    valid_mask: np.ndarray
    quality: np.ndarray
    coverage: np.ndarray
    support_features: np.ndarray
    time_features: np.ndarray
    target_response: np.ndarray
    target_valid_mask: np.ndarray
    target_area_coverage: np.ndarray


def _quad_area(vertices: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape[1] >= 3:
        first = 0.5 * np.linalg.norm(
            np.cross(points[1] - points[0], points[2] - points[0])
        )
        second = 0.5 * np.linalg.norm(
            np.cross(points[2] - points[0], points[3] - points[0])
        )
        return float(first + second)
    x = points[:, 0]
    y = points[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
    )


def _nodal_area_weights(
    node_coords: np.ndarray,
    node_ids: np.ndarray,
    deck_element_connectivity: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(node_coords, dtype=np.float64)
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    elements = np.asarray(deck_element_connectivity, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[0] != ids.size or coords.shape[1] < 2:
        raise ValueError("node_coords must have shape (nodes, >=2) and match node_ids")
    if np.unique(ids).size != ids.size:
        raise ValueError("node_ids must be unique")
    if elements.ndim != 2 or elements.shape[1] != 4 or elements.size == 0:
        raise ValueError("deck_element_connectivity must contain four-node quads")

    index_by_id = {int(node_id): index for index, node_id in enumerate(ids)}
    nodal = np.zeros(ids.size, dtype=np.float64)
    for element_index, element in enumerate(elements):
        try:
            indices = np.asarray([index_by_id[int(node_id)] for node_id in element])
        except KeyError as exc:
            raise ValueError(
                f"deck element {element_index} references an unknown node id {exc.args[0]}"
            ) from exc
        area = _quad_area(coords[indices])
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError(f"deck element {element_index} has invalid area {area}")
        nodal[indices] += area / 4.0
    if not np.isfinite(nodal).all() or nodal.sum() <= 0.0:
        raise ValueError("mesh has no finite positive deck area")
    return nodal.astype(np.float32)


def build_spatial_support_layout(
    node_coords: np.ndarray,
    deck_node_mask: np.ndarray,
    node_ids: np.ndarray,
    deck_element_connectivity: np.ndarray,
    *,
    rows: int = 4,
    cols: int = 2,
    bandwidth_scale: float = 0.75,
) -> SpatialSupportLayout:
    """Build a reproducible soft grid; model features contain only coarse supports.

    Exact mesh coordinates are used here by the data adapter to define the grid and
    memberships. They are not included in the returned per-sample model inputs.
    """
    coords = np.asarray(node_coords, dtype=np.float64)
    deck = np.asarray(deck_node_mask, dtype=bool).reshape(-1)
    if coords.ndim != 2 or coords.shape[0] != deck.size or coords.shape[1] < 2:
        raise ValueError("deck_node_mask must match node_coords")
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    if not np.isfinite(bandwidth_scale) or bandwidth_scale <= 0.0:
        raise ValueError("bandwidth_scale must be finite and positive")
    deck_xy = coords[deck, :2]
    if deck_xy.size == 0 or not np.isfinite(deck_xy).all():
        raise ValueError("deck mesh coordinates must be non-empty and finite")
    minimum = deck_xy.min(axis=0)
    maximum = deck_xy.max(axis=0)
    span = maximum - minimum
    if np.any(span <= np.finfo(np.float64).eps):
        raise ValueError("deck mesh must span both spatial axes")

    x = (np.arange(rows, dtype=np.float64) + 0.5) / rows
    y = (np.arange(cols, dtype=np.float64) + 0.5) / cols
    xx, yy = np.meshgrid(x, y, indexing="ij")
    centers = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    cell_size = np.tile(np.asarray([1.0 / rows, 1.0 / cols]), (len(centers), 1))
    bandwidths = cell_size * float(bandwidth_scale)
    features = np.column_stack(
        (2.0 * centers - 1.0, cell_size)
    ).astype(np.float32)
    bounds = (
        float(minimum[0]),
        float(maximum[0]),
        float(minimum[1]),
        float(maximum[1]),
    )
    return SpatialSupportLayout(
        bounds_xy=bounds,
        centers_unit=centers.astype(np.float32),
        bandwidths_unit=bandwidths.astype(np.float32),
        support_features=features,
        nodal_area_weights=_nodal_area_weights(
            coords, node_ids, deck_element_connectivity
        ),
    )


def _normalize_xy(coords: np.ndarray, layout: SpatialSupportLayout) -> np.ndarray:
    value = np.asarray(coords, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 2:
        raise ValueError("coordinates must have shape (items, >=2)")
    if not np.isfinite(value[:, :2]).all():
        raise ValueError("coordinates must be finite")
    xmin, xmax, ymin, ymax = layout.bounds_xy
    return np.column_stack(
        ((value[:, 0] - xmin) / (xmax - xmin), (value[:, 1] - ymin) / (ymax - ymin))
    )


def _soft_memberships(
    coords: np.ndarray, layout: SpatialSupportLayout
) -> np.ndarray:
    normalized = _normalize_xy(coords, layout)
    return soft_support_memberships(
        normalized, layout.centers_unit, layout.bandwidths_unit
    )


def aggregate_observation_tokens(
    point_values: np.ndarray,
    point_coords: np.ndarray,
    valid_mask: np.ndarray,
    quality: np.ndarray,
    layout: SpatialSupportLayout,
    *,
    minimum_coverage: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool point observations with the shared quality and coverage semantics."""
    values = np.asarray(point_values, dtype=np.float32)
    points = np.asarray(point_coords, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    q = np.asarray(quality, dtype=np.float32)
    if values.ndim != 2 or valid.shape != values.shape or q.shape != values.shape:
        raise ValueError("point values, validity mask, and quality must share shape (time, points)")
    if points.ndim != 2 or points.shape[0] != values.shape[1]:
        raise ValueError("point_coords must contain one coordinate per observed point")
    membership = _soft_memberships(points, layout)
    return aggregate_support_observations(
        values,
        valid,
        q,
        membership,
        minimum_coverage=minimum_coverage,
    )


def build_support_token_sample(
    point_values: np.ndarray,
    point_coords: np.ndarray,
    valid_mask: np.ndarray,
    quality: np.ndarray,
    time_features: np.ndarray,
    node_coords: np.ndarray,
    dense_target: np.ndarray,
    layout: SpatialSupportLayout,
    *,
    minimum_observation_coverage: float = 0.05,
    minimum_target_area_coverage: float = 0.99,
) -> SupportTokenSample:
    """Aggregate sparse inputs and area-weighted FEM targets over the same supports.

    Observed values and dense targets must already use the same reference and units.
    The returned model-side fields omit exact point coordinates and dense targets.
    """
    values = np.asarray(point_values, dtype=np.float32)
    points = np.asarray(point_coords, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    q = np.asarray(quality, dtype=np.float32)
    times = np.asarray(time_features, dtype=np.float32)
    target = np.asarray(dense_target, dtype=np.float32)
    mesh_coords = np.asarray(node_coords, dtype=np.float32)
    if values.ndim != 2 or valid.shape != values.shape or q.shape != values.shape:
        raise ValueError("point values, validity mask, and quality must share shape (time, points)")
    if points.ndim != 2 or points.shape[0] != values.shape[1]:
        raise ValueError("point_coords must contain one coordinate per observed point")
    if times.ndim != 2 or times.shape[0] != values.shape[0] or not np.isfinite(times).all():
        raise ValueError("time_features must be finite and share the observation time axis")
    if target.ndim != 2 or target.shape != (values.shape[0], mesh_coords.shape[0]):
        raise ValueError("dense_target must have shape (time, mesh_nodes)")
    if layout.nodal_area_weights.size != mesh_coords.shape[0]:
        raise ValueError("layout nodal areas do not match node_coords")
    if not 0.0 < minimum_target_area_coverage <= 1.0:
        raise ValueError("minimum_target_area_coverage must be in (0, 1]")
    token_values, token_valid, token_quality, coverage = aggregate_observation_tokens(
        values,
        points,
        valid,
        q,
        layout,
        minimum_coverage=minimum_observation_coverage,
    )

    target_membership = _soft_memberships(mesh_coords, layout)
    target_response, target_mask, target_area_coverage, _ = aggregate_area_weighted_responses(
        target,
        np.isfinite(target),
        target_membership,
        layout.nodal_area_weights,
        minimum_area_coverage=minimum_target_area_coverage,
    )

    return SupportTokenSample(
        values=token_values,
        valid_mask=token_valid,
        quality=token_quality,
        coverage=coverage,
        support_features=layout.support_features.copy(),
        time_features=times,
        target_response=target_response,
        target_valid_mask=target_mask,
        target_area_coverage=target_area_coverage,
    )


def collate_support_token_samples(
    samples: list[SupportTokenSample],
) -> tuple[SupportTokenInputs, SupportTokenTargets]:
    if not samples:
        raise ValueError("cannot collate an empty support-token batch")
    shapes = {
        (
            sample.values.shape,
            sample.support_features.shape,
            sample.time_features.shape,
            sample.target_response.shape,
        )
        for sample in samples
    }
    if len(shapes) != 1:
        raise ValueError("support-token samples in one batch must have matching shapes")

    def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(
            np.stack([getattr(sample, name) for sample in samples]), dtype=dtype
        )

    inputs = SupportTokenInputs(
        values=stack("values", torch.float32),
        valid_mask=stack("valid_mask", torch.bool),
        quality=stack("quality", torch.float32),
        coverage=stack("coverage", torch.float32),
        support_features=stack("support_features", torch.float32),
        time_features=stack("time_features", torch.float32),
    )
    targets = SupportTokenTargets(
        response=stack("target_response", torch.float32),
        valid_mask=stack("target_valid_mask", torch.bool),
        area_coverage=stack("target_area_coverage", torch.float32),
    )
    return inputs, targets


def _sample_qianhai_state(
    ref: WindowRef,
    observation: ObservationConfig,
    seed: int,
    realization: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    arrays = HDF5StateSource(ref.record.path).read(cache=False)
    rng = np.random.default_rng(
        int(seed)
        + sum(ord(char) for char in ref.record.structural_state_id)
        + int(realization) * _REALIZATION_STRIDE
    )
    point_count = int(
        rng.integers(observation.point_count_min, observation.point_count_max + 1)
    )
    view = sample_virtual_observation(
        arrays["node_coords"],
        arrays["deck_node_mask"],
        arrays["displacement_uz"],
        observation,
        rng,
        node_ids=arrays["node_ids"],
        deck_element_connectivity=arrays["deck_element_connectivity"],
        point_count=point_count,
    )
    dense_target = np.asarray(arrays["displacement_uz"], dtype=np.float32)
    if observation.reference_mode == "first_step":
        dense_target = dense_target - dense_target[0:1]
    elif observation.reference_mode != "none":
        raise ValueError(f"Unsupported reference mode: {observation.reference_mode}")
    return arrays, view, dense_target


def fit_support_token_value_normalization(
    train_refs: list[WindowRef],
    observation: ObservationConfig,
    seed: int,
    layout: SpatialSupportLayout,
    *,
    minimum_coverage: float = 0.05,
) -> SupportTokenValueNormalization:
    """Fit pooled-value statistics on unique training states, without targets."""
    by_state: dict[str, WindowRef] = {}
    for ref in train_refs:
        by_state.setdefault(ref.record.structural_state_id, ref)
    if not by_state:
        raise ValueError("cannot fit support-token normalization without training states")
    chunks: list[np.ndarray] = []
    fitted_states: list[str] = []
    for state_id, ref in sorted(by_state.items()):
        _, view, _ = _sample_qianhai_state(ref, observation, seed, 0)
        values, valid, _, _ = aggregate_observation_tokens(
            view["values"],
            view["point_coords"][:, :2],
            view["valid_mask"],
            view["quality"],
            layout,
            minimum_coverage=minimum_coverage,
        )
        if not valid.any():
            raise ValueError(
                f"training state {state_id} produced no valid support-token observations"
            )
        chunks.append(values[valid].astype(np.float64))
        fitted_states.append(state_id)
    if not chunks:
        raise ValueError("training states produced no valid support-token observations")
    flattened = np.concatenate(chunks)
    return SupportTokenValueNormalization(
        mean=float(flattened.mean()),
        std=max(float(flattened.std()), 1.0e-6),
        fitted_state_ids=tuple(fitted_states),
    )


class QianhaiSupportTokenDataset(Dataset[SupportTokenSample]):
    """Windowed Qianhai observations and FEM targets adapted to support tokens."""

    def __init__(
        self,
        refs: list[WindowRef],
        observation: ObservationConfig,
        seed: int,
        time_normalization: NormalizationStats,
        value_normalization: SupportTokenValueNormalization,
        layout: SpatialSupportLayout,
        *,
        realizations: int = 1,
        minimum_observation_coverage: float = 0.05,
        minimum_target_area_coverage: float = 0.99,
    ) -> None:
        self.refs = refs
        self.observation = observation
        self.seed = int(seed)
        self.time_normalization = time_normalization
        self.value_normalization = value_normalization
        self.layout = layout
        self.realizations = max(int(realizations), 1)
        self.minimum_observation_coverage = float(minimum_observation_coverage)
        self.minimum_target_area_coverage = float(minimum_target_area_coverage)
        self._state_cache: OrderedDict[
            tuple[str, int], tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self.refs) * self.realizations

    def __getitem__(self, index: int) -> SupportTokenSample:
        if not self.refs:
            raise IndexError("empty support-token dataset")
        ref_index = int(index) % len(self.refs)
        realization = int(index) // len(self.refs)
        ref = self.refs[ref_index]
        key = (ref.record.structural_state_id, realization)
        state = self._state_cache.get(key)
        if state is None:
            state = _sample_qianhai_state(ref, self.observation, self.seed, realization)
            self._state_cache[key] = state
            if len(self._state_cache) > 2:
                self._state_cache.popitem(last=False)
        else:
            self._state_cache.move_to_end(key)
        arrays, view, dense_target = state
        sl = slice(ref.start, ref.start + ref.length)
        temperature = np.asarray(arrays["temperature_C"][sl], dtype=np.float32)
        delta_days = np.asarray(arrays["delta_days"][sl], dtype=np.float32)
        temperature = np.nan_to_num(temperature, nan=0.0, posinf=0.0, neginf=0.0)
        delta_days = np.nan_to_num(delta_days, nan=0.0, posinf=0.0, neginf=0.0)
        time_features = np.c_[
            self.time_normalization.transform_temperature(temperature),
            self.time_normalization.transform_delta_days(delta_days),
        ].astype(np.float32)
        sample = build_support_token_sample(
            point_values=view["values"][sl],
            point_coords=view["point_coords"][:, :2],
            valid_mask=view["valid_mask"][sl],
            quality=view["quality"][sl],
            time_features=time_features,
            node_coords=arrays["node_coords"],
            dense_target=dense_target[sl],
            layout=self.layout,
            minimum_observation_coverage=self.minimum_observation_coverage,
            minimum_target_area_coverage=self.minimum_target_area_coverage,
        )
        return SupportTokenSample(
            values=self.value_normalization.transform_numpy(
                sample.values, sample.valid_mask
            ),
            valid_mask=sample.valid_mask,
            quality=sample.quality,
            coverage=sample.coverage,
            support_features=sample.support_features,
            time_features=sample.time_features,
            target_response=sample.target_response,
            target_valid_mask=sample.target_valid_mask,
            target_area_coverage=sample.target_area_coverage,
        )
