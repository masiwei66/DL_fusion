"""Soft spatial-support tokens for KW51 FEM LOS response prediction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import WindowBatch
from .fuzzy_range import FuzzyNormalization, FuzzyQueryRef
from .fuzzy_region import FuzzyRegionSource, RegionLayout
from spatial_support_contract import (
    SupportTokenInputs,
    SupportTokenTargets,
    SupportTokenValueNormalization,
    aggregate_area_weighted_responses,
    aggregate_support_observations,
    soft_support_memberships,
)


@dataclass(frozen=True)
class SupportTokenLayout:
    centers_unit: np.ndarray
    bandwidths_unit: np.ndarray
    features: np.ndarray

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers_unit)
        widths = np.asarray(self.bandwidths_unit)
        features = np.asarray(self.features)
        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
            raise ValueError("support centers must have shape [R, 2] with R > 0")
        if widths.shape != centers.shape or features.ndim != 2 or features.shape[0] != len(centers):
            raise ValueError("support widths and features must align with support centers")
        if not np.isfinite(centers).all() or not np.isfinite(widths).all() or not np.isfinite(features).all():
            raise ValueError("support layout values must be finite")
        if np.any(widths <= 0.0):
            raise ValueError("support bandwidths must be positive")

    @property
    def support_count(self) -> int:
        return int(self.centers_unit.shape[0])


def build_support_token_layout(
    layout: RegionLayout, *, bandwidth_scale: float = 0.75
) -> SupportTokenLayout:
    """Represent each coarse KW51 region as an overlapping fuzzy footprint."""
    if not np.isfinite(bandwidth_scale) or bandwidth_scale <= 0.0:
        raise ValueError("bandwidth_scale must be finite and positive")
    centers_st = np.zeros((layout.region_count, 2), dtype=np.float32)
    widths: list[tuple[float, float]] = []
    for index in range(layout.region_count):
        s_low, s_high, t_low, t_high = layout.bounds(index)
        centers_st[index] = (0.5 * (s_low + s_high), 0.5 * (t_low + t_high))
        widths.append((s_high - s_low, t_high - t_low))
    cell_widths_st = np.asarray(widths, dtype=np.float32)
    if not np.isfinite(centers_st).all() or not np.isfinite(cell_widths_st).all():
        raise ValueError("support layout must be finite")
    if np.any(cell_widths_st <= 0.0):
        raise ValueError("every support footprint must have positive width")
    centers_unit = centers_st.copy()
    centers_unit[:, 1] = (centers_unit[:, 1] + 1.0) / 2.0
    cell_widths_unit = cell_widths_st.copy()
    cell_widths_unit[:, 1] /= 2.0
    bandwidths = cell_widths_unit * np.float32(bandwidth_scale)
    features = np.c_[
        2.0 * centers_unit[:, 0] - 1.0,
        2.0 * centers_unit[:, 1] - 1.0,
        cell_widths_unit[:, 0],
        cell_widths_unit[:, 1],
    ].astype(np.float32)
    return SupportTokenLayout(centers_unit, bandwidths, features)


def _soft_memberships(
    coordinates_st: np.ndarray, layout: SupportTokenLayout
) -> np.ndarray:
    coordinates_st = np.asarray(coordinates_st, dtype=np.float64)
    if coordinates_st.ndim != 2 or coordinates_st.shape[1] != 2:
        raise ValueError("coordinates_st must have shape [items, 2]")
    coordinates = coordinates_st.copy()
    coordinates[:, 1] = (coordinates[:, 1] + 1.0) / 2.0
    if not np.isfinite(coordinates).all():
        raise ValueError("support assignment coordinates must be finite")
    return soft_support_memberships(
        coordinates, layout.centers_unit, layout.bandwidths_unit
    )


def aggregate_observation_tokens(
    values: np.ndarray,
    valid_mask: np.ndarray,
    quality: np.ndarray,
    coordinates_st: np.ndarray,
    layout: SupportTokenLayout,
    *,
    minimum_coverage: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Soft-pool sparse LOS points into fixed supports without returning points."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    quality = np.asarray(quality, dtype=np.float32)
    coordinates = np.asarray(coordinates_st, dtype=np.float32)
    if values.ndim != 2 or valid.shape != values.shape or quality.shape != values.shape:
        raise ValueError("values, valid_mask and quality must share shape [T, P]")
    if coordinates.shape != (values.shape[1], 2):
        raise ValueError("coordinates_st must contain one [s,t] pair per point")
    memberships = _soft_memberships(coordinates, layout)
    return aggregate_support_observations(
        values,
        valid,
        quality,
        memberships,
        minimum_coverage=minimum_coverage,
    )


def fit_support_token_value_normalization(
    source: FuzzyRegionSource,
    records: list[object],
    region_layout: RegionLayout,
    support_layout: SupportTokenLayout,
    *,
    minimum_coverage: float = 0.05,
) -> SupportTokenValueNormalization:
    """Fit value statistics on pooled tokens from unique training states only."""
    by_state: dict[str, list[object]] = {}
    for record in records:
        by_state.setdefault(str(record.structural_state_id), []).append(record)
    if not by_state:
        raise ValueError("cannot fit support-token normalization without training states")
    chunks: list[np.ndarray] = []
    fitted_states: list[str] = []
    maximum_records = int(source.config.normalization_records_per_state)
    for state_id in sorted(by_state):
        state_records = sorted(by_state[state_id], key=lambda record: record.sample_id)
        if maximum_records > 0:
            state_records = state_records[:maximum_records]
        state_chunks: list[np.ndarray] = []
        for record in state_records:
            query = source.sample_region_query(record, 0, region_layout)
            values, valid, _, _ = aggregate_observation_tokens(
                np.asarray(query["values"]),
                np.asarray(query["valid_mask"]),
                np.asarray(query["quality"]),
                np.asarray(query["point_coords"]),
                support_layout,
                minimum_coverage=minimum_coverage,
            )
            if valid.any():
                state_chunks.append(values[valid].astype(np.float64))
        if state_chunks:
            chunks.extend(state_chunks)
            fitted_states.append(state_id)
    if not chunks:
        raise ValueError("training states produced no valid support-token observations")
    flattened = np.concatenate(chunks)
    return SupportTokenValueNormalization(
        mean=float(flattened.mean()),
        std=max(float(flattened.std()), 1.0e-6),
        fitted_state_ids=tuple(fitted_states),
    )


def aggregate_area_weighted_targets(
    values: np.ndarray,
    valid_mask: np.ndarray,
    coordinates_st: np.ndarray,
    nodal_area_weights: np.ndarray,
    layout: SupportTokenLayout,
    *,
    minimum_area_coverage: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate referenced LOS millimetres over soft supports by deck area."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    coordinates = np.asarray(coordinates_st, dtype=np.float32)
    areas = np.asarray(nodal_area_weights, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("target values and mask must share shape [T, N]")
    if coordinates.shape != (values.shape[1], 2) or areas.size != values.shape[1]:
        raise ValueError("target coordinates and nodal areas must match target nodes")
    memberships = _soft_memberships(coordinates, layout)
    return aggregate_area_weighted_responses(
        values,
        valid,
        memberships,
        areas,
        minimum_area_coverage=minimum_area_coverage,
    )


class FuzzySupportTokenDataset(Dataset[dict[str, object]]):
    """Adapt KW51 regional fuzzy bags into Qianhai-style support-token batches."""

    def __init__(
        self,
        refs: list[FuzzyQueryRef],
        source: FuzzyRegionSource,
        normalization: FuzzyNormalization,
        region_layout: RegionLayout,
        support_layout: SupportTokenLayout,
        value_normalization: SupportTokenValueNormalization,
        *,
        minimum_observation_coverage: float = 0.05,
        minimum_target_area_coverage: float = 0.99,
    ) -> None:
        self.refs = refs
        self.source = source
        self.normalization = normalization
        self.region_layout = region_layout
        self.support_layout = support_layout
        self.value_normalization = value_normalization
        self.minimum_observation_coverage = float(minimum_observation_coverage)
        self.minimum_target_area_coverage = float(minimum_target_area_coverage)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        query = self.source.sample_region_query(
            ref.record, ref.realization, self.region_layout
        )
        token_values, token_valid, token_quality, coverage = aggregate_observation_tokens(
            np.asarray(query["values"]),
            np.asarray(query["valid_mask"]),
            np.asarray(query["quality"]),
            np.asarray(query["point_coords"]),
            self.support_layout,
            minimum_coverage=self.minimum_observation_coverage,
        )
        dense = self.source._dense_region_data(ref.record, self.region_layout)
        target, target_valid, target_coverage, target_area = aggregate_area_weighted_targets(
            np.asarray(dense["deck_values_mm"]),
            np.asarray(dense["deck_valid_mask"]),
            np.c_[self.source.geometry.deck_s, self.source.geometry.deck_t],
            np.asarray(dense["node_area_weights"]),
            self.support_layout,
            minimum_area_coverage=self.minimum_target_area_coverage,
        )
        values = self.value_normalization.transform_numpy(token_values, token_valid)
        time_features = self.normalization.transform_time_features(
            np.asarray(query["time_features"])[:, 0],
            np.asarray(query["time_features"])[:, 1],
        )
        metadata = dict(query["metadata"])
        metadata["point_budget"] = int(self.source.config.region_point_budget)
        return {
            "values": values,
            "valid_mask": token_valid,
            "quality": token_quality,
            "coverage": coverage,
            "support_features": self.support_layout.features.copy(),
            "time_features": time_features,
            "target_response": target,
            "target_valid_mask": target_valid,
            "target_area_coverage": target_coverage,
            "target_area_weights": target_area,
            "metadata": metadata,
        }


def collate_fuzzy_support_tokens(
    items: list[dict[str, object]],
) -> WindowBatch:
    if not items:
        raise ValueError("cannot collate an empty support-token batch")
    values = [np.asarray(item["values"], dtype=np.float32) for item in items]
    token_shape = values[0].shape
    feature_shape = np.asarray(items[0]["support_features"]).shape
    for item in items:
        if np.asarray(item["values"]).shape != token_shape:
            raise ValueError("support token time and support axes must match in a batch")
        if np.asarray(item["support_features"]).shape != feature_shape:
            raise ValueError("support feature shapes must match in a batch")
    values_tensor = torch.from_numpy(np.stack(values)).float()
    valid_tensor = torch.from_numpy(
        np.stack([np.asarray(item["valid_mask"], dtype=bool) for item in items])
    ).bool()
    quality_tensor = torch.from_numpy(
        np.stack([np.asarray(item["quality"], dtype=np.float32) for item in items])
    ).float()
    coverage_tensor = torch.from_numpy(
        np.stack([np.asarray(item["coverage"], dtype=np.float32) for item in items])
    ).float()
    support_features = torch.from_numpy(
        np.stack(
            [np.asarray(item["support_features"], dtype=np.float32) for item in items]
        )
    ).float()
    target = torch.from_numpy(
        np.stack(
            [np.asarray(item["target_response"], dtype=np.float32).T for item in items]
        )
    ).float()
    target_valid = torch.from_numpy(
        np.stack(
            [np.asarray(item["target_valid_mask"], dtype=bool).T for item in items]
        )
    ).bool()
    target_coverage = torch.from_numpy(
        np.stack(
            [np.asarray(item["target_area_coverage"], dtype=np.float32).T for item in items]
        )
    ).float()
    target_area = torch.from_numpy(
        np.stack(
            [np.asarray(item["target_area_weights"], dtype=np.float32) for item in items]
        )
    ).float()
    time_features = torch.from_numpy(
        np.stack([np.asarray(item["time_features"], dtype=np.float32) for item in items])
    ).float()
    present = valid_tensor.any(dim=1)
    support_centers = torch.from_numpy(
        np.stack(
            [
                np.asarray(item["support_features"], dtype=np.float32)[:, :2]
                for item in items
            ]
        )
    ).float()
    return WindowBatch(
        values=values_tensor,
        valid_mask=valid_tensor,
        quality=quality_tensor,
        point_features=support_centers,
        time_features=time_features,
        point_present_mask=present,
        mapping_distance=torch.zeros_like(present, dtype=torch.float32),
        mapping_valid_mask=present,
        temperature_valid_mask=torch.ones(
            (len(items), token_shape[0], 1), dtype=torch.bool
        ),
        teacher_features={},
        targets={
            "region_response_target": target,
            "region_response_valid_mask": target_valid,
            "region_area_coverage": target_coverage,
            "region_area_weights": target_area,
        },
        metadata=[dict(item["metadata"]) for item in items],
        support_features=support_features,
        support_coverage=coverage_tensor,
    )


def support_token_inputs_from_window_batch(batch: WindowBatch) -> SupportTokenInputs:
    """Expose only canonical support-token fields to the shared student model."""
    if batch.support_features is None or batch.support_coverage is None:
        raise ValueError("support-token inputs require support features and coverage")
    return SupportTokenInputs(
        values=batch.values,
        valid_mask=batch.valid_mask,
        quality=batch.quality,
        coverage=batch.support_coverage,
        support_features=batch.support_features,
        time_features=batch.time_features,
    )


def support_token_targets_from_window_batch(batch: WindowBatch) -> SupportTokenTargets:
    """Transpose KW51's report layout [B,R,T] to the canonical [B,T,R] target."""
    required = (
        "region_response_target",
        "region_response_valid_mask",
        "region_area_coverage",
    )
    missing = [key for key in required if key not in batch.targets]
    if missing:
        raise ValueError(f"support-token batch is missing target fields {missing}")
    return SupportTokenTargets(
        response=batch.targets["region_response_target"].transpose(1, 2),
        valid_mask=batch.targets["region_response_valid_mask"].transpose(1, 2),
        area_coverage=batch.targets["region_area_coverage"].transpose(1, 2),
    )
