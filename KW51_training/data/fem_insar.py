"""FEM-only sparse observation and candidate-ranking data path for KW51.

The archive stores complete XYZ displacement fields.  This module materializes
the deployment-visible source view lazily: sparse node selection, LOS
projection, optional fixed-grid aggregation, reference removal, synthetic
degradation and train-only normalization. Structural parameters and complete
node fields remain metadata/supervision and never enter student features.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset, Sampler

from dl_fusion.data.regions import SquareGrid, aggregate_sparse_observations, build_square_grid


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _manifest_from_dataset(value: Any) -> dict[str, Any]:
    array = np.asarray(value).reshape(-1)
    raw = array.tobytes().decode("utf-8")
    return json.loads(raw)


@dataclass(frozen=True)
class FEMArchiveRecord:
    path: Path
    sample_id: str
    structural_state_id: str
    trajectory_id: str
    split_group: str
    metadata: dict[str, Any]
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FEMArchiveIndex:
    records: tuple[FEMArchiveRecord, ...]
    run_manifest: dict[str, Any]

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.structural_state_id for record in self.records}))


@dataclass(frozen=True)
class FEMTrajectorySplit:
    """Disjoint environment keys for reference galleries and query roles."""

    gallery: tuple[str, ...]
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "gallery": list(self.gallery),
            "train_query": list(self.train),
            "validation_query": list(self.validation),
            "test_query": list(self.test),
        }


def split_fem_trajectory_keys(
    index: FEMArchiveIndex,
    *,
    gallery_fraction: float,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> FEMTrajectorySplit:
    """Partition shared trajectory keys and fail if states do not share them."""
    fractions = (gallery_fraction, train_fraction, validation_fraction, test_fraction)
    if any(not 0.0 < float(value) < 1.0 for value in fractions):
        raise ValueError("trajectory split fractions must be in (0, 1)")
    if abs(sum(float(value) for value in fractions) - 1.0) > 1.0e-8:
        raise ValueError("trajectory split fractions must sum to 1")

    keys_by_state: dict[str, set[str]] = defaultdict(set)
    for record in index.records:
        keys_by_state[record.structural_state_id].add(trajectory_key(record))
    if not keys_by_state:
        raise ValueError("Cannot split trajectory keys from an empty FEM archive")
    state_key_sets = list(keys_by_state.values())
    reference_keys = state_key_sets[0]
    if any(keys != reference_keys for keys in state_key_sets[1:]):
        raise ValueError(
            "KW51 formal ranking requires every structural state to share the same "
            "trajectory-key set"
        )
    keys = sorted(reference_keys)
    if len(keys) < 4:
        raise ValueError(
            "KW51 ranking needs at least four shared trajectory keys for gallery, "
            "train, validation and test"
        )

    exact_counts = np.asarray(fractions, dtype=np.float64) * len(keys)
    counts = np.floor(exact_counts).astype(np.int64)
    counts[counts == 0] = 1
    while int(counts.sum()) > len(keys):
        eligible = np.flatnonzero(counts > 1)
        if not eligible.size:
            raise ValueError("Not enough trajectory keys for the requested split")
        index_to_reduce = eligible[np.argmin(exact_counts[eligible] - counts[eligible])]
        counts[index_to_reduce] -= 1
    remainder_order = np.argsort(-(exact_counts - np.floor(exact_counts)))
    for slot in remainder_order:
        if int(counts.sum()) >= len(keys):
            break
        counts[slot] += 1

    shuffled = np.asarray(keys, dtype=object)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    offsets = np.cumsum([0, *counts.tolist()])
    parts = [
        tuple(sorted(str(value) for value in shuffled[offsets[i]:offsets[i + 1]]))
        for i in range(4)
    ]
    if sum(len(part) for part in parts) != len(keys):
        raise RuntimeError("Trajectory split did not assign every key exactly once")
    return FEMTrajectorySplit(*parts)


def _load_archive_audit(root: Path) -> dict[str, Any] | None:
    """Load the merge archive audit when present, without trusting it blindly."""
    for name in ("fem_archive_audit.json", "fem_archive_audit_live.json"):
        path = root / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: audit report must be a JSON object")
            return payload
    return None


def build_fem_archive_index(data_root: str | Path) -> FEMArchiveIndex:
    """Index the canonical ``kw51_state_*.h5`` archive without loading fields."""
    import h5py

    root = Path(data_root)
    paths = sorted(root.glob("kw51_state_*.h5"))
    if not paths:
        raise FileNotFoundError(f"No kw51_state_*.h5 files found under {root}")
    run_path = root / "_run_manifest.json"
    run_manifest = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {
        "schema_version": "inferred_kw51_fem_archive_v2",
        "run_manifest_missing": True,
    }
    audit = _load_archive_audit(root)
    if audit is not None:
        run_manifest["audit_report"] = audit
        run_manifest.setdefault("status", audit.get("status", "unknown"))
        run_manifest.setdefault("sample_count", audit.get("sample_count"))
        run_manifest.setdefault("structural_state_count", audit.get("structural_state_count"))
    records: list[FEMArchiveRecord] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            manifest = _manifest_from_dataset(handle["manifest_json"][...])
            parameters: dict[str, float] = {}
            for name in ("Es_rel", "hb_rel", "Id_rel", "Ad_rel", "kxx", "fixed", "addmass"):
                key = f"state/parameters/{name}"
                if key in handle:
                    parameters[name] = float(np.asarray(handle[key]).reshape(-1)[0])
        sample = manifest.get("sample", {})
        state_id = _decode(sample.get("structural_state_id", "unknown"))
        records.append(
            FEMArchiveRecord(
                path=path,
                sample_id=_decode(sample.get("sample_id", path.stem)),
                structural_state_id=state_id,
                trajectory_id=_decode(sample.get("trajectory_id", path.stem)),
                split_group=_decode(sample.get("state_split_group", state_id)),
                metadata=manifest,
                parameters=parameters,
            )
        )
    return FEMArchiveIndex(tuple(records), run_manifest)


def _state_sort_key(state_id: str) -> tuple[int, Any]:
    try:
        return (0, int(state_id))
    except (TypeError, ValueError):
        return (1, str(state_id))


def trajectory_key(record: FEMArchiveRecord) -> str:
    """Environment realization id shared by every state at the same index."""
    suffix = str(record.trajectory_id).rsplit("_", 1)[-1]
    return suffix if suffix else str(record.trajectory_id)


@dataclass(frozen=True)
class AdGroup:
    group_id: int
    value: float
    state_ids: tuple[str, ...]


@dataclass(frozen=True)
class AdGroupMap:
    parameter: str
    groups: tuple[AdGroup, ...]
    state_to_group: dict[str, int]
    state_to_value: dict[str, float]

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def group_ids(self) -> tuple[int, ...]:
        return tuple(group.group_id for group in self.groups)

    def group_value(self, group_id: int) -> float:
        return self.groups[group_id].value

    def group_states(self, group_id: int) -> tuple[str, ...]:
        return self.groups[group_id].state_ids


def derive_ad_groups(
    index: FEMArchiveIndex,
    parameter: str = "Ad_rel",
    expected_count: int = 0,
    tolerance: float = 1e-9,
) -> AdGroupMap:
    """Group structural states by one scalar FEM parameter.

    The KW51 archive stores a 5x5 parameter grid. Under the static seasonal
    displacement observable the Id_rel axis is effectively inactive, so the
    identifiable candidate set is the Ad_rel axis. Group ids follow the sorted
    unique parameter values.
    """
    if tolerance <= 0.0:
        raise ValueError("group tolerance must be positive")
    state_values: dict[str, float] = {}
    for record in index.records:
        value = record.parameters.get(parameter)
        if value is None:
            raise ValueError(
                f"{record.path}: missing state/parameters/{parameter} for ad grouping"
            )
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{record.path}: non-finite {parameter}={value!r}")
        previous = state_values.get(record.structural_state_id)
        if previous is None:
            state_values[record.structural_state_id] = value
        elif abs(previous - value) > tolerance:
            raise ValueError(
                f"structural_state_id={record.structural_state_id} has inconsistent "
                f"{parameter}: {previous} vs {value}"
            )
    unique_values: list[float] = []
    for value in sorted(state_values.values()):
        if not unique_values or abs(value - unique_values[-1]) > tolerance:
            unique_values.append(value)
    if len(unique_values) < 2:
        raise ValueError(f"{parameter} grouping needs at least two distinct values")
    if expected_count and len(unique_values) != int(expected_count):
        raise ValueError(
            f"{parameter} has {len(unique_values)} groups, expected {expected_count}"
        )
    members: dict[int, list[str]] = {group_id: [] for group_id in range(len(unique_values))}
    state_to_group: dict[str, int] = {}
    state_to_value: dict[str, float] = {}
    for state_id in sorted(state_values, key=_state_sort_key):
        value = state_values[state_id]
        group_id = int(np.argmin([abs(value - unique) for unique in unique_values]))
        if abs(value - unique_values[group_id]) > tolerance:
            raise ValueError(
                f"structural_state_id={state_id} value {value} does not match "
                f"any {parameter} group"
            )
        members[group_id].append(state_id)
        state_to_group[state_id] = group_id
        state_to_value[state_id] = value
    groups = tuple(
        AdGroup(
            group_id,
            unique_values[group_id],
            tuple(sorted(members[group_id], key=_state_sort_key)),
        )
        for group_id in range(len(unique_values))
    )
    return AdGroupMap(parameter, groups, state_to_group, state_to_value)


def read_fem_record(record: FEMArchiveRecord) -> dict[str, np.ndarray | dict[str, Any]]:
    """Read one complete FEM field and the geometry needed by the operator."""
    import h5py

    with h5py.File(record.path, "r") as handle:
        field = np.asarray(handle["responses/fem/disp_at_acquisition"], dtype=np.float64)
        valid = np.asarray(handle["responses/fem/disp_at_acquisition_valid_mask"], dtype=bool)
        node_coords = np.asarray(handle["mesh/node_coords"], dtype=np.float64)
        node_ids = np.asarray(handle["mesh/node_ids"], dtype=np.int64).reshape(-1)
        deck_mask = np.asarray(handle["mesh/deck_node_mask"], dtype=bool).reshape(-1)
        temperature = np.asarray(handle["environment/temperature_C"], dtype=np.float32).reshape(-1)
        delta_days = np.asarray(handle["environment/delta_days"], dtype=np.float32).reshape(-1)
        element_connectivity = np.asarray(handle["mesh/element_connectivity"], dtype=np.int64)
        element_type = np.asarray(handle["mesh/element_type"], dtype=np.int64).reshape(-1)
        element_section = np.asarray(handle["mesh/element_section"], dtype=np.int64).reshape(-1)
        sample_seed = float(np.asarray(handle["sample/random_seed"]).reshape(-1)[0])
        parameters = {
            name: float(np.asarray(handle[f"state/parameters/{name}"]).reshape(-1)[0])
            for name in ("Es_rel", "hb_rel", "Id_rel", "Ad_rel", "kxx", "fixed", "addmass")
            if f"state/parameters/{name}" in handle
        }
    if field.ndim != 3 or field.shape[-1] != 3:
        raise ValueError(f"{record.path}: expected disp_at_acquisition [T,N,3], got {field.shape}")
    if valid.shape != field.shape:
        raise ValueError(f"{record.path}: displacement validity mask shape {valid.shape} != {field.shape}")
    return {
        "field": field,
        "valid": valid,
        "node_coords": node_coords,
        "node_ids": node_ids,
        "deck_mask": deck_mask,
        "temperature": temperature,
        "delta_days": delta_days,
        "element_connectivity": element_connectivity,
        "element_type": element_type,
        "element_section": element_section,
        "random_seed": np.asarray(sample_seed, dtype=np.float64),
        "parameters": parameters,
    }


@dataclass(frozen=True)
class RegionConfig:
    los_vector: tuple[float, float, float] = (-0.534, -0.105, 0.839)
    reference_mode: str = "region_mean"
    unit_scale: float = 1000.0
    noise_std_mm: float = 1.5
    point_bias_std_mm: float = 0.5
    missing_probability: float = 0.03
    block_dropout_probability: float = 0.10
    block_dropout_fraction: float = 0.15
    min_quality: float = 0.05
    normalization_max_records: int = 512
    input_mode: str = "region"
    region_grid_size_m: float = 10.0
    coordinate_scale_to_m: float = 1.0
    sparse_point_count: int = 80
    min_points_per_region: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_mode not in ("node", "region"):
            raise ValueError("input_mode must be node or region")
        if not np.isfinite(self.region_grid_size_m) or self.region_grid_size_m <= 0:
            raise ValueError("region_grid_size_m must be positive and finite")
        if not np.isfinite(self.coordinate_scale_to_m) or self.coordinate_scale_to_m <= 0:
            raise ValueError("coordinate_scale_to_m must be positive and finite")
        if self.sparse_point_count < 1 or self.min_points_per_region < 1:
            raise ValueError("grid size, sparse point count and minimum cell count must be positive")
        if np.linalg.norm(np.asarray(self.los_vector, dtype=np.float64)) <= 1.0e-12:
            raise ValueError("los_vector must not be zero")
        if self.unit_scale <= 0.0:
            raise ValueError("unit_scale must be positive")
        for name in ("missing_probability", "block_dropout_probability", "block_dropout_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.reference_mode not in ("none", "region_mean", "first_region"):
            raise ValueError("reference_mode must be none, region_mean or first_region")


@dataclass(frozen=True)
class RegionMap:
    grid: SquareGrid
    effective_node_count: np.ndarray

    @property
    def point_features(self) -> np.ndarray:
        return self.grid.centers

    @property
    def valid_mask(self) -> np.ndarray:
        return self.grid.present_mask


def build_region_map(node_coords: np.ndarray, deck_mask: np.ndarray, config: RegionConfig) -> RegionMap:
    """Create fixed square cells from the full deck geometry."""
    coords = np.asarray(node_coords, dtype=np.float64)
    mask = np.asarray(deck_mask, dtype=bool).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] < 2 or mask.shape != (coords.shape[0],):
        raise ValueError("node_coords and deck_mask have incompatible shapes")
    usable = np.flatnonzero(mask & np.all(np.isfinite(coords[:, :2]), axis=1))
    metric_coords = coords[:, :2] * float(config.coordinate_scale_to_m)
    grid = build_square_grid(metric_coords[usable], config.region_grid_size_m)
    cell_ids = grid.assign(metric_coords)
    effective = np.bincount(cell_ids[usable], minlength=grid.region_count).astype(np.int64)
    return RegionMap(grid, effective)


def _normalise_los(los_vector: Iterable[float]) -> np.ndarray:
    los = np.asarray(tuple(los_vector), dtype=np.float64).reshape(-1)
    if los.size != 3 or not np.all(np.isfinite(los)):
        raise ValueError("los_vector must contain three finite components")
    return los / np.linalg.norm(los)


def materialize_query(
    clean_values_mm: np.ndarray,
    clean_valid: np.ndarray,
    *,
    seed: int,
    config: RegionConfig,
) -> dict[str, np.ndarray]:
    """Generate one deterministic noisy/masked/quality-degraded query view."""
    values = np.asarray(clean_values_mm, dtype=np.float32)
    valid = np.asarray(clean_valid, dtype=bool).copy() & np.isfinite(values)
    rng = np.random.default_rng(int(seed))
    bias = rng.normal(0.0, float(config.point_bias_std_mm), size=(values.shape[1],)).astype(np.float32)
    observed = np.nan_to_num(values, nan=0.0) + bias[None, :]
    observed += rng.normal(0.0, float(config.noise_std_mm), size=values.shape).astype(np.float32)
    dropout = rng.random(values.shape) < float(config.missing_probability)
    valid &= ~dropout
    block_len = max(1, int(round(values.shape[0] * float(config.block_dropout_fraction))))
    if values.shape[0] > 1:
        for region in range(values.shape[1]):
            if rng.random() < float(config.block_dropout_probability):
                start = int(rng.integers(0, max(1, values.shape[0] - block_len + 1)))
                valid[start:min(values.shape[0], start + block_len), region] = False
    quality = np.ones_like(values, dtype=np.float32)
    quality *= np.float32(max(float(config.min_quality), 1.0 - float(config.noise_std_mm) / 10.0))
    quality[~valid] = 0.0
    observed[~valid] = 0.0
    return {
        "values": observed.astype(np.float32),
        "valid_mask": valid,
        "quality": quality,
    }


@dataclass(frozen=True)
class RegionalNormalization:
    mean_mm: float
    std_mm: float
    coordinate_min: np.ndarray
    coordinate_scale: np.ndarray
    fitted_state_ids: tuple[str, ...]
    scope: str = "global"
    region_offset_mm: np.ndarray | None = None
    residual_std_mm: float = 0.0
    time_feature_mode: str = "raw"
    temperature_mean: float = 0.0
    temperature_std: float = 1.0
    delta_days_mean: float = 0.0
    delta_days_std: float = 1.0

    def transform_values(self, values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.scope == "per_region_offset":
            if self.region_offset_mm is None or self.residual_std_mm <= 0.0:
                raise ValueError(
                    "per_region_offset normalization requires region_offset_mm and residual_std_mm"
                )
            offset = self.region_offset_mm.astype(np.float32)
            if offset.shape[-1] != values.shape[-1]:
                raise ValueError(
                    f"region offset width {offset.shape[-1]} does not match values width "
                    f"{values.shape[-1]}"
                )
            result = (values - offset[None, :]) / np.float32(self.residual_std_mm)
        else:
            result = (values - self.mean_mm) / self.std_mm
        return np.where(np.asarray(valid_mask, dtype=bool), result, 0.0).astype(np.float32)

    def transform_time_features(
        self, temperature: np.ndarray, delta_days: np.ndarray
    ) -> np.ndarray:
        temperature = np.nan_to_num(
            np.asarray(temperature, dtype=np.float32).reshape(-1), nan=0.0
        )
        delta_days = np.nan_to_num(
            np.asarray(delta_days, dtype=np.float32).reshape(-1), nan=0.0
        )
        if self.time_feature_mode == "disabled":
            return np.zeros((temperature.size, 2), dtype=np.float32)
        if self.time_feature_mode == "standardized":
            temperature = (temperature - self.temperature_mean) / self.temperature_std
            delta_days = (delta_days - self.delta_days_mean) / self.delta_days_std
        return np.c_[temperature, delta_days].astype(np.float32)

    def transform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        return ((np.asarray(coordinates, dtype=np.float32) - self.coordinate_min) / self.coordinate_scale).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "kw51_regional_normalization_v2",
            "scope": self.scope,
            "mean_mm": float(self.mean_mm),
            "std_mm": float(self.std_mm),
            "coordinate_min": self.coordinate_min.astype(float).tolist(),
            "coordinate_scale": self.coordinate_scale.astype(float).tolist(),
            "fitted_state_ids": list(self.fitted_state_ids),
            "time_feature_mode": self.time_feature_mode,
            "temperature_mean": float(self.temperature_mean),
            "temperature_std": float(self.temperature_std),
            "delta_days_mean": float(self.delta_days_mean),
            "delta_days_std": float(self.delta_days_std),
        }
        if self.scope == "per_region_offset":
            payload["region_offset_mm"] = (
                None if self.region_offset_mm is None
                else self.region_offset_mm.astype(float).tolist()
            )
            payload["residual_std_mm"] = float(self.residual_std_mm)
        return payload


def _sample_normalization_records(
    records: list[FEMArchiveRecord], max_records: int
) -> list[FEMArchiveRecord]:
    """Select normalization records with deterministic coverage across states.

    Archive ordering is grouped by structural state, so ``records[:N]`` can
    silently fit the global scale on only the first few states. Round-robin
    selection gives every state a chance before consuming additional records.
    """
    if max_records < 1:
        return list(records)
    by_state: dict[str, list[FEMArchiveRecord]] = defaultdict(list)
    for record in records:
        by_state[record.structural_state_id].append(record)
    state_ids = sorted(by_state, key=_state_sort_key)
    selected: list[FEMArchiveRecord] = []
    offset = 0
    while len(selected) < max_records:
        added = False
        for state_id in state_ids:
            items = by_state[state_id]
            if offset < len(items):
                selected.append(items[offset])
                added = True
                if len(selected) >= max_records:
                    break
        if not added:
            break
        offset += 1
    return selected


def fit_regional_normalization(
    source: "FEMRegionalSource",
    records: Iterable[FEMArchiveRecord],
    max_records: int = 512,
    *,
    per_region_offset: bool = False,
    records_per_state: int = 0,
    time_feature_mode: str = "raw",
) -> RegionalNormalization:
    records = list(records)
    if not records:
        raise ValueError("No records available for normalization")
    if per_region_offset:
        by_state: dict[str, list[FEMArchiveRecord]] = defaultdict(list)
        for record in records:
            by_state[record.structural_state_id].append(record)
        sampled: list[FEMArchiveRecord] = []
        for state_id in sorted(by_state, key=_state_sort_key):
            items = by_state[state_id]
            if records_per_state > 0:
                items = items[: int(records_per_state)]
            sampled.extend(items)
    elif max_records > 0:
        sampled = _sample_normalization_records(records, int(max_records))
    else:
        sampled = records
    views: list[dict[str, np.ndarray]] = []
    state_ids: set[str] = set()
    for record in sampled:
        view = source.clean_view(record)
        views.append(view)
        state_ids.add(record.structural_state_id)
    if not views:
        raise ValueError("No valid FEM regional values available for normalization")
    finite_values = [view["values"][view["valid_mask"]] for view in views]
    finite_values = [np.asarray(value, dtype=np.float64) for value in finite_values if value.size]
    if not finite_values:
        raise ValueError("No valid FEM regional values available for normalization")
    flattened = np.concatenate(finite_values)
    mean_mm = float(np.mean(flattened))
    std_mm = max(float(np.std(flattened)), 1.0e-6)
    region_offset = None
    residual_std = 0.0
    if per_region_offset:
        stacked = np.stack(
            [np.nan_to_num(view["values"], nan=0.0).astype(np.float64) for view in views],
            axis=0,
        )
        masks = np.stack([np.asarray(view["valid_mask"], dtype=bool) for view in views], axis=0)
        counts = masks.sum(axis=(0, 1))
        region_offset = (stacked * masks).sum(axis=(0, 1)) / np.maximum(counts, 1.0)
        residuals = (stacked - region_offset[None, None, :]) * masks
        residual_std = max(float(np.std(residuals[masks])), 1.0e-6)
        mean_mm = 0.0
        std_mm = residual_std
    temperatures = np.concatenate(
        [np.asarray(view["temperature"], dtype=np.float64).reshape(-1) for view in views]
    )
    delta_days = np.concatenate(
        [np.asarray(view["delta_days"], dtype=np.float64).reshape(-1) for view in views]
    )
    temperatures = temperatures[np.isfinite(temperatures)]
    delta_days = delta_days[np.isfinite(delta_days)]
    temperature_mean = float(np.mean(temperatures)) if temperatures.size else 0.0
    temperature_std = max(float(np.std(temperatures)), 1.0e-6) if temperatures.size else 1.0
    delta_mean = float(np.mean(delta_days)) if delta_days.size else 0.0
    delta_std = max(float(np.std(delta_days)), 1.0e-6) if delta_days.size else 1.0
    coordinates = source.observation_coordinates[source.point_present_mask].astype(np.float32)
    coordinate_min = coordinates.min(axis=0)
    coordinate_scale = np.maximum(coordinates.max(axis=0) - coordinate_min, 1.0e-6)
    return RegionalNormalization(
        mean_mm=mean_mm,
        std_mm=std_mm,
        coordinate_min=coordinate_min,
        coordinate_scale=coordinate_scale,
        fitted_state_ids=tuple(sorted(state_ids, key=_state_sort_key)),
        scope="per_region_offset" if per_region_offset else "global",
        region_offset_mm=None if region_offset is None else region_offset.astype(np.float32),
        residual_std_mm=residual_std,
        time_feature_mode=time_feature_mode,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
        delta_days_mean=delta_mean,
        delta_days_std=delta_std,
    )


class FEMRegionalSource:
    """Lazy FEM source with fixed sparse sensors and shared region aggregation."""

    def __init__(self, index: FEMArchiveIndex, config: RegionConfig):
        self.index = index
        self.config = config
        first = read_fem_record(index.records[0])
        self.region_map = build_region_map(first["node_coords"], first["deck_mask"], config)
        deck_ids = first["node_ids"][first["deck_mask"]]
        point_count = min(int(config.sparse_point_count), len(deck_ids))
        if point_count < 1:
            raise ValueError("KW51 mesh contains no deck nodes for sparse observation")
        rng = np.random.default_rng(int(config.seed))
        self.sample_node_ids = np.sort(rng.choice(deck_ids, size=point_count, replace=False))
        self.sample_point_coordinates = self._sample_coordinates(first)
        self.point_present_mask = (
            self.region_map.grid.present_mask.copy()
            if config.input_mode == "region"
            else np.ones(point_count, dtype=bool)
        )
        self.observation_coordinates = (
            self.region_map.point_features
            if config.input_mode == "region"
            else self.sample_point_coordinates
        ).astype(np.float32)
        self.region_count = int(self.observation_coordinates.shape[0])
        assigned_cells = self.region_map.grid.assign(self.sample_point_coordinates)
        self.sampled_points_per_region = np.bincount(
            assigned_cells[assigned_cells >= 0],
            minlength=self.region_map.grid.region_count,
        ).astype(np.int64)
        self._clean_cache: dict[str, dict[str, np.ndarray]] = {}
        self._state_cache: dict[str, dict[str, np.ndarray]] = {}
        self._group_totals: dict[
            int, tuple[np.ndarray, np.ndarray, tuple[str, ...], int, np.ndarray, np.ndarray]
        ] = {}
        self._group_traj_totals: dict[
            tuple[int, str], tuple[np.ndarray, np.ndarray, int]
        ] = {}
        self._group_state_totals: dict[
            tuple[int, str], tuple[np.ndarray, np.ndarray, int]
        ] = {}
        self._group_state_traj_totals: dict[
            tuple[int, str, str], tuple[np.ndarray, np.ndarray, int]
        ] = {}

    def _sample_coordinates(self, arrays: dict[str, Any]) -> np.ndarray:
        id_to_index = {int(node_id): index for index, node_id in enumerate(arrays["node_ids"])}
        try:
            indices = np.asarray([id_to_index[int(node_id)] for node_id in self.sample_node_ids])
        except KeyError as exc:
            raise ValueError(f"sampled sensor node {exc.args[0]} is absent from a KW51 record") from exc
        coordinates = (
            np.asarray(arrays["node_coords"], dtype=np.float64)[indices, :2]
            * float(self.config.coordinate_scale_to_m)
        )
        if not np.isfinite(coordinates).all():
            raise ValueError("sampled KW51 sensor coordinates must be finite")
        return coordinates

    def _observation_from_sparse(
        self, values: np.ndarray, valid: np.ndarray, quality: np.ndarray, coordinates: np.ndarray
    ) -> dict[str, np.ndarray]:
        if self.config.input_mode == "region":
            output, output_valid, output_quality, coverage = aggregate_sparse_observations(
                values,
                valid,
                quality,
                coordinates,
                self.region_map.grid,
                minimum_points=self.config.min_points_per_region,
            )
            present = self.region_map.grid.present_mask.copy()
            output_coordinates = self.region_map.grid.centers
        else:
            output = np.asarray(values, dtype=np.float32).copy()
            output_valid = np.asarray(valid, dtype=bool).copy()
            output_quality = np.asarray(quality, dtype=np.float32).copy()
            coverage = output_valid.astype(np.float32)
            present = np.ones(output.shape[1], dtype=bool)
            output_coordinates = np.asarray(coordinates, dtype=np.float32)

        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output_valid &= np.isfinite(output)
        if self.config.reference_mode == "region_mean":
            denominator = output_valid.sum(axis=1).clip(min=1)
            reference = (output * output_valid).sum(axis=1) / denominator
        elif self.config.reference_mode == "first_region":
            first_valid = output_valid.argmax(axis=1)
            reference = output[np.arange(len(output)), first_valid]
            reference = np.where(output_valid.any(axis=1), reference, 0.0)
        else:
            reference = np.zeros(output.shape[0], dtype=np.float32)
        # ``clean_view`` applies the FEM-to-mm conversion before sparse
        # observations are generated. Noise, aggregation and reference removal
        # therefore stay in millimetres and must not rescale a second time.
        output = output - reference[:, None]
        output[~output_valid] = 0.0
        output_quality = np.where(output_valid, output_quality, 0.0).astype(np.float32)
        return {
            "values": output.astype(np.float32),
            "valid_mask": output_valid,
            "quality": output_quality,
            "region_coords": output_coordinates.astype(np.float32),
            "point_present_mask": present,
            "coverage": coverage.astype(np.float32),
        }

    def clean_view(self, record: FEMArchiveRecord) -> dict[str, np.ndarray]:
        cached = self._clean_cache.get(record.sample_id)
        if cached is not None:
            return cached
        arrays = read_fem_record(record)
        id_to_index = {int(node_id): index for index, node_id in enumerate(arrays["node_ids"])}
        try:
            indices = np.asarray([id_to_index[int(node_id)] for node_id in self.sample_node_ids])
        except KeyError as exc:
            raise ValueError(f"sampled sensor node {exc.args[0]} is absent from {record.path}") from exc
        coordinates = (
            np.asarray(arrays["node_coords"], dtype=np.float64)[indices, :2]
            * float(self.config.coordinate_scale_to_m)
        )
        los = _normalise_los(self.config.los_vector)
        point_values = np.einsum("tnc,c->tn", arrays["field"][:, indices], los)
        point_values = np.nan_to_num(point_values, nan=0.0, posinf=0.0, neginf=0.0)
        point_values = (point_values * float(self.config.unit_scale)).astype(np.float32)
        point_valid = arrays["valid"][:, indices].all(axis=2) & np.isfinite(point_values)
        point_quality = point_valid.astype(np.float32)
        observed = self._observation_from_sparse(point_values, point_valid, point_quality, coordinates)
        result = {
            **observed,
            "point_values": point_values,
            "point_valid": point_valid,
            "point_quality": point_quality,
            "point_coordinates": coordinates,
            "temperature": np.asarray(arrays["temperature"], dtype=np.float32),
            "delta_days": np.asarray(arrays["delta_days"], dtype=np.float32),
        }
        self._clean_cache[record.sample_id] = result
        return result

    def aggregate_state(
        self,
        state_id: str,
        *,
        exclude_sample_id: str | None = None,
        exclude_trajectory_key: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Aggregate a structural state with symmetric gallery exclusions."""
        cache_key = (
            f"{state_id}::{exclude_sample_id or ''}::"
            f"{exclude_trajectory_key or ''}"
        )
        cached = self._state_cache.get(cache_key)
        if cached is not None:
            return cached
        records = [
            record for record in self.index.records
            if record.structural_state_id == state_id
            and record.sample_id != exclude_sample_id
            and trajectory_key(record) != exclude_trajectory_key
        ]
        if not records:
            raise KeyError(
                f"No candidate trajectories remain for structural_state_id={state_id} "
                f"after excluding sample_id={exclude_sample_id} "
                f"trajectory_key={exclude_trajectory_key}"
            )
        views = [self.clean_view(record) for record in records]
        stack = np.stack([np.nan_to_num(view["values"], nan=0.0) for view in views], axis=0)
        masks = np.stack([view["valid_mask"] for view in views], axis=0)
        weights = masks.astype(np.float32)
        denom = weights.sum(axis=0)
        values = (stack * weights).sum(axis=0) / np.maximum(denom, 1.0)
        valid = denom > 0
        quality_stack = np.stack([view["quality"] for view in views], axis=0)
        quality = (quality_stack * weights).sum(axis=0) / np.maximum(denom, 1.0)
        coverage = np.stack([view["coverage"] for view in views], axis=0).mean(axis=0)
        first = views[0]
        result = {
            "values": values.astype(np.float32),
            "valid_mask": valid,
            "quality": np.where(valid, quality, 0.0).astype(np.float32),
            "coverage": coverage.astype(np.float32),
            "region_coords": self.observation_coordinates.astype(np.float32),
            "point_present_mask": self.point_present_mask.copy(),
            "temperature": first["temperature"],
            "delta_days": first["delta_days"],
            "trajectory_count": np.asarray(len(records), dtype=np.int64),
        }
        self._state_cache[cache_key] = result
        return result

    def _ensure_group_totals(self, group_id: int, state_ids: Iterable[str]) -> None:
        state_tuple = tuple(sorted(set(state_ids), key=_state_sort_key))
        cached = self._group_totals.get(group_id)
        if cached is not None:
            if cached[2] != state_tuple:
                raise ValueError(
                    f"Ad group {group_id} was already built with states {cached[2]}, "
                    f"got {state_tuple}"
                )
            return
        records = [
            record for record in self.index.records
            if record.structural_state_id in set(state_tuple)
        ]
        if not records:
            raise KeyError(f"No FEM records for Ad group {group_id}: {state_tuple}")
        records.sort(
            key=lambda record: (
                _state_sort_key(record.structural_state_id),
                trajectory_key(record),
            )
        )
        total_values: np.ndarray | None = None
        total_mask: np.ndarray | None = None
        first_temperature: np.ndarray | None = None
        first_delta_days: np.ndarray | None = None
        for record in records:
            view = self.clean_view(record)
            values = np.nan_to_num(view["values"], nan=0.0).astype(np.float64)
            mask = np.asarray(view["valid_mask"], dtype=np.float64)
            if total_values is None:
                total_values = np.zeros_like(values)
                total_mask = np.zeros_like(mask)
                first_temperature = np.asarray(view["temperature"], dtype=np.float32)
                first_delta_days = np.asarray(view["delta_days"], dtype=np.float32)
            weighted = values * mask
            total_values += weighted
            total_mask += mask
            key = (group_id, trajectory_key(record))
            partial = self._group_traj_totals.get(key)
            if partial is None:
                self._group_traj_totals[key] = (weighted, mask, 1)
            else:
                self._group_traj_totals[key] = (
                    partial[0] + weighted,
                    partial[1] + mask,
                    partial[2] + 1,
                )
            state_key = (group_id, record.structural_state_id)
            state_partial = self._group_state_totals.get(state_key)
            if state_partial is None:
                self._group_state_totals[state_key] = (weighted, mask, 1)
            else:
                self._group_state_totals[state_key] = (
                    state_partial[0] + weighted,
                    state_partial[1] + mask,
                    state_partial[2] + 1,
                )
            state_traj_key = (
                group_id,
                record.structural_state_id,
                trajectory_key(record),
            )
            state_traj_partial = self._group_state_traj_totals.get(state_traj_key)
            if state_traj_partial is None:
                self._group_state_traj_totals[state_traj_key] = (weighted, mask, 1)
            else:
                self._group_state_traj_totals[state_traj_key] = (
                    state_traj_partial[0] + weighted,
                    state_traj_partial[1] + mask,
                    state_traj_partial[2] + 1,
                )
        assert total_values is not None and total_mask is not None
        assert first_temperature is not None and first_delta_days is not None
        self._group_totals[group_id] = (
            total_values,
            total_mask,
            state_tuple,
            len(records),
            first_temperature,
            first_delta_days,
        )

    def aggregate_group(
        self,
        group_id: int,
        state_ids: Iterable[str],
        *,
        exclude_trajectory_key: str | None = None,
        exclude_state_id: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Aggregate every trajectory of one Ad group with symmetric exclusion.

        The query's environment realization is shared by all states at the same
        trajectory index, so excluding it from every candidate (positive and
        negative) prevents the gallery from leaking the query's environment.
        ``exclude_state_id`` additionally removes every trajectory of one
        structural state, which is the open-set unseen-state setting. When both
        exclusions are used, their union is removed exactly once.
        """
        self._ensure_group_totals(group_id, state_ids)
        total_values, total_mask, _, record_count, temperature, delta_days = (
            self._group_totals[group_id]
        )
        values = total_values.copy()
        mask = total_mask.copy()
        count = record_count
        if exclude_state_id:
            partial = self._group_state_totals.get((group_id, exclude_state_id))
            if partial is not None:
                values -= partial[0]
                mask -= partial[1]
                count -= partial[2]
        if exclude_trajectory_key:
            partial = self._group_traj_totals.get((group_id, exclude_trajectory_key))
            if partial is not None:
                values -= partial[0]
                mask -= partial[1]
                count -= partial[2]
        if exclude_state_id and exclude_trajectory_key:
            partial = self._group_state_traj_totals.get(
                (group_id, exclude_state_id, exclude_trajectory_key)
            )
            if partial is not None:
                values += partial[0]
                mask += partial[1]
                count += partial[2]
        if count <= 0:
            raise KeyError(
                f"No candidate trajectories remain for Ad group {group_id} after "
                f"excluding state={exclude_state_id!r} "
                f"trajectory_key={exclude_trajectory_key!r}"
            )
        averaged = values / np.maximum(mask, 1.0e-12)
        valid = mask > 0
        averaged[~valid] = np.nan
        coverage = np.clip(mask / max(count, 1), 0.0, 1.0).astype(np.float32)
        return {
            "values": averaged.astype(np.float32),
            "valid_mask": valid,
            "quality": coverage,
            "coverage": coverage,
            "temperature": temperature,
            "delta_days": delta_days,
            "trajectory_count": np.asarray(count, dtype=np.int64),
        }

    def query_window(self, ref: "FEMWindowRef", realization: int) -> dict[str, np.ndarray]:
        view = self.clean_view(ref.record)
        seed = int(self.config.sparse_point_count * 1000003 + ref.record.metadata["sample"].get("random_seed", 0) + realization * 1009 + ref.start * 9176)
        point_slice = slice(ref.start, ref.start + ref.length)
        degraded = materialize_query(
            view["point_values"][point_slice],
            view["point_valid"][point_slice],
            seed=seed,
            config=self.config,
        )
        observed = self._observation_from_sparse(
            degraded["values"], degraded["valid_mask"], degraded["quality"],
            view["point_coordinates"],
        )
        degraded = {**observed, "point_values": degraded["values"], "point_valid": degraded["valid_mask"]}
        degraded["temperature"] = view["temperature"][ref.start:ref.start + ref.length]
        degraded["delta_days"] = view["delta_days"][ref.start:ref.start + ref.length]
        return degraded


@dataclass(frozen=True)
class FEMWindowRef:
    record: FEMArchiveRecord
    start: int
    length: int


@dataclass(frozen=True)
class FEMCandidatePair:
    query: FEMWindowRef
    candidate_state_id: str
    label: int
    realization: int
    candidate_kind: str = "state"
    candidate_group_id: int | None = None
    candidate_group_value: float | None = None


def make_fem_window_refs(records: Iterable[FEMArchiveRecord], length: int, stride: int) -> list[FEMWindowRef]:
    if length < 1 or stride < 1:
        raise ValueError("window length and stride must be positive")
    refs: list[FEMWindowRef] = []
    for record in records:
        steps = int(record.metadata.get("acquisition_count", 26))
        refs.extend(FEMWindowRef(record, start, length) for start in range(0, steps - length + 1, stride))
    return refs


def split_fem_records(
    index: FEMArchiveIndex,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    *,
    group_map: AdGroupMap | None = None,
    split_mode: str = "random",
    fold: int | None = None,
) -> dict[str, list[FEMArchiveRecord]]:
    if fold is not None and split_mode != "ad_group_stratified":
        raise ValueError("fold requires ad_group_stratified split mode")
    if split_mode == "ad_group_stratified":
        if group_map is None:
            raise ValueError("ad_group_stratified split requires an AdGroupMap")
        rng = np.random.default_rng(seed)
        train_states: set[str] = set()
        validation_states: set[str] = set()
        test_states: set[str] = set()
        for group in group_map.groups:
            states = list(group.state_ids)
            if fold is None:
                rng.shuffle(states)
                if len(states) < 3:
                    raise ValueError(
                        f"Ad group {group.group_id} needs at least three states for a "
                        f"train/validation/test stratified split, got {len(states)}"
                    )
                validation = states[0]
                test = states[1]
                train = states[2:]
            else:
                if not 0 <= int(fold) < len(states):
                    raise ValueError(
                        f"fold={fold} is out of range for Ad group {group.group_id} "
                        f"with {len(states)} states"
                    )
                local_rng = np.random.default_rng(int(seed) + 1009 * (group.group_id + 1))
                local_rng.shuffle(states)
                test = states[int(fold)]
                validation = states[(int(fold) + 1) % len(states)]
                train = [
                    state for state in states
                    if state not in {validation, test}
                ]
                if not train:
                    raise ValueError(
                        f"Ad group {group.group_id} fold {fold} leaves no training states"
                    )
            validation_states.add(validation)
            test_states.add(test)
            train_states.update(train)
        split = {"train": [], "validation": [], "test": []}
        for record in index.records:
            state_id = record.structural_state_id
            role = (
                "train" if state_id in train_states
                else "validation" if state_id in validation_states
                else "test"
            )
            split[role].append(record)
        if not split["validation"] or not split["test"]:
            raise ValueError("State-level split needs non-empty validation and test groups")
        return split
    if split_mode != "random":
        raise ValueError("split_mode must be random or ad_group_stratified")
    states = list(index.state_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(states)
    n_train = max(1, int(round(len(states) * train_fraction)))
    remaining = len(states) - n_train
    n_validation = max(1, int(round(remaining * validation_fraction / max(1.0 - train_fraction, 1.0e-6)))) if remaining else 0
    if n_train + n_validation >= len(states) and len(states) > 2:
        n_validation = max(1, len(states) - n_train - 1)
    train_states = set(states[:n_train])
    validation_states = set(states[n_train:n_train + n_validation])
    split = {"train": [], "validation": [], "test": []}
    for record in index.records:
        role = "train" if record.structural_state_id in train_states else "validation" if record.structural_state_id in validation_states else "test"
        split[role].append(record)
    if not split["validation"] or not split["test"]:
        raise ValueError("State-level split needs non-empty validation and test groups")
    return split


def make_fem_candidate_pairs(
    query_refs: list[FEMWindowRef],
    candidate_state_ids: Iterable[str],
    *,
    negative_candidates_per_window: int,
    realizations: int,
    seed: int,
) -> list[FEMCandidatePair]:
    candidates = sorted(set(candidate_state_ids))
    if len(candidates) < 2:
        raise ValueError("Candidate ranking requires at least two structural states")
    rng = np.random.default_rng(seed)
    pairs: list[FEMCandidatePair] = []
    for ref in query_refs:
        positive = ref.record.structural_state_id
        negatives = [state for state in candidates if state != positive]
        if negative_candidates_per_window >= len(negatives):
            chosen = negatives
        else:
            chosen = rng.choice(negatives, size=negative_candidates_per_window, replace=False).tolist()
        for realization in range(max(1, int(realizations))):
            pairs.append(FEMCandidatePair(ref, positive, 1, realization))
            pairs.extend(FEMCandidatePair(ref, state, 0, realization) for state in chosen)
    return pairs


def make_fem_group_pairs(
    query_refs: list[FEMWindowRef],
    group_map: AdGroupMap,
    *,
    realizations: int = 1,
) -> list[FEMCandidatePair]:
    """Build a complete Ad-group gallery for every query window."""
    if group_map.group_count < 2:
        raise ValueError("Ad-group ranking requires at least two groups")
    pairs: list[FEMCandidatePair] = []
    for ref in query_refs:
        positive_group = group_map.state_to_group[ref.record.structural_state_id]
        for realization in range(max(1, int(realizations))):
            for group in group_map.groups:
                pairs.append(
                    FEMCandidatePair(
                        query=ref,
                        candidate_state_id=f"ad_group_{group.group_id}",
                        label=1 if group.group_id == positive_group else 0,
                        realization=realization,
                        candidate_kind="group",
                        candidate_group_id=group.group_id,
                        candidate_group_value=group.value,
                    )
                )
    return pairs


class FEMRankingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pairs: list[FEMCandidatePair],
        source: FEMRegionalSource,
        normalization: RegionalNormalization,
        group_map: AdGroupMap | None = None,
        gallery_exclusion: str = "query_trajectory",
        candidate_galleries: dict[
            str | None, dict[int, dict[str, np.ndarray]]
        ] | None = None,
    ):
        self.pairs = pairs
        self.source = source
        self.normalization = normalization
        self.group_map = group_map
        if gallery_exclusion not in ("query_trajectory", "query_state"):
            raise ValueError(
                "gallery_exclusion must be query_trajectory or query_state"
            )
        if gallery_exclusion == "query_state" and group_map is None:
            raise ValueError("gallery_exclusion=query_state requires an AdGroupMap")
        self.gallery_exclusion = gallery_exclusion
        self.candidate_galleries = candidate_galleries
        if any(pair.candidate_kind == "group" for pair in pairs) and group_map is None:
            raise ValueError("group candidate pairs require an AdGroupMap")
        if candidate_galleries is not None and group_map is None:
            raise ValueError("frozen group galleries require an AdGroupMap")

    def __len__(self) -> int:
        return len(self.pairs)

    def _side(
        self,
        values: np.ndarray,
        valid: np.ndarray,
        quality: np.ndarray,
        temperature: np.ndarray,
        delta_days: np.ndarray,
        coverage: np.ndarray | None = None,
    ) -> dict[str, Any]:
        valid = np.asarray(valid, dtype=bool)
        coverage = valid.astype(np.float32) if coverage is None else np.asarray(coverage, dtype=np.float32)
        temperature = np.asarray(temperature, dtype=np.float32)
        delta_days = np.asarray(delta_days, dtype=np.float32)
        return {
            "values": self.normalization.transform_values(values, valid),
            "valid_mask": valid,
            "quality": np.asarray(quality, dtype=np.float32),
            "region_coords": self.normalization.transform_coordinates(self.source.observation_coordinates),
            "point_present_mask": self.source.point_present_mask.copy(),
            "coverage": coverage,
            "time_features": self.normalization.transform_time_features(temperature, delta_days),
            "temperature_valid_mask": np.isfinite(temperature) & np.isfinite(delta_days),
            "input_mode": self.source.config.input_mode,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        query = self.source.query_window(pair.query, pair.realization)
        start = pair.query.start
        stop = start + pair.query.length
        if pair.candidate_kind == "group":
            if self.group_map is None or pair.candidate_group_id is None:
                raise ValueError("group candidate pair is missing its AdGroupMap")
            group = self.group_map.groups[pair.candidate_group_id]
            if self.candidate_galleries is not None:
                query_state_id = pair.query.record.structural_state_id
                gallery_key = (
                    query_state_id
                    if self.gallery_exclusion == "query_state"
                    and query_state_id in self.candidate_galleries
                    else None
                )
                try:
                    candidate = self.candidate_galleries[gallery_key][group.group_id]
                except KeyError as exc:
                    raise KeyError(
                        f"Frozen candidate gallery is missing group={group.group_id} "
                        f"for query_state={query_state_id}"
                    ) from exc
            else:
                exclude_state_id = (
                    pair.query.record.structural_state_id
                    if self.gallery_exclusion == "query_state"
                    else None
                )
                candidate = self.source.aggregate_group(
                    group.group_id,
                    group.state_ids,
                    exclude_trajectory_key=trajectory_key(pair.query.record),
                    exclude_state_id=exclude_state_id,
                )
            candidate_temperature = query["temperature"]
            candidate_delta_days = query["delta_days"]
            candidate_trajectory_count = int(candidate["trajectory_count"])
        else:
            candidate = self.source.aggregate_state(
                pair.candidate_state_id,
                exclude_sample_id=pair.query.record.sample_id
                if pair.candidate_state_id == pair.query.record.structural_state_id
                else None,
                exclude_trajectory_key=trajectory_key(pair.query.record),
            )
            candidate_temperature = candidate["temperature"][start:stop]
            candidate_delta_days = candidate["delta_days"][start:stop]
            candidate_trajectory_count = int(candidate["trajectory_count"])
        candidate_side = self._side(
            candidate["values"][start:stop], candidate["valid_mask"][start:stop],
            candidate["quality"][start:stop], candidate_temperature, candidate_delta_days,
            candidate.get("coverage", candidate["valid_mask"])[start:stop],
        )
        query_side = self._side(
            query["values"], query["valid_mask"], query["quality"],
            query["temperature"], query["delta_days"], query.get("coverage"),
        )
        query_trajectory = trajectory_key(pair.query.record)
        if pair.candidate_kind == "group":
            exclusion_key = query_trajectory
            if self.gallery_exclusion == "query_state":
                exclusion_key = (
                    f"{pair.query.record.structural_state_id}|{query_trajectory}"
                )
        else:
            exclusion_key = query_trajectory
        metadata = {
            "query_state_id": pair.query.record.structural_state_id,
            "query_sample_id": pair.query.record.sample_id,
            "candidate_state_id": pair.candidate_state_id,
            "query_event": 0,
            "candidate_event": 0,
            "query_start": pair.query.start,
            "candidate_start": start,
            "query_trajectory_id": pair.query.record.trajectory_id,
            "query_trajectory_key": query_trajectory,
            "observation_realization_id": pair.realization,
            "candidate_trajectory_count": candidate_trajectory_count,
            "candidate_kind": pair.candidate_kind,
            "candidate_exclusion_key": exclusion_key,
            "gallery_exclusion": self.gallery_exclusion,
            "query_state_excluded": bool(
                pair.candidate_kind == "group"
                and self.gallery_exclusion == "query_state"
            ),
        }
        if self.group_map is not None:
            query_group = self.group_map.state_to_group.get(pair.query.record.structural_state_id)
            metadata["query_ad_group_id"] = query_group
            metadata["query_ad_rel"] = self.group_map.state_to_value.get(
                pair.query.record.structural_state_id
            )
        if pair.candidate_kind == "group":
            metadata["candidate_ad_group_id"] = pair.candidate_group_id
            metadata["candidate_ad_rel"] = pair.candidate_group_value
        return {"query": query_side, "candidate": candidate_side, "label": np.int64(pair.label), "metadata": metadata}


class FEMQueryBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: FEMRankingDataset, batch_size: int, seed: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0
        self._groups = self._make_groups()

    def _make_groups(self) -> list[list[int]]:
        groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for index, pair in enumerate(self.dataset.pairs):
            groups[(pair.query.record.sample_id, pair.query.start, pair.realization)].append(index)
        return list(groups.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        groups = [group[:] for group in self._groups]
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(groups)
        batch: list[int] = []
        for group in groups:
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        count = 0
        current = 0
        for group in self._groups:
            if current and current + len(group) > self.batch_size:
                count += 1
                current = 0
            if len(group) > self.batch_size:
                count += 1
                continue
            current += len(group)
            if current >= self.batch_size:
                count += 1
                current = 0
        return max(1, count + int(current > 0))


def collate_fem_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from ..contracts import WindowBatch

    def stack_side(side: str) -> WindowBatch:
        keys = (
            "values", "valid_mask", "quality", "region_coords",
            "point_present_mask", "coverage", "time_features", "temperature_valid_mask",
        )
        stacked = {
            key: torch.as_tensor(np.stack([item[side][key] for item in items]))
            for key in keys
        }
        modes = {str(item[side]["input_mode"]) for item in items}
        if len(modes) != 1:
            raise ValueError(f"a KW51 batch cannot mix input modes: {sorted(modes)}")
        return WindowBatch.from_observations(
            values=stacked["values"],
            valid_mask=stacked["valid_mask"],
            quality=stacked["quality"],
            coordinates=stacked["region_coords"],
            time_features=stacked["time_features"],
            point_present_mask=stacked["point_present_mask"],
            coverage=stacked["coverage"],
            temperature_valid_mask=stacked["temperature_valid_mask"].unsqueeze(-1),
            input_mode=modes.pop(),
            metadata=[item["metadata"] for item in items],
        )

    return {
        "query": stack_side("query"),
        "candidate": stack_side("candidate"),
        "label": torch.as_tensor([item["label"] for item in items], dtype=torch.float32),
        "metadata": [item["metadata"] for item in items],
    }

