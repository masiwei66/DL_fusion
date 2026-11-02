"""Region-stratified fuzzy bags for KW51 state identification.

A query is a fixed set of bridge regions. Each region carries its own variable
point bag, validity mask, quality, and coverage. Region identity is preserved
all the way to the model; points are pooled only inside their region.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import WindowBatch
from .fem_insar import AdGroupMap, FEMArchiveRecord, read_fem_record
from .fuzzy_range import (
    DeckGeometry,
    FuzzyNormalization,
    FuzzyQueryRef,
    FuzzyRangeSource,
)


@dataclass(frozen=True)
class RegionLayout:
    """Deterministic deck partition shared by every query."""

    mode: str
    rows: int
    cols: int
    s_edges: np.ndarray
    t_edges: np.ndarray
    centers_s: np.ndarray
    centers_t: np.ndarray

    def __post_init__(self) -> None:
        if self.mode not in ("longitudinal", "grid"):
            raise ValueError("region mode must be longitudinal or grid")
        if len(self.s_edges) != self.rows + 1:
            raise ValueError("s_edges must have rows+1 entries")
        if len(self.t_edges) != self.cols + 1:
            raise ValueError("t_edges must have cols+1 entries")
        if self.region_count != len(self.centers_s):
            raise ValueError("region centers must match region_count")

    @property
    def region_count(self) -> int:
        return int(self.rows) * int(self.cols)

    def bounds(self, index: int) -> tuple[float, float, float, float]:
        index = int(index)
        if not 0 <= index < self.region_count:
            raise IndexError(f"region index {index} out of range")
        row, col = divmod(index, self.cols)
        return (
            float(self.s_edges[row]),
            float(self.s_edges[row + 1]),
            float(self.t_edges[col]),
            float(self.t_edges[col + 1]),
        )

    def center(self, index: int) -> tuple[float, float]:
        return float(self.centers_s[index]), float(self.centers_t[index])

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "region_count": self.region_count,
            "s_edges": self.s_edges.tolist(),
            "t_edges": self.t_edges.tolist(),
            "centers_s": self.centers_s.tolist(),
            "centers_t": self.centers_t.tolist(),
        }


def _edges(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if high <= low:
        high = low + 1.0e-6
    edges = np.linspace(low, high, int(count) + 1, dtype=np.float64)
    # Ensure the final interval is closed even when a node sits exactly on high.
    edges[-1] = max(edges[-1], high)
    return edges


def build_region_layout(
    geometry: DeckGeometry,
    *,
    mode: str = "longitudinal",
    rows: int = 8,
    cols: int = 1,
) -> RegionLayout:
    """Build fixed longitudinal or 2-D grid regions on the deck surface."""
    if mode not in ("longitudinal", "grid"):
        raise ValueError("region mode must be longitudinal or grid")
    rows = int(rows)
    cols = 1 if mode == "longitudinal" else int(cols)
    if rows < 1 or cols < 1:
        raise ValueError("region rows and cols must be >= 1")
    s = np.asarray(geometry.deck_s, dtype=np.float64)
    t = np.asarray(geometry.deck_t, dtype=np.float64)
    s_edges = _edges(s, rows)
    t_edges = (
        np.asarray([-1.0, 1.0], dtype=np.float64)
        if mode == "longitudinal"
        else _edges(t, cols)
    )
    centers_s = np.zeros(rows * cols, dtype=np.float32)
    centers_t = np.zeros(rows * cols, dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            s_low, s_high = s_edges[row], s_edges[row + 1]
            t_low, t_high = t_edges[col], t_edges[col + 1]
            mask = (
                (s >= s_low)
                & ((s < s_high) if row < rows - 1 else (s <= s_high))
                & (t >= t_low)
                & ((t < t_high) if col < cols - 1 else (t <= t_high))
            )
            if not np.any(mask):
                centers_s[index] = 0.5 * (s_low + s_high)
                centers_t[index] = 0.5 * (t_low + t_high)
            else:
                centers_s[index] = float(np.mean(s[mask]))
                centers_t[index] = float(np.mean(t[mask]))
    return RegionLayout(
        mode=mode,
        rows=rows,
        cols=cols,
        s_edges=s_edges,
        t_edges=t_edges,
        centers_s=centers_s,
        centers_t=centers_t,
    )


def compute_dense_region_target(
    arrays: dict[str, object], geometry: DeckGeometry, layout: RegionLayout
) -> dict[str, object]:
    """Project a complete FEM field and area-average its valid deck response."""
    field = np.asarray(arrays["field"], dtype=np.float64)
    valid = np.asarray(arrays["valid"], dtype=bool)
    coords = np.asarray(arrays["node_coords"], dtype=np.float64)
    node_ids_raw = np.asarray(arrays["node_ids"]).reshape(-1)
    deck_mask = np.asarray(arrays["deck_mask"], dtype=bool).reshape(-1)
    connectivity = np.asarray(arrays["element_connectivity"], dtype=np.float64)
    element_type = np.asarray(arrays["element_type"], dtype=np.float64).reshape(-1)
    element_section = np.asarray(arrays["element_section"], dtype=np.float64).reshape(-1)
    if field.ndim != 3 or field.shape[-1] != 3 or valid.shape != field.shape:
        raise ValueError("dense region target requires field and valid arrays shaped [T,N,3]")
    node_count = field.shape[1]
    if coords.shape != (node_count, 3):
        raise ValueError("node_coords must have shape [N,3] matching the FEM field")
    if node_ids_raw.size != node_count or deck_mask.size != node_count:
        raise ValueError("node_ids and deck mask must match the FEM field node axis")
    if not np.all(np.isfinite(node_ids_raw)) or not np.all(node_ids_raw == np.floor(node_ids_raw)):
        raise ValueError("node_ids must contain finite integer identifiers")
    node_ids = node_ids_raw.astype(np.int64)
    if np.unique(node_ids).size != node_count:
        raise ValueError("node_ids must be unique for topology mapping")
    if connectivity.ndim != 2 or connectivity.shape[0] != element_type.size:
        raise ValueError("element connectivity and type arrays have incompatible shapes")
    if element_type.size != element_section.size:
        raise ValueError("element type and section arrays have incompatible shapes")
    if not np.all(np.isfinite(element_type)) or not np.all(np.isfinite(element_section)):
        raise ValueError("element type and section arrays must be finite")
    deck_indices = np.asarray(geometry.deck_indices, dtype=np.int64).reshape(-1)
    if np.any((deck_indices < 0) | (deck_indices >= node_count)):
        raise ValueError("deck geometry contains node indices outside the FEM field")

    selected = (element_type.astype(np.int64) == 3) & np.isin(
        element_section.astype(np.int64), [7, 8]
    )
    if not np.any(selected):
        raise ValueError("no audited shell4 deck/ballast surface elements are available")
    node_lookup = {int(node_id): index for index, node_id in enumerate(node_ids)}
    node_area_full = np.zeros(node_count, dtype=np.float64)
    surface_area = 0.0
    for element_index in np.flatnonzero(selected):
        raw_ids = connectivity[element_index]
        finite_ids = raw_ids[np.isfinite(raw_ids)]
        if finite_ids.size != 4 or not np.all(finite_ids == np.floor(finite_ids)):
            raise ValueError(
                f"shell4 surface element {element_index} does not have four integer node ids"
            )
        try:
            indices = np.asarray(
                [node_lookup[int(node_id)] for node_id in finite_ids], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError(
                f"surface connectivity references unknown node id {exc.args[0]}"
            ) from exc
        if not np.all(deck_mask[indices]):
            raise ValueError("surface connectivity includes nodes outside the deck mask")
        points = coords[indices]
        if not np.all(np.isfinite(points)):
            raise ValueError("surface element has non-finite node coordinates")
        area = 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
        area += 0.5 * np.linalg.norm(
            np.cross(points[2] - points[0], points[3] - points[0])
        )
        if not np.isfinite(area) or area <= 1.0e-12:
            raise ValueError(f"surface element {element_index} has invalid area")
        node_area_full[indices] += area / 4.0
        surface_area += float(area)

    if not np.all(np.isin(np.flatnonzero(node_area_full > 0.0), deck_indices)):
        raise ValueError("surface area nodes are missing from deck geometry")
    node_area = node_area_full[deck_indices]
    if not np.any(node_area > 0.0):
        raise ValueError("surface elements produced no area weights on deck geometry")

    rows = np.searchsorted(layout.s_edges, geometry.deck_s, side="right") - 1
    cols = np.searchsorted(layout.t_edges, geometry.deck_t, side="right") - 1
    rows = np.clip(rows, 0, layout.rows - 1)
    cols = np.clip(cols, 0, layout.cols - 1)
    region_indices = rows * layout.cols + cols
    region_area = np.bincount(
        region_indices, weights=node_area, minlength=layout.region_count
    ).astype(np.float64)
    if region_area.size != layout.region_count or np.any(region_area <= 0.0):
        raise ValueError("area-weighted target has a region with no usable surface area")

    los = np.asarray(geometry.los, dtype=np.float64).reshape(-1)
    if los.size != 3 or not np.all(np.isfinite(los)) or np.linalg.norm(los) <= 1.0e-12:
        raise ValueError("dense target LOS vector must contain three finite non-zero values")
    projected = np.einsum(
        "tnc,c->tn",
        np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0),
        los / np.linalg.norm(los),
    )
    projected_valid = valid.all(axis=2) & np.isfinite(field).all(axis=2)
    deck_values = projected[:, deck_indices]
    deck_valid = projected_valid[:, deck_indices]
    reference = np.where(deck_valid, deck_values, 0.0).sum(axis=1) / np.maximum(
        deck_valid.sum(axis=1), 1
    )
    deck_values_mm = np.where(
        deck_valid, (deck_values - reference[:, None]) * 1000.0, 0.0
    ).astype(np.float32)

    target = np.zeros((layout.region_count, field.shape[0]), dtype=np.float32)
    target_valid = np.zeros((layout.region_count, field.shape[0]), dtype=bool)
    area_coverage = np.zeros_like(target)
    for region in range(layout.region_count):
        members = region_indices == region
        weights = node_area[members]
        weighted_valid = deck_valid[:, members] * weights[None, :]
        denominator = weighted_valid.sum(axis=1)
        numerator = (deck_values_mm[:, members] * weighted_valid).sum(axis=1)
        target_valid[region] = denominator > 0.0
        target[region] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > 0.0,
        ).astype(np.float32)
        area_coverage[region] = np.clip(denominator / region_area[region], 0.0, 1.0)
    return {
        "target_mm": target,
        "valid_mask": target_valid,
        "region_area_m2": region_area.astype(np.float32),
        "area_coverage": area_coverage,
        "node_area_weights": node_area.astype(np.float32),
        "surface_area_m2": float(surface_area),
        "surface_element_count": int(selected.sum()),
        "surface_sections": [7, 8],
        "area_method": "shell4_3d_quad_area_lumped_equally_to_four_nodes",
        "deck_values_mm": deck_values_mm,
        "deck_valid_mask": deck_valid,
    }


class FuzzyRegionSource(FuzzyRangeSource):
    """Fuzzy bag sampler that emits one bag per fixed deck region."""

    def __init__(self, *args, region_cache_limit: int = 4096, **kwargs):
        super().__init__(*args, **kwargs)
        self._region_cache: OrderedDict[tuple, dict[str, object]] = OrderedDict()
        self._dense_region_cache: OrderedDict[tuple, dict[str, object]] = OrderedDict()
        self._region_cache_limit = int(region_cache_limit)

    def _dense_region_data(
        self, record: FEMArchiveRecord, layout: RegionLayout
    ) -> dict[str, object]:
        key = (record.sample_id, layout.mode, int(layout.rows), int(layout.cols))
        cached = self._dense_region_cache.get(key)
        if cached is not None:
            self._dense_region_cache.move_to_end(key)
            return cached
        arrays = read_fem_record(record)
        result = compute_dense_region_target(arrays, self.geometry, layout)
        result["temperature"] = np.nan_to_num(
            np.asarray(arrays["temperature"], dtype=np.float32), nan=0.0
        )
        result["delta_days"] = np.nan_to_num(
            np.asarray(arrays["delta_days"], dtype=np.float32), nan=0.0
        )
        self._dense_region_cache[key] = result
        self._dense_region_cache.move_to_end(key)
        while len(self._dense_region_cache) > self._region_cache_limit:
            self._dense_region_cache.popitem(last=False)
        return result

    def sample_region_query(
        self,
        record: FEMArchiveRecord,
        realization: int,
        layout: RegionLayout,
    ) -> dict[str, object]:
        key = (
            record.sample_id,
            int(realization),
            int(self.seed),
            layout.mode,
            int(layout.rows),
            int(layout.cols),
            int(self.config.region_point_budget),
            float(self.config.region_missing_probability),
            int(self.config.region_min_valid_points),
            int(self.config.point_interpolation_k),
            float(self.config.coordinate_jitter_m),
            float(self.config.noise_std_mm),
            float(self.config.point_bias_std_mm),
            float(self.config.missing_probability),
            float(self.config.block_dropout_probability),
            float(self.config.block_dropout_fraction),
            float(self.config.min_quality),
        )
        cached = self._region_cache.get(key)
        if cached is not None:
            self._region_cache.move_to_end(key)
            return cached
        rng = np.random.default_rng(self._record_seed(record, realization))
        dense = self._dense_region_data(record, layout)
        los_values = np.asarray(dense["deck_values_mm"])
        los_valid = np.asarray(dense["deck_valid_mask"], dtype=bool)
        temperature = np.asarray(dense["temperature"], dtype=np.float32)
        delta_days = np.asarray(dense["delta_days"], dtype=np.float32)
        geometry = self.geometry
        region_count = layout.region_count
        base_count = max(1, int(self.config.region_point_budget) // region_count)
        counts = np.full(region_count, base_count, dtype=np.int64)
        remainder = int(self.config.region_point_budget) - base_count * region_count
        if remainder > 0:
            counts[: min(remainder, region_count)] += 1
        dropped = rng.random(region_count) < self.config.region_missing_probability

        true_s_parts: list[np.ndarray] = []
        true_t_parts: list[np.ndarray] = []
        for index in range(region_count):
            s_low, s_high, t_low, t_high = layout.bounds(index)
            count = int(counts[index])
            true_s_parts.append(rng.uniform(s_low, s_high, count))
            true_t_parts.append(rng.uniform(t_low, t_high, count))
        true_s = np.clip(np.concatenate(true_s_parts), 0.0, 1.0)
        true_t = np.clip(np.concatenate(true_t_parts), -1.0, 1.0)
        point_region = np.repeat(np.arange(region_count), counts).astype(np.int64)
        true_xy = np.c_[
            geometry.x_min + true_s * geometry.deck_length_m,
            geometry.y_min + (true_t + 1.0) / 2.0 * geometry.deck_width_m,
        ]
        interpolated, point_valid, mapping_distance = self._interpolate(
            los_values, los_valid, true_xy
        )
        observed, valid, quality, observed_s, observed_t = self._degrade_observations(
            interpolated, point_valid, true_s, true_t, rng
        )

        dropped_points = dropped[point_region]
        if np.any(dropped_points):
            valid[:, dropped_points] = False
            observed[:, dropped_points] = 0.0
            quality[:, dropped_points] = 0.0
            point_valid[:, dropped_points] = False

        valid_any = valid.any(axis=0)
        region_present = np.zeros(region_count, dtype=bool)
        region_coverage = np.zeros(region_count, dtype=np.float32)
        region_valid_points = np.zeros(region_count, dtype=np.int64)
        for index in range(region_count):
            members = point_region == index
            if dropped[index] or not np.any(members):
                continue
            region_valid_points[index] = int(valid_any[members].sum())
            region_coverage[index] = float(valid[:, members].mean())
            region_present[index] = (
                region_valid_points[index] >= int(self.config.region_min_valid_points)
            )
        # Regions with too few valid points are treated as missing; their
        # points must not leak into the region pool.
        weak_regions = ~region_present & ~dropped
        if np.any(weak_regions):
            weak_points = weak_regions[point_region]
            valid[:, weak_points] = False
            observed[:, weak_points] = 0.0
            quality[:, weak_points] = 0.0
            point_valid[:, weak_points] = False
            region_coverage[weak_regions] = 0.0

        present_count = int(region_present.sum())
        mean_coverage = (
            float(region_coverage[region_present].mean()) if present_count else 0.0
        )
        mean_valid_points = (
            float(region_valid_points[region_present].mean())
            if present_count
            else 0.0
        )
        mean_quality = float(quality[valid].mean()) if valid.any() else 0.0
        total_points = int(true_s.size)
        max_region_count = max(int(counts.max()), 1)
        bag_features = np.asarray(
            [
                float(total_points),
                float(present_count) / max(region_count, 1),
                mean_coverage,
                mean_valid_points / max_region_count,
                mean_quality,
                float(np.log1p(total_points)),
                float(counts[region_present].mean()) / max_region_count
                if present_count
                else 0.0,
                float(np.mean(region_coverage >= 0.5)),
            ],
            dtype=np.float32,
        )
        result = {
            "values": observed,
            "valid_mask": valid,
            "quality": quality,
            "point_coords": np.c_[observed_s, observed_t].astype(np.float32),
            "point_present": np.ones(total_points, dtype=bool),
            "mapping_distance": mapping_distance,
            "mapping_valid": point_valid.any(axis=0),
            "time_features": np.c_[temperature, delta_days].astype(np.float32),
            "bag_features": bag_features,
            "region_index": point_region,
            "region_present": region_present,
            "region_coverage": region_coverage,
            "dense_region_target": dense["target_mm"],
            "dense_region_valid": dense["valid_mask"],
            "dense_region_area_m2": dense["region_area_m2"],
            "dense_region_area_coverage": dense["area_coverage"],
            "metadata": {
                "sample_id": record.sample_id,
                "structural_state_id": record.structural_state_id,
                "trajectory_id": record.trajectory_id,
                "observation_realization_id": int(realization),
                "point_count": total_points,
                "region_count": region_count,
                "region_layout": layout.to_dict(),
                "region_present": region_present.tolist(),
                "region_coverage": region_coverage.tolist(),
                "region_point_counts": counts.tolist(),
                "region_valid_points": region_valid_points.tolist(),
                "dense_target_area_method": dense["area_method"],
                "dense_target_surface_element_count": dense["surface_element_count"],
                "dense_target_surface_area_m2": dense["surface_area_m2"],
            },
        }
        self._region_cache[key] = result
        self._region_cache.move_to_end(key)
        while len(self._region_cache) > self._region_cache_limit:
            self._region_cache.popitem(last=False)
        return result


def estimate_region_values(
    values: np.ndarray,
    valid: np.ndarray,
    region_index: np.ndarray,
    region_present: np.ndarray,
    point_coords: np.ndarray,
    layout: RegionLayout,
    estimator: str = "mean",
    ridge: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one response series per region from its point bag.

    ``mean`` is the raw point-time mean. ``plane``/``quadratic`` fit a local
    spatial surface per acquisition and evaluate it at the region center, which
    reduces the sampling variance of sparse bags. The returned validity marks
    entries that had at least one valid observation.
    """
    if estimator not in ("mean", "plane", "quadratic"):
        raise ValueError("estimator must be mean, plane or quadratic")
    region_count = layout.region_count
    estimates = np.zeros((region_count, values.shape[0]), dtype=np.float64)
    estimate_valid = np.zeros((region_count, values.shape[0]), dtype=bool)
    for index in range(region_count):
        if not bool(region_present[index]):
            continue
        members = region_index == index
        if not np.any(members):
            continue
        point_values = values[:, members]
        point_valid = valid[:, members]
        if estimator == "mean":
            estimates[index] = np.where(point_valid, point_values, 0.0).sum(
                axis=1
            ) / np.maximum(point_valid.sum(axis=1), 1)
            estimate_valid[index] = point_valid.any(axis=1)
            continue
        s = point_coords[members, 0]
        t = point_coords[members, 1]
        if estimator == "plane":
            design = np.c_[np.ones(s.size), s, t]
        else:
            design = np.c_[np.ones(s.size), s, t, s * s, t * t, s * t]
        center_s, center_t = layout.center(index)
        if estimator == "plane":
            center_design = np.asarray([1.0, center_s, center_t])
        else:
            center_design = np.asarray(
                [1.0, center_s, center_t, center_s**2, center_t**2, center_s * center_t]
            )
        active_points = point_valid.any(axis=0)
        if int(active_points.sum()) < design.shape[1] + 1:
            raw = np.where(point_valid, point_values, 0.0).sum(axis=1) / np.maximum(
                point_valid.sum(axis=1), 1
            )
            estimates[index] = raw
            estimate_valid[index] = point_valid.any(axis=1)
            continue
        local_design = design[active_points]
        local_values = point_values[:, active_points]
        local_valid = point_valid[:, active_points]
        point_means = np.where(local_valid, local_values, 0.0).sum(
            axis=0
        ) / np.maximum(local_valid.sum(axis=0), 1)
        filled = np.where(local_valid, local_values, point_means[None, :])
        if estimator == "plane":
            coefficients = np.linalg.lstsq(
                local_design, filled.T, rcond=None
            )[0]
        else:
            gram = local_design.T @ local_design + float(ridge) * np.eye(
                local_design.shape[1]
            )
            coefficients = np.linalg.solve(
                gram, local_design.T @ filled.T
            )
        estimates[index] = center_design @ coefficients
        estimate_valid[index] = point_valid.any(axis=1)
        # Times with too few valid points fall back to the raw mean.
        sparse_times = point_valid.sum(axis=1) < design.shape[1] + 1
        if np.any(sparse_times):
            raw = np.where(point_valid, point_values, 0.0).sum(
                axis=1
            ) / np.maximum(point_valid.sum(axis=1), 1)
            estimates[index, sparse_times] = raw[sparse_times]
    return estimates, estimate_valid


class FuzzyRegionQueryDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        refs: list[FuzzyQueryRef],
        source: FuzzyRegionSource,
        normalization: FuzzyNormalization,
        layout: RegionLayout,
        group_map: AdGroupMap | None = None,
    ):
        self.refs = refs
        self.source = source
        self.normalization = normalization
        self.layout = layout
        self.group_map = group_map

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        result = self.source.sample_region_query(
            ref.record, ref.realization, self.layout
        )
        values = self.normalization.transform_values(
            result["values"], result["valid_mask"]
        )
        time_features = self.normalization.transform_time_features(
            result["time_features"][:, 0], result["time_features"][:, 1]
        )
        metadata = dict(result["metadata"])
        state_id = ref.record.structural_state_id
        raw_values = np.asarray(result["values"], dtype=np.float64)
        raw_valid = np.asarray(result["valid_mask"], dtype=bool)
        bag_mean_mm = (
            np.where(raw_valid, raw_values, 0.0).sum(axis=1)
            / np.maximum(raw_valid.sum(axis=1), 1)
        ).astype(np.float32)
        item: dict[str, object] = {
            "values": values,
            "valid_mask": result["valid_mask"],
            "quality": result["quality"],
            "point_coords": result["point_coords"],
            "point_present": result["point_present"],
            "mapping_distance": result["mapping_distance"],
            "mapping_valid": result["mapping_valid"],
            "time_features": time_features,
            "bag_features": result["bag_features"],
            "region_index": result["region_index"],
            "region_present": result["region_present"],
            "region_coverage": result["region_coverage"],
            "bag_mean_mm": bag_mean_mm,
            "target_fields": {
                "region_response_target": np.asarray(
                    result["dense_region_target"], dtype=np.float32
                ),
                "region_response_valid_mask": np.asarray(
                    result["dense_region_valid"], dtype=bool
                ),
                "region_area_weights": np.asarray(
                    result["dense_region_area_m2"], dtype=np.float32
                ),
                "region_area_coverage": np.asarray(
                    result["dense_region_area_coverage"], dtype=np.float32
                ),
            },
            "metadata": metadata,
        }
        if self.group_map is not None:
            group_id = int(self.group_map.state_to_group[state_id])
            item["ad_group_id"] = np.int64(group_id)
            item["ad_rel"] = np.float32(self.group_map.state_to_value[state_id])
            item["id_rel"] = np.float32(
                ref.record.parameters.get("Id_rel", 0.0)
            )
            metadata["ad_group_id"] = group_id
            metadata["ad_rel"] = float(self.group_map.state_to_value[state_id])
        return item


def _collate_target_fields(items: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    fields = [item.get("target_fields") for item in items]
    if not any(value is not None for value in fields):
        return {}
    if any(not isinstance(value, dict) for value in fields):
        raise ValueError("all region items must provide the same target_fields mapping")
    keys = set(fields[0])
    if any(set(value) != keys for value in fields[1:]):
        raise ValueError("region target_fields keys differ within a batch")
    result: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        stacked = np.stack([np.asarray(value[key]) for value in fields])
        if key.endswith("valid_mask"):
            result[key] = torch.from_numpy(stacked.astype(bool, copy=False)).bool()
        else:
            result[key] = torch.from_numpy(stacked.astype(np.float32, copy=False)).float()
    return result


def collate_fuzzy_region_queries(items: list[dict[str, object]]) -> WindowBatch:
    batch_size = len(items)
    window_length = int(np.asarray(items[0]["values"]).shape[0])
    max_points = max(int(np.asarray(item["values"]).shape[1]) for item in items)
    region_count = int(np.asarray(items[0]["region_present"]).shape[0])

    def pad_matrix(key: str, fill: float, dtype: np.dtype) -> np.ndarray:
        padded = np.full((batch_size, window_length, max_points), fill, dtype=dtype)
        for row, item in enumerate(items):
            value = np.asarray(item[key])
            padded[row, :, : value.shape[1]] = value
        return padded

    def pad_points(key: str, fill: float, dtype: np.dtype) -> np.ndarray:
        width = np.asarray(items[0][key]).shape[-1]
        padded = np.full((batch_size, max_points, width), fill, dtype=dtype)
        for row, item in enumerate(items):
            value = np.asarray(item[key])
            padded[row, : value.shape[0], :] = value
        return padded

    point_present = np.zeros((batch_size, max_points), dtype=bool)
    mapping_distance = np.zeros((batch_size, max_points), dtype=np.float32)
    mapping_valid = np.zeros((batch_size, max_points), dtype=bool)
    region_index = np.full((batch_size, max_points), -1, dtype=np.int64)
    for row, item in enumerate(items):
        count = int(np.asarray(item["point_present"]).shape[0])
        point_present[row, :count] = np.asarray(item["point_present"], dtype=bool)
        mapping_distance[row, :count] = np.asarray(
            item["mapping_distance"], dtype=np.float32
        )
        mapping_valid[row, :count] = np.asarray(item["mapping_valid"], dtype=bool)
        region_index[row, :count] = np.asarray(item["region_index"], dtype=np.int64)

    targets: dict[str, torch.Tensor] = {
        "bag_mean_mm": torch.from_numpy(
            np.stack(
                [np.asarray(item["bag_mean_mm"], dtype=np.float32) for item in items]
            )
        ).float(),
    }
    if "ad_group_id" in items[0]:
        targets["ad_group_id"] = torch.as_tensor(
            [int(item["ad_group_id"]) for item in items], dtype=torch.long
        )
        targets["ad_rel"] = torch.as_tensor(
            [float(item["ad_rel"]) for item in items], dtype=torch.float32
        )
        targets["id_rel"] = torch.as_tensor(
            [float(item["id_rel"]) for item in items], dtype=torch.float32
        )
    targets.update(_collate_target_fields(items))
    return WindowBatch(
        values=torch.from_numpy(pad_matrix("values", 0.0, np.float32)).float(),
        valid_mask=torch.from_numpy(pad_matrix("valid_mask", False, bool)).bool(),
        quality=torch.from_numpy(pad_matrix("quality", 0.0, np.float32)).float(),
        point_features=torch.from_numpy(
            pad_points("point_coords", 0.0, np.float32)
        ).float(),
        time_features=torch.from_numpy(
            np.stack([np.asarray(item["time_features"], dtype=np.float32) for item in items])
        ).float(),
        point_present_mask=torch.from_numpy(point_present).bool(),
        mapping_distance=torch.from_numpy(mapping_distance).float(),
        mapping_valid_mask=torch.from_numpy(mapping_valid).bool(),
        temperature_valid_mask=torch.ones(
            (batch_size, window_length, 1), dtype=torch.bool
        ),
        teacher_features={},
        targets=targets,
        metadata=[dict(item["metadata"]) for item in items],
        bag_features=torch.from_numpy(
            np.stack(
                [np.asarray(item["bag_features"], dtype=np.float32) for item in items]
            )
        ).float(),
        bag_feature_mask=torch.ones((batch_size, 1), dtype=torch.bool),
        region_index=torch.from_numpy(region_index).long(),
        region_present=torch.from_numpy(
            np.stack(
                [np.asarray(item["region_present"], dtype=bool) for item in items]
            )
        ).bool(),
        region_coverage=torch.from_numpy(
            np.stack(
                [np.asarray(item["region_coverage"], dtype=np.float32) for item in items]
            )
        ).float(),
    )


class FuzzyRegionSummaryDataset(Dataset[dict[str, object]]):
    """Region-level summary view: one estimated response series per region."""

    def __init__(
        self,
        refs: list[FuzzyQueryRef],
        source: FuzzyRegionSource,
        normalization: FuzzyNormalization,
        layout: RegionLayout,
        group_map: AdGroupMap | None = None,
        estimator: str = "mean",
    ):
        self.refs = refs
        self.source = source
        self.normalization = normalization
        self.layout = layout
        self.group_map = group_map
        self.estimator = estimator

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        query = self.source.sample_region_query(
            ref.record, ref.realization, self.layout
        )
        values = np.asarray(query["values"], dtype=np.float64)
        valid = np.asarray(query["valid_mask"], dtype=bool)
        region_index = np.asarray(query["region_index"], dtype=np.int64)
        present = np.asarray(query["region_present"], dtype=bool)
        coverage = np.asarray(query["region_coverage"], dtype=np.float64)
        coords = np.asarray(query["point_coords"], dtype=np.float64)
        estimates, estimate_valid = estimate_region_values(
            values,
            valid,
            region_index,
            present,
            coords,
            self.layout,
            estimator=self.estimator,
        )
        transformed = self.normalization.transform_values(
            estimates.T, estimate_valid.T
        )
        quality = np.where(
            estimate_valid.T, coverage[None, :], 0.0
        ).astype(np.float32)
        time_features = self.normalization.transform_time_features(
            query["time_features"][:, 0], query["time_features"][:, 1]
        )
        metadata = dict(query["metadata"])
        state_id = ref.record.structural_state_id
        bag_mean_mm = (
            np.where(estimate_valid, estimates, 0.0).sum(axis=0)
            / np.maximum(estimate_valid.sum(axis=0), 1)
        ).astype(np.float32)
        item: dict[str, object] = {
            "values": transformed,
            "valid_mask": estimate_valid.T,
            "quality": quality,
            "point_coords": np.c_[
                self.layout.centers_s, self.layout.centers_t
            ].astype(np.float32),
            "point_present": present,
            "mapping_distance": np.zeros(self.layout.region_count, dtype=np.float32),
            "mapping_valid": present,
            "time_features": time_features,
            "bag_features": query["bag_features"],
            "region_present": present,
            "region_coverage": coverage.astype(np.float32),
            "bag_mean_mm": bag_mean_mm,
            "target_fields": {
                "region_response_target": np.asarray(
                    query["dense_region_target"], dtype=np.float32
                ),
                "region_response_valid_mask": np.asarray(
                    query["dense_region_valid"], dtype=bool
                ),
                "region_area_weights": np.asarray(
                    query["dense_region_area_m2"], dtype=np.float32
                ),
                "region_area_coverage": np.asarray(
                    query["dense_region_area_coverage"], dtype=np.float32
                ),
            },
            "metadata": metadata,
        }
        if self.group_map is not None:
            group_id = int(self.group_map.state_to_group[state_id])
            item["ad_group_id"] = np.int64(group_id)
            item["ad_rel"] = np.float32(self.group_map.state_to_value[state_id])
            item["id_rel"] = np.float32(ref.record.parameters.get("Id_rel", 0.0))
            metadata["ad_group_id"] = group_id
            metadata["ad_rel"] = float(self.group_map.state_to_value[state_id])
        return item


def collate_fuzzy_region_summaries(
    items: list[dict[str, object]],
) -> WindowBatch:
    batch_size = len(items)
    window_length = int(np.asarray(items[0]["values"]).shape[0])
    region_count = int(np.asarray(items[0]["point_present"]).shape[0])
    targets: dict[str, torch.Tensor] = {
        "bag_mean_mm": torch.from_numpy(
            np.stack(
                [np.asarray(item["bag_mean_mm"], dtype=np.float32) for item in items]
            )
        ).float(),
    }
    if "ad_group_id" in items[0]:
        targets["ad_group_id"] = torch.as_tensor(
            [int(item["ad_group_id"]) for item in items], dtype=torch.long
        )
        targets["ad_rel"] = torch.as_tensor(
            [float(item["ad_rel"]) for item in items], dtype=torch.float32
        )
        targets["id_rel"] = torch.as_tensor(
            [float(item["id_rel"]) for item in items], dtype=torch.float32
        )
    targets.update(_collate_target_fields(items))
    return WindowBatch(
        values=torch.from_numpy(
            np.stack([np.asarray(item["values"], dtype=np.float32) for item in items])
        ).float(),
        valid_mask=torch.from_numpy(
            np.stack(
                [np.asarray(item["valid_mask"], dtype=bool) for item in items]
            )
        ).bool(),
        quality=torch.from_numpy(
            np.stack(
                [np.asarray(item["quality"], dtype=np.float32) for item in items]
            )
        ).float(),
        point_features=torch.from_numpy(
            np.stack(
                [np.asarray(item["point_coords"], dtype=np.float32) for item in items]
            )
        ).float(),
        time_features=torch.from_numpy(
            np.stack(
                [np.asarray(item["time_features"], dtype=np.float32) for item in items]
            )
        ).float(),
        point_present_mask=torch.from_numpy(
            np.stack(
                [np.asarray(item["point_present"], dtype=bool) for item in items]
            )
        ).bool(),
        mapping_distance=torch.from_numpy(
            np.stack(
                [
                    np.asarray(item["mapping_distance"], dtype=np.float32)
                    for item in items
                ]
            )
        ).float(),
        mapping_valid_mask=torch.from_numpy(
            np.stack(
                [np.asarray(item["mapping_valid"], dtype=bool) for item in items]
            )
        ).bool(),
        temperature_valid_mask=torch.ones(
            (batch_size, window_length, 1), dtype=torch.bool
        ),
        teacher_features={},
        targets=targets,
        metadata=[dict(item["metadata"]) for item in items],
        bag_features=torch.from_numpy(
            np.stack(
                [np.asarray(item["bag_features"], dtype=np.float32) for item in items]
            )
        ).float(),
        bag_feature_mask=torch.ones((batch_size, 1), dtype=torch.bool),
        region_present=torch.from_numpy(
            np.stack(
                [np.asarray(item["region_present"], dtype=bool) for item in items]
            )
        ).bool(),
        region_coverage=torch.from_numpy(
            np.stack(
                [
                    np.asarray(item["region_coverage"], dtype=np.float32)
                    for item in items
                ]
            )
        ).float(),
    )
