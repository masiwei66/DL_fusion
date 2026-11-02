"""Read-only, track-preserving adapter for KW51 EGMS Level-2b CSV files."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_TRACK_PATTERN = re.compile(r"l2b_(\d{3})_(\d{4})_iw(\d+)_([a-z0-9]+)", re.IGNORECASE)
_QUALITY_COLUMNS = (
    "temporal_coherence",
    "rmse",
    "mean_velocity_std",
    "amplitude_dispersion",
)


@dataclass(frozen=True)
class RealInSARObservation:
    native_values: np.ndarray
    valid_mask: np.ndarray
    coordinates_xy: np.ndarray
    dates: tuple[str, ...]
    quality: np.ndarray
    los_vector: np.ndarray
    audit: dict[str, Any]
    track_id: str = "unknown"
    coordinates_wgs84: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    coordinates_epsg3035_from_wgs84: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    coordinates_fem_local_xy: np.ndarray | None = None
    los_vectors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    product_metadata: dict[str, np.ndarray] = field(default_factory=dict)
    raw_values_text: tuple[tuple[str, ...], ...] = ()
    reference_metadata: dict[str, Any] = field(default_factory=dict)
    raw_unit_status: str = "native_units_unverified"

    @property
    def point_count(self) -> int:
        return int(self.native_values.shape[1])

    @property
    def acquisition_count(self) -> int:
        return int(self.native_values.shape[0])


def _float(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _track_id(path: Path) -> str:
    match = _TRACK_PATTERN.search(path.name)
    if match is None:
        return path.stem
    return f"{match.group(1)}_{match.group(2)}_iw{match.group(3)}_{match.group(4)}"


def _wgs84_to_epsg3035(
    longitude_latitude: np.ndarray,
) -> tuple[np.ndarray, str, str | None]:
    coordinates = np.asarray(longitude_latitude, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("WGS84 coordinates must have shape [point, (longitude, latitude)]")
    if not np.all(np.isfinite(coordinates)):
        return np.full_like(coordinates, np.nan), "BLOCKED", "wgs84_coordinates_missing_or_nonfinite"
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
        projected = np.c_[x, y].astype(np.float64)
    except Exception as exc:
        return np.full_like(coordinates, np.nan), "BLOCKED", f"epsg3035_transform_error:{type(exc).__name__}"
    if not np.all(np.isfinite(projected)):
        return np.full_like(coordinates, np.nan), "BLOCKED", "epsg3035_transform_nonfinite"
    return projected, "PASS", None


def load_real_insar_csv(
    path: str | Path,
    *,
    reference_metadata: dict[str, Any] | None = None,
    control_points_path: str | Path | None = None,
) -> RealInSARObservation:
    """Read one orbit/product without converting units or merging tracks."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: CSV has no header")
        headers = list(reader.fieldnames)
        dates = sorted(
            [name for name in headers if len(name) == 8 and name.isdigit()],
            key=lambda value: datetime.strptime(value, "%Y%m%d"),
        )
        required = {
            "easting",
            "northing",
            "los_east",
            "los_north",
            "los_up",
            "temporal_coherence",
        }
        missing = sorted(required - set(headers))
        if missing:
            raise ValueError(f"{csv_path}: missing required InSAR columns {missing}")
        if not dates:
            raise ValueError(f"{csv_path}: no YYYYMMDD acquisition columns found")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path}: no scattering points found")

    native_values = np.asarray(
        [[_float(row.get(date, "")) for row in rows] for date in dates],
        dtype=np.float64,
    )
    raw_values_text = tuple(
        tuple(str(row.get(date, "")) for row in rows) for date in dates
    )
    valid = np.isfinite(native_values)
    projected_native = np.asarray(
        [[_float(row.get("easting")), _float(row.get("northing"))] for row in rows],
        dtype=np.float64,
    )
    if "latitude" in headers and "longitude" in headers:
        wgs84 = np.asarray(
            [[_float(row.get("longitude")), _float(row.get("latitude"))] for row in rows],
            dtype=np.float64,
        )
        projected_from_wgs84, wgs84_status, wgs84_reason = _wgs84_to_epsg3035(wgs84)
    else:
        wgs84 = np.full((len(rows), 2), np.nan, dtype=np.float64)
        projected_from_wgs84 = np.full((len(rows), 2), np.nan, dtype=np.float64)
        wgs84_status, wgs84_reason = "BLOCKED", "latitude_longitude_columns_missing"

    los_vectors = np.asarray(
        [
            [_float(row.get("los_east")), _float(row.get("los_north")), _float(row.get("los_up"))]
            for row in rows
        ],
        dtype=np.float32,
    )
    los = np.nanmedian(los_vectors.astype(np.float64), axis=0)
    los_norm = float(np.linalg.norm(los))
    if not np.isfinite(los_norm) or los_norm <= 1.0e-12:
        raise ValueError(f"{csv_path}: per-track LOS vector is missing or degenerate")
    los /= los_norm
    quality_by_column: dict[str, np.ndarray] = {}
    for column in _QUALITY_COLUMNS:
        if column in headers:
            quality_by_column[column] = np.asarray(
                [_float(row.get(column)) for row in rows], dtype=np.float32
            )
    coherence = quality_by_column["temporal_coherence"]
    quality = np.broadcast_to(
        np.clip(np.nan_to_num(coherence, nan=0.0), 0.0, 1.0)[None, :],
        native_values.shape,
    ).copy()
    product_metadata: dict[str, np.ndarray] = {}
    date_set = set(dates)
    for column in headers:
        if column in date_set:
            continue
        values = [row.get(column, "") for row in rows]
        numeric = np.asarray([_float(value) for value in values], dtype=np.float64)
        product_metadata[column] = (
            numeric if np.isfinite(numeric).all() else np.asarray(values, dtype=object)
        )

    parsed_dates = [datetime.strptime(date, "%Y%m%d") for date in dates]
    finite_crosscheck = np.isfinite(projected_from_wgs84).all(axis=1) & np.isfinite(
        projected_native
    ).all(axis=1)
    crosscheck = np.linalg.norm(
        projected_from_wgs84[finite_crosscheck] - projected_native[finite_crosscheck],
        axis=1,
    )
    refs = dict(reference_metadata or {})
    audit: dict[str, Any] = {
        "source": str(csv_path),
        "track_id": _track_id(csv_path),
        "point_count": int(native_values.shape[1]),
        "acquisition_count": int(native_values.shape[0]),
        "date_start": dates[0],
        "date_end": dates[-1],
        "date_span_days": int((parsed_dates[-1] - parsed_dates[0]).days),
        "delta_days": [
            int((right - left).days) for left, right in zip(parsed_dates[:-1], parsed_dates[1:])
        ],
        "los_vector_normalized": los.astype(float).tolist(),
        "per_point_los_preserved": True,
        "finite_fraction": float(valid.mean()),
        "temporal_profile_compatible_with_fem_26": False,
        "training_inclusion": False,
        "source_crs": "EPSG:3035",
        "wgs84_to_epsg3035_status": wgs84_status,
        "wgs84_to_epsg3035_reason": wgs84_reason,
        "native_projected_crosscheck_median_m": float(np.median(crosscheck))
        if crosscheck.size
        else None,
        "fem_local_transform_status": "BLOCKED",
        "unit_status": "native_units_unverified",
        "reference_status": refs.get("status", "product_reference_unverified"),
        "reference_metadata_source": refs.get("source", "not_supplied"),
        "reference_image": refs.get("reference_image"),
    }
    observation = RealInSARObservation(
        native_values=native_values,
        valid_mask=valid,
        coordinates_xy=projected_native.astype(np.float32),
        dates=tuple(dates),
        quality=quality.astype(np.float32),
        los_vector=los.astype(np.float32),
        audit=audit,
        track_id=_track_id(csv_path),
        coordinates_wgs84=wgs84,
        coordinates_epsg3035_from_wgs84=projected_from_wgs84,
        los_vectors=los_vectors,
        product_metadata=product_metadata,
        raw_values_text=raw_values_text,
        reference_metadata=refs,
    )
    if control_points_path is not None:
        transform = fit_fem_local_transform(control_points_path)
        fem_coordinates, transform_audit = apply_fem_local_transform(observation, transform)
        observation = replace(
            observation,
            coordinates_fem_local_xy=fem_coordinates,
            audit={**observation.audit, **transform_audit},
        )
    return observation


def load_real_insar_tracks(
    paths: Iterable[str | Path],
    *,
    reference_metadata: dict[str, dict[str, Any]] | None = None,
    control_points_path: str | Path | None = None,
) -> list[RealInSARObservation]:
    """Load each orbit independently, retaining its own dates, LOS and reference."""
    references = reference_metadata or {}
    observations = [
        load_real_insar_csv(
            path,
            reference_metadata=references.get(
                _track_id(Path(path)),
                next(
                    (
                        value
                        for key, value in references.items()
                        if _track_id(Path(path)).startswith(str(key) + "_")
                    ),
                    {},
                ),
            ),
            control_points_path=control_points_path,
        )
        for path in paths
    ]
    ids = [item.track_id for item in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("track CSVs must have unique track_id values")
    return observations


def fit_fem_local_transform(
    control_points_path: str | Path,
    *,
    validation_residual_max_m: float = 5.0,
) -> dict[str, Any]:
    """Fit a 2D rigid transform from surveyed controls; return BLOCKED on gaps."""
    path = Path(control_points_path)
    base: dict[str, Any] = {
        "status": "BLOCKED",
        "control_points_path": str(path),
        "source_crs": "EPSG:4326",
        "intermediate_crs": "EPSG:3035",
        "target_crs": "FEM_LOCAL_XY_METRES",
        "model": "2d_rigid_rotation_translation",
        "validation_residual_max_m": float(validation_residual_max_m),
    }
    if not path.exists():
        return {**base, "reason": "control_points_file_missing"}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        return {**base, "reason": "surveyed_control_points_missing", "control_point_count": 0}
    projected_points: list[list[float]] = []
    fem_points: list[list[float]] = []
    roles: list[str] = []
    groups: list[str] = []
    ids: list[str] = []
    for row in rows:
        source = [_float(row.get("longitude")), _float(row.get("latitude"))]
        target = [_float(row.get("fem_x_m")), _float(row.get("fem_y_m"))]
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            continue
        projected, status, reason = _wgs84_to_epsg3035(np.asarray([source]))
        if status != "PASS":
            return {**base, "reason": reason or "control_coordinate_projection_failed"}
        projected_points.append(projected[0].tolist())
        fem_points.append(target)
        roles.append(str(row.get("role", "")).strip().lower())
        groups.append(str(row.get("fit_group", "")).strip().lower())
        ids.append(str(row.get("point_id", "")))
    source_xy = np.asarray(projected_points, dtype=np.float64)
    target_xy = np.asarray(fem_points, dtype=np.float64)
    fit_mask = np.asarray(
        [group == "fit" or role in {"bridge_endpoint", "bridge_axis_midpoint"} for group, role in zip(groups, roles)]
    )
    validation_mask = np.asarray(
        [group == "validation" or role == "independent_validation" for group, role in zip(groups, roles)]
    )
    if int(fit_mask.sum()) < 3 or int(validation_mask.sum()) < 1:
        return {
            **base,
            "reason": "need_three_fit_controls_and_one_independent_validation_point",
            "control_point_count": len(source_xy),
            "fit_count": int(fit_mask.sum()),
            "validation_count": int(validation_mask.sum()),
        }
    fit_source = source_xy[fit_mask]
    fit_target = target_xy[fit_mask]
    source_center = fit_source.mean(axis=0)
    target_center = fit_target.mean(axis=0)
    if np.linalg.matrix_rank(fit_source - source_center) < 2:
        return {**base, "reason": "fit_controls_are_collinear", "control_point_count": len(source_xy)}
    covariance = (fit_source - source_center).T @ (fit_target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    fit_residual = fit_source @ rotation + translation - fit_target
    validation_residual = (
        source_xy[validation_mask] @ rotation
        + translation
        - target_xy[validation_mask]
    )
    fit_rmse = float(np.sqrt(np.mean(np.sum(fit_residual**2, axis=1))))
    validation_rmse = float(np.sqrt(np.mean(np.sum(validation_residual**2, axis=1))))
    if validation_rmse > float(validation_residual_max_m):
        return {
            **base,
            "reason": "independent_validation_residual_exceeds_threshold",
            "control_point_count": len(source_xy),
            "fit_count": int(fit_mask.sum()),
            "validation_count": int(validation_mask.sum()),
            "fit_rmse_m": fit_rmse,
            "validation_rmse_m": validation_rmse,
        }
    return {
        **base,
        "status": "PASS",
        "reason": None,
        "control_point_count": len(source_xy),
        "fit_count": int(fit_mask.sum()),
        "validation_count": int(validation_mask.sum()),
        "fit_point_ids": [point_id for point_id, keep in zip(ids, fit_mask) if keep],
        "validation_point_ids": [point_id for point_id, keep in zip(ids, validation_mask) if keep],
        "rotation_2x2": rotation.tolist(),
        "translation_xy_m": translation.tolist(),
        "fit_rmse_m": fit_rmse,
        "validation_rmse_m": validation_rmse,
    }


def apply_fem_local_transform(
    coordinates_or_observation: np.ndarray | RealInSARObservation,
    transform: dict[str, Any],
) -> np.ndarray | tuple[np.ndarray | None, dict[str, Any]]:
    """Apply the audited EPSG:3035 to FEM-local stage, or fail closed."""
    observation = (
        coordinates_or_observation
        if isinstance(coordinates_or_observation, RealInSARObservation)
        else None
    )
    if transform.get("status") != "PASS":
        audit = {
            "transform_status": "BLOCKED",
            "fem_local_transform_status": "BLOCKED",
            "fem_local_transform_reason": transform.get("reason", "transform_not_passed"),
            "region_assignment_status": "BLOCKED",
        }
        return (None, audit) if observation is not None else _raise_blocked_transform(transform)
    coordinates = (
        observation.coordinates_epsg3035_from_wgs84
        if observation is not None
        else np.asarray(coordinates_or_observation, dtype=np.float64)
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        audit = {
            "transform_status": "BLOCKED",
            "fem_local_transform_status": "BLOCKED",
            "fem_local_transform_reason": "epsg3035_coordinates_missing_or_nonfinite",
            "region_assignment_status": "BLOCKED",
        }
        return (None, audit) if observation is not None else _raise_blocked_transform(audit)
    rotation = np.asarray(transform["rotation_2x2"], dtype=np.float64)
    translation = np.asarray(transform["translation_xy_m"], dtype=np.float64)
    mapped = coordinates @ rotation + translation
    if observation is None:
        return mapped
    audit = {
        "transform_status": "PASS",
        "fem_local_transform_status": "PASS",
        "fem_local_transform_reason": None,
        "fem_local_validation_rmse_m": float(transform["validation_rmse_m"]),
        "region_assignment_status": "NOT_RUN_REQUIRES_AUDITED_DECK_LAYOUT",
    }
    return mapped.astype(np.float32), audit


def _raise_blocked_transform(transform: dict[str, Any]) -> np.ndarray:
    raise ValueError(
        "FEM local coordinate transform is blocked: "
        f"{transform.get('reason', 'transform_not_passed')}"
    )


def map_real_points_to_regions(
    observation: RealInSARObservation,
    region_layout: Any,
    *,
    fem_bounds: dict[str, float],
    minimum_points_per_region: int = 3,
    minimum_quality: float = 0.0,
) -> dict[str, Any]:
    """Assign transformed points to the declared FEM layout, retaining coverage."""
    if observation.coordinates_fem_local_xy is None or observation.audit.get(
        "transform_status"
    ) != "PASS":
        return {
            "track_id": observation.track_id,
            "transform_status": "BLOCKED",
            "region_mapping_status": "BLOCKED",
            "reason": observation.audit.get(
                "fem_local_transform_reason", "fem_local_transform_not_passed"
            ),
            "region_assignments": None,
            "mapping_distance_to_region_center_m": None,
            "per_region": {},
            "region_outside_point_count": None,
            "low_coverage_regions": None,
        }
    if region_layout is None or fem_bounds is None:
        return {
            "track_id": observation.track_id,
            "transform_status": "PASS",
            "region_mapping_status": "BLOCKED",
            "reason": "audited_fem_region_layout_unavailable",
            "region_assignments": None,
            "mapping_distance_to_region_center_m": None,
            "per_region": {},
            "region_outside_point_count": None,
            "low_coverage_regions": None,
        }
    x_min, x_max = float(fem_bounds["x_min"]), float(fem_bounds["x_max"])
    y_min, y_max = float(fem_bounds["y_min"]), float(fem_bounds["y_max"])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("FEM bounds must have positive longitudinal and transverse extents")
    coordinates = np.asarray(observation.coordinates_fem_local_xy, dtype=np.float64)
    s = (coordinates[:, 0] - x_min) / (x_max - x_min)
    t = 2.0 * (coordinates[:, 1] - y_min) / (y_max - y_min) - 1.0
    inside = (s >= 0.0) & (s <= 1.0) & (t >= -1.0) & (t <= 1.0)
    rows = np.searchsorted(region_layout.s_edges, s, side="right") - 1
    cols = np.searchsorted(region_layout.t_edges, t, side="right") - 1
    rows = np.clip(rows, 0, int(region_layout.rows) - 1)
    cols = np.clip(cols, 0, int(region_layout.cols) - 1)
    assignments = np.full(observation.point_count, -1, dtype=np.int64)
    assignments[inside] = rows[inside] * int(region_layout.cols) + cols[inside]
    centers = np.c_[
        x_min + np.asarray(region_layout.centers_s) * (x_max - x_min),
        y_min + (np.asarray(region_layout.centers_t) + 1.0) * 0.5 * (y_max - y_min),
    ]
    distances = np.full(observation.point_count, np.nan, dtype=np.float64)
    for point_index in np.flatnonzero(inside):
        region = int(assignments[point_index])
        distances[point_index] = np.linalg.norm(coordinates[point_index] - centers[region])

    usable_point = observation.valid_mask.any(axis=0) & (
        observation.quality.max(axis=0) >= float(minimum_quality)
    )
    per_region: dict[str, Any] = {}
    low_coverage: list[int] = []
    for region in range(int(region_layout.region_count)):
        members = assignments == region
        count = int(members.sum())
        valid_count = int((members & usable_point).sum())
        if valid_count < int(minimum_points_per_region):
            low_coverage.append(region)
        per_region[str(region)] = {
            "point_count": count,
            "valid_point_count": valid_count,
            "point_coverage": float(valid_count / count) if count else 0.0,
            "observation_valid_fraction": float(observation.valid_mask[:, members].mean())
            if count
            else 0.0,
            "mean_quality": float(observation.quality[:, members].mean()) if count else 0.0,
            "region_center_distance_m": {
                "median": float(np.nanmedian(distances[members])) if count else None,
                "max": float(np.nanmax(distances[members])) if count else None,
            },
            "density_status": "PASS" if valid_count >= minimum_points_per_region else "INSUFFICIENT_DENSITY",
        }
    return {
        "track_id": observation.track_id,
        "transform_status": "PASS",
        "region_mapping_status": "PASS",
        "region_assignments": assignments.tolist(),
        "mapping_distance_to_region_center_m": [
            float(value) if np.isfinite(value) else None for value in distances
        ],
        "per_region": per_region,
        "region_outside_point_count": int((~inside).sum()),
        "low_coverage_regions": low_coverage,
    }


def aggregate_track_time_windows(
    observation: RealInSARObservation,
    *,
    window_days: int,
    anchor_date: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate one track into explicit non-overlapping calendar windows.

    Values remain in their native product units. The caller must audit and
    freeze ``window_days`` before using these summaries for model inference.
    """
    window_days = int(window_days)
    if window_days < 1:
        raise ValueError("window_days must be a positive integer")
    dates = [datetime.strptime(value, "%Y%m%d") for value in observation.dates]
    anchor = datetime.strptime(anchor_date, "%Y%m%d") if anchor_date else dates[0]
    buckets: dict[int, list[int]] = {}
    for index, date in enumerate(dates):
        bucket = (date - anchor).days // window_days
        buckets.setdefault(bucket, []).append(index)
    result: list[dict[str, Any]] = []
    for bucket, indices in sorted(buckets.items()):
        index_array = np.asarray(indices, dtype=np.int64)
        valid = observation.valid_mask[index_array]
        values = observation.native_values[index_array]
        counts = valid.sum(axis=0)
        mean_values = np.where(valid, values, 0.0).sum(axis=0) / np.maximum(counts, 1)
        point_valid = counts > 0
        mean_quality = observation.quality[index_array].mean(axis=0)
        dates_in_window = [dates[index] for index in indices]
        representative = dates_in_window[0] + (
            dates_in_window[-1] - dates_in_window[0]
        ) / 2
        # Acquisition gaps remain explicit in the dates included in each bin.
        window_start = anchor + timedelta(days=bucket * window_days)
        window_end = window_start + timedelta(days=window_days - 1)
        result.append(
            {
                "window_index": int(bucket),
                "date_start": min(dates_in_window).strftime("%Y%m%d"),
                "date_end": max(dates_in_window).strftime("%Y%m%d"),
                "calendar_window_start": window_start.strftime("%Y%m%d"),
                "calendar_window_end": window_end.strftime("%Y%m%d"),
                "time_representative": representative.isoformat(),
                "acquisition_count": int(len(indices)),
                "acquisition_indices": indices,
                "values_native": mean_values.astype(float).tolist(),
                "valid_mask": point_valid.astype(bool).tolist(),
                "valid_observation_count_per_point": counts.astype(int).tolist(),
                "quality_mean_per_point": mean_quality.astype(float).tolist(),
                "mean_quality": float(mean_quality.mean()) if mean_quality.size else 0.0,
                "raw_unit_status": observation.raw_unit_status,
                "aggregation_rule": f"fixed_non_overlapping_{window_days}_day_bins_from_{anchor.strftime('%Y%m%d')}",
            }
        )
    return result


def assert_temporal_alignment(
    observation: RealInSARObservation, expected_acquisition_count: int = 26
) -> None:
    """Fail closed until an explicit temporal aggregation is supplied."""
    if observation.acquisition_count != int(expected_acquisition_count):
        raise ValueError(
            "Real InSAR has "
            f"{observation.acquisition_count} acquisitions while FEM has {expected_acquisition_count}; "
            "provide an audited temporal aggregation operator before ranking."
        )
