"""Read-only density audit for the deployable KW51 InSAR CSVs.

The bridge-origin/axis transform is still marked pending in
``kw51_insar_observation_protocol.json``. Until it is frozen, per-region point
counts cannot be computed reliably; this module reports track-level counts,
coverage and a conservative per-region upper bound.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .real_insar_adapter import (
    RealInSARObservation,
    load_real_insar_tracks,
)


def _pca_extent(coordinates: np.ndarray) -> dict[str, float]:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.shape[0] < 2:
        return {"major_axis_m": 0.0, "minor_axis_m": 0.0}
    centered = coords - coords.mean(axis=0, keepdims=True)
    try:
        _, _, components = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"major_axis_m": 0.0, "minor_axis_m": 0.0}
    projected = centered @ components.T
    return {
        "major_axis_m": float(projected[:, 0].max() - projected[:, 0].min()),
        "minor_axis_m": float(
            projected[:, 1].max() - projected[:, 1].min()
        )
        if projected.shape[1] > 1
        else 0.0,
    }


def _nearest_neighbour_stats(coordinates: np.ndarray) -> dict[str, float]:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.shape[0] < 2:
        return {"min_m": 0.0, "median_m": 0.0}
    distances = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return {
        "min_m": float(np.min(nearest)),
        "median_m": float(np.median(nearest)),
    }


def _audit_track(observation: RealInSARObservation) -> dict[str, Any]:
    coords = np.asarray(observation.coordinates_xy, dtype=np.float64)
    valid = np.asarray(observation.valid_mask, dtype=bool)
    quality = np.asarray(observation.quality, dtype=np.float64)
    coherence = quality[0] if quality.ndim == 2 else quality
    unique = np.unique(np.round(coords, decimals=3), axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0) if len(coords) else np.zeros(2)
    pca = _pca_extent(coords)
    nearest = _nearest_neighbour_stats(coords)
    dates = list(observation.dates)
    point_count = int(observation.point_count)
    return {
        "point_count": point_count,
        "track_id": observation.track_id,
        "acquisition_count": int(observation.acquisition_count),
        "finite_fraction": float(valid.mean()) if valid.size else 0.0,
        "unique_coordinate_count": int(unique.shape[0]),
        "duplicate_coordinate_count": int(point_count - unique.shape[0]),
        "easting_northing_extent_m": [float(extent[0]), float(extent[1])],
        **pca,
        **nearest,
        "coherence_median": float(np.nanmedian(coherence)),
        "date_start": dates[0] if dates else "",
        "date_end": dates[-1] if dates else "",
        "los_vector": observation.los_vector.astype(float).tolist(),
        "los_vectors_per_point": observation.los_vectors.astype(float).tolist(),
        "delta_days": observation.audit.get("delta_days", []),
        "source_crs": observation.audit.get("source_crs", "unknown"),
        "transform_status": observation.audit.get("transform_status", "BLOCKED"),
        "unit_status": observation.raw_unit_status,
        "reference_status": observation.audit.get("reference_status", "product_reference_unverified"),
        "reference_metadata": observation.reference_metadata,
        "per_region_density_status": (
            "unavailable_pending_geo_transform"
            if observation.coordinates_fem_local_xy is None
            else "mapped_pending_region_coverage_gate"
        ),
        "upper_bound_points_per_region_8": float(point_count) / 8.0,
    }


def audit_real_insar_density(
    paths: Iterable[str | Path],
    *,
    control_points_path: str | Path | None = None,
    reference_metadata: dict[str, dict[str, Any]] | None = None,
    region_count: int = 8,
    bridge_length_m: float = 116.316,
    bridge_width_m: float = 11.75,
    synthetic_budgets: Iterable[int] = (40, 80, 160, 640),
    configured_synthetic_budget: int = 160,
) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = []
    all_coords: list[np.ndarray] = []
    total_points = 0
    observations = load_real_insar_tracks(
        paths,
        control_points_path=control_points_path,
        reference_metadata=reference_metadata,
    )
    for observation in observations:
        track = _audit_track(observation)
        track["source"] = observation.audit["source"]
        tracks.append(track)
        total_points += track["point_count"]
        all_coords.append(np.asarray(observation.coordinates_xy, dtype=np.float64))
    if all_coords:
        combined = np.concatenate(all_coords, axis=0)
        combined_unique = np.unique(np.round(combined, decimals=3), axis=0)
        combined_extent = _pca_extent(combined)
    else:
        combined_unique = np.zeros((0, 2))
        combined_extent = {"major_axis_m": 0.0, "minor_axis_m": 0.0}
    point_counts = [track["point_count"] for track in tracks]
    budgets = [int(value) for value in synthetic_budgets]
    per_track_region = [
        float(count) / max(region_count, 1) for count in point_counts
    ]
    median_track_points = (
        float(np.median(point_counts)) if point_counts else 0.0
    )
    if median_track_points < min(budgets):
        density_support = "below_lowest_synthetic_budget_per_track"
    elif median_track_points < int(configured_synthetic_budget):
        density_support = "below_configured_synthetic_budget_per_track"
    else:
        density_support = "at_or_above_configured_synthetic_budget_per_track"
    return {
        "schema": "kw51_real_insar_density_audit_v1",
        "status": "AUDIT_ONLY",
        "coordinate_status": (
            "PASS"
            if tracks and all(track["transform_status"] == "PASS" for track in tracks)
            else "BLOCKED"
        ),
        "region_count": int(region_count),
        "track_count": len(tracks),
        "total_points_across_tracks": int(total_points),
        "unique_points_across_tracks": int(combined_unique.shape[0]),
        "points_per_track": {
            "min": int(min(point_counts)) if point_counts else 0,
            "median": median_track_points,
            "max": int(max(point_counts)) if point_counts else 0,
        },
        "configured_synthetic_budget": int(configured_synthetic_budget),
        "upper_bound_points_per_region_8": float(total_points) / max(region_count, 1),
        "per_track_upper_bound_points_per_region": {
            "min": float(min(per_track_region)) if per_track_region else 0.0,
            "median": float(np.median(per_track_region))
            if per_track_region
            else 0.0,
            "max": float(max(per_track_region)) if per_track_region else 0.0,
        },
        "combined_pca_extent": combined_extent,
        "bridge_footprint_m": {
            "length": float(bridge_length_m),
            "width": float(bridge_width_m),
        },
        "synthetic_budget_comparison": {
            str(budget): {
                "total_to_synthetic_ratio": float(total_points) / float(budget)
                if budget
                else 0.0,
                "median_track_to_synthetic_ratio": median_track_points
                / float(budget)
                if budget
                else 0.0,
                "synthetic_per_region": float(budget) / max(region_count, 1),
            }
            for budget in budgets
        },
        "density_support": density_support,
        "tracks": tracks,
        "notes": [
            "The bridge-origin/axis transform is pending; per-region point counts "
            "are not computed and the total point count is an upper bound.",
            "Tracks have different acquisition dates and LOS vectors; their total "
            "point count is not a simultaneous observation.",
            "Per-track point count is the relevant effective density for one "
            "acquisition; multi-track fusion requires an explicit temporal "
            "aggregation operator.",
            "Public EGMS ROI CSVs are not a complete PS inventory; a denser "
            "processing product may change the deployable density.",
            "This audit is read-only and does not use structural-state labels.",
        ],
    }
