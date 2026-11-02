"""Readiness-first Track R reporting for unlabeled EGMS observations."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..data.fem_insar import build_fem_archive_index
from ..data.fuzzy_range import build_deck_geometry
from ..data.fuzzy_region import build_region_layout
from ..data.real_insar_adapter import (
    RealInSARObservation,
    aggregate_track_time_windows,
    fit_fem_local_transform,
    load_real_insar_tracks,
    map_real_points_to_regions,
)


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _date_text(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _reference_metadata(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    settings = config.get("real_insar", {})
    source = str(settings.get("reference_metadata_source", "not_supplied"))
    status = str(settings.get("reference_status", "product_reference_unverified"))
    result: dict[str, dict[str, Any]] = {}
    for key, value in settings.get("reference_images", {}).items():
        track = str(key).replace("track_", "")
        result[track] = {
            "reference_image": str(value),
            "source": source,
            "status": status,
        }
    return result


def _phase_summary(
    observation: RealInSARObservation, phase_windows: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    parsed = [datetime.strptime(item, "%Y%m%d").date() for item in observation.dates]
    result: dict[str, dict[str, Any]] = {}
    for name, bounds in phase_windows.items():
        start = datetime.strptime(_date_text(bounds[0]), "%Y-%m-%d").date()
        end = datetime.strptime(_date_text(bounds[1]), "%Y-%m-%d").date()
        selected = [index for index, value in enumerate(parsed) if start <= value <= end]
        if selected:
            valid = observation.valid_mask[selected]
            quality = observation.quality[selected]
            result[str(name)] = {
                "date_start": min(parsed[index] for index in selected).isoformat(),
                "date_end": max(parsed[index] for index in selected).isoformat(),
                "configured_start": start.isoformat(),
                "configured_end": end.isoformat(),
                "acquisition_count": len(selected),
                "valid_observation_fraction": float(valid.mean()),
                "mean_quality": float(quality.mean()) if quality.size else 0.0,
                "use": "descriptive_phase_window_only",
            }
        else:
            result[str(name)] = {
                "date_start": None,
                "date_end": None,
                "configured_start": start.isoformat(),
                "configured_end": end.isoformat(),
                "acquisition_count": 0,
                "valid_observation_fraction": 0.0,
                "mean_quality": 0.0,
                "use": "descriptive_phase_window_only",
            }
    return result


def _region_layout_if_ready(
    config: dict[str, Any], observations: list[RealInSARObservation], root: Path
) -> tuple[Any | None, dict[str, float] | None]:
    if not observations or not all(item.coordinates_fem_local_xy is not None for item in observations):
        return None, None
    archive_path = _resolve(config["source_archive"], root)
    index = build_fem_archive_index(archive_path)
    geometry = build_deck_geometry(index.records[0], (0.0, 0.0, 1.0))
    region = config.get("real_insar", {}).get("region_layout", {})
    layout = build_region_layout(
        geometry,
        mode=str(region.get("mode", "longitudinal")),
        rows=int(region.get("rows", 8)),
        cols=int(region.get("cols", 1)),
    )
    bounds = {
        "x_min": float(geometry.x_min),
        "x_max": float(geometry.x_max),
        "y_min": float(geometry.y_min),
        "y_max": float(geometry.y_max),
    }
    return layout, bounds


def _track_report(
    observation: RealInSARObservation,
    *,
    phase_windows: dict[str, Any],
    aggregation_setting: Any,
    window_days: int | None,
    layout: Any | None,
    fem_bounds: dict[str, float] | None,
    minimum_points_per_region: int,
) -> dict[str, Any]:
    transform_status = str(observation.audit.get("transform_status", "BLOCKED"))
    if layout is None or fem_bounds is None:
        mapping = map_real_points_to_regions(
            observation,
            None,
            fem_bounds={"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
            minimum_points_per_region=minimum_points_per_region,
        )
    else:
        mapping = map_real_points_to_regions(
            observation,
            layout,
            fem_bounds=fem_bounds,
            minimum_points_per_region=minimum_points_per_region,
            minimum_quality=0.4,
        )
    if window_days is not None:
        windows = aggregate_track_time_windows(observation, window_days=window_days)
        aggregation_status = "AUDIT_ONLY_PENDING_RULE_APPROVAL"
        aggregation_reason = "configured_window_has_not_been_frozen_for_fem_profile_alignment"
    else:
        windows = []
        aggregation_status = "BLOCKED"
        aggregation_reason = (
            "time_aggregation_pending_audit"
            if str(aggregation_setting).lower() == "pending_audit"
            else "no_explicit_window_rule"
        )
    return {
        "track_id": observation.track_id,
        "source": observation.audit["source"],
        "point_count": observation.point_count,
        "acquisition_count": observation.acquisition_count,
        "date_range": {
            "start": observation.dates[0],
            "end": observation.dates[-1],
            "delta_days": observation.audit.get("delta_days", []),
        },
        "valid_observation_fraction": float(observation.valid_mask.mean()),
        "valid_point_count": int(observation.valid_mask.any(axis=0).sum()),
        "quality": {
            "mean": float(observation.quality.mean()),
            "per_point": observation.quality.mean(axis=0).astype(float).tolist(),
            "fields_preserved": sorted(observation.product_metadata.keys()),
        },
        "los": {
            "track_vector_normalized": observation.los_vector.astype(float).tolist(),
            "per_point_direction_cosines": observation.los_vectors.astype(float).tolist(),
            "sign_policy": "preserve_product_sign",
        },
        "coordinates": {
            "source_crs": "EPSG:3035",
            "wgs84_to_epsg3035_status": observation.audit.get(
                "wgs84_to_epsg3035_status", "BLOCKED"
            ),
            "native_projected_crosscheck_median_m": observation.audit.get(
                "native_projected_crosscheck_median_m"
            ),
            "transform_status": transform_status,
        },
        "transform_status": transform_status,
        "unit_status": observation.raw_unit_status,
        "reference_status": observation.audit.get(
            "reference_status", "product_reference_unverified"
        ),
        "reference_metadata": observation.reference_metadata,
        "region_mapping": mapping,
        "window_aggregation": {
            "status": aggregation_status,
            "rule": f"fixed_calendar_{window_days}_day_bins" if window_days else "BLOCKED_NOT_CONFIGURED",
            "reason": aggregation_reason,
            "windows": windows,
        },
        "phase_windows": _phase_summary(observation, phase_windows),
    }


def build_real_insar_benchmark_report(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    track_paths: Iterable[str | Path] | None = None,
    control_points_path: str | Path | None = None,
    window_days: int | None = None,
) -> tuple[dict[str, Any], Path]:
    config_file = Path(config_path).resolve()
    root = config_file.parents[1].parent
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    real = config.get("real_insar", {})
    tracks = [Path(value) for value in (track_paths or real.get("track_csvs", []))]
    control_path = Path(control_points_path) if control_points_path else Path(real["control_points"])
    reference_metadata = _reference_metadata(config)
    observations = load_real_insar_tracks(
        tracks,
        reference_metadata=reference_metadata,
        control_points_path=control_path,
    )
    transform = fit_fem_local_transform(control_path)
    layout, fem_bounds = _region_layout_if_ready(config, observations, root)
    phase_windows = real.get("phase_windows", {})
    aggregation_setting = real.get("time_aggregation", "pending_audit")
    configured_window = window_days
    if configured_window is None and isinstance(aggregation_setting, int):
        configured_window = int(aggregation_setting)
    minimum_points = int(real.get("minimum_points_per_region", 3))
    track_reports = [
        _track_report(
            observation,
            phase_windows=phase_windows,
            aggregation_setting=aggregation_setting,
            window_days=configured_window,
            layout=layout,
            fem_bounds=fem_bounds,
            minimum_points_per_region=minimum_points,
        )
        for observation in observations
    ]
    transform_status = "PASS" if transform.get("status") == "PASS" else "BLOCKED"
    unit_status = "PASS" if all(
        item.raw_unit_status == "verified" for item in observations
    ) else "BLOCKED"
    reference_status = "PASS" if all(
        item.audit.get("reference_status") == "verified" for item in observations
    ) else "BLOCKED"
    time_status = (
        "PASS"
        if real.get("time_aggregation_status") == "PASS"
        else "BLOCKED"
    )
    density_status = (
        "INSUFFICIENT_DENSITY"
        if layout is not None
        and any(item["region_mapping"].get("low_coverage_regions") for item in track_reports)
        else "PASS"
        if layout is not None
        else "BLOCKED_TRANSFORM_OR_REGION_MAPPING"
    )
    readiness_checks = {
        "transform_status": {
            "status": transform_status,
            "reason": transform.get("reason"),
            "audit": transform,
        },
        "native_units": {
            "status": unit_status,
            "per_track_status": {item.track_id: item.raw_unit_status for item in observations},
        },
        "track_reference": {
            "status": reference_status,
            "per_track_status": {
                item.track_id: item.audit.get("reference_status") for item in observations
            },
        },
        "time_aggregation": {
            "status": time_status,
            "configured_rule": aggregation_setting,
            "window_days": configured_window,
        },
        "per_track_region_density": {
            "status": density_status,
            "reason": "region counts require an audited FEM-local transform and region mapping",
        },
    }
    blocked_checks = [
        name for name, item in readiness_checks.items() if item["status"] != "PASS"
    ]
    report: dict[str, Any] = {
        "schema_version": "kw51_real_insar_benchmark_report_v1",
        "claim_scope": {
            "track_r": "Unlabeled EGMS consistency and candidate-score stability only.",
            "real_state_accuracy": False,
            "component_damage_truth": "unavailable",
            "retrofit_phase_use": "weak_descriptive_evidence_only",
        },
        "transform_status": transform_status,
        "coordinate_pipeline": {
            "stage_1": "WGS84_EPSG4326_to_EPSG3035",
            "stage_2": "EPSG3035_to_FEM_local_xy_using_surveyed_controls",
            "fem_local_transform": transform,
        },
        "readiness": {
            "status": "BLOCKED" if blocked_checks else "READY_FOR_AUDITED_INFERENCE",
            **readiness_checks,
            "blocked_checks": blocked_checks,
        },
        "time_aggregation": {
            "configured_rule": aggregation_setting,
            "window_days": configured_window,
            "track_independent": True,
            "phase_windows": {
                name: [_date_text(value) for value in bounds]
                for name, bounds in phase_windows.items()
            },
            "phase_labels_used_for_reference_or_training": False,
        },
        "tracks": track_reports,
        "l1": {
            "status": "BLOCKED" if blocked_checks else "NOT_RUN",
            "metrics": {},
            "allowed_metrics": ["trend_direction_agreement", "time_correlation", "rank_consistency"],
            "not_run_reason": ";".join(blocked_checks) if blocked_checks else "inference_not_requested",
            "physical_error_metrics_reported": False,
        },
        "l2": {
            "status": "BLOCKED" if blocked_checks else "NOT_RUN",
            "candidate_scores_by_phase": {},
            "candidate_order_changes": {},
            "stability": {},
            "uncertainty": {},
            "not_run_reason": ";".join(blocked_checks) if blocked_checks else "inference_not_requested",
        },
        "run_status": "BLOCKED" if blocked_checks else "READY_NO_INFERENCE",
        "not_run_reasons": blocked_checks,
    }
    output_dir = Path(output_root).resolve() if output_root else _resolve(
        real.get("output_root", "runs/public_benchmark_v1/track_r"), root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "real_insar_benchmark_report.json"
    csv_path = output_dir / "real_insar_track_summary.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "track_id",
            "point_count",
            "acquisition_count",
            "date_start",
            "date_end",
            "transform_status",
            "unit_status",
            "reference_status",
            "region_mapping_status",
            "region_outside_point_count",
            "phase_windows_json",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in track_reports:
            writer.writerow(
                {
                    "track_id": item["track_id"],
                    "point_count": item["point_count"],
                    "acquisition_count": item["acquisition_count"],
                    "date_start": item["date_range"]["start"],
                    "date_end": item["date_range"]["end"],
                    "transform_status": item["coordinates"]["transform_status"],
                    "unit_status": item["unit_status"],
                    "reference_status": item["reference_status"],
                    "region_mapping_status": item["region_mapping"]["region_mapping_status"],
                    "region_outside_point_count": item["region_mapping"]["region_outside_point_count"],
                    "phase_windows_json": json.dumps(item["phase_windows"], sort_keys=True),
                }
            )
    return report, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="KW51_training/configs/kw51_public_benchmark_v1.yaml",
    )
    parser.add_argument("--output-root")
    parser.add_argument("--track-csv", action="append", dest="track_paths")
    parser.add_argument("--control-points")
    parser.add_argument("--window-days", type=int)
    args = parser.parse_args(argv)
    report, csv_path = build_real_insar_benchmark_report(
        args.config,
        output_root=args.output_root,
        track_paths=args.track_paths,
        control_points_path=args.control_points,
        window_days=args.window_days,
    )
    print(
        f"run_status={report['run_status']} "
        f"transform_status={report['transform_status']} "
        f"tracks={len(report['tracks'])} csv={csv_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
