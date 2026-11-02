from __future__ import annotations

from pathlib import Path

from KW51_training.training.real_insar_benchmark_runner import (
    build_real_insar_benchmark_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _track(path: Path) -> Path:
    path.write_text(
        "pid,latitude,longitude,easting,northing,los_east,los_north,los_up,"
        "rmse,temporal_coherence,amplitude_dispersion,20200101,20200108\n"
        "1,50.9004,4.7066,6500000,5500000,0.1,-0.1,0.9,1.5,0.8,0.12,1,2\n"
        "2,50.9005,4.7067,6500010,5500010,0.2,-0.1,0.8,1.2,0.7,0.10,2,3\n",
        encoding="utf-8",
    )
    return path


def test_real_report_blocks_without_control_points_units_reference_and_time_rule(
    tmp_path: Path,
) -> None:
    track = _track(tmp_path / "KW51_l2b_088_0302_iw1_vv_2019_2023_1.csv")
    controls = tmp_path / "control_points.csv"
    controls.write_text(
        "point_id,role,latitude,longitude,fem_x_m,fem_y_m,fit_group,description\n",
        encoding="utf-8",
    )
    report, csv_path = build_real_insar_benchmark_report(
        ROOT / "configs" / "kw51_public_benchmark_v1.yaml",
        output_root=tmp_path / "report",
        track_paths=[track],
        control_points_path=controls,
    )
    assert report["run_status"] == "BLOCKED"
    assert report["readiness"]["transform_status"]["status"] == "BLOCKED"
    assert report["readiness"]["time_aggregation"]["status"] == "BLOCKED"
    assert report["l1"]["status"] == "BLOCKED"
    assert report["l2"]["status"] == "BLOCKED"
    assert "state_accuracy" not in report["claim_scope"]
    row = report["tracks"][0]
    assert row["track_id"] == "088_0302_iw1_vv"
    assert row["unit_status"] == "native_units_unverified"
    assert row["transform_status"] == "BLOCKED"
    assert row["region_mapping"]["region_assignments"] is None
    assert row["region_mapping"]["mapping_distance_to_region_center_m"] is None
    assert row["window_aggregation"]["status"] == "BLOCKED"
    assert row["phase_windows"]["post_retrofit"]["acquisition_count"] == 2
    assert csv_path.exists()
    assert "mae" not in report["l1"]
    assert "rmse" not in report["l1"]
