from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from KW51_training.data.real_insar_adapter import (
    aggregate_track_time_windows,
    apply_fem_local_transform,
    fit_fem_local_transform,
    load_real_insar_csv,
    load_real_insar_tracks,
    map_real_points_to_regions,
)


def _track_csv(path: Path) -> Path:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    rows = []
    for index in range(3):
        longitude = 4.7066 + index * 0.0001
        latitude = 50.9004 + index * 0.0001
        easting, northing = transformer.transform(longitude, latitude)
        rows.append(
            f"{index},{latitude},{longitude},{easting},{northing},"
            f"{0.1 + index * 0.01},-0.1,0.9,1.5,0.8,0.12,{index + 2},{index + 1}"
        )
    path.write_text(
        "pid,latitude,longitude,easting,northing,los_east,los_north,los_up,"
        "rmse,temporal_coherence,amplitude_dispersion,20200102,20200101\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_track_csv_preserves_native_values_dates_los_and_metadata(tmp_path: Path) -> None:
    path = _track_csv(tmp_path / "KW51_l2b_088_0302_iw1_vv_2019_2023_1.csv")
    observation = load_real_insar_csv(path)
    assert observation.track_id == "088_0302_iw1_vv"
    assert observation.dates == ("20200101", "20200102")
    assert observation.native_values[:, 0].tolist() == [1.0, 2.0]
    assert observation.raw_unit_status == "native_units_unverified"
    assert observation.los_vectors.shape == (3, 3)
    assert observation.coordinates_wgs84.shape == (3, 2)
    assert observation.coordinates_epsg3035_from_wgs84.shape == (3, 2)
    assert observation.audit["wgs84_to_epsg3035_status"] == "PASS"
    assert observation.audit["native_projected_crosscheck_median_m"] < 0.01
    assert observation.product_metadata["rmse"].tolist() == [1.5] * 3


def test_empty_control_points_block_fem_mapping_and_preserve_source_xy(tmp_path: Path) -> None:
    path = _track_csv(tmp_path / "track_088.csv")
    observation = load_real_insar_csv(path)
    controls = tmp_path / "control_points.csv"
    controls.write_text(
        "point_id,role,latitude,longitude,fem_x_m,fem_y_m,fit_group,description\n",
        encoding="utf-8",
    )
    result = fit_fem_local_transform(controls)
    local, audit = apply_fem_local_transform(observation, result)
    assert result["status"] == "BLOCKED"
    assert audit["transform_status"] == "BLOCKED"
    assert local is None
    assert observation.coordinates_xy.shape == (3, 2)
    mapping = map_real_points_to_regions(
        observation,
        object(),
        fem_bounds={"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 10.0},
    )
    assert mapping["region_assignments"] is None
    assert mapping["region_mapping_status"] == "BLOCKED"


def test_control_point_fit_uses_independent_validation_point(tmp_path: Path) -> None:
    controls = tmp_path / "control_points.csv"
    inverse = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    projected = [
        (6500000.0, 5500000.0),
        (6500010.0, 5500000.0),
        (6500000.0, 5500010.0),
        (6500010.0, 5500010.0),
    ]
    geographic = [inverse.transform(x, y) for x, y in projected]
    rows = [
        ("A", "bridge_endpoint", 100.0, 200.0, "fit", "a"),
        ("B", "bridge_endpoint", 100.0, 210.0, "fit", "b"),
        ("C", "bridge_axis_midpoint", 90.0, 200.0, "fit", "c"),
        ("V", "independent_validation", 90.0, 210.0, "validation", "v"),
    ]
    csv_rows = []
    for (point_id, role, fem_x, fem_y, fit_group, description), (longitude, latitude) in zip(rows, geographic):
        csv_rows.append(
            f"{point_id},{role},{latitude},{longitude},{fem_x},{fem_y},{fit_group},{description}"
        )
    controls.write_text(
        "point_id,role,latitude,longitude,fem_x_m,fem_y_m,fit_group,description\n"
        + "\n".join(csv_rows)
        + "\n",
        encoding="utf-8",
    )
    transform = fit_fem_local_transform(controls)
    assert transform["status"] == "PASS"
    source = np.asarray([[6500005.0, 5500005.0], [6500000.0, 5500000.0]])
    mapped = apply_fem_local_transform(source, transform)
    assert np.allclose(mapped[0], [95.0, 205.0], atol=1e-6)
    assert transform["validation_rmse_m"] < 1e-6


def test_four_track_loader_keeps_tracks_separate(tmp_path: Path) -> None:
    paths = [
        _track_csv(tmp_path / f"KW51_l2b_{track}_0302_iw1_vv_2019_2023_1.csv")
        for track in ("037", "088", "110", "161")
    ]
    observations = load_real_insar_tracks(paths)
    assert [item.track_id for item in observations] == [
        "037_0302_iw1_vv", "088_0302_iw1_vv", "110_0302_iw1_vv", "161_0302_iw1_vv"
    ]
    assert all(item.native_values.shape == (2, 3) for item in observations)


def test_irregular_acquisitions_aggregate_into_explicit_calendar_bins(tmp_path: Path) -> None:
    path = tmp_path / "track_088.csv"
    path.write_text(
        "pid,easting,northing,los_east,los_north,los_up,temporal_coherence,"
        "20200101,20200102,20200110\n"
        "1,0,0,0,0,1,0.8,1,3,5\n"
        "2,1,0,0,0,1,0.6,2,,6\n",
        encoding="utf-8",
    )
    observation = load_real_insar_csv(path)
    windows = aggregate_track_time_windows(observation, window_days=5)
    assert len(windows) == 2
    assert windows[0]["acquisition_count"] == 2
    assert windows[1]["acquisition_count"] == 1
    assert windows[0]["date_start"] == "20200101"
    assert windows[0]["date_end"] == "20200102"
    assert windows[0]["valid_mask"] == [True, True]
    assert windows[0]["values_native"] == pytest.approx([2.0, 2.0])
