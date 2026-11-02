from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from KW51_training.config import FuzzyConfig
from KW51_training.data.fem_insar import build_fem_archive_index
from KW51_training.data.fuzzy_range import (
    FuzzyNormalization,
    FuzzyQueryRef,
    build_deck_geometry,
)
from KW51_training.data.fuzzy_region import (
    FuzzyRegionQueryDataset,
    FuzzyRegionSource,
    FuzzyRegionSummaryDataset,
    build_region_layout,
    collate_fuzzy_region_queries,
    collate_fuzzy_region_summaries,
    compute_dense_region_target,
)


def _surface_arrays():
    quads = [
        [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.4, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [(0.4, 0.0, 0.0), (0.9, 0.0, 0.0), (0.9, 1.0, 0.0), (0.4, 1.0, 0.0)],
        [(1.4, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.4, 1.0, 0.0)],
    ]
    coords = np.asarray([point for quad in quads for point in quad], dtype=np.float64)
    node_ids = np.arange(101001, 101001 + len(coords), dtype=np.int64)
    field = np.zeros((2, len(coords), 3), dtype=np.float64)
    field[0, :4, 0] = 1.0
    field[0, 4:8, 0] = 5.0
    field[0, 8:12, 0] = 9.0
    field[1, :8, 0] = np.asarray([1.0] * 4 + [5.0] * 4)
    valid = np.ones_like(field, dtype=bool)
    valid[1, 8:, :] = False
    connectivity = np.asarray(
        [
            node_ids[0:4],
            node_ids[4:8],
            node_ids[8:12],
            [999999.0, 999999.0, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    arrays = {
        "field": field,
        "valid": valid,
        "node_coords": coords,
        "node_ids": node_ids,
        "deck_mask": np.ones(len(coords), dtype=bool),
        "element_connectivity": connectivity,
        "element_type": np.asarray([3, 3, 3, 1]),
        "element_section": np.asarray([7, 8, 7, 1]),
    }
    s = coords[:, 0] / 2.0
    t = 2.0 * coords[:, 1] - 1.0
    geometry = SimpleNamespace(
        node_coords=coords,
        deck_indices=np.arange(len(coords)),
        deck_xy=coords[:, :2],
        deck_s=s,
        deck_t=t,
        x_min=0.0,
        x_max=2.0,
        y_min=0.0,
        y_max=1.0,
        los=np.asarray([1.0, 0.0, 0.0]),
    )
    layout = build_region_layout(geometry, mode="longitudinal", rows=2, cols=1)
    return arrays, geometry, layout


def test_dense_target_uses_los_shell_area_weights_and_area_coverage() -> None:
    arrays, geometry, layout = _surface_arrays()

    result = compute_dense_region_target(arrays, geometry, layout)

    assert result["target_mm"].shape == (2, 2)
    assert np.allclose(result["region_area_m2"], [0.9, 0.6])
    assert np.allclose(result["area_coverage"][:, 0], [1.0, 1.0])
    assert result["valid_mask"].tolist() == [[True, True], [True, False]]
    assert result["area_coverage"][1, 1] == pytest.approx(0.0)
    assert result["target_mm"][0, 0] == pytest.approx((0.4 * 1.0 + 0.5 * 5.0) / 0.9 * 1000.0 - 5000.0)
    assert result["target_mm"][1, 0] == pytest.approx(4000.0)
    assert result["node_area_weights"].shape == (len(geometry.deck_indices),)
    assert result["node_area_weights"].sum() == pytest.approx(1.5)


def test_dense_target_fails_closed_when_surface_connectivity_is_unmappable() -> None:
    arrays, geometry, layout = _surface_arrays()
    arrays["element_connectivity"] = arrays["element_connectivity"].copy()
    arrays["element_connectivity"][0, 0] = 999999.0

    with pytest.raises(ValueError, match="unknown node id|node_ids"):
        compute_dense_region_target(arrays, geometry, layout)


@pytest.mark.skipif(
    not Path("KW51_training/runs/_fuzzy_smoke_archive").exists(),
    reason="local six-file smoke archive not present",
)
def test_dense_region_target_is_collated_as_target_not_model_input() -> None:
    index = build_fem_archive_index("KW51_training/runs/_fuzzy_smoke_archive")
    geometry = build_deck_geometry(index.records[0], [0.0, 0.0, 1.0])
    layout = build_region_layout(geometry, mode="longitudinal", rows=8, cols=1)
    config = FuzzyConfig(region_point_budget=80, region_missing_probability=0.0)
    source = FuzzyRegionSource(index, geometry, config, seed=0)
    normalization = FuzzyNormalization(
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, "disabled", ()
    )
    dataset = FuzzyRegionQueryDataset(
        [FuzzyQueryRef(index.records[0], 0)],
        source,
        normalization,
        layout,
    )

    item = dataset[0]
    batch = collate_fuzzy_region_queries([item])

    assert "region_response_target" not in item
    assert batch.targets["region_response_target"].shape == (1, 8, 26)
    assert batch.targets["region_response_valid_mask"].shape == (1, 8, 26)
    assert batch.targets["region_area_weights"].shape == (1, 8)
    assert batch.targets["region_area_coverage"].shape == (1, 8, 26)
    assert batch.values.shape == (1, 26, 80)
    query = source.sample_region_query(index.records[0], 0, layout)
    assert np.allclose(
        batch.targets["region_response_target"].numpy()[0],
        query["dense_region_target"],
    )

    summary_dataset = FuzzyRegionSummaryDataset(
        [FuzzyQueryRef(index.records[0], 0)],
        source,
        normalization,
        layout,
        estimator="mean",
    )
    summary_item = summary_dataset[0]
    summary_batch = collate_fuzzy_region_summaries([summary_item])
    assert "region_response_target" not in summary_item
    assert summary_batch.targets["region_response_target"].shape == (1, 8, 26)
    assert np.allclose(
        summary_batch.targets["region_response_target"].numpy()[0],
        query["dense_region_target"],
    )
