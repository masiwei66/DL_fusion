from pathlib import Path

import numpy as np
import pytest
import torch

from KW51_training.config import FuzzyConfig, load_experiment_config
from KW51_training.data.fem_insar import (
    build_fem_archive_index,
    derive_ad_groups,
)
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
    estimate_region_values,
)
from KW51_training.data.real_insar_density import audit_real_insar_density
from KW51_training.models.fuzzy_region import FuzzyRegionModel
from KW51_training.models.fuzzy_region_summary import FuzzyRegionSummaryModel
from KW51_training.training.fuzzy_region_audit import (
    configured_protocol,
    run_fuzzy_region_audit,
)
from KW51_training.training.fuzzy_region_runner import _cap_records_per_state
from types import SimpleNamespace

_SMOKE = Path("KW51_training/runs/_fuzzy_smoke_archive")
_SKIP = pytest.mark.skipif(
    not _SMOKE.exists(), reason="local 6-file smoke archive not present"
)


@pytest.fixture(scope="module")
def smoke_setup():
    index = build_fem_archive_index(_SMOKE)
    geometry = build_deck_geometry(
        index.records[0], np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    return index, geometry


def _normalization() -> FuzzyNormalization:
    return FuzzyNormalization(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, "disabled", ())


def _model_params() -> dict[str, object]:
    return {
        "input_dim": 6,
        "input_mode": "node",
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
    }


@_SKIP
def test_region_layout_is_deterministic_and_covers_deck(smoke_setup) -> None:
    _, geometry = smoke_setup
    first = build_region_layout(geometry, mode="longitudinal", rows=8, cols=1)
    second = build_region_layout(geometry, mode="longitudinal", rows=8, cols=1)
    assert first.to_dict() == second.to_dict()
    assert first.region_count == 8
    assert len(first.s_edges) == 9
    assert np.all(np.diff(first.s_edges) > 0.0)
    assert first.s_edges[0] <= geometry.deck_s.min()
    assert first.s_edges[-1] >= geometry.deck_s.max()
    grid = build_region_layout(geometry, mode="grid", rows=4, cols=2)
    assert grid.region_count == 8
    assert len(grid.t_edges) == 3


@_SKIP
def test_region_source_shapes_and_masks(smoke_setup) -> None:
    index, geometry = smoke_setup
    layout = build_region_layout(geometry, mode="longitudinal", rows=8, cols=1)
    config = FuzzyConfig(
        region_point_budget=80,
        region_missing_probability=0.0,
        region_min_valid_points=1,
    )
    source = FuzzyRegionSource(index, geometry, config, seed=0)
    query = source.sample_region_query(index.records[0], 0, layout)
    assert query["values"].shape == (26, 80)
    assert query["region_index"].shape == (80,)
    assert np.bincount(query["region_index"]).tolist() == [10] * 8
    assert query["region_present"].all()
    assert np.all(query["region_coverage"] > 0.0)
    assert query["bag_features"].shape == (8,)

    missing = FuzzyConfig(
        region_point_budget=80,
        region_missing_probability=1.0,
        region_min_valid_points=1,
    )
    source_missing = FuzzyRegionSource(index, geometry, missing, seed=0)
    query_missing = source_missing.sample_region_query(
        index.records[0], 0, layout
    )
    assert not query_missing["region_present"].any()
    assert not query_missing["mapping_valid"].any()
    assert np.allclose(query_missing["region_coverage"], 0.0)


def test_region_estimator_recovers_linear_surface() -> None:
    layout = build_region_layout(
        type(
            "Geometry",
            (),
            {
                "deck_s": np.asarray([0.0, 1.0], dtype=np.float32),
                "deck_t": np.asarray([-1.0, 1.0], dtype=np.float32),
            },
        )(),
        mode="longitudinal",
        rows=1,
        cols=1,
    )
    rng = np.random.default_rng(0)
    count = 20
    s = rng.uniform(0.0, 1.0, count)
    t = rng.uniform(-1.0, 1.0, count)
    coords = np.c_[s, t].astype(np.float32)
    values = 2.0 + 3.0 * s - 4.0 * t
    values = np.stack([values, values + 1.0, values + 2.0], axis=0)
    valid = np.ones_like(values, dtype=bool)
    estimates, estimate_valid = estimate_region_values(
        values,
        valid,
        np.zeros(count, dtype=np.int64),
        np.asarray([True]),
        coords,
        layout,
        estimator="plane",
    )
    center_s, center_t = layout.center(0)
    expected = np.asarray([2.0 + 3.0 * center_s - 4.0 * center_t])
    assert estimate_valid.all()
    assert np.allclose(estimates[:, 0], expected[:1], atol=1e-5)
    assert np.allclose(estimates[:, 1], expected[:1] + 1.0, atol=1e-5)


def test_summary_and_point_batches_feed_models(smoke_setup) -> None:
    index, geometry = smoke_setup
    layout = build_region_layout(geometry, mode="longitudinal", rows=8, cols=1)
    config = FuzzyConfig(
        region_point_budget=80,
        region_missing_probability=0.0,
        region_min_valid_points=1,
    )
    source = FuzzyRegionSource(index, geometry, config, seed=0)
    refs = [FuzzyQueryRef(index.records[0], 0)]
    normalization = _normalization()
    point_dataset = FuzzyRegionQueryDataset(
        refs, source, normalization, layout
    )
    point_batch = collate_fuzzy_region_queries(
        [point_dataset[0], point_dataset[0]]
    )
    point_model = FuzzyRegionModel(
        "lstm",
        _model_params(),
        region_count=layout.region_count,
        group_count=5,
        head_hidden=8,
        dropout=0.0,
        bag_feature_dropout=0.0,
    )
    assert point_model(point_batch)["group_logits"].shape == (2, 5)

    summary_dataset = FuzzyRegionSummaryDataset(
        refs,
        source,
        normalization,
        layout,
        estimator="quadratic",
    )
    summary_batch = collate_fuzzy_region_summaries(
        [summary_dataset[0], summary_dataset[0]]
    )
    assert summary_batch.values.shape == (2, 26, 8)
    summary_params = _model_params() | {
        "input_mode": "region",
        "region_pooling": "flatten",
    }
    summary_model = FuzzyRegionSummaryModel(
        "lstm",
        summary_params,
        region_count=layout.region_count,
        group_count=5,
        head_hidden=8,
        dropout=0.0,
        bag_feature_dropout=0.0,
    )
    assert summary_model(summary_batch)["group_logits"].shape == (2, 5)


@_SKIP
def test_region_audit_schema(smoke_setup) -> None:
    index, geometry = smoke_setup
    group_map = derive_ad_groups(index, expected_count=0)
    config = FuzzyConfig(
        region_point_budget=80,
        region_missing_probability=0.0,
        region_min_valid_points=1,
        region_audit_realizations=1,
        region_audit_seeds=1,
        region_audit_folds=1,
        region_estimator="quadratic",
    )
    protocol = configured_protocol(config)
    audit = run_fuzzy_region_audit(
        index,
        geometry,
        group_map,
        config,
        seed=0,
        split_seed=0,
        train_fraction=0.6,
        validation_fraction=0.2,
        split_mode="random",
        fold_indices=[0],
        protocols=[protocol],
        configured=protocol,
    )
    assert audit["schema"] == "kw51_fuzzy_region_audit_v1"
    assert audit["configured_protocol"]["name"] == protocol.name
    assert protocol.name in audit["protocols"]
    summary = audit["protocols"][protocol.name]["summary"]
    assert summary["folds"] == 1
    assert summary["seeds"] == 1
    assert 0.0 <= summary["state_macro_top1"] <= 1.0


def test_region_configs_load() -> None:
    formal = load_experiment_config(
        "KW51_training/configs/kw51_fuzzy_region_summary_lstm.yaml"
    )
    assert formal.fuzzy is not None
    assert formal.fuzzy.observation_mode == "region_bag"
    assert formal.fuzzy.region_model == "region_summary"
    assert formal.fuzzy.region_estimator == "quadratic"
    smoke = load_experiment_config(
        "KW51_training/configs/kw51_fuzzy_region_summary_lstm_smoke.yaml"
    )
    assert smoke.fuzzy is not None
    assert smoke.fuzzy.region_model == "region_summary"
    assert smoke.fuzzy.region_audit is False


def test_region_config_validation() -> None:
    with pytest.raises(ValueError):
        FuzzyConfig(region_estimator="bad")
    with pytest.raises(ValueError):
        FuzzyConfig(region_model="bad")
    with pytest.raises(ValueError):
        FuzzyConfig(region_rows=8, region_cols=2, region_mode="longitudinal")
    with pytest.raises(ValueError):
        FuzzyConfig(region_rows=8, region_cols=1, region_point_budget=8, region_min_valid_points=3)


def test_cap_records_per_state() -> None:
    records = [
        SimpleNamespace(structural_state_id=str(state), sample=index)
        for index, state in enumerate((1, 1, 1, 1, 2, 2, 2, 3))
    ]
    capped = _cap_records_per_state(records, 2, seed=27)
    repeated = _cap_records_per_state(records, 2, seed=27)
    assert [record.structural_state_id for record in capped] == ["1", "1", "2", "2", "3"]
    assert [record.sample for record in capped] == [record.sample for record in repeated]
    assert len(_cap_records_per_state(records, 0)) == len(records)


def test_real_insar_density_audit(tmp_path) -> None:
    path = tmp_path / "synthetic_track.csv"
    path.write_text(
        "pid,easting,northing,los_east,los_north,los_up,temporal_coherence,20200101,20200102\n"
        "1,0,0,0,0,1,0.9,1.0,1.1\n"
        "2,10,0,0,0,1,0.8,2.0,2.1\n"
        "3,20,2,0,0,1,0.7,3.0,3.1\n",
        encoding="utf-8",
    )
    audit = audit_real_insar_density([path])
    assert audit["track_count"] == 1
    assert audit["total_points_across_tracks"] == 3
    assert audit["unique_points_across_tracks"] == 3
    assert audit["points_per_track"]["median"] == 3.0
    assert audit["upper_bound_points_per_region_8"] == pytest.approx(3.0 / 8.0)
    assert audit["density_support"] == "below_lowest_synthetic_budget_per_track"
    assert audit["tracks"][0]["per_region_density_status"] == (
        "unavailable_pending_geo_transform"
    )


@_SKIP
def test_region_budget_audit_sweep(smoke_setup) -> None:
    index, geometry = smoke_setup
    group_map = derive_ad_groups(index, expected_count=0)
    config = FuzzyConfig(
        region_point_budget=40,
        region_audit_point_budgets=(40, 80),
        region_missing_probability=0.0,
        region_min_valid_points=1,
        region_audit_realizations=1,
        region_audit_seeds=1,
        region_audit_folds=1,
        region_estimator="quadratic",
    )
    protocol = configured_protocol(config)
    audit = run_fuzzy_region_audit(
        index,
        geometry,
        group_map,
        config,
        seed=0,
        split_seed=0,
        train_fraction=0.6,
        validation_fraction=0.2,
        split_mode="random",
        fold_indices=[0],
        protocols=[protocol],
        configured=protocol,
    )
    assert audit["point_budgets"] == [40, 80]
    assert audit["configured_budget"] == 40
    entry = audit["protocols"][protocol.name]
    assert set(entry["budgets"]) == {"40", "80"}


def test_budget_audit_config_load_and_validation() -> None:
    config = load_experiment_config(
        "KW51_training/configs/kw51_fuzzy_region_budget_audit.yaml"
    )
    assert config.fuzzy is not None
    assert list(config.fuzzy.region_audit_point_budgets) == [40, 80, 160, 640]
    assert config.fuzzy.train_trajectories_per_state == 10
    with pytest.raises(ValueError):
        FuzzyConfig(train_trajectories_per_state=-1)
    with pytest.raises(ValueError):
        FuzzyConfig(region_audit_point_budgets=(0, 40))
