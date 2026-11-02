import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from KW51_training.config import FuzzyConfig, load_experiment_config
from KW51_training.data.fem_insar import (
    AdGroup,
    AdGroupMap,
    build_fem_archive_index,
)
from KW51_training.data.fuzzy_range import (
    FuzzyQueryRef,
    FuzzyRangeSource,
    build_deck_geometry,
    collate_fuzzy_queries,
)
from KW51_training.models.fuzzy_response import FuzzyResponseModel
from KW51_training.training.fuzzy_group_audit import run_fuzzy_group_audit
from KW51_training.training.fuzzy_losses import (
    fuzzy_state_loss,
    group_cross_entropy,
)
from KW51_training.training.fuzzy_metrics import (
    group_metrics,
    group_per_state,
)


def _item(group_id: int | None = None, point_count: int = 4) -> dict[str, object]:
    rng = np.random.default_rng(point_count)
    window = 4
    metadata = {
        "sample_id": f"sample-{point_count}",
        "structural_state_id": "1",
        "trajectory_id": "traj",
        "observation_realization_id": 0,
        "point_count": point_count,
        "target_mean_mm": 0.0,
        "target_std_mm": 1.0,
    }
    item: dict[str, object] = {
        "values": rng.normal(size=(window, point_count)).astype(np.float32),
        "valid_mask": np.ones((window, point_count), dtype=bool),
        "quality": np.ones((window, point_count), dtype=np.float32),
        "point_coords": rng.normal(size=(point_count, 2)).astype(np.float32),
        "point_present": np.ones(point_count, dtype=bool),
        "mapping_distance": np.zeros(point_count, dtype=np.float32),
        "mapping_valid": np.ones(point_count, dtype=bool),
        "time_features": np.zeros((window, 2), dtype=np.float32),
        "bag_features": rng.normal(size=9).astype(np.float32),
        "target_samples": rng.normal(size=(3, window)).astype(np.float32),
        "target_mean": rng.normal(size=window).astype(np.float32),
        "target_std": np.ones(window, dtype=np.float32),
        "target_quantiles": rng.normal(size=(window, 3)).astype(np.float32),
        "target_sample_valid": np.ones((3, window), dtype=bool),
        "target_time_valid": np.ones(window, dtype=bool),
        "bag_mean_mm": rng.normal(size=window).astype(np.float32),
        "metadata": metadata,
    }
    if group_id is not None:
        item["ad_group_id"] = np.int64(group_id)
        item["ad_rel"] = np.float32(0.5 + 0.25 * group_id)
        item["id_rel"] = np.float32(1.0)
    return item


def _model(target_length: int, group_count: int | None) -> FuzzyResponseModel:
    return FuzzyResponseModel(
        "lstm",
        {
            "input_dim": 6,
            "input_mode": "node",
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False,
        },
        target_length,
        group_count=group_count,
        head_hidden=8,
        dropout=0.0,
        bag_feature_dropout=0.0,
    )


def test_collate_carries_group_targets() -> None:
    batch = collate_fuzzy_queries([_item(0), _item(3)])
    assert batch.targets["ad_group_id"].tolist() == [0, 3]
    assert batch.targets["ad_rel"].tolist() == pytest.approx([0.5, 1.25])
    assert batch.targets["id_rel"].tolist() == pytest.approx([1.0, 1.0])
    plain = collate_fuzzy_queries([_item(), _item()])
    assert "ad_group_id" not in plain.targets


def test_state_only_model_has_no_response_head() -> None:
    model = _model(target_length=0, group_count=5)
    outputs = model(collate_fuzzy_queries([_item(0), _item(2)]))
    assert set(outputs) == {"group_logits"}
    assert outputs["group_logits"].shape == (2, 5)
    assert model.response_head is None


def test_state_aux_model_outputs_both_heads() -> None:
    model = _model(target_length=4, group_count=5)
    outputs = model(collate_fuzzy_queries([_item(0), _item(2)]))
    assert set(outputs) == {"group_logits", "mean", "logvar"}
    assert outputs["group_logits"].shape == (2, 5)
    assert outputs["mean"].shape == (2, 4)
    assert outputs["logvar"].shape == (2, 4)


def test_group_cross_entropy_and_state_loss() -> None:
    logits = torch.tensor([[8.0, 0.0], [0.0, 8.0]])
    ids = torch.tensor([0, 1])
    loss = group_cross_entropy(logits, ids)
    assert float(loss) < 1.0e-3
    batch = collate_fuzzy_queries([_item(0), _item(1)])
    outputs = _model(target_length=4, group_count=2)(batch)
    state_loss, components = fuzzy_state_loss(outputs, batch)
    assert torch.isfinite(state_loss)
    assert set(components) == {"group_ce", "group_accuracy"}
    aux_loss, aux_components = fuzzy_state_loss(
        outputs, batch, response_auxiliary_weight=0.5
    )
    assert torch.isfinite(aux_loss)
    assert set(aux_components) == {
        "group_ce",
        "group_accuracy",
        "response_nll",
        "response_crps",
    }


def test_group_metrics_perfect_predictions() -> None:
    values = [0.5, 0.75, 1.0, 1.25, 1.5]
    logits = torch.full((5, 5), -10.0)
    for index in range(5):
        logits[index, index] = 10.0
    ids = torch.arange(5)
    metrics = group_metrics(logits, ids, values)
    assert metrics["top1_accuracy"] == pytest.approx(1.0)
    assert metrics["top3_accuracy"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["nll"] < 1.0e-3
    assert metrics["ece"] == pytest.approx(0.0, abs=1.0e-4)
    assert metrics["brier"] < 1.0e-3
    assert np.asarray(metrics["confusion_matrix"]).diagonal().tolist() == [1] * 5
    assert metrics["ad_rel_abs_error"] == pytest.approx(0.0, abs=1.0e-6)
    assert metrics["ad_rel_std_mean"] == pytest.approx(0.0, abs=1.0e-4)


def test_group_metrics_detects_uniform_prior() -> None:
    values = [0.5, 0.75, 1.0, 1.25, 1.5]
    logits = torch.zeros(5, 5)
    ids = torch.arange(5)
    metrics = group_metrics(logits, ids, values)
    assert metrics["top1_accuracy"] == pytest.approx(0.2)
    assert metrics["nll"] == pytest.approx(math.log(5.0), rel=1.0e-5)
    chance = float(np.mean(np.asarray(values)))
    assert metrics["ad_rel_expected_mean"] == pytest.approx(chance)


def test_group_per_state_groups_rows() -> None:
    rows = [
        {"structural_state_id": "1", "correct": 1.0, "group_nll": 0.1},
        {"structural_state_id": "1", "correct": 0.0, "group_nll": 0.5},
        {"structural_state_id": "2", "correct": 1.0, "group_nll": 0.2},
    ]
    result = group_per_state(rows)
    assert result["1"]["queries"] == 2
    assert result["1"]["accuracy"] == pytest.approx(0.5)
    assert result["2"]["accuracy"] == pytest.approx(1.0)
    assert len(result["1"]["accuracy_ci95"]) == 2


def _group_map() -> AdGroupMap:
    groups = (
        AdGroup(0, 0.5, ("1", "2")),
        AdGroup(1, 1.5, ("3", "4")),
    )
    return AdGroupMap(
        parameter="Ad_rel",
        groups=groups,
        state_to_group={"1": 0, "2": 0, "3": 1, "4": 1},
        state_to_value={"1": 0.5, "2": 0.5, "3": 1.5, "4": 1.5},
    )


class _FakeSource:
    def __init__(self, features: dict[str, float]) -> None:
        self.features = features

    def sample_query(self, record, realization: int, *, with_targets: bool = True):
        value = float(self.features[record.structural_state_id])
        values = np.full((4, 6), value, dtype=np.float64)
        return {
            "values": values,
            "valid_mask": np.ones((4, 6), dtype=bool),
        }


def _refs(states: list[str]) -> list[FuzzyQueryRef]:
    return [
        FuzzyQueryRef(
            record=SimpleNamespace(structural_state_id=state), realization=0
        )
        for state in states
    ]


def test_group_audit_separable_and_deterministic() -> None:
    group_map = _group_map()
    source = _FakeSource({"1": 0.0, "2": 0.0, "3": 10.0, "4": 10.0})
    train_refs = _refs(["1", "2", "3", "4"])
    test_refs = _refs(["1", "3"])
    first = run_fuzzy_group_audit(
        source,
        group_map,
        train_refs,
        test_refs,
        max_refs_per_state=2,
        noise_std_mm=0.1,
        point_bias_std_mm=0.1,
        min_accuracy=0.9,
    )
    second = run_fuzzy_group_audit(
        source,
        group_map,
        train_refs,
        test_refs,
        max_refs_per_state=2,
        noise_std_mm=0.1,
        point_bias_std_mm=0.1,
        min_accuracy=0.9,
    )
    assert first == second
    assert first["status"] == "PASS"
    assert first["top1_accuracy"] == pytest.approx(1.0)
    assert first["mrr"] == pytest.approx(1.0)
    assert np.asarray(first["confusion_matrix"]).diagonal().tolist() == [1, 1]


def test_group_audit_fails_on_indistinguishable_groups() -> None:
    group_map = _group_map()
    source = _FakeSource({"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0})
    result = run_fuzzy_group_audit(
        source,
        group_map,
        _refs(["1", "2", "3", "4"]),
        _refs(["1", "3"]),
        max_refs_per_state=1,
        noise_std_mm=0.1,
        point_bias_std_mm=0.1,
        min_accuracy=0.9,
    )
    assert result["status"] == "FAIL"
    assert result["reasons"]


def test_group_audit_snr_threshold_is_configurable() -> None:
    group_map = _group_map()
    source = _FakeSource({"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0})
    result = run_fuzzy_group_audit(
        source,
        group_map,
        _refs(["1", "2", "3", "4"]),
        _refs(["1", "3"]),
        max_refs_per_state=1,
        noise_std_mm=0.1,
        point_bias_std_mm=0.1,
        min_accuracy=0.0,
        min_snr=0.0,
    )
    assert result["status"] == "PASS"
    assert result["thresholds"]["min_snr"] == pytest.approx(0.0)


def test_state_configs_load() -> None:
    smoke = load_experiment_config(
        "KW51_training/configs/kw51_fuzzy_state_lstm_smoke.yaml"
    )
    assert smoke.fuzzy is not None
    assert smoke.fuzzy.primary_target == "state"
    assert smoke.fuzzy.formal_run is False
    assert smoke.fuzzy.group_audit is False
    formal = load_experiment_config(
        "KW51_training/configs/kw51_fuzzy_state_lstm.yaml"
    )
    assert formal.fuzzy is not None
    assert formal.fuzzy.primary_target == "state"
    assert formal.fuzzy.formal_run is True
    assert formal.fuzzy.response_auxiliary_weight == pytest.approx(0.0)
    assert formal.fuzzy.group_audit_min_snr == pytest.approx(1.0)
    aux = load_experiment_config(
        "KW51_training/configs/kw51_fuzzy_state_lstm_aux.yaml"
    )
    assert aux.fuzzy is not None
    assert aux.fuzzy.response_auxiliary_weight > 0.0


_SMOKE_ARCHIVE = Path("KW51_training/runs/_fuzzy_smoke_archive")


@pytest.mark.skipif(
    not _SMOKE_ARCHIVE.exists(), reason="local 6-file smoke archive not present"
)
def test_sample_query_can_skip_response_targets() -> None:
    index = build_fem_archive_index(_SMOKE_ARCHIVE)
    geometry = build_deck_geometry(
        index.records[0], np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    config = FuzzyConfig(target_range_count=4)
    source = FuzzyRangeSource(index, geometry, config, seed=0)
    record = index.records[0]
    plain = source.sample_query(record, 0, with_targets=False)
    assert plain["target_samples"].shape == (4, 26)
    assert not plain["target_sample_valid"].any()
    assert plain["metadata"]["target_range_count"] == 0
    full = source.sample_query(record, 0, with_targets=True)
    assert full["metadata"]["target_range_count"] == 4
    assert np.any(full["target_time_valid"])


def test_fuzzy_config_state_validation() -> None:
    with pytest.raises(ValueError):
        FuzzyConfig(primary_target="bad")
    with pytest.raises(ValueError):
        FuzzyConfig(response_auxiliary_weight=-0.1)
    with pytest.raises(ValueError):
        FuzzyConfig(group_audit_min_accuracy=1.5)
    with pytest.raises(ValueError):
        FuzzyConfig(group_audit_min_snr=-1.0)
