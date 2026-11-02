from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import numpy as np

import KW51_training.training.public_benchmark_runner as runner
from KW51_training.training.public_benchmark_runner import (
    _assert_dense_targets_not_model_inputs,
    audit_l1_stratified_target_preflight,
    build_public_benchmark_report,
    select_stratified_preflight_records,
    validate_state_cv_splits,
)


ROOT = Path(__file__).resolve().parents[1]


def _manifests() -> tuple[dict, dict]:
    manifest = json.loads(
        (ROOT / "benchmark" / "state_manifest_v1.json").read_text(encoding="utf-8")
    )
    split = json.loads(
        (ROOT / "benchmark" / "splits" / "state_cv5.json").read_text(encoding="utf-8")
    )
    return manifest, split


def test_fixed_state_cv_split_is_disjoint_complete_and_train_only() -> None:
    manifest, split = _manifests()
    folds = validate_state_cv_splits(manifest, split)
    assert len(folds) == 5
    for fold in folds:
        assert len(fold["train_state_ids"]) == 15
        assert len(fold["validation_state_ids"]) == 5
        assert len(fold["test_state_ids"]) == 5
        assert fold["fitted_state_ids"] == fold["train_state_ids"]


def test_fixed_state_cv_rejects_state_leakage() -> None:
    manifest, split = _manifests()
    invalid = copy.deepcopy(split)
    invalid["folds"][0]["validation_state_ids"][0] = invalid["folds"][0]["train_state_ids"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_state_cv_splits(manifest, invalid)


def test_gate_report_is_complete_and_does_not_claim_performance(tmp_path: Path) -> None:
    report, csv_path = build_public_benchmark_report(
        ROOT / "configs" / "kw51_public_benchmark_v1.yaml",
        output_root=tmp_path,
        archive_root=tmp_path / "archive_unavailable",
    )
    assert report["schema_version"] == "kw51_public_benchmark_report_v1"
    assert report["run_status"] == "NOT_RUN_GATED"
    assert report["claim_scope"]["track_r_state_accuracy"] is False
    assert report["folds"][0]["fitted_state_ids"] == report["folds"][0]["train_state_ids"]
    assert report["task_status"]["L2"]["status"] == "not_run_identifiability_fail"
    assert report["task_status"]["L1"]["status"] == "not_run_archive_validation_fail"
    assert all(not row["metrics"] for row in report["model_runs"])
    assert csv_path.exists()


def test_l1_pilot_config_is_bounded_and_keeps_other_gates_closed() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "kw51_public_benchmark_l1_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["models"] == ["global_mean", "region_mean_mlp"]
    assert config["seeds"] == [20261015]
    assert config["density"]["point_budgets"] == [80]
    assert config["split"]["folds"] == [0, 1, 2, 3, 4]
    assert config["training"]["epochs"] == 2
    assert config["gates"]["L2"]["status"] == "FAIL"
    assert config["gates"]["L2"]["formal_network_training"] is False
    assert config["gates"]["track_r"]["status"] == "BLOCKED"
    assert config["pilot"]["purpose"] == "process_validation_not_benchmark_performance"


def test_l1_pilot_manifest_marks_output_non_publication() -> None:
    path = ROOT / "runs" / "public_benchmark_v1_pilot_l1_80" / "pilot_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["purpose"] == "process_validation_not_benchmark_performance"
    assert manifest["not_for_publication"] is True
    assert manifest["model"] == "region_mean_mlp"
    assert manifest["seed"] == 20261015
    assert manifest["point_budget"] == 80
    assert manifest["folds"] == [0, 1, 2, 3, 4]
    assert manifest["epochs"] == 2


def test_support_token_pilot_config_keeps_kw51_quality_gates_closed() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "kw51_public_benchmark_support_tokens_pilot.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["models"] == ["global_mean", "support_token_gru"]
    assert config["support_tokens"]["bandwidth_scale"] == 0.75
    assert config["support_tokens"]["time_feature_mode"] == "standardized"
    assert config["split"]["folds"] == [0]
    assert config["pilot"]["model"] == "support_token_gru"
    assert config["pilot"]["not_for_publication"] is True
    assert config["gates"]["L2"]["status"] == "FAIL"
    assert config["gates"]["track_r"]["status"] == "BLOCKED"


def _preflight_config() -> dict:
    return {
        "pilot": {
            "seed": 20261015,
            "point_budget": 80,
            "preflight": {
                "trajectories_per_state": 5,
                "minimum_target_valid_fraction": 0.99,
                "minimum_area_coverage": 0.99,
                "minimum_region_present_fraction": 0.90,
                "minimum_state_region_present_fraction": 0.75,
            },
        },
        "region_layout": {"mode": "longitudinal", "rows": 8, "cols": 1},
    }


def _preflight_records(trajectories_per_state: int = 7) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            sample_id=f"state-{state}-sample-{trajectory}",
            structural_state_id=str(state),
            trajectory_id=f"trajectory-{trajectory}",
        )
        for state in range(1, 26)
        for trajectory in range(trajectories_per_state)
    ]


def _mock_preflight_dependencies(monkeypatch, *, nonfinite_target: bool = False) -> None:
    records = _preflight_records()
    monkeypatch.setattr(
        runner, "build_fem_archive_index", lambda _: SimpleNamespace(records=records)
    )
    monkeypatch.setattr(
        runner,
        "_load_training_config",
        lambda *_: SimpleNamespace(
            data=SimpleNamespace(los_vector=(-0.534, -0.105, 0.839), window_length=26),
            fuzzy=object(),
        ),
    )
    monkeypatch.setattr(runner, "_region_fuzzy_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "build_deck_geometry", lambda *args: object())
    monkeypatch.setattr(
        runner, "build_region_layout", lambda *args, **kwargs: SimpleNamespace(region_count=8)
    )

    class FakeSource:
        def __init__(self, *args, **kwargs):
            pass

        def sample_region_query(self, record, realization, layout):
            target = np.zeros((8, 26), dtype=np.float32)
            if nonfinite_target:
                target[0, 0] = np.nan
            return {
                "dense_region_target": target,
                "dense_region_valid": np.ones((8, 26), dtype=bool),
                "dense_region_area_coverage": np.ones((8, 26), dtype=np.float32),
                "region_present": np.ones(8, dtype=bool),
            }

    monkeypatch.setattr(runner, "FuzzyRegionSource", FakeSource)


def test_stratified_preflight_samples_five_distinct_trajectories_per_state(
    monkeypatch, tmp_path: Path
) -> None:
    _mock_preflight_dependencies(monkeypatch)
    result = audit_l1_stratified_target_preflight(
        _preflight_config(), project_root=ROOT.parent, archive_root=tmp_path
    )
    assert result["status"] == "PASS"
    assert result["sampled_record_count"] == 125
    assert result["sampled_state_count"] == 25
    assert all(item["sampled_trajectory_count"] == 5 for item in result["per_state"].values())
    for state_id in result["per_state"]:
        trajectories = {
            item["trajectory_id"]
            for item in result["records"]
            if item["structural_state_id"] == state_id
        }
        assert len(trajectories) == 5


def test_stratified_preflight_fails_closed_on_nonfinite_targets(
    monkeypatch, tmp_path: Path
) -> None:
    _mock_preflight_dependencies(monkeypatch, nonfinite_target=True)
    result = audit_l1_stratified_target_preflight(
        _preflight_config(), project_root=ROOT.parent, archive_root=tmp_path
    )
    assert result["status"] == "FAIL"
    assert len(result["failures"]) == 125
    assert all("nonfinite_dense_target" in item["reason"] for item in result["failures"])


def test_stratified_preflight_selection_fails_when_state_lacks_unique_trajectories() -> None:
    records = _preflight_records(trajectories_per_state=4)
    with pytest.raises(ValueError, match="distinct trajectories"):
        select_stratified_preflight_records(
            records, trajectories_per_state=5, seed=20261015
        )


def test_pilot_runner_writes_blocked_preflight_without_starting_training(
    tmp_path: Path,
) -> None:
    report, _ = build_public_benchmark_report(
        ROOT / "configs" / "kw51_public_benchmark_l1_pilot.yaml",
        output_root=tmp_path,
        archive_root=tmp_path / "archive_unavailable",
        run_l1_audit=True,
        execute_l1=True,
    )
    assert report["run_status"] == "NOT_RUN_L1_GATE_FAIL"
    assert report["L1_pilot_target_preflight"]["status"] == "BLOCKED"
    assert all(
        run["tasks"]["L1"]["metrics_status"] == "not_evaluated"
        for run in report["model_runs"]
    )
    manifest = json.loads((tmp_path / "pilot_manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads(
        (tmp_path / "pilot_target_preflight.json").read_text(encoding="utf-8")
    )
    assert manifest["run_status"] == "NOT_RUN_L1_GATE_FAIL"
    assert preflight["status"] == "BLOCKED"


def test_pilot_model_outputs_are_independent_of_dense_targets() -> None:
    from KW51_training.models.fuzzy_region_mlp import FuzzyRegionMLPModel
    from KW51_training.tests.test_kw51_public_multitask import _batch

    batch = _batch(point_mode=False)
    model = FuzzyRegionMLPModel(
        region_count=3, group_count=2, hidden_dim=8, dropout=0.0
    )
    assert _assert_dense_targets_not_model_inputs(model, batch) == "PASS"
