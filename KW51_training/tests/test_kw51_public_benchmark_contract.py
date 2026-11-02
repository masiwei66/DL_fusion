from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from KW51_training.benchmark.validation import (
    load_public_contract,
    validate_public_split,
    validate_state_manifest,
)


_ROOT = Path("KW51_training/benchmark")


def _frozen_artifacts():
    contract = load_public_contract(_ROOT / "contract_v1.yaml")
    state_manifest = json.loads(
        (_ROOT / "state_manifest_v1.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (_ROOT / "splits/state_cv5.json").read_text(encoding="utf-8")
    )
    return contract, state_manifest, split_manifest


def test_public_contract_separates_track_claims_and_forbids_label_inputs() -> None:
    contract, _, _ = _frozen_artifacts()

    assert contract["schema_version"] == "kw51_public_benchmark_v1"
    assert contract["tracks"]["S"]["labels_available"] is True
    assert contract["tracks"]["R"]["labels_available"] is False
    assert contract["tracks"]["R"]["state_accuracy"] is False
    assert {
        "structural_state_id",
        "Id_rel",
        "Ad_rel",
        "complete_node_field",
        "oracle_derived_labels",
    } <= set(contract["input_contract"]["forbidden_fields"])


def test_state_manifest_and_all_five_fixed_folds_are_complete_and_stratified() -> None:
    _, state_manifest, split_manifest = _frozen_artifacts()

    states = validate_state_manifest(state_manifest)
    validate_public_split(split_manifest, state_manifest)

    assert len(states) == 25
    assert state_manifest["trajectory_count"] == 2500
    assert len(split_manifest["folds"]) == 5
    for fold in split_manifest["folds"]:
        assert len(fold["train_state_ids"]) == 15
        assert len(fold["validation_state_ids"]) == 5
        assert len(fold["test_state_ids"]) == 5


def test_split_rejects_duplicate_or_cross_role_state_ids() -> None:
    _, state_manifest, split_manifest = _frozen_artifacts()
    broken = copy.deepcopy(split_manifest)
    broken["folds"][0]["test_state_ids"].append(
        broken["folds"][0]["train_state_ids"][0]
    )

    with pytest.raises(ValueError, match="disjoint|duplicate|partition"):
        validate_public_split(broken, state_manifest)


def test_split_rejects_state_manifest_membership_drift() -> None:
    _, state_manifest, split_manifest = _frozen_artifacts()
    broken = copy.deepcopy(split_manifest)
    broken["folds"][0]["test_state_ids"][0] = "unlisted_state"

    with pytest.raises(ValueError, match="partition|state manifest|unknown"):
        validate_public_split(broken, state_manifest)
