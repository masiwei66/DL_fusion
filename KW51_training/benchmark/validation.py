from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_public_contract(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Loading the public benchmark contract requires PyYAML") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: public benchmark contract must be a mapping")
    if payload.get("schema_version") != "kw51_public_benchmark_v1":
        raise ValueError(f"{path}: unsupported public benchmark schema_version")
    input_contract = payload.get("input_contract")
    if not isinstance(input_contract, dict):
        raise ValueError(f"{path}: input_contract must be a mapping")
    allowed = set(input_contract.get("allowed_fields", ()))
    forbidden = set(input_contract.get("forbidden_fields", ()))
    if not allowed or not forbidden or allowed & forbidden:
        raise ValueError("input_contract allowed/forbidden fields must be disjoint and non-empty")
    required_forbidden = {
        "structural_state_id",
        "Id_rel",
        "Ad_rel",
        "complete_node_field",
        "oracle_derived_labels",
    }
    if not required_forbidden <= forbidden:
        raise ValueError(
            "input_contract is missing forbidden fields: "
            + ", ".join(sorted(required_forbidden - forbidden))
        )
    if allowed & required_forbidden:
        raise ValueError("a forbidden supervision field is also declared as an input")
    tracks = payload.get("tracks", {})
    if not isinstance(tracks, dict) or not {"S", "R"} <= set(tracks):
        raise ValueError("contract must define Track S and Track R")
    if tracks["S"].get("labels_available") is not True:
        raise ValueError("Track S must declare labels_available=true")
    if tracks["R"].get("labels_available") is not False:
        raise ValueError("Track R must declare labels_available=false")
    if tracks["R"].get("state_accuracy") is not False:
        raise ValueError("Track R must prohibit real-side state accuracy")
    return payload


def validate_state_manifest(payload: dict[str, Any]) -> tuple[str, ...]:
    if payload.get("schema_version") != "kw51_state_manifest_v1":
        raise ValueError("unsupported state manifest schema_version")
    states = payload.get("states")
    families = payload.get("candidate_families")
    if not isinstance(states, list) or not isinstance(families, list):
        raise ValueError("state manifest must contain states and candidate_families lists")
    state_ids = [str(row.get("structural_state_id", "")) for row in states]
    if any(not state_id for state_id in state_ids) or len(set(state_ids)) != len(state_ids):
        raise ValueError("state manifest structural_state_id values must be unique and non-empty")
    expected_count = int(payload.get("state_count", -1))
    if expected_count != len(state_ids):
        raise ValueError("state manifest state_count does not match states")
    if int(payload.get("trajectory_count", -1)) != sum(
        int(row.get("trajectory_count", 0)) for row in states
    ):
        raise ValueError("state manifest trajectory_count does not match per-state counts")

    family_ids = [str(row.get("family_id", "")) for row in families]
    if any(not family_id for family_id in family_ids) or len(set(family_ids)) != len(family_ids):
        raise ValueError("candidate family ids must be unique and non-empty")
    family_by_id = {str(row["family_id"]): row for row in families}
    family_members: dict[str, set[str]] = {family_id: set() for family_id in family_ids}
    for row in states:
        state_id = str(row["structural_state_id"])
        family_id = str(row.get("candidate_family_id", ""))
        if family_id not in family_by_id:
            raise ValueError(f"state {state_id} references an unknown candidate family")
        if int(row.get("trajectory_count", 0)) < 1:
            raise ValueError(f"state {state_id} must have at least one trajectory")
        family_members[family_id].add(state_id)
    for family in families:
        family_id = str(family["family_id"])
        declared = {str(state_id) for state_id in family.get("state_ids", ())}
        if declared != family_members[family_id]:
            raise ValueError(f"candidate family {family_id} state membership is inconsistent")
        if not math.isfinite(float(family.get("Ad_rel", float("nan")))):
            raise ValueError(f"candidate family {family_id} has a non-finite Ad_rel label")
    if set().union(*family_members.values()) != set(state_ids):
        raise ValueError("candidate families do not cover the state manifest")
    return tuple(sorted(state_ids, key=_state_sort_key))


def validate_public_split(
    split_payload: dict[str, Any], state_manifest: dict[str, Any]
) -> None:
    state_ids = set(validate_state_manifest(state_manifest))
    if split_payload.get("schema_version") != "kw51_state_cv5_v1":
        raise ValueError("unsupported state CV split schema_version")
    if split_payload.get("split_unit") != "structural_state_id":
        raise ValueError("state CV split_unit must be structural_state_id")
    if split_payload.get("state_manifest_sha256") != canonical_sha256(state_manifest):
        raise ValueError("split state manifest hash does not match state manifest")
    folds = split_payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("state CV split must contain exactly five folds")

    family_for_state = {
        str(row["structural_state_id"]): str(row["candidate_family_id"])
        for row in state_manifest["states"]
    }
    test_appearances: list[str] = []
    validation_appearances: list[str] = []
    for expected_fold, fold in enumerate(folds):
        if int(fold.get("fold", -1)) != expected_fold:
            raise ValueError("folds must be ordered and numbered 0 through 4")
        roles = {
            "train": [str(value) for value in fold.get("train_state_ids", ())],
            "validation": [str(value) for value in fold.get("validation_state_ids", ())],
            "test": [str(value) for value in fold.get("test_state_ids", ())],
        }
        for role, values in roles.items():
            if len(values) != len(set(values)):
                raise ValueError(f"fold {expected_fold} has duplicate {role} states")
            unknown = set(values) - state_ids
            if unknown:
                raise ValueError(f"fold {expected_fold} contains unknown states: {sorted(unknown)}")
        train, validation, test = (set(roles[name]) for name in ("train", "validation", "test"))
        if train & validation or train & test or validation & test:
            raise ValueError(f"fold {expected_fold} state roles must be disjoint")
        if train | validation | test != state_ids:
            raise ValueError(f"fold {expected_fold} does not partition the state manifest")
        if (len(train), len(validation), len(test)) != (15, 5, 5):
            raise ValueError(f"fold {expected_fold} must contain 15/5/5 train/validation/test states")
        for family_id in set(family_for_state.values()):
            members = {state for state, family in family_for_state.items() if family == family_id}
            if (len(train & members), len(validation & members), len(test & members)) != (3, 1, 1):
                raise ValueError(f"fold {expected_fold} is not stratified for {family_id}")
        test_appearances.extend(roles["test"])
        validation_appearances.extend(roles["validation"])
    if set(test_appearances) != state_ids or len(test_appearances) != len(state_ids):
        raise ValueError("every state must appear in test exactly once across five folds")
    if set(validation_appearances) != state_ids or len(validation_appearances) != len(state_ids):
        raise ValueError("every state must appear in validation exactly once across five folds")


def _state_sort_key(state_id: str) -> tuple[int, int | str]:
    try:
        return 0, int(state_id)
    except ValueError:
        return 1, state_id

