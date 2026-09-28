from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifest import ArchiveIndex, ArchiveRecord


@dataclass(frozen=True)
class WindowRef:
    record: ArchiveRecord
    start: int
    length: int
    target_index: int


def make_window_refs(index: ArchiveIndex, length: int = 6, stride: int = 1) -> list[WindowRef]:
    refs: list[WindowRef] = []
    for record in index.records:
        shape = record.metadata.get("array_shapes", {}).get("responses/fem/disp_at_acquisition")
        steps = int(shape[0]) if shape else 26
        if length > steps:
            continue
        for start in range(0, steps - length + 1, stride):
            refs.append(WindowRef(record, start, length, start + length - 1))
    return refs


def _allocate_states(
    states_by_family: dict[str, set[str]],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Assign every structural state to exactly one split, one family at a time.

    State families are tiny and heavily skewed, so a global shuffle can drop a
    whole mechanism out of training - and a 40-class head then cannot predict it,
    which is what made validation loss exceed the uniform-guessing baseline.
    Allocating per family guarantees every family reaches the training split.
    """
    del validation_fraction  # the remainder is split evenly between validation and test
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for family_index, family in enumerate(sorted(states_by_family)):
        states = sorted(states_by_family[family])
        rng.shuffle(states)
        count = len(states)
        if count <= 2:
            # Too small to split three ways: keep one in training so it stays learnable.
            train, val = count, max(count - 1, 0)
        else:
            train = max(1, int(count * train_fraction + 0.5))
            rest = count - train
            val = rest // 2
            # An odd remainder would always favour test; alternate to stay balanced.
            if rest % 2 and family_index % 2 == 0:
                val += 1
        for position, state in enumerate(states):
            if position < train:
                assignment[state] = "train"
            elif position < train + val:
                assignment[state] = "validation"
            else:
                assignment[state] = "test"
    return assignment


def split_by_state(
    refs: list[WindowRef],
    seed: int = 20260924,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    *,
    stratify: bool = True,
) -> dict[str, list[WindowRef]]:
    if stratify:
        states_by_family: dict[str, set[str]] = {}
        for ref in refs:
            states_by_family.setdefault(ref.record.state_family_id, set()).add(ref.record.structural_state_id)
        assignment = _allocate_states(states_by_family, train_fraction, validation_fraction, seed)
    else:
        groups = sorted({ref.record.structural_state_id for ref in refs})
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        n_train = int(len(groups) * train_fraction)
        n_val = int(len(groups) * validation_fraction)
        assignment = {state: "train" for state in groups[:n_train]}
        assignment.update({state: "validation" for state in groups[n_train : n_train + n_val]})
        assignment.update({state: "test" for state in groups[n_train + n_val :]})
    result: dict[str, list[WindowRef]] = {"train": [], "validation": [], "test": []}
    for ref in refs:
        result[assignment[ref.record.structural_state_id]].append(ref)
    return result


def missing_train_families(splits: dict[str, list[WindowRef]]) -> list[str]:
    """Families that appear in validation or test but never in training."""
    seen = {name: {ref.record.state_family_id for ref in refs} for name, refs in splits.items()}
    return sorted((seen["validation"] | seen["test"]) - seen["train"])


def split_distribution(splits: dict[str, list[WindowRef]]) -> dict[str, Any]:
    """Summarize state-family and structural-state counts for split auditing."""
    result: dict[str, Any] = {}
    for name, refs in splits.items():
        families = {ref.record.state_family_id for ref in refs}
        states = {ref.record.structural_state_id for ref in refs}
        result[name] = {"windows": len(refs), "unique_states": len(states), "unique_state_families": len(families)}
    result["missing_from_train"] = missing_train_families(splits)
    return result
