from __future__ import annotations

from dataclasses import dataclass

import torch

from dl_fusion.experiment import (
    SmokeConfig,
    evaluation_candidate_pools,
    limit_per_state,
    limit_states,
)
from dl_fusion.training.runtime import loader_generator


@dataclass(frozen=True)
class Record:
    structural_state_id: str
    trajectory_id: str


def test_smoke_limits_records_per_state_and_can_require_distinct_trajectories() -> None:
    records = [
        Record("state-a", "trajectory-1"),
        Record("state-a", "trajectory-1"),
        Record("state-a", "trajectory-2"),
        Record("state-b", "trajectory-1"),
    ]

    selected = limit_per_state(
        records, 2, distinct_by=lambda item: item.trajectory_id
    )

    assert selected == [records[0], records[2], records[3]]


def test_smoke_state_cap_is_applied_independently_after_split() -> None:
    splits = {
        "train": [Record("a", "1"), Record("b", "1"), Record("c", "1")],
        "validation": [Record("d", "1"), Record("e", "1"), Record("f", "1")],
        "test": [Record("g", "1"), Record("h", "1"), Record("i", "1")],
    }

    selected = limit_states(splits, SmokeConfig(enabled=True, states_per_split=2))

    assert {role: {item.structural_state_id for item in rows} for role, rows in selected.items()} == {
        "train": {"a", "b"},
        "validation": {"d", "e"},
        "test": {"g", "h"},
    }


def test_eval_split_candidate_pools_do_not_cross_reuse_validation_states() -> None:
    records = {
        "validation": [Record("val-a", "1"), Record("val-b", "1")],
        "test": [Record("test-a", "1"), Record("test-b", "1")],
    }

    pools = evaluation_candidate_pools(records, ["all-a", "all-b"], "split")
    all_state_pools = evaluation_candidate_pools(records, ["all-a", "all-b"], "all_states")

    assert pools == {"validation": ["val-a", "val-b"], "test": ["test-a", "test-b"]}
    assert all_state_pools["validation"] == all_state_pools["test"] == ["all-a", "all-b"]


def test_loader_generator_reproduces_shuffle_order() -> None:
    def order(seed: int) -> list[int]:
        return list(torch.randperm(20, generator=loader_generator(seed)).tolist())

    assert order(17) == order(17)
    assert order(17) != order(18)
