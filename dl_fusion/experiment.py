"""Configuration conventions and bounded data selection shared by both tasks."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SmokeConfig:
    enabled: bool = False
    states_per_split: int = 2
    records_per_state: int = 2
    windows_per_state: int = 2

    def __post_init__(self) -> None:
        if self.states_per_split < 2:
            raise ValueError("smoke.states_per_split must be >= 2 for candidate ranking")
        if self.records_per_state < 2:
            raise ValueError("smoke.records_per_state must be >= 2 for query exclusion")
        if self.windows_per_state < 1:
            raise ValueError("smoke.windows_per_state must be >= 1")


def model_parameters(raw_model: dict[str, Any]) -> dict[str, Any]:
    """Use kwargs for new YAMLs; accept params/flat legacy forms without ambiguity."""
    explicit = [key for key in ("kwargs", "params") if key in raw_model]
    if explicit:
        values = [dict(raw_model[key] or {}) for key in explicit]
        if any(value != values[0] for value in values[1:]):
            raise ValueError("model.kwargs and model.params must agree when both are set")
        extra = set(raw_model) - set(explicit)
        if extra:
            raise ValueError(f"model parameters must be inside model.kwargs: {sorted(extra)}")
        return values[0]
    return dict(raw_model)


def validate_training_settings(train: Any) -> None:
    if train.epochs < 1 or train.batch_size < 1:
        raise ValueError("train.epochs and train.batch_size must be >= 1")
    if not 0 <= train.seed < 2**32:
        raise ValueError("train.seed must be in [0, 2**32)")
    if train.cpu_threads is not None and train.cpu_threads < 1:
        raise ValueError("train.cpu_threads must be >= 1 when set")
    if train.device not in ("auto", "cpu", "cuda") and not str(train.device).startswith("cuda:"):
        raise ValueError("train.device must be auto, cpu, cuda, or cuda:N")


def limit_states(splits: dict[str, list[Any]], smoke: SmokeConfig) -> dict[str, list[Any]]:
    """Select states only AFTER the real state-level split, preserving disjointness."""
    if not smoke.enabled:
        return splits
    result = {}
    for role, items in splits.items():
        def state(item):
            return str(getattr(item, "record", item).structural_state_id)

        selected = set(sorted({state(item) for item in items})[:smoke.states_per_split])
        result[role] = [item for item in items if state(item) in selected]
        if len(selected) < 2:
            raise ValueError(f"smoke needs at least two states in {role}, found {len(selected)}")
    return result


def limit_per_state(
    items: Iterable[Any], count: int, *, distinct_by: Any | None = None
) -> list[Any]:
    counts: dict[str, int] = defaultdict(int)
    seen: dict[str, set[Any]] = defaultdict(set)
    selected = []
    for item in items:
        state = str(getattr(item, "record", item).structural_state_id)
        if counts[state] >= count:
            continue
        if distinct_by is not None:
            key = distinct_by(item)
            if key in seen[state]:
                continue
            seen[state].add(key)
        selected.append(item)
        counts[state] += 1
    return selected


def evaluation_candidate_pools(
    records: dict[str, list[Any]], all_states: Iterable[str], scope: str
) -> dict[str, list[str]]:
    if scope not in ("split", "all_states"):
        raise ValueError("candidate_pool_scope must be split or all_states")
    shared_gallery = sorted({str(state) for state in all_states})
    pools = {}
    for role in ("validation", "test"):
        states = (
            shared_gallery
            if scope == "all_states"
            else sorted({str(record.structural_state_id) for record in records[role]})
        )
        if not states:
            raise ValueError(f"No candidate states available for {role}")
        pools[role] = states
    return pools
