from __future__ import annotations

from dataclasses import dataclass

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


def split_by_state(
    refs: list[WindowRef],
    seed: int = 20260924,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, list[WindowRef]]:
    groups = sorted({ref.record.structural_state_id for ref in refs})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_train = int(len(groups) * train_fraction)
    n_val = int(len(groups) * validation_fraction)
    train_ids = set(groups[:n_train])
    val_ids = set(groups[n_train : n_train + n_val])
    result = {"train": [], "validation": [], "test": []}
    for ref in refs:
        key = "train" if ref.record.structural_state_id in train_ids else "validation" if ref.record.structural_state_id in val_ids else "test"
        result[key].append(ref)
    return result
