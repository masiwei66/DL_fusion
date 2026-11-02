from __future__ import annotations

from typing import Any

import numpy as np


TASK_LABEL_NAMES = (
    "deck_material",
    "pier_material",
    "pier_settlement",
    "bearing_stiffness",
)
TASK_LABEL_LEVEL_NAMES = ("safe", "warning", "risk", "danger")
TASK_LABEL_SCHEMA = "qianhai_four_task_ordinal_v1"
TASK_CLASS_COUNT = len(TASK_LABEL_LEVEL_NAMES)


def _decode_names(values: Any) -> tuple[str, ...]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return tuple(result)


def task_targets_at_index(arrays: dict[str, Any], index: int) -> dict[str, int]:
    """Read and validate the four v3 task classes for one acquisition."""
    names = _decode_names(arrays.get("task_label_names", []))
    levels = _decode_names(arrays.get("task_label_level_names", []))
    if names != TASK_LABEL_NAMES:
        raise ValueError(
            f"task label order mismatch: expected {TASK_LABEL_NAMES}, got {names}"
        )
    if levels != TASK_LABEL_LEVEL_NAMES:
        raise ValueError(
            f"task label vocabulary mismatch: expected {TASK_LABEL_LEVEL_NAMES}, got {levels}"
        )
    matrix = np.asarray(arrays.get("task_level_by_acquisition", []), dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] != len(TASK_LABEL_NAMES):
        raise ValueError(
            "task_level_by_acquisition must have shape (acquisitions, 4); "
            f"got {matrix.shape}"
        )
    if index < 0 or index >= matrix.shape[0]:
        raise IndexError(
            f"target acquisition {index} is outside label range 0..{matrix.shape[0] - 1}"
        )
    row = matrix[index]
    if np.any((row < 0) | (row >= TASK_CLASS_COUNT)):
        raise ValueError(f"task labels at acquisition {index} must be in 0..3: {row.tolist()}")
    return {name: int(row[column]) for column, name in enumerate(TASK_LABEL_NAMES)}
