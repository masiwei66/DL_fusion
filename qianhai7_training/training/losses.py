from __future__ import annotations

import torch
from torch import nn, Tensor

from ..contracts import WindowBatch
from ..task_labels import TASK_CLASS_COUNT, TASK_LABEL_NAMES


def supervised_loss(outputs: dict[str, torch.Tensor], batch: WindowBatch) -> Tensor:
    missing_outputs = [key for key in TASK_LABEL_NAMES if key not in outputs]
    missing_targets = [f"target_{key}" for key in TASK_LABEL_NAMES if f"target_{key}" not in batch.targets]
    if missing_outputs or missing_targets:
        raise KeyError(f"four-task target contract incomplete: outputs={missing_outputs}, targets={missing_targets}")
    for key in TASK_LABEL_NAMES:
        logits = outputs[key]
        targets = batch.targets[f"target_{key}"]
        if logits.ndim != 2 or logits.shape[1] != TASK_CLASS_COUNT:
            raise ValueError(f"{key} logits must have shape (batch, 4), got {tuple(logits.shape)}")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(f"{key} targets must have shape ({logits.shape[0]},), got {tuple(targets.shape)}")
    return sum(
        nn.functional.cross_entropy(outputs[key], batch.targets[f"target_{key}"].long())
        for key in TASK_LABEL_NAMES
    )
