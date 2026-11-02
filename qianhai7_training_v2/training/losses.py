from __future__ import annotations

import torch
from torch import nn, Tensor

from ..contracts import WindowBatch
from ..task_labels import TASK_CLASS_COUNT, TASK_LABEL_NAMES


def supervised_loss(
    outputs: dict[str, torch.Tensor],
    batch: WindowBatch,
    *,
    task_weights: dict[str, float] | None = None,
) -> Tensor:
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
    weights = {key: 1.0 for key in TASK_LABEL_NAMES}
    if task_weights is not None:
        unknown = set(task_weights) - set(TASK_LABEL_NAMES)
        if unknown:
            raise ValueError(f"Unknown task loss weights: {sorted(unknown)}")
        for key, value in task_weights.items():
            if float(value) <= 0.0:
                raise ValueError(f"Task loss weight for {key} must be > 0")
            weights[key] = float(value)
    # Keep the original four-CE sum scale when all weights are one. This avoids
    # changing the effective learning rate just by enabling the option.
    scale = len(TASK_LABEL_NAMES) / sum(weights.values())
    return scale * sum(
        weights[key]
        * nn.functional.cross_entropy(outputs[key], batch.targets[f"target_{key}"].long())
        for key in TASK_LABEL_NAMES
    )
