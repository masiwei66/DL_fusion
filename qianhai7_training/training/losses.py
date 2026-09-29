from __future__ import annotations

import torch
from torch import nn, Tensor

from ..contracts import WindowBatch


def supervised_loss(outputs: dict[str, torch.Tensor], batch: WindowBatch) -> int:
    regional_keys = ("deck_material", "pier_settlement", "pier_material", "bearing_stiffness")
    missing_outputs = [key for key in regional_keys if key not in outputs]
    missing_targets = [f"target_{key}" for key in regional_keys if f"target_{key}" not in batch.targets]
    if missing_outputs or missing_targets:
        raise KeyError(f"regional target contract incomplete: outputs={missing_outputs}, targets={missing_targets}")
    return sum(
        nn.functional.binary_cross_entropy_with_logits(outputs[key], batch.targets[f"target_{key}"].float())
        for key in regional_keys
    )
