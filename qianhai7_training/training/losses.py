from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch


def supervised_loss(outputs: dict[str, torch.Tensor], batch: WindowBatch) -> torch.Tensor:
    return nn.functional.cross_entropy(outputs["state_family"], batch.targets["state_family"]) + nn.functional.cross_entropy(outputs["global_severity"], batch.targets["global_severity"])
