from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch


def supervised_loss(outputs: dict[str, torch.Tensor], batch: WindowBatch) -> torch.Tensor:
    return (
        nn.functional.cross_entropy(outputs["fine_state_family"], batch.targets["fine_state_family"])
        + nn.functional.cross_entropy(outputs["global_severity"], batch.targets["global_severity"])
        + 0.5 * nn.functional.cross_entropy(outputs["material_active"], batch.targets["material_active"])
        + 0.5 * nn.functional.cross_entropy(outputs["settlement_type"], batch.targets["settlement_type"])
        + 0.5 * nn.functional.cross_entropy(outputs["stiffness_reduction"], batch.targets["stiffness_reduction"])
    )
