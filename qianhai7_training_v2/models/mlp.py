"""Qianhai four-task head over the shared point-set MLP backbone."""
from __future__ import annotations

import torch
from torch import nn

from dl_fusion.models.mlp import PointSetMLP as _PointSetBackbone
from ..contracts import WindowBatch
from ..task_labels import TASK_CLASS_COUNT, TASK_LABEL_NAMES


class PointSetMLP(_PointSetBackbone):
    def __init__(self, point_feature_dim: int = 2, time_feature_dim: int = 2,
                 regional_targets: bool = True, input_mode: str = "node"):
        super().__init__(point_feature_dim, time_feature_dim, input_mode)
        if not regional_targets:
            raise ValueError("four-task classification heads must be enabled")
        self.task_heads = nn.ModuleDict(
            {name: nn.Linear(self.representation_dim, TASK_CLASS_COUNT) for name in TASK_LABEL_NAMES}
        )

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        representation = self.forward_representation(batch)
        return {
            "representation": representation,
            **{name: head(representation) for name, head in self.task_heads.items()},
        }
