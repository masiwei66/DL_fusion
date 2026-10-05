from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch
from ..task_labels import TASK_CLASS_COUNT, TASK_LABEL_NAMES
from .base import StudentModel


class PointSetMLP(StudentModel):
    """Permutation-safe smoke baseline over point/time summary features."""

    def __init__(self, point_feature_dim: int = 2, time_feature_dim: int = 2, regional_targets: bool = True):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(point_feature_dim + time_feature_dim + 2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        if not regional_targets:
            raise ValueError("four-task classification heads must be enabled")
        self.task_heads = nn.ModuleDict({
            name: nn.Linear(64, TASK_CLASS_COUNT) for name in TASK_LABEL_NAMES
        })

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        mask = (batch.valid_mask & batch.point_present_mask[:, None, :]).float()
        denom = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        value_mean = (batch.values * mask).sum(dim=(1, 2), keepdim=True) / denom
        value_std = (((batch.values - value_mean) * mask) ** 2).sum(dim=(1, 2), keepdim=True).div(denom).sqrt()
        point_mask = batch.point_present_mask.float().unsqueeze(-1)
        point_mean = (batch.point_features * point_mask).sum(dim=1) / point_mask.sum(dim=1).clamp_min(1.0)
        time_mask = batch.temperature_valid_mask.float()
        time_mean = (batch.time_features * time_mask).sum(dim=1) / time_mask.sum(dim=1).clamp_min(1.0)
        features = torch.cat([point_mean, time_mean, value_mean.squeeze(-1), value_std.squeeze(-1)], dim=-1)
        representation = self.encoder(features)
        return {
            "representation": representation,
            **{name: head(representation) for name, head in self.task_heads.items()},
        }
