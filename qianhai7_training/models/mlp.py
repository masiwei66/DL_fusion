from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch
from .base import StudentModel


class PointSetMLP(StudentModel):
    """Permutation-safe smoke baseline over point/time summary features."""

    def __init__(self, point_feature_dim: int = 2, time_feature_dim: int = 2, num_classes: int = 40, severity_classes: int = 4, settlement_classes: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(point_feature_dim + time_feature_dim + 2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.state_head = nn.Linear(64, num_classes)
        self.severity_head = nn.Linear(64, 6)
        self.severity_head = nn.Linear(64, severity_classes)
        self.material_head = nn.Linear(64, 2)
        self.settlement_head = nn.Linear(64, settlement_classes)
        self.stiffness_head = nn.Linear(64, 2)

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
        return {"representation": representation, "state_family": self.state_head(representation), "fine_state_family": self.state_head(representation), "global_severity": self.severity_head(representation), "material_active": self.material_head(representation), "settlement_type": self.settlement_head(representation), "stiffness_reduction": self.stiffness_head(representation)}
