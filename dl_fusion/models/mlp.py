from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch
from .base import StudentModel, validate_input_mode


class PointSetMLP(StudentModel):
    """Permutation-safe smoke baseline over point/time summary features."""

    def __init__(
        self,
        point_feature_dim: int = 2,
        time_feature_dim: int = 2,
        input_mode: str = "node",
    ):
        super().__init__()
        if input_mode not in ("node", "region"):
            raise ValueError("input_mode must be node or region")
        self.input_mode = input_mode
        self.encoder = nn.Sequential(nn.Linear(point_feature_dim + time_feature_dim + 2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.representation_dim = 64

    def forward_representation(self, batch: WindowBatch) -> torch.Tensor:
        validate_input_mode(batch, self.input_mode)
        mask = (batch.valid_mask & batch.point_present_mask[:, None, :]).float()
        denom = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        value_mean = (batch.values * mask).sum(dim=(1, 2), keepdim=True) / denom
        value_std = (((batch.values - value_mean) * mask) ** 2).sum(dim=(1, 2), keepdim=True).div(denom).sqrt()
        point_mask = batch.point_present_mask.float().unsqueeze(-1)
        point_mean = (batch.point_features * point_mask).sum(dim=1) / point_mask.sum(dim=1).clamp_min(1.0)
        time_mask = batch.temperature_valid_mask.float()
        time_mean = (batch.time_features * time_mask).sum(dim=1) / time_mask.sum(dim=1).clamp_min(1.0)
        features = torch.cat([point_mean, time_mean, value_mean.squeeze(-1), value_std.squeeze(-1)], dim=-1)
        return self.encoder(features)

