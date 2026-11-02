"""Region-time MLP baseline for dense regional response prediction."""
from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch


class FuzzyRegionMLPModel(nn.Module):
    def __init__(
        self,
        *,
        region_count: int,
        group_count: int,
        hidden_dim: int = 96,
        region_embedding_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.region_count = int(region_count)
        self.region_embedding = nn.Embedding(self.region_count, int(region_embedding_dim))
        feature_dim = 3 + 2 + 2 + 2 + int(region_embedding_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(int(hidden_dim), 1)
        self.scale_head = nn.Linear(int(hidden_dim), 1)
        self.region_state_head = nn.Linear(int(hidden_dim), int(group_count))

    def forward(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        if batch.region_present is None:
            raise ValueError("FuzzyRegionMLPModel requires region batch fields")
        if batch.values.shape[2] != self.region_count:
            raise ValueError("region summary axis does not match region_count")
        values = batch.values.permute(0, 2, 1)
        valid = batch.valid_mask.permute(0, 2, 1).bool()
        quality = batch.quality.permute(0, 2, 1)
        batch_size, region_count, time_count = values.shape
        region_ids = torch.arange(region_count, device=values.device)
        embedding = self.region_embedding(region_ids)[None, :, None, :].expand(
            batch_size, -1, time_count, -1
        )
        coordinates = batch.point_features[:, :, :2, None].permute(0, 1, 3, 2)
        coordinates = coordinates.expand(-1, -1, time_count, -1)
        times = batch.time_features[:, None, :, :].expand(-1, region_count, -1, -1)
        time_valid = batch.temperature_valid_mask.squeeze(-1)[:, None, :, None]
        time_valid = time_valid.to(values.dtype).expand(-1, region_count, -1, -1)
        present = batch.region_present[:, :, None, None].to(values.dtype).expand(
            -1, -1, time_count, -1
        )
        features = torch.cat(
            [
                values.unsqueeze(-1),
                valid.to(values.dtype).unsqueeze(-1),
                quality.unsqueeze(-1),
                coordinates,
                times,
                time_valid,
                present,
                embedding,
            ],
            dim=-1,
        )
        encoded = self.token_encoder(features)
        mean = self.mean_head(encoded).squeeze(-1)
        scale = torch.nn.functional.softplus(self.scale_head(encoded).squeeze(-1))
        scale = scale.clamp_min(1.0e-4)
        time_mask = batch.temperature_valid_mask.squeeze(-1)[:, None, :].to(encoded.dtype)
        region_summary = (encoded * time_mask.unsqueeze(-1)).sum(dim=2)
        region_summary = region_summary / time_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        region_logits = self.region_state_head(region_summary)
        region_mask = batch.region_present.to(region_logits.dtype).unsqueeze(-1)
        state_logits = (region_logits * region_mask).sum(dim=1) / region_mask.sum(dim=1).clamp_min(1.0)
        return {
            "region_response_mean": mean,
            "region_response_scale": scale,
            "region_state_logits": region_logits,
            "state_logits": state_logits,
            "group_logits": state_logits,
        }
