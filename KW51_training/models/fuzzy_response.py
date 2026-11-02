"""Fuzzy-range bag encoder and Gaussian response-distribution head."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..contracts import WindowBatch
from .registry import build_model


class BagAttentionPool(nn.Module):
    """Permutation-invariant attention pooling over a variable-size point bag."""

    def __init__(self) -> None:
        super().__init__()
        self.score = nn.LazyLinear(1)

    def forward(
        self, representations: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        scores = self.score(representations).squeeze(-1)
        scores = scores.masked_fill(~mask, -1.0e9)
        attention = torch.softmax(scores, dim=1)
        attention = attention * mask.to(attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        attention_out = (representations * attention.unsqueeze(-1)).sum(dim=1)
        mask_float = mask.to(representations.dtype).unsqueeze(-1)
        denominator = mask_float.sum(dim=1).clamp_min(1.0)
        mean = (representations * mask_float).sum(dim=1) / denominator
        masked = representations.masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(representations.dtype).min
        )
        maximum = masked.max(dim=1).values
        maximum = torch.where(
            mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        return torch.cat([attention_out, mean, maximum], dim=-1)


class RegionTemporalResponseHead(nn.Module):
    """Shared temporal encoder for dense region responses and local scores."""

    def __init__(
        self,
        region_count: int,
        group_count: int,
        coordinate_dim: int,
        time_dim: int,
        *,
        hidden_dim: int,
        region_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.region_count = int(region_count)
        self.region_embedding = nn.Embedding(
            self.region_count, int(region_embedding_dim)
        )
        input_dim = 3 + int(coordinate_dim) + int(time_dim) + 2 + int(region_embedding_dim)
        self.temporal = nn.GRU(input_dim, int(hidden_dim), batch_first=True)
        self.mean_head = nn.Linear(int(hidden_dim), 1)
        self.scale_head = nn.Linear(int(hidden_dim), 1)
        self.state_head = nn.Linear(int(hidden_dim), int(group_count))

    def forward(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        quality: torch.Tensor,
        coordinates: torch.Tensor,
        time_features: torch.Tensor,
        time_valid: torch.Tensor,
        region_present: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode ``[B,R,T]`` inputs without collapsing either output axis."""
        batch_size, region_count, time_count = values.shape
        if region_count != self.region_count:
            raise ValueError(
                f"expected {self.region_count} regions, received {region_count}"
            )
        region_ids = torch.arange(region_count, device=values.device)
        embedding = self.region_embedding(region_ids)[None, :, None, :].expand(
            batch_size, -1, time_count, -1
        )
        coordinates = coordinates[:, :, None, :].expand(-1, -1, time_count, -1)
        times = time_features[:, None, :, :].expand(-1, region_count, -1, -1)
        time_mask = time_valid[:, None, :, None].to(values.dtype).expand(
            -1, region_count, -1, -1
        )
        presence = region_present[:, :, None, None].to(values.dtype).expand(
            -1, -1, time_count, -1
        )
        features = torch.cat(
            [
                values.unsqueeze(-1),
                valid.to(values.dtype).unsqueeze(-1),
                quality.to(values.dtype).unsqueeze(-1),
                coordinates,
                times,
                time_mask,
                presence,
                embedding,
            ],
            dim=-1,
        )
        encoded, _ = self.temporal(
            features.reshape(batch_size * region_count, time_count, -1)
        )
        encoded = encoded.reshape(batch_size, region_count, time_count, -1)
        mean = self.mean_head(encoded).squeeze(-1)
        scale = torch.nn.functional.softplus(self.scale_head(encoded).squeeze(-1))
        scale = scale.clamp_min(1.0e-4)
        time_weights = time_valid[:, None, :].to(encoded.dtype)
        pooled = (encoded * time_weights.unsqueeze(-1)).sum(dim=2)
        pooled = pooled / time_weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        region_logits = self.state_head(pooled)
        return {
            "region_response_mean": mean,
            "region_response_scale": scale,
            "region_state_logits": region_logits,
        }


class FuzzyResponseModel(nn.Module):
    """Shared point encoder, bag attention pooling and Gaussian response head."""

    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any] | None,
        target_length: int,
        *,
        group_count: int | None = None,
        head_hidden: int = 128,
        dropout: float = 0.1,
        bag_feature_dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_params = dict(model_params or {})
        if self.model_params.get("input_mode", "node") != "node":
            raise ValueError("FuzzyResponseModel requires input_mode=node")
        self.encoder = build_model(model_name, **self.model_params)
        if not hasattr(self.encoder, "forward_point_representations"):
            raise ValueError(
                f"{model_name} does not expose per-point representations for fuzzy bags"
            )
        self.target_length = int(target_length)
        self.group_count = None if group_count is None else int(group_count)
        self.bag_pool = BagAttentionPool()
        self.trunk = nn.Sequential(
            nn.LazyLinear(head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.group_head = (
            nn.Linear(head_hidden, self.group_count)
            if self.group_count is not None
            else None
        )
        self.response_head = (
            nn.Linear(head_hidden, 2 * self.target_length)
            if self.target_length > 0
            else None
        )
        self.bag_feature_dropout = float(bag_feature_dropout)

    def forward(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        representations, mask = self.encoder.forward_point_representations(batch)
        pooled = self.bag_pool(representations, mask)
        if batch.bag_features is not None:
            bag_features = batch.bag_features
            if self.training and self.bag_feature_dropout > 0.0:
                keep = (
                    torch.rand((bag_features.shape[0], 1), device=bag_features.device)
                    >= self.bag_feature_dropout
                ).to(bag_features.dtype)
                bag_features = bag_features * keep
            pooled = torch.cat([pooled, bag_features], dim=-1)
        hidden = self.trunk(pooled)
        outputs: dict[str, torch.Tensor] = {}
        if self.group_head is not None:
            outputs["group_logits"] = self.group_head(hidden)
        if self.response_head is not None:
            raw = self.response_head(hidden)
            outputs["mean"] = raw[..., : self.target_length]
            outputs["logvar"] = torch.clamp(
                raw[..., self.target_length:], -10.0, 10.0
            )
        return outputs
