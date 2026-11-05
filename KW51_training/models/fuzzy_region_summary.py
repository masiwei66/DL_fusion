"""Region-summary sequence model for the KW51 5-group task.

The fuzzy bag is first reduced to one estimated response series per region
(raw mean, local plane, or local quadratic surface), then the proven
region-sequence encoder processes the ordered region tokens.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..contracts import WindowBatch
from .fuzzy_response import RegionTemporalResponseHead
from .registry import build_model


class FuzzyRegionSummaryModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any] | None,
        *,
        region_count: int,
        group_count: int,
        head_hidden: int = 128,
        dropout: float = 0.1,
        bag_feature_dropout: float = 0.1,
        include_response_head: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_params = dict(model_params or {})
        self.encoder = build_model(model_name, **self.model_params)
        if not hasattr(self.encoder, "forward_representation"):
            raise ValueError(
                f"{model_name} does not expose a region-level representation"
            )
        self.group_count = int(group_count)
        self.region_count = int(region_count)
        self.bag_feature_dropout = float(bag_feature_dropout)
        self.trunk = nn.Sequential(
            nn.LazyLinear(head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.group_head = nn.Linear(head_hidden, self.group_count)
        self.response_head = (
            RegionTemporalResponseHead(
                self.region_count,
                self.group_count,
                coordinate_dim=2,
                time_dim=2,
                hidden_dim=head_hidden,
            )
            if include_response_head
            else None
        )

    def forward(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        representation = self.encoder.forward_representation(batch)
        if batch.bag_features is not None:
            bag_features = batch.bag_features
            if self.training and self.bag_feature_dropout > 0.0:
                keep = (
                    torch.rand(
                        (bag_features.shape[0], 1), device=bag_features.device
                    )
                    >= self.bag_feature_dropout
                ).to(bag_features.dtype)
                bag_features = bag_features * keep
            representation = torch.cat([representation, bag_features], dim=-1)
        hidden = self.trunk(representation)
        state_logits = self.group_head(hidden)
        if self.response_head is None:
            return {"state_logits": state_logits, "group_logits": state_logits}
        if batch.region_present is None or batch.region_coverage is None:
            raise ValueError("FuzzyRegionSummaryModel requires region batch fields")
        values = batch.values.permute(0, 2, 1)
        valid = batch.valid_mask.permute(0, 2, 1).bool()
        quality = batch.quality.permute(0, 2, 1)
        coordinates = batch.point_features[..., :2]
        outputs = self.response_head(
            values,
            valid,
            quality,
            coordinates,
            batch.time_features,
            batch.temperature_valid_mask.squeeze(-1).bool(),
            batch.region_present.bool(),
        )
        outputs["state_logits"] = state_logits
        outputs["group_logits"] = state_logits
        return outputs
