"""Region-aware fuzzy bag encoder for the 5-group Ad compatibility task."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..contracts import WindowBatch
from .fuzzy_response import BagAttentionPool, RegionTemporalResponseHead
from .registry import build_model


class FuzzyRegionModel(nn.Module):
    """Pool points inside each region, then model the ordered region sequence.

    Region identity is explicit through a learned embedding and the fixed
    flatten order. Points are only pooled within their own region, so shuffling
    points inside a region must not change the output while changing the region
    assignment must.
    """

    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any] | None,
        *,
        region_count: int,
        group_count: int,
        region_pooling: str = "flatten",
        region_embedding_dim: int = 16,
        head_hidden: int = 128,
        dropout: float = 0.1,
        bag_feature_dropout: float = 0.1,
        target_length: int = 0,
        include_response_head: bool = True,
    ):
        super().__init__()
        if int(target_length) != 0:
            raise ValueError(
                "FuzzyRegionModel currently supports state-only training "
                "(target_length=0)"
            )
        if region_pooling not in ("flatten", "mean"):
            raise ValueError("region_pooling must be flatten or mean")
        self.model_name = model_name
        self.model_params = dict(model_params or {})
        if self.model_params.get("input_mode", "node") != "node":
            raise ValueError("FuzzyRegionModel requires input_mode=node")
        self.encoder = build_model(model_name, **self.model_params)
        if not hasattr(self.encoder, "forward_point_representations"):
            raise ValueError(
                f"{model_name} does not expose per-point representations"
            )
        self.region_count = int(region_count)
        self.group_count = int(group_count)
        self.region_pooling = region_pooling
        self.bag_feature_dropout = float(bag_feature_dropout)
        self.region_pool = BagAttentionPool()
        self.region_embedding = nn.Embedding(
            self.region_count, int(region_embedding_dim)
        )
        self.region_trunk = nn.Sequential(
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

    def _regional_sequences(
        self, batch: WindowBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert batch.region_index is not None
        assert batch.region_present is not None
        assert batch.region_coverage is not None
        region_values: list[torch.Tensor] = []
        region_valid: list[torch.Tensor] = []
        region_quality: list[torch.Tensor] = []
        region_coordinates: list[torch.Tensor] = []
        point_mask = batch.point_present_mask & batch.mapping_valid_mask
        for index in range(self.region_count):
            members = (batch.region_index == index) & point_mask
            present = batch.region_present[:, index : index + 1].bool()
            observed = batch.valid_mask & members[:, None, :] & present[:, :, None]
            weights = batch.quality.clamp_min(0.0) * observed.to(batch.quality.dtype)
            denominator = weights.sum(dim=2)
            region_values.append(
                (batch.values * weights).sum(dim=2) / denominator.clamp_min(1.0e-6)
            )
            region_valid.append(denominator > 0.0)
            region_quality.append(
                (batch.quality * observed.to(batch.quality.dtype)).sum(dim=2)
                / observed.sum(dim=2).clamp_min(1)
            )
            coordinate_weights = members.to(batch.point_features.dtype)
            coordinate_denominator = coordinate_weights.sum(dim=1, keepdim=True)
            coordinates = (
                batch.point_features[..., :2] * coordinate_weights.unsqueeze(-1)
            ).sum(dim=1) / coordinate_denominator.clamp_min(1.0)
            region_coordinates.append(coordinates)
        return (
            torch.stack(region_values, dim=1),
            torch.stack(region_valid, dim=1),
            torch.stack(region_quality, dim=1),
            torch.stack(region_coordinates, dim=1),
        )

    def forward(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        if (
            batch.region_index is None
            or batch.region_present is None
            or batch.region_coverage is None
        ):
            raise ValueError("FuzzyRegionModel requires region batch fields")
        representations, point_mask = self.encoder.forward_point_representations(
            batch
        )
        if point_mask is None:
            raise ValueError("FuzzyRegionModel requires a per-point mask")
        region_index = batch.region_index
        present = batch.region_present.bool()
        coverage = batch.region_coverage.float()
        pooled_regions: list[torch.Tensor] = []
        for index in range(self.region_count):
            region_mask = (
                point_mask
                & (region_index == index)
                & present[:, index : index + 1]
            )
            pooled_regions.append(self.region_pool(representations, region_mask))
        pooled = torch.stack(pooled_regions, dim=1)  # [B, R, 3H]
        present_float = present.to(pooled.dtype).unsqueeze(-1)
        embedding = self.region_embedding.weight.unsqueeze(0).expand(
            pooled.shape[0], -1, -1
        )
        region_features = torch.cat(
            [
                pooled * present_float,
                embedding * present_float,
                coverage.unsqueeze(-1) * present_float,
                present_float,
            ],
            dim=-1,
        )
        if self.region_pooling == "mean":
            denominator = present_float.sum(dim=1).clamp_min(1.0)
            region_features = region_features.sum(dim=1) / denominator
        else:
            region_features = region_features.reshape(region_features.shape[0], -1)
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
            region_features = torch.cat([region_features, bag_features], dim=-1)
        hidden = self.region_trunk(region_features)
        state_logits = self.group_head(hidden)
        if self.response_head is None:
            return {"state_logits": state_logits, "group_logits": state_logits}
        values, valid, quality, coordinates = self._regional_sequences(batch)
        outputs = self.response_head(
            values,
            valid,
            quality,
            coordinates,
            batch.time_features,
            batch.temperature_valid_mask.squeeze(-1).bool(),
            present,
        )
        outputs["state_logits"] = state_logits
        outputs["group_logits"] = state_logits
        return outputs
