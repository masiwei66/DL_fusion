"""Candidate-state ranking wrapper around the unchanged comparison backbones."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .registry import build_model


def _side_context(side) -> torch.Tensor:
    """Summarize deployment-visible quality and train-condition context."""
    valid = side.valid_mask.float()
    denominator = valid.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    quality_mean = (side.quality * valid).sum(dim=(1, 2), keepdim=True) / denominator
    valid_ratio = valid.mean(dim=(1, 2), keepdim=True)
    coverage = side.effective_coverage
    if coverage is None:
        coverage_mean = valid_ratio
    else:
        coverage_mean = (coverage.float() * valid).sum(dim=(1, 2), keepdim=True) / denominator
    return torch.cat(
        [quality_mean.reshape(quality_mean.shape[0], 1),
         valid_ratio.reshape(valid_ratio.shape[0], 1),
         coverage_mean.reshape(coverage_mean.shape[0], 1)],
        dim=-1,
    )


class KW51CandidateScorer(nn.Module):
    """Shared encoder for query and candidate, followed by compatibility head."""

    def __init__(self, model_name: str, model_params: dict[str, Any] | None = None):
        super().__init__()
        if model_name == "xgboost":
            raise ValueError("xgboost is not a neural KW51 candidate-ranking backbone")
        self.model_name = model_name
        self.model_params = dict(model_params or {})
        self.encoder = build_model(model_name, **self.model_params)
        self.score_head = nn.Sequential(
            nn.LazyLinear(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode(self, side) -> torch.Tensor:
        return self.encoder.forward_representation(side)

    def _encode_by_metadata(
        self,
        side,
        metadata: list[dict[str, Any]],
        *,
        prefix: str,
    ) -> torch.Tensor:
        """Encode each repeated query/candidate window once per batch.

        The key must include the observation realization and, for candidate
        aggregates, the query exclusion. Different noisy query realizations of
        the same state/window and positive candidates built with different
        trajectory exclusions are distinct inputs even when their state ids and
        window starts match.
        """
        if len(metadata) != side.batch_size:
            raise ValueError(
                f"{prefix} metadata has {len(metadata)} rows for batch size {side.batch_size}"
            )
        if prefix == "query":
            keys = [
                (
                    str(item.get("query_sample_id", "")),
                    int(item.get("query_start", -1)),
                    int(item.get("observation_realization_id", 0)),
                )
                for item in metadata
            ]
        elif prefix == "candidate":
            keys = [
                (
                    str(
                        item.get(
                            "candidate_group_id",
                            item.get("candidate_state_id", ""),
                        )
                    ),
                    int(item.get("candidate_start", -1)),
                    int(item.get("observation_realization_id", 0)),
                    str(item.get("candidate_exclusion_key", "")),
                )
                for item in metadata
            ]
        else:
            raise ValueError("prefix must be query or candidate")
        unique_indices: list[int] = []
        key_to_unique: dict[tuple[Any, ...], int] = {}
        inverse: list[int] = []
        for index, key in enumerate(keys):
            unique_index = key_to_unique.get(key)
            if unique_index is None:
                unique_index = len(unique_indices)
                key_to_unique[key] = unique_index
                unique_indices.append(index)
            inverse.append(unique_index)
        if len(unique_indices) == len(keys):
            return self.encode(side)
        unique = torch.as_tensor(unique_indices, dtype=torch.long, device=side.values.device)
        encoded_unique = self.encode(side.index_select(unique))
        inverse_tensor = torch.as_tensor(inverse, dtype=torch.long, device=side.values.device)
        return encoded_unique.index_select(0, inverse_tensor)

    def forward_pair(self, batch: dict[str, Any]) -> torch.Tensor:
        query = self._encode_by_metadata(
            batch["query"], batch["metadata"], prefix="query"
        )
        candidate = self._encode_by_metadata(
            batch["candidate"], batch["metadata"], prefix="candidate"
        )
        context = torch.cat([_side_context(batch["query"]), _side_context(batch["candidate"])], dim=-1)
        features = torch.cat(
            [query, candidate, torch.abs(query - candidate), query * candidate, context], dim=-1
        )
        return self.score_head(features).squeeze(-1)
