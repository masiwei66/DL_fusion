"""Qianhai four-task heads over the shared temporal backbones."""
from __future__ import annotations

import torch
from torch import nn

from dl_fusion.models.temporal import (
    ENCODER_INPUT_CHANNELS,
    NODE_FEATURE_DIM,
    POOLED_NODE_FEATURE_DIM,
    REGIONAL_FEATURE_DIM,
    GRUModel as _GRUBackbone,
    LSTMModel as _LSTMBackbone,
    ResNet1DModel as _ResNet1DBackbone,
    ResNet50_1DModel as _ResNet50Backbone,
    TemporalTransformerModel as _TransformerBackbone,
    encoder_sequence,
    expected_input_dim,
    flatten_node_sequences,
    last_valid_token,
    node_features,
    pool_node_features,
    pool_node_representations,
    pool_points,
    pool_region_representations,
    temporal_features,
    valid_time_lengths,
)
from ..contracts import WindowBatch
from ..task_labels import TASK_CLASS_COUNT, TASK_LABEL_NAMES


class TemporalHead(nn.Module):
    """Qianhai's four independent classification heads."""

    def __init__(self, input_dim: int, representation_dim: int | None, regional_targets: bool = True):
        super().__init__()
        if not regional_targets:
            raise ValueError("four-task classification heads must be enabled")
        if representation_dim is None:
            self.task_heads = nn.ModuleDict({
                name: nn.LazyLinear(TASK_CLASS_COUNT) for name in TASK_LABEL_NAMES
            })
        else:
            self.task_heads = nn.ModuleDict({
                name: nn.Linear(representation_dim, TASK_CLASS_COUNT)
                for name in TASK_LABEL_NAMES
            })

    def outputs(self, representation: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "representation": representation,
            **{name: head(representation) for name, head in self.task_heads.items()},
        }


class _FourTaskAdapter:
    def _init_task_head(self, input_dim: int, regional_targets: bool) -> None:
        representation_dim = (
            None if getattr(self, "region_pooling", "mean") == "flatten"
            else self.representation_dim
        )
        self.head = TemporalHead(input_dim, representation_dim, regional_targets)

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        return self.head.outputs(self.forward_representation(batch))


class LSTMModel(_FourTaskAdapter, _LSTMBackbone):
    def __init__(self, input_dim=NODE_FEATURE_DIM, hidden_size=128, num_layers=2, dropout=0.1,
                 bidirectional=False, regional_targets=True, input_mode="node", region_pooling="mean"):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional, input_mode, region_pooling)
        self._init_task_head(input_dim, regional_targets)


class GRUModel(_FourTaskAdapter, _GRUBackbone):
    def __init__(self, input_dim=NODE_FEATURE_DIM, hidden_size=128, num_layers=2, dropout=0.1,
                 bidirectional=False, regional_targets=True, input_mode="node", region_pooling="mean"):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional, input_mode, region_pooling)
        self._init_task_head(input_dim, regional_targets)


class TemporalTransformerModel(_FourTaskAdapter, _TransformerBackbone):
    def __init__(self, input_dim=NODE_FEATURE_DIM, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, regional_targets=True, input_mode="node",
                 region_pooling="mean"):
        super().__init__(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout,
                         input_mode, region_pooling)
        self._init_task_head(input_dim, regional_targets)


class ResNet50_1DModel(_FourTaskAdapter, _ResNet50Backbone):
    def __init__(self, input_dim=NODE_FEATURE_DIM, base_width=32, regional_targets=True,
                 layers=(3, 4, 6, 3), stage_strides=(1, 1, 1, 1), input_mode="node",
                 region_pooling="mean"):
        super().__init__(input_dim, base_width, layers, stage_strides, input_mode, region_pooling)
        self._init_task_head(input_dim, regional_targets)


ResNet1DModel = ResNet50_1DModel

__all__ = [
    "ENCODER_INPUT_CHANNELS", "NODE_FEATURE_DIM", "POOLED_NODE_FEATURE_DIM",
    "REGIONAL_FEATURE_DIM", "GRUModel", "LSTMModel", "ResNet1DModel",
    "ResNet50_1DModel", "TemporalHead", "TemporalTransformerModel",
    "encoder_sequence", "expected_input_dim", "flatten_node_sequences",
    "last_valid_token", "node_features", "pool_node_features",
    "pool_node_representations", "pool_points", "pool_region_representations",
    "temporal_features", "valid_time_lengths",
]
