"""Qianhai four-task wrapper around the shared coordinate-kNN GCN-LSTM."""
from __future__ import annotations

import torch

from dl_fusion.models.spatiotemporal import (
    GCNLSTMModel as _GCNBackbone,
    GraphConv,
    _build_knn_adjacency,
)
from ..contracts import WindowBatch
from .temporal import NODE_FEATURE_DIM, TemporalHead, node_features


class GCNLSTMModel(_GCNBackbone):
    def __init__(self, node_input_dim=NODE_FEATURE_DIM, graph_hidden_dim=64,
                 lstm_hidden_size=128, graph_layers=2, lstm_layers=2, knn_k=8,
                 distance_sigma=None, dropout=0.1, bidirectional=False,
                 regional_targets=True, input_mode="node"):
        super().__init__(node_input_dim, graph_hidden_dim, lstm_hidden_size,
                         graph_layers, lstm_layers, knn_k, distance_sigma,
                         dropout, bidirectional, input_mode)
        self.head = TemporalHead(self.graph_hidden_dim, self.representation_dim, regional_targets)

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        return self.head.outputs(self.forward_representation(batch))


__all__ = ["GCNLSTMModel", "GraphConv", "_build_knn_adjacency", "NODE_FEATURE_DIM", "node_features"]
