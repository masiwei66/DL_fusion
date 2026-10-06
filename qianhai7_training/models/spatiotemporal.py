from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch
from .base import StudentModel
from .temporal import TemporalHead


def _build_knn_adjacency(
    coordinates: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    knn_k: int,
    distance_sigma: float | None,
) -> torch.Tensor:
    """Build a masked, symmetrized kNN adjacency for a padded point batch.

    Coordinates are the deployment-visible point geometry.  The graph is built
    per sample so changing the virtual point layout cannot be mistaken for a
    change in node identity.  Invalid and padded points never receive edges.
    """
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(
            "coordinates must have shape (batch, points, 2), "
            f"got {tuple(coordinates.shape)}"
        )
    if node_mask.shape != coordinates.shape[:2]:
        raise ValueError(
            "node_mask must have shape (batch, points), "
            f"got {tuple(node_mask.shape)}"
        )

    batch_size, point_count, _ = coordinates.shape
    adjacency = coordinates.new_zeros((batch_size, point_count, point_count))
    if point_count == 0:
        return adjacency

    if point_count > 1 and knn_k > 0:
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        pair_mask = pair_mask & ~torch.eye(
            point_count, dtype=torch.bool, device=coordinates.device
        ).unsqueeze(0)
        distances = torch.cdist(coordinates, coordinates)
        candidate_distances = distances.masked_fill(~pair_mask, float("inf"))
        neighbor_count = min(int(knn_k), point_count - 1)
        neighbor_distances, neighbor_indices = torch.topk(
            candidate_distances, k=neighbor_count, dim=-1, largest=False
        )
        finite_neighbors = torch.isfinite(neighbor_distances)

        if distance_sigma is None:
            finite_sum = neighbor_distances.masked_fill(~finite_neighbors, 0.0).sum(dim=(1, 2))
            finite_count = finite_neighbors.sum(dim=(1, 2)).clamp_min(1)
            sigma = (finite_sum / finite_count).clamp_min(1e-6).view(batch_size, 1, 1)
        else:
            if float(distance_sigma) <= 0.0:
                raise ValueError("distance_sigma must be positive when provided")
            sigma = coordinates.new_tensor(float(distance_sigma)).view(1, 1, 1)

        edge_weights = torch.exp(-neighbor_distances.square() / (2.0 * sigma.square()))
        edge_weights = edge_weights * finite_neighbors.to(edge_weights.dtype)
        directed = adjacency.clone()
        directed.scatter_(2, neighbor_indices, edge_weights)
        adjacency = torch.maximum(directed, directed.transpose(1, 2))

    # Self loops preserve each point's own observation after spatial mixing.
    adjacency = adjacency + torch.diag_embed(node_mask.to(adjacency.dtype))
    adjacency = adjacency * (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).to(adjacency.dtype)
    degree = adjacency.sum(dim=-1).clamp_min(1e-6)
    normalization = degree.rsqrt()
    return normalization.unsqueeze(-1) * adjacency * normalization.unsqueeze(-2)


class GraphConv(nn.Module):
    """A dependency-free GCN layer for a dense, padded batch of small graphs."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.bmm(adjacency, self.linear(features))


class GCNLSTMModel(StudentModel):
    """Dynamic coordinate kNN-GCN followed by an LSTM temporal encoder.

    Each time step is represented as a graph of observed virtual points.  GCN
    embeddings are masked-pooled to a regional vector, and the sequence of
    regional vectors is consumed by the same terminal-state convention used by
    the existing LSTM baseline.
    """

    def __init__(
        self,
        node_input_dim: int = 9,
        graph_hidden_dim: int = 64,
        lstm_hidden_size: int = 128,
        graph_layers: int = 2,
        lstm_layers: int = 2,
        knn_k: int = 8,
        distance_sigma: float | None = None,
        dropout: float = 0.1,
        bidirectional: bool = False,
        regional_targets: bool = True,
    ):
        super().__init__()
        if node_input_dim != 9:
            raise ValueError(
                "The current WindowBatch adapter produces exactly 9 GCN node features"
            )
        if graph_hidden_dim < 1 or lstm_hidden_size < 1:
            raise ValueError("graph_hidden_dim and lstm_hidden_size must be positive")
        if graph_layers < 1 or lstm_layers < 1:
            raise ValueError("graph_layers and lstm_layers must be positive")
        if knn_k < 0:
            raise ValueError("knn_k must be non-negative")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if distance_sigma is not None and float(distance_sigma) <= 0.0:
            raise ValueError("distance_sigma must be positive when provided")

        self.node_input_dim = int(node_input_dim)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.knn_k = int(knn_k)
        self.distance_sigma = distance_sigma
        self.graph_layers = nn.ModuleList(
            [
                GraphConv(
                    self.node_input_dim if layer == 0 else self.graph_hidden_dim,
                    self.graph_hidden_dim,
                )
                for layer in range(graph_layers)
            ]
        )
        self.graph_dropout = nn.Dropout(float(dropout))
        self.temporal_encoder = nn.LSTM(
            input_size=self.graph_hidden_dim,
            hidden_size=int(lstm_hidden_size),
            num_layers=int(lstm_layers),
            batch_first=True,
            dropout=float(dropout) if lstm_layers > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        temporal_dim = int(lstm_hidden_size) * (2 if bidirectional else 1)
        self.head = TemporalHead(self.graph_hidden_dim, temporal_dim, regional_targets)

    @staticmethod
    def _node_features(batch: WindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B, W, P, 9] features and the static valid-node mask."""
        if batch.point_features.shape[-1] < 2:
            raise ValueError("WindowBatch.point_features must contain x/y coordinates")
        point_mask = batch.point_present_mask & batch.mapping_valid_mask
        point_mask_float = point_mask.float()
        observation_mask = batch.valid_mask & point_mask[:, None, :]
        observation_mask_float = observation_mask.float()
        values = batch.values * observation_mask_float
        quality = batch.quality * observation_mask_float
        coordinates = batch.point_features[..., :2]
        mapping_distance = batch.mapping_distance * point_mask_float
        mapping_valid = batch.mapping_valid_mask.float() * point_mask_float
        time_mask = batch.temperature_valid_mask.float()
        time_features = batch.time_features * time_mask

        _, window_length, point_count = batch.values.shape
        coordinate_sequence = coordinates[:, None, :, :].expand(-1, window_length, -1, -1)
        mapping_distance_sequence = mapping_distance[:, None, :, None].expand(
            -1, window_length, -1, -1
        )
        mapping_valid_sequence = mapping_valid[:, None, :, None].expand(
            -1, window_length, -1, -1
        )
        time_sequence = time_features[:, :, None, :].expand(-1, -1, point_count, -1)
        features = torch.cat(
            [
                values.unsqueeze(-1),
                observation_mask_float.unsqueeze(-1),
                quality.unsqueeze(-1),
                coordinate_sequence,
                mapping_distance_sequence,
                mapping_valid_sequence,
                time_sequence,
            ],
            dim=-1,
        )
        return features, point_mask

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        features, node_mask = self._node_features(batch)
        adjacency = _build_knn_adjacency(
            batch.point_features[..., :2],
            node_mask,
            knn_k=self.knn_k,
            distance_sigma=self.distance_sigma,
        )

        graph_sequence: list[torch.Tensor] = []
        node_mask_float = node_mask.float()
        for time_index in range(features.shape[1]):
            encoded = features[:, time_index]
            for layer_index, layer in enumerate(self.graph_layers):
                encoded = layer(encoded, adjacency)
                encoded = torch.relu(encoded)
                if layer_index < len(self.graph_layers) - 1:
                    encoded = self.graph_dropout(encoded)
                encoded = encoded * node_mask_float.unsqueeze(-1)
            denominator = node_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            graph_sequence.append(encoded.sum(dim=1) / denominator)

        sequence = torch.stack(graph_sequence, dim=1)
        encoded, _ = self.temporal_encoder(sequence)
        valid_time = batch.temperature_valid_mask.squeeze(-1).float()
        lengths = valid_time.sum(dim=1).long().clamp_min(1)
        representation = encoded[
            torch.arange(encoded.shape[0], device=encoded.device), lengths - 1
        ]
        return self.head.outputs(representation)
