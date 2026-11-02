from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch
from .base import StudentModel, validate_input_mode


# The shared per-node observation contract. ``node_features`` returns
# ``[B, W, P, NODE_FEATURE_DIM]`` with this fixed channel order:
#
#   0 displacement       normalised vertical displacement, 0 where missing
#   1 observation mask   1.0 where a real observation exists at (t, point)
#   2 x coordinate       normalised deck coordinate of that point
#   3 y coordinate       normalised deck coordinate of that point
#   4 temperature_C      normalised air temperature at that acquisition
#   5 delta_days         normalised interval since the previous acquisition
#
# Every neural model sees exactly these channels in this order. Model-specific
# feature engineering must not happen outside this function, otherwise the
# comparison between architectures stops being apples-to-apples.
NODE_FEATURE_DIM = 6


def node_features(batch: WindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the shared node-level input contract ``[B, W, P, 6]``.

    ``node_mask`` identifies real, successfully mapped points.  Observation
    quality and mapping distance/validity are deliberately not channels: quality
    is identical to the observation mask, and both mapping quantities are
    constant over the mapped points that reach the model.
    """
    if batch.point_features.shape[-1] < 2:
        raise ValueError("WindowBatch.point_features must contain x/y coordinates")
    point_mask = batch.point_present_mask & batch.mapping_valid_mask
    observation_mask = batch.valid_mask & point_mask[:, None, :]
    observation_mask_float = observation_mask.float()
    values = batch.values * observation_mask_float
    coordinates = batch.point_features[..., :2]
    time_features = batch.time_features * batch.temperature_valid_mask.float()

    _, window_length, point_count = batch.values.shape
    coordinate_sequence = coordinates[:, None, :, :].expand(-1, window_length, -1, -1)
    time_sequence = time_features[:, :, None, :].expand(-1, -1, point_count, -1)
    features = torch.cat(
        [
            values.unsqueeze(-1),
            observation_mask_float.unsqueeze(-1),
            coordinate_sequence,
            time_sequence,
        ],
        dim=-1,
    )
    if features.shape[-1] != NODE_FEATURE_DIM:
        raise RuntimeError(
            f"shared node feature contract must have {NODE_FEATURE_DIM} channels, "
            f"got {features.shape[-1]}"
        )
    return features, point_mask


# Width of the pooled [B, W, F] contract produced by ``pool_node_features``:
# four displacement summaries plus the remaining node channels.
POOLED_NODE_FEATURE_DIM = 4 + NODE_FEATURE_DIM - 1


def pool_node_features(features: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Pool shared node features to ``[B, W, 9]``, the no-point-set ablation.

    Displacement is summarised as mean/std/min/max over the *observed* points
    only, so a missing sample is excluded instead of being averaged in as a zero
    displacement.  The remaining node channels are averaged over the mapped
    points.  This collapses the point axis before the encoder, so it is the
    "what if the model cannot see the point set" counterpart of the per-point
    ``node`` path.
    """
    point_mask = node_mask[:, None, :].to(features.dtype)
    observation_mask = features[..., 1] * point_mask
    point_denominator = point_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    observation_denominator = observation_mask.sum(dim=2, keepdim=True).clamp_min(1.0)

    displacement = features[..., 0]
    mean = (displacement * observation_mask).sum(dim=2, keepdim=True) / observation_denominator
    centered = (displacement - mean) * observation_mask
    std = (centered.square().sum(dim=2, keepdim=True) / observation_denominator).clamp_min(0.0).sqrt()
    missing = observation_mask == 0
    minimum = displacement.masked_fill(missing, float("inf")).amin(dim=2, keepdim=True)
    maximum = displacement.masked_fill(missing, float("-inf")).amax(dim=2, keepdim=True)
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

    others = [
        (features[..., index] * point_mask).sum(dim=2, keepdim=True) / point_denominator
        for index in range(1, features.shape[-1])
    ]
    return torch.cat([mean, std, minimum, maximum, *others], dim=-1)


def temporal_features(batch: WindowBatch) -> torch.Tensor:
    """Legacy regional adapter: [B, W, 9] hand-made aggregate statistics.

    Kept only as an ablation input so a report can show what changes when the
    point set is replaced by fixed summary statistics.
    """
    mask = (batch.valid_mask & batch.point_present_mask[:, None, :]).float()
    values = batch.values
    denom = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=2, keepdim=True) / denom
    centered = (values - mean) * mask
    std = centered.square().sum(dim=2, keepdim=True).div(denom).sqrt()
    minimum = values.masked_fill(mask == 0, float("inf")).amin(dim=2, keepdim=True)
    maximum = values.masked_fill(mask == 0, float("-inf")).amax(dim=2, keepdim=True)
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    # Divide by the number of real points in each sample, rather than the
    # batch-padded point capacity. The latter changes when another sample in
    # the batch has more virtual points, which makes this feature depend on
    # batch composition and can expose realization-specific point counts.
    point_count = batch.point_present_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    valid_ratio = mask.sum(dim=2, keepdim=True) / point_count[:, None, :]
    time_mask = batch.temperature_valid_mask.float()
    time = batch.time_features * time_mask
    point_mask = batch.point_present_mask.float().unsqueeze(-1)
    coordinate_mean = (batch.point_features * point_mask).sum(dim=1)
    coordinate_mean = coordinate_mean / point_mask.sum(dim=1).clamp_min(1.0)
    coordinate_sequence = coordinate_mean.unsqueeze(1).expand(-1, values.shape[1], -1)
    return torch.cat([mean, std, minimum, maximum, valid_ratio, time, coordinate_sequence], dim=-1)


REGIONAL_FEATURE_DIM = 9

# Encoder input width implied by each ``input_mode``. The per-point ``node``
# path feeds the raw contract channels; the two aggregate paths substitute a
# fixed-width summary vector.
ENCODER_INPUT_CHANNELS = {
    "node": NODE_FEATURE_DIM,
    # ``region`` uses the same tensor contract as ``node`` but the point axis
    # contains low-resolution footprint regions, never FEM nodes.
    "region": NODE_FEATURE_DIM,
    "pooled": POOLED_NODE_FEATURE_DIM,
    "regional": REGIONAL_FEATURE_DIM,
}


def expected_input_dim(input_mode: str) -> int:
    if input_mode not in ENCODER_INPUT_CHANNELS:
        choices = ", ".join(ENCODER_INPUT_CHANNELS)
        raise ValueError(f"input_mode must be one of: {choices}")
    return ENCODER_INPUT_CHANNELS[input_mode]


def valid_time_lengths(batch: WindowBatch) -> torch.Tensor:
    """Number of acquisitions with valid time metadata, per sample."""
    return batch.temperature_valid_mask.squeeze(-1).bool().sum(dim=1).long().clamp_min(1)


def last_valid_token(encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(encoded.shape[0], device=encoded.device)
    return encoded[positions, lengths - 1]


def flatten_node_sequences(features: torch.Tensor, batch: WindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold the point axis into the batch axis: ``[B, W, P, F] -> [B*P, W, F]``.

    Every point becomes an independent temporal sequence that keeps its own
    displacement, coordinates and time metadata.  The point axis is folded back
    by :func:`pool_node_representations` after encoding.
    """
    batch_size, window_length, point_count, feature_dim = features.shape
    sequence = features.permute(0, 2, 1, 3).reshape(batch_size * point_count, window_length, feature_dim)
    return sequence, valid_time_lengths(batch).repeat_interleave(point_count, dim=0)


def pool_node_representations(representations: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the point axis: ``[B, P, H] -> [B, H]``.

    Padded and unmapped points contribute nothing because ``node_mask`` is zero
    for them.
    """
    mask = node_mask.float().unsqueeze(-1)
    return (representations * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def pool_points(representation: torch.Tensor, node_mask: torch.Tensor | None, batch: WindowBatch) -> torch.Tensor:
    """Undo the point-axis fold for ``node`` mode; identity for the others."""
    if node_mask is None:
        return representation
    return pool_node_representations(
        representation.reshape(batch.batch_size, batch.point_capacity, -1), node_mask
    )


def pool_region_representations(
    representation: torch.Tensor,
    node_mask: torch.Tensor | None,
    batch: WindowBatch,
    mode: str,
) -> torch.Tensor:
    """Pool the point axis for the region adapter.

    ``mean`` keeps the historical behaviour. ``flatten`` keeps one vector per
    region so the compatibility head can compare spatial patterns instead of
    only their average. The latter is required when the state signature lives
    in the relative response of different bridge regions.
    """
    if mode == "mean":
        return pool_points(representation, node_mask, batch)
    if mode == "flatten":
        if node_mask is None:
            raise ValueError("flatten region pooling requires per-region representations")
        expected = batch.batch_size * batch.point_capacity
        if representation.shape[0] != expected:
            raise ValueError(
                f"flatten region pooling expected {expected} rows, got {representation.shape[0]}"
            )
        region_representations = representation.reshape(
            batch.batch_size, batch.point_capacity, -1
        )
        region_representations = region_representations * node_mask.unsqueeze(-1).to(
            region_representations.dtype
        )
        return region_representations.reshape(batch.batch_size, -1)
    raise ValueError("region_pooling must be mean or flatten")


def encoder_sequence(batch: WindowBatch, input_mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build the tensor a neural encoder consumes, for one ``input_mode``.

    Returns ``(sequence, lengths, time_valid, node_mask)``.  ``sequence`` is
    ``[B, W, F]`` for the aggregate modes and ``[B*P, W, F]`` for the per-point
    ``node`` mode; ``node_mask`` is not None only for ``node``, where the point
    axis has to be pooled back after encoding.
    """
    validate_input_mode(batch, input_mode)
    time_valid = batch.temperature_valid_mask.squeeze(-1).bool()
    if input_mode in ("node", "region"):
        features, node_mask = node_features(batch)
        sequence, lengths = flatten_node_sequences(features, batch)
        return sequence, lengths, time_valid.repeat_interleave(batch.point_capacity, dim=0), node_mask
    if input_mode == "pooled":
        features, node_mask = node_features(batch)
        return pool_node_features(features, node_mask), valid_time_lengths(batch), time_valid, None
    if input_mode == "regional":
        return temporal_features(batch), valid_time_lengths(batch), time_valid, None
    choices = ", ".join(ENCODER_INPUT_CHANNELS)
    raise ValueError(f"input_mode must be one of: {choices}")


class LSTMModel(StudentModel):
    def __init__(self, input_dim: int = NODE_FEATURE_DIM, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False,
                 input_mode: str = "node", region_pooling: str = "mean"):
        super().__init__()
        expected = expected_input_dim(input_mode)
        if input_dim != expected:
            raise ValueError(f"input_mode={input_mode!r} requires input_dim={expected}")
        if region_pooling not in ("mean", "flatten"):
            raise ValueError("region_pooling must be mean or flatten")
        if region_pooling == "flatten" and input_mode != "region":
            raise ValueError("region_pooling=flatten requires input_mode=region")
        self.input_dim = input_dim
        self.input_mode = input_mode
        self.region_pooling = region_pooling
        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.representation_dim = output_dim

    def _encode_sequence(self, batch: WindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, lengths, _, node_mask = encoder_sequence(batch, self.input_mode)
        encoded, _ = self.encoder(sequence)
        return last_valid_token(encoded, lengths), node_mask

    def forward_representation(self, batch: WindowBatch) -> torch.Tensor:
        representation, node_mask = self._encode_sequence(batch)
        return pool_region_representations(
            representation, node_mask, batch, self.region_pooling
        )

    def forward_point_representations(
        self, batch: WindowBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation, node_mask = self._encode_sequence(batch)
        if node_mask is None:
            raise ValueError("forward_point_representations requires per-point input mode")
        return representation.reshape(
            batch.batch_size, batch.point_capacity, -1
        ), node_mask

class GRUModel(StudentModel):
    """GRU temporal baseline using the same input contract as LSTM."""

    def __init__(self, input_dim: int = NODE_FEATURE_DIM, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False,
                 input_mode: str = "node", region_pooling: str = "mean"):
        super().__init__()
        expected = expected_input_dim(input_mode)
        if input_dim != expected:
            raise ValueError(f"input_mode={input_mode!r} requires input_dim={expected}")
        if region_pooling not in ("mean", "flatten"):
            raise ValueError("region_pooling must be mean or flatten")
        if region_pooling == "flatten" and input_mode != "region":
            raise ValueError("region_pooling=flatten requires input_mode=region")
        self.input_dim = input_dim
        self.input_mode = input_mode
        self.region_pooling = region_pooling
        self.encoder = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.representation_dim = output_dim

    def _encode_sequence(self, batch: WindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, lengths, _, node_mask = encoder_sequence(batch, self.input_mode)
        encoded, _ = self.encoder(sequence)
        return last_valid_token(encoded, lengths), node_mask

    def forward_representation(self, batch: WindowBatch) -> torch.Tensor:
        representation, node_mask = self._encode_sequence(batch)
        return pool_region_representations(
            representation, node_mask, batch, self.region_pooling
        )

    def forward_point_representations(
        self, batch: WindowBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation, node_mask = self._encode_sequence(batch)
        if node_mask is None:
            raise ValueError("forward_point_representations requires per-point input mode")
        return representation.reshape(
            batch.batch_size, batch.point_capacity, -1
        ), node_mask

class TemporalTransformerModel(StudentModel):
    """Transformer encoder over the shared observation contract."""

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        input_mode: str = "node",
        region_pooling: str = "mean",
    ):
        super().__init__()
        expected = expected_input_dim(input_mode)
        if input_dim != expected:
            raise ValueError(f"input_mode={input_mode!r} requires input_dim={expected}")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if region_pooling not in ("mean", "flatten"):
            raise ValueError("region_pooling must be mean or flatten")
        if region_pooling == "flatten" and input_mode != "region":
            raise ValueError("region_pooling=flatten requires input_mode=region")
        self.input_dim = int(input_dim)
        self.input_mode = input_mode
        self.region_pooling = region_pooling
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.representation_dim = d_model

    def _encode_sequence(self, batch: WindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, time_valid, node_mask = encoder_sequence(batch, self.input_mode)
        if sequence.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("window length exceeds configured positional embedding capacity")
        sequence = self.input_projection(sequence)
        sequence = sequence + self.position_embedding[:, :sequence.shape[1]]
        padding_mask = ~time_valid
        # Keep one token unmasked for samples whose time metadata is entirely missing.
        empty = padding_mask.all(dim=1)
        if empty.any():
            padding_mask = padding_mask.clone()
            padding_mask[empty, 0] = False
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        representation = last_valid_token(encoded, (~padding_mask).sum(dim=1).clamp_min(1))
        return representation, node_mask

    def forward_representation(self, batch: WindowBatch) -> torch.Tensor:
        representation, node_mask = self._encode_sequence(batch)
        return pool_region_representations(
            representation, node_mask, batch, self.region_pooling
        )

    def forward_point_representations(
        self, batch: WindowBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation, node_mask = self._encode_sequence(batch)
        if node_mask is None:
            raise ValueError("forward_point_representations requires per-point input mode")
        return representation.reshape(
            batch.batch_size, batch.point_capacity, -1
        ), node_mask

class ResNet50_1DModel(StudentModel):
    """A configurable ResNet-50 bottleneck encoder for short time windows.

    The default stage strides preserve the temporal resolution. With the
    current window length of six acquisitions, the image-style ``1, 2, 2, 2``
    schedule would collapse the time axis too early.
    """

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        base_width: int = 32,
        layers: tuple[int, int, int, int] = (3, 4, 6, 3),
        stage_strides: tuple[int, int, int, int] = (1, 1, 1, 1),
        input_mode: str = "node",
        region_pooling: str = "mean",
    ):
        super().__init__()
        expected = expected_input_dim(input_mode)
        if input_dim != expected:
            raise ValueError(f"input_mode={input_mode!r} requires input_dim={expected}")
        if region_pooling not in ("mean", "flatten"):
            raise ValueError("region_pooling must be mean or flatten")
        if region_pooling == "flatten" and input_mode != "region":
            raise ValueError("region_pooling=flatten requires input_mode=region")
        self.input_dim = int(input_dim)
        self.input_mode = input_mode
        self.region_pooling = region_pooling
        self.base_width = int(base_width)
        self.layers = tuple(int(count) for count in layers)
        self.stage_strides = tuple(int(stride) for stride in stage_strides)
        if len(self.layers) != 4:
            raise ValueError("ResNet-50 1D requires four stage depths, e.g. [3, 4, 6, 3]")
        if len(self.stage_strides) != 4 or any(stride not in (1, 2) for stride in self.stage_strides):
            raise ValueError("stage_strides must contain four values, each 1 or 2")
        if self.base_width < 1 or any(count < 1 for count in self.layers):
            raise ValueError("base_width and every ResNet stage depth must be positive")

        self.stem = nn.Sequential(
            nn.Conv1d(self.input_dim, self.base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.base_width),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        channels = self.base_width
        for stage, (count, stride) in enumerate(zip(self.layers, self.stage_strides)):
            width = self.base_width * (2 ** stage)
            blocks.append(_ResNetStage(channels, width, count, stride))
            channels = width * _Bottleneck1D.expansion
        self.stages = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.representation_dim = channels

    def forward_representation(self, batch: WindowBatch) -> torch.Tensor:
        sequence, _, _, node_mask = encoder_sequence(batch, self.input_mode)
        representation = self.pool(self.stages(self.stem(sequence.transpose(1, 2)))).squeeze(-1)
        return pool_region_representations(
            representation, node_mask, batch, self.region_pooling
        )

    def forward_point_representations(
        self, batch: WindowBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, _, node_mask = encoder_sequence(batch, self.input_mode)
        if node_mask is None:
            raise ValueError("forward_point_representations requires per-point input mode")
        representation = self.pool(
            self.stages(self.stem(sequence.transpose(1, 2)))
        ).squeeze(-1)
        return representation.reshape(
            batch.batch_size, batch.point_capacity, -1
        ), node_mask


# Keep the original registry name working while exposing an explicit name for
# new experiments and reports.
ResNet1DModel = ResNet50_1DModel


class _Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, width: int, stride: int):
        super().__init__()
        out_channels = width * self.expansion
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, width, 1, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 3, stride=stride, padding=1, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.skip(x))


class _ResNetStage(nn.Module):
    def __init__(self, in_channels: int, width: int, count: int, stride: int):
        super().__init__()
        blocks = [_Bottleneck1D(in_channels, width, stride)]
        blocks.extend(_Bottleneck1D(width * 4, width, 1) for _ in range(count - 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
