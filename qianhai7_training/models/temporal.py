from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch
from .base import StudentModel


def temporal_features(batch: WindowBatch) -> torch.Tensor:
    """Convert the variable-point contract to fixed-width [B, W, F] features.

    The normalized mean x/y coordinates are static over the window and are
    repeated across time so the temporal encoder can use absolute location.
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
    valid_ratio = mask.mean(dim=2, keepdim=True)
    time_mask = batch.temperature_valid_mask.float()
    time = batch.time_features * time_mask
    point_mask = batch.point_present_mask.float().unsqueeze(-1)
    coordinate_mean = (batch.point_features * point_mask).sum(dim=1)
    coordinate_mean = coordinate_mean / point_mask.sum(dim=1).clamp_min(1.0)
    coordinate_sequence = coordinate_mean.unsqueeze(1).expand(-1, values.shape[1], -1)
    return torch.cat([mean, std, minimum, maximum, valid_ratio, time, coordinate_sequence], dim=-1)


class TemporalHead(nn.Module):
    def __init__(self, input_dim: int, representation_dim: int, regional_targets: bool = True):
        super().__init__()
        if not regional_targets:
            raise ValueError("regional_targets must be enabled; legacy classification heads are removed")
        self.deck_material_regional_head = nn.Linear(representation_dim, 3)
        self.pier_settlement_regional_head = nn.Linear(representation_dim, 3)
        self.pier_material_regional_head = nn.Linear(representation_dim, 3)
        self.bearing_stiffness_regional_head = nn.Linear(representation_dim, 8)

    def outputs(self, representation: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "representation": representation,
            "deck_material": self.deck_material_regional_head(representation),
            "pier_settlement": self.pier_settlement_regional_head(representation),
            "pier_material": self.pier_material_regional_head(representation),
            "bearing_stiffness": self.bearing_stiffness_regional_head(representation),
        }


class LSTMModel(StudentModel):
    def __init__(self, input_dim: int = 9, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False, regional_targets: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.head = TemporalHead(input_dim, output_dim, regional_targets)

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        sequence = temporal_features(batch)
        encoded, _ = self.encoder(sequence)
        valid_time = batch.temperature_valid_mask.squeeze(-1).float()
        lengths = valid_time.sum(dim=1).long().clamp_min(1)
        representation = encoded[torch.arange(encoded.shape[0], device=encoded.device), lengths - 1]
        return self.head.outputs(representation)


class ResNet50_1DModel(StudentModel):
    """A configurable ResNet-50 bottleneck encoder for short time windows.

    The default stage strides preserve the temporal resolution. With the
    current window length of six acquisitions, the image-style ``1, 2, 2, 2``
    schedule would collapse the time axis too early.
    """

    def __init__(
        self,
        input_dim: int = 9,
        base_width: int = 32,
        regional_targets: bool = True,
        layers: tuple[int, int, int, int] = (3, 4, 6, 3),
        stage_strides: tuple[int, int, int, int] = (1, 1, 1, 1),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.base_width = int(base_width)
        self.layers = tuple(int(count) for count in layers)
        self.stage_strides = tuple(int(stride) for stride in stage_strides)
        if len(self.layers) != 4:
            raise ValueError("ResNet-50 1D requires four stage depths, e.g. [3, 4, 6, 3]")
        if len(self.stage_strides) != 4 or any(stride not in (1, 2) for stride in self.stage_strides):
            raise ValueError("stage_strides must contain four values, each 1 or 2")
        if self.input_dim != 9:
            raise ValueError("temporal_features currently produces exactly 9 input channels")
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
        self.head = TemporalHead(self.input_dim, self.representation_dim, regional_targets)

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        sequence = temporal_features(batch).transpose(1, 2)
        representation = self.pool(self.stages(self.stem(sequence))).squeeze(-1)
        return self.head.outputs(representation)


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
