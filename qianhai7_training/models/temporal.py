from __future__ import annotations

import torch
from torch import nn

from ..contracts import WindowBatch
from .base import StudentModel


def temporal_features(batch: WindowBatch) -> torch.Tensor:
    """Convert the variable-point contract to fixed-width [B, W, F] features."""
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
    return torch.cat([mean, std, minimum, maximum, valid_ratio, time], dim=-1)


class TemporalHead(nn.Module):
    def __init__(self, input_dim: int, representation_dim: int, num_classes: int, severity_classes: int):
        super().__init__()
        self.state_head = nn.Linear(representation_dim, num_classes)
        self.severity_head = nn.Linear(representation_dim, severity_classes)

    def outputs(self, representation: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "representation": representation,
            "state_family": self.state_head(representation),
            "global_severity": self.severity_head(representation),
        }


class LSTMModel(StudentModel):
    def __init__(self, input_dim: int = 7, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False, num_classes: int = 40,
                 severity_classes: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.head = TemporalHead(input_dim, output_dim, num_classes, severity_classes)

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        sequence = temporal_features(batch)
        encoded, _ = self.encoder(sequence)
        valid_time = batch.temperature_valid_mask.squeeze(-1).float()
        lengths = valid_time.sum(dim=1).long().clamp_min(1)
        representation = encoded[torch.arange(encoded.shape[0], device=encoded.device), lengths - 1]
        return self.head.outputs(representation)


class ResNet1DModel(StudentModel):
    """ResNet-1D over point-pooled time features; default depth is ResNet-50."""

    def __init__(self, input_dim: int = 7, base_width: int = 32, num_classes: int = 40,
                 severity_classes: int = 6, layers: tuple[int, int, int, int] = (3, 4, 6, 3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_width), nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        channels = base_width
        for stage, count in enumerate(layers):
            out_channels = base_width * (2 ** stage)
            stride = 1 if stage == 0 else 2
            blocks.append(_ResNetStage(channels, out_channels, count, stride))
            channels = out_channels * 4
        self.stages = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        representation_dim = channels
        self.head = TemporalHead(input_dim, representation_dim, num_classes, severity_classes)

    def forward_batch(self, batch: WindowBatch) -> dict[str, torch.Tensor]:
        sequence = temporal_features(batch).transpose(1, 2)
        representation = self.pool(self.stages(self.stem(sequence))).squeeze(-1)
        return self.head.outputs(representation)


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
