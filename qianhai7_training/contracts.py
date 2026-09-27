from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class WindowBatch:
    """The only input contract shared by all student models.

    ``W`` is fixed for one experiment. ``P`` may vary by sample and is
    padded to the largest point count in the current batch.

    values, valid_mask, quality: [B, W, P]
    point_features: [B, P, Fp]
    time_features: [B, W, Ft]
    point_present_mask: [B, P]
    targets: tensors indexed by batch item
    """

    values: torch.Tensor
    valid_mask: torch.Tensor
    quality: torch.Tensor
    point_features: torch.Tensor
    time_features: torch.Tensor
    point_present_mask: torch.Tensor
    mapping_distance: torch.Tensor
    mapping_valid_mask: torch.Tensor
    temperature_valid_mask: torch.Tensor
    teacher_features: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    metadata: list[dict[str, Any]]

    def to(self, device: torch.device | str) -> "WindowBatch":
        return WindowBatch(
            values=self.values.to(device),
            valid_mask=self.valid_mask.to(device),
            quality=self.quality.to(device),
            point_features=self.point_features.to(device),
            time_features=self.time_features.to(device),
            point_present_mask=self.point_present_mask.to(device),
            mapping_distance=self.mapping_distance.to(device),
            mapping_valid_mask=self.mapping_valid_mask.to(device),
            temperature_valid_mask=self.temperature_valid_mask.to(device),
            teacher_features={key: value.to(device) for key, value in self.teacher_features.items()},
            targets={key: value.to(device) for key, value in self.targets.items()},
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def window_length(self) -> int:
        return self.values.shape[1]

    @property
    def point_capacity(self) -> int:
        return self.values.shape[2]
