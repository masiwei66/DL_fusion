from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class WindowBatch:
    """Common observation batch consumed by every student backbone.

    ``point_features`` is the coordinate axis for both modes: it contains
    sparse sensor/node coordinates in ``node`` mode and fixed-grid region
    centers in ``region`` mode.  Dataset adapters own that distinction; model
    code only sees the masks and tensors below.

    Required tensors use the shapes ``values/valid_mask/quality: [B, W, P]``,
    ``point_features: [B, P, Fp]``, ``time_features: [B, W, Ft]`` and
    ``point_present_mask: [B, P]``.  Optional region fields are deliberately
    additive so legacy node and auxiliary KW51 loaders remain compatible.
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
    bag_features: torch.Tensor | None = None
    bag_feature_mask: torch.Tensor | None = None
    region_index: torch.Tensor | None = None
    region_present: torch.Tensor | None = None
    region_coverage: torch.Tensor | None = None
    support_features: torch.Tensor | None = None
    support_coverage: torch.Tensor | None = None
    coverage: torch.Tensor | None = None
    input_mode: str | None = None

    def to(self, device: torch.device | str) -> "WindowBatch":
        def move_optional(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(device)

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
            bag_features=move_optional(self.bag_features),
            bag_feature_mask=move_optional(self.bag_feature_mask),
            region_index=move_optional(self.region_index),
            region_present=move_optional(self.region_present),
            region_coverage=move_optional(self.region_coverage),
            support_features=move_optional(self.support_features),
            support_coverage=move_optional(self.support_coverage),
            coverage=move_optional(self.coverage),
            input_mode=self.input_mode,
        )

    def index_select(self, indices: torch.Tensor) -> "WindowBatch":
        """Select batch rows while preserving the complete observation contract."""
        def select_optional(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.index_select(0, indices)

        return WindowBatch(
            values=self.values.index_select(0, indices),
            valid_mask=self.valid_mask.index_select(0, indices),
            quality=self.quality.index_select(0, indices),
            point_features=self.point_features.index_select(0, indices),
            time_features=self.time_features.index_select(0, indices),
            point_present_mask=self.point_present_mask.index_select(0, indices),
            mapping_distance=self.mapping_distance.index_select(0, indices),
            mapping_valid_mask=self.mapping_valid_mask.index_select(0, indices),
            temperature_valid_mask=self.temperature_valid_mask.index_select(0, indices),
            teacher_features={key: value.index_select(0, indices) for key, value in self.teacher_features.items()},
            targets={key: value.index_select(0, indices) for key, value in self.targets.items()},
            metadata=[self.metadata[index] for index in indices.tolist()] if self.metadata else [],
            bag_features=select_optional(self.bag_features),
            bag_feature_mask=select_optional(self.bag_feature_mask),
            region_index=select_optional(self.region_index),
            region_present=select_optional(self.region_present),
            region_coverage=select_optional(self.region_coverage),
            support_features=select_optional(self.support_features),
            support_coverage=select_optional(self.support_coverage),
            coverage=select_optional(self.coverage),
            input_mode=self.input_mode,
        )

    @classmethod
    def from_observations(
        cls,
        *,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        quality: torch.Tensor,
        coordinates: torch.Tensor,
        time_features: torch.Tensor,
        point_present_mask: torch.Tensor,
        coverage: torch.Tensor | None = None,
        temperature_valid_mask: torch.Tensor | None = None,
        input_mode: str,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "WindowBatch":
        """Build the common student input from a collated observation view."""
        if input_mode not in ("node", "region"):
            raise ValueError("input_mode must be node or region")
        if values.ndim != 3:
            raise ValueError("observation values must have shape [B, W, P]")
        batch_size, window_length, point_count = values.shape
        if valid_mask.shape != values.shape or quality.shape != values.shape:
            raise ValueError("values, valid_mask and quality must share [B, W, P] shape")
        if coordinates.ndim != 3 or coordinates.shape[:2] != (batch_size, point_count):
            raise ValueError("coordinates must have shape [B, P, F]")
        if time_features.ndim != 3 or time_features.shape[:2] != (batch_size, window_length):
            raise ValueError("time_features must have shape [B, W, Ft]")
        if point_present_mask.shape != (batch_size, point_count):
            raise ValueError("point_present_mask must have shape [B, P]")
        point_present_mask = point_present_mask.bool()
        mapping_distance = values.new_zeros((batch_size, point_count))
        if coverage is None:
            coverage = valid_mask.to(values.dtype)
        elif coverage.shape != values.shape:
            raise ValueError("coverage must have shape [B, W, P]")
        if temperature_valid_mask is None:
            temperature_valid_mask = torch.ones(
                (batch_size, window_length, 1), dtype=torch.bool, device=values.device
            )
        elif temperature_valid_mask.ndim == 2:
            temperature_valid_mask = temperature_valid_mask.unsqueeze(-1)
        if temperature_valid_mask.shape != (batch_size, window_length, 1):
            raise ValueError("temperature_valid_mask must have shape [B, W, 1]")
        return cls(
            values=values,
            valid_mask=valid_mask.bool(),
            quality=quality,
            point_features=coordinates,
            time_features=time_features,
            point_present_mask=point_present_mask,
            mapping_distance=mapping_distance,
            mapping_valid_mask=point_present_mask.clone(),
            temperature_valid_mask=temperature_valid_mask.bool(),
            teacher_features={},
            targets={},
            metadata=list(metadata or []),
            coverage=coverage,
            region_present=point_present_mask if input_mode == "region" else None,
            region_coverage=coverage if input_mode == "region" else None,
            input_mode=input_mode,
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

    @property
    def region_coords(self) -> torch.Tensor:
        """Return the coordinate axis under the shared region-aware name."""
        return self.point_features

    @property
    def effective_coverage(self) -> torch.Tensor | None:
        """Return the first available coverage field for adapters/models."""
        if self.coverage is not None:
            return self.coverage
        if self.region_coverage is not None:
            return self.region_coverage
        return self.support_coverage
