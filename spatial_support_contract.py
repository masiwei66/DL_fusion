"""Versioned, dataset-agnostic contract for spatial support-token responses."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


SUPPORT_TOKEN_SCHEMA = "spatial_support_tokens_v1"
SUPPORT_NORMALIZATION_POLICY = "fit_training_states_only_apply_after_support_pooling"
SUPPORT_FEATURE_ORDER = (
    "center_s_unit_minus1_to1",
    "center_t_unit_minus1_to1",
    "extent_s_fraction_of_domain",
    "extent_t_fraction_of_domain",
)


@dataclass(frozen=True)
class SupportTokenValueNormalization:
    """Per-dataset value statistics fitted from pooled training-state tokens."""

    mean: float
    std: float
    fitted_state_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean) or not math.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("support-token normalization statistics must be finite with std > 0")
        if not self.fitted_state_ids:
            raise ValueError("support-token normalization requires training-state provenance")

    def transform(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if values.shape != valid_mask.shape:
            raise ValueError("values and valid_mask must have matching shapes")
        transformed = (values - float(self.mean)) / float(self.std)
        return torch.where(valid_mask.bool(), transformed, torch.zeros_like(transformed))

    def transform_numpy(self, values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if values.shape != valid.shape:
            raise ValueError("values and valid_mask must have matching shapes")
        return np.where(valid, (values - self.mean) / self.std, 0.0).astype(np.float32)


@dataclass(frozen=True)
class SupportTokenInputs:
    """Model inputs; observation tensors use [B,T,R], supports use [B,R,F]."""

    values: torch.Tensor
    valid_mask: torch.Tensor
    quality: torch.Tensor
    coverage: torch.Tensor
    support_features: torch.Tensor
    time_features: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("values must have shape [B,T,R]")
        batch, time, supports = self.values.shape
        if self.valid_mask.shape != self.values.shape:
            raise ValueError("valid_mask must match values [B,T,R]")
        if self.quality.shape != self.values.shape or self.coverage.shape != self.values.shape:
            raise ValueError("quality and coverage must match values [B,T,R]")
        if self.support_features.ndim != 3 or self.support_features.shape[:2] != (batch, supports):
            raise ValueError("support_features must have shape [B,R,F]")
        if self.support_features.shape[-1] != len(SUPPORT_FEATURE_ORDER):
            raise ValueError(f"support_features must follow {SUPPORT_FEATURE_ORDER}")
        if self.time_features.ndim != 3 or self.time_features.shape[:2] != (batch, time):
            raise ValueError("time_features must have shape [B,T,Ft]")
        for name, value in (
            ("values", self.values),
            ("quality", self.quality),
            ("coverage", self.coverage),
            ("support_features", self.support_features),
            ("time_features", self.time_features),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if torch.any((self.quality < 0.0) | (self.quality > 1.0)):
            raise ValueError("quality must be in [0,1]")
        if torch.any((self.coverage < 0.0) | (self.coverage > 1.0)):
            raise ValueError("coverage must be in [0,1]")
        centers = self.support_features[..., :2]
        extents = self.support_features[..., 2:]
        if torch.any((centers < -1.0) | (centers > 1.0)):
            raise ValueError("support centers must be normalized to [-1,1]")
        if torch.any((extents <= 0.0) | (extents > 1.0)):
            raise ValueError("support extents must be positive fractions no greater than 1")

    def to(self, device: torch.device | str) -> SupportTokenInputs:
        return SupportTokenInputs(
            values=self.values.to(device),
            valid_mask=self.valid_mask.to(device),
            quality=self.quality.to(device),
            coverage=self.coverage.to(device),
            support_features=self.support_features.to(device),
            time_features=self.time_features.to(device),
        )


@dataclass(frozen=True)
class SupportTokenTargets:
    """Loss/metric-only response targets, also indexed as [B,T,R]."""

    response: torch.Tensor
    valid_mask: torch.Tensor
    area_coverage: torch.Tensor

    def __post_init__(self) -> None:
        if self.response.ndim != 3:
            raise ValueError("response must have shape [B,T,R]")
        if self.valid_mask.shape != self.response.shape:
            raise ValueError("target valid_mask must match response [B,T,R]")
        if self.area_coverage.shape != self.response.shape:
            raise ValueError("area_coverage must match response [B,T,R]")
        if not torch.isfinite(self.response).all() or not torch.isfinite(self.area_coverage).all():
            raise ValueError("support-token targets must be finite")
        if torch.any((self.area_coverage < 0.0) | (self.area_coverage > 1.0)):
            raise ValueError("area_coverage must be in [0,1]")

    def to(self, device: torch.device | str) -> SupportTokenTargets:
        return SupportTokenTargets(
            response=self.response.to(device),
            valid_mask=self.valid_mask.to(device),
            area_coverage=self.area_coverage.to(device),
        )


class SupportTokenResponseModel(nn.Module):
    """Shared temporal GRU response head for the versioned support-token input."""

    def __init__(
        self,
        *,
        time_feature_count: int,
        support_feature_count: int = 4,
        hidden_size: int = 48,
    ) -> None:
        super().__init__()
        if time_feature_count < 0 or support_feature_count != 4 or hidden_size < 1:
            raise ValueError("invalid feature count or hidden_size for support-token v1")
        self.input_projection = nn.Sequential(
            nn.Linear(4 + time_feature_count + support_feature_count, hidden_size),
            nn.GELU(),
        )
        self.temporal_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_head = nn.Linear(hidden_size, 2)

    def forward(self, inputs: SupportTokenInputs) -> dict[str, torch.Tensor]:
        if not isinstance(inputs, SupportTokenInputs):
            raise TypeError("SupportTokenResponseModel requires SupportTokenInputs")
        values = inputs.values
        batch, time, supports = values.shape
        observed = torch.stack(
            (
                values,
                inputs.valid_mask.to(values.dtype),
                inputs.quality,
                inputs.coverage,
            ),
            dim=-1,
        )
        times = inputs.time_features[:, :, None, :].expand(-1, -1, supports, -1)
        support = inputs.support_features[:, None, :, :].expand(-1, time, -1, -1)
        features = torch.cat((observed, times, support), dim=-1)
        sequences = features.permute(0, 2, 1, 3).reshape(
            batch * supports, time, features.shape[-1]
        )
        encoded, _ = self.temporal_encoder(self.input_projection(sequences))
        output = self.output_head(encoded).reshape(batch, supports, time, 2)
        return {
            "mean": output[..., 0].permute(0, 2, 1),
            "scale": torch.nn.functional.softplus(output[..., 1]).permute(0, 2, 1)
            + 1.0e-4,
        }


def masked_gaussian_nll(
    predictions: dict[str, torch.Tensor], targets: SupportTokenTargets
) -> torch.Tensor:
    mean = predictions["mean"]
    scale = predictions["scale"].clamp_min(1.0e-4)
    if mean.shape != targets.response.shape or scale.shape != targets.response.shape:
        raise ValueError("prediction and support target shapes must match [B,T,R]")
    valid = targets.valid_mask.bool() & torch.isfinite(targets.response)
    if not valid.any():
        raise ValueError("no valid support targets are available for loss")
    standardized = (targets.response[valid] - mean[valid]) / scale[valid]
    nll = (
        0.5 * standardized.square()
        + torch.log(scale[valid])
        + 0.5 * math.log(2.0 * math.pi)
    )
    weights = targets.area_coverage[valid]
    return (nll * weights).sum() / weights.sum().clamp_min(1.0e-12)


def soft_support_memberships(
    coordinates_unit: np.ndarray,
    centers_unit: np.ndarray,
    bandwidths_unit: np.ndarray,
) -> np.ndarray:
    """Shared Gaussian soft assignment for coordinates already mapped to [0,1]^2."""
    coordinates = np.asarray(coordinates_unit, dtype=np.float64)
    centers = np.asarray(centers_unit, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_unit, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_unit must have shape [N,2]")
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
        raise ValueError("centers_unit must have shape [R,2] with R > 0")
    if bandwidths.shape != centers.shape:
        raise ValueError("bandwidths_unit must match centers_unit")
    if not np.isfinite(coordinates).all() or not np.isfinite(centers).all():
        raise ValueError("support assignment coordinates must be finite")
    if not np.isfinite(bandwidths).all() or np.any(bandwidths <= 0.0):
        raise ValueError("support bandwidths must be finite and positive")
    delta = (coordinates[:, None, :] - centers[None, :, :]) / bandwidths[None, :, :]
    log_weights = -0.5 * np.square(delta).sum(axis=-1)
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(np.maximum(log_weights, -80.0))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)
    return weights.astype(np.float32)


def aggregate_support_observations(
    values: np.ndarray,
    valid_mask: np.ndarray,
    quality: np.ndarray,
    memberships: np.ndarray,
    *,
    minimum_coverage: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared quality-weighted pooling and observed-point coverage semantics."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    quality = np.asarray(quality, dtype=np.float32)
    memberships = np.asarray(memberships, dtype=np.float32)
    if values.ndim != 2 or valid.shape != values.shape or quality.shape != values.shape:
        raise ValueError("values, valid_mask and quality must share shape [T,P]")
    if memberships.ndim != 2 or memberships.shape[0] != values.shape[1]:
        raise ValueError("memberships must have shape [P,R]")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in [0,1]")
    quality_valid = np.isfinite(quality) & (quality >= 0.0) & (quality <= 1.0)
    valid &= np.isfinite(values) & quality_valid
    quality = np.where(quality_valid, quality, 0.0)
    total_mass = memberships.sum(axis=0)
    valid_mass = valid.astype(np.float32) @ memberships
    quality_mass = (valid * quality).astype(np.float32) @ memberships
    coverage = valid_mass / np.maximum(total_mass[None, :], 1.0e-12)
    token_quality = quality_mass / np.maximum(valid_mass, 1.0e-12)
    numerator = (valid * quality * values).astype(np.float32) @ memberships
    token_values = numerator / np.maximum(quality_mass, 1.0e-12)
    token_valid = (coverage >= minimum_coverage) & (quality_mass > 0.0)
    return (
        np.where(token_valid, token_values, 0.0).astype(np.float32),
        token_valid,
        np.where(token_valid, token_quality, 0.0).astype(np.float32),
        np.clip(coverage, 0.0, 1.0).astype(np.float32),
    )


def aggregate_area_weighted_responses(
    values: np.ndarray,
    valid_mask: np.ndarray,
    memberships: np.ndarray,
    nodal_area_weights: np.ndarray,
    *,
    minimum_area_coverage: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared area-weighted response aggregation and target coverage semantics."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    memberships = np.asarray(memberships, dtype=np.float32)
    areas = np.asarray(nodal_area_weights, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("target values and mask must share shape [T,N]")
    if memberships.ndim != 2 or memberships.shape[0] != values.shape[1]:
        raise ValueError("target memberships must have shape [N,R]")
    if areas.size != values.shape[1]:
        raise ValueError("nodal area weights must match target nodes")
    if not 0.0 < minimum_area_coverage <= 1.0:
        raise ValueError("minimum_area_coverage must be in (0,1]")
    if not np.isfinite(areas).all() or np.any(areas < 0.0) or areas.sum() <= 0.0:
        raise ValueError("nodal area weights must be finite, non-negative and non-empty")
    valid &= np.isfinite(values)
    area_memberships = memberships * areas[:, None]
    total_area = area_memberships.sum(axis=0)
    if np.any(total_area <= 0.0) or not np.isfinite(total_area).all():
        raise ValueError("one or more supports have no finite positive mesh area")
    valid_area = valid.astype(np.float32) @ area_memberships
    area_coverage = np.clip(valid_area / total_area[None, :], 0.0, 1.0)
    numerator = np.where(valid, values, 0.0).astype(np.float32) @ area_memberships
    response = numerator / np.maximum(valid_area, 1.0e-12)
    target_valid = area_coverage >= minimum_area_coverage
    response = np.where(target_valid, response, 0.0).astype(np.float32)
    if not target_valid.any():
        raise ValueError("no support target meets minimum_area_coverage")
    return (
        response,
        target_valid,
        area_coverage.astype(np.float32),
        total_area.astype(np.float32),
    )
