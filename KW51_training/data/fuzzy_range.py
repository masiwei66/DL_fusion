"""Fuzzy-range multi-instance FEM response data for KW51.

A query is a variable-size bag of approximate InSAR points on/near the bridge.
The exact range is latent; a target distribution is built by sampling K ranges
consistent with the bag and aggregating the clean FEM LOS response over each
range. The model sees only bridge-relative bag coordinates and bag statistics,
never absolute FEM node ids or exact point positions.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import FuzzyConfig
from ..contracts import WindowBatch
from .fem_insar import (
    AdGroupMap,
    FEMArchiveRecord,
    FEMArchiveIndex,
    _normalise_los,
    read_fem_record,
)


@dataclass(frozen=True)
class DeckGeometry:
    node_coords: np.ndarray
    deck_indices: np.ndarray
    deck_xy: np.ndarray
    deck_s: np.ndarray
    deck_t: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    los: np.ndarray

    @property
    def deck_length_m(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def deck_width_m(self) -> float:
        return float(self.y_max - self.y_min)


@dataclass(frozen=True)
class FuzzyRange:
    center_s: float
    center_t: float
    half_s: float
    half_t: float


@dataclass(frozen=True)
class FuzzyTarget:
    samples: np.ndarray  # [K, T] mm
    mean: np.ndarray  # [T] mm
    std: np.ndarray  # [T] mm
    quantiles: np.ndarray  # [T, 3] mm
    sample_valid: np.ndarray  # [K, T] bool
    time_valid: np.ndarray  # [T] bool


@dataclass(frozen=True)
class FuzzyQueryRef:
    record: FEMArchiveRecord
    realization: int


@dataclass(frozen=True)
class FuzzyNormalization:
    value_mean: float
    value_std: float
    target_mean: float
    target_std: float
    temperature_mean: float
    temperature_std: float
    delta_days_mean: float
    delta_days_std: float
    time_feature_mode: str
    fitted_state_ids: tuple[str, ...]

    def transform_values(self, values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        transformed = (values - self.value_mean) / self.value_std
        return np.where(np.asarray(valid_mask, dtype=bool), transformed, 0.0).astype(np.float32)

    def transform_targets(self, samples: np.ndarray) -> np.ndarray:
        return ((np.asarray(samples, dtype=np.float32) - self.target_mean) / self.target_std).astype(np.float32)

    def transform_time_features(self, temperature: np.ndarray, delta_days: np.ndarray) -> np.ndarray:
        temperature = np.nan_to_num(np.asarray(temperature, dtype=np.float32), nan=0.0)
        delta_days = np.nan_to_num(np.asarray(delta_days, dtype=np.float32), nan=0.0)
        if self.time_feature_mode == "disabled":
            return np.zeros((temperature.size, 2), dtype=np.float32)
        if self.time_feature_mode == "standardized":
            temperature = (temperature - self.temperature_mean) / self.temperature_std
            delta_days = (delta_days - self.delta_days_mean) / self.delta_days_std
        return np.c_[temperature, delta_days].astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "kw51_fuzzy_normalization_v1",
            "value_mean": float(self.value_mean),
            "value_std": float(self.value_std),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
            "temperature_mean": float(self.temperature_mean),
            "temperature_std": float(self.temperature_std),
            "delta_days_mean": float(self.delta_days_mean),
            "delta_days_std": float(self.delta_days_std),
            "time_feature_mode": self.time_feature_mode,
            "fitted_state_ids": list(self.fitted_state_ids),
        }


def build_deck_geometry(record: FEMArchiveRecord, los_vector: Iterable[float]) -> DeckGeometry:
    arrays = read_fem_record(record)
    coords = np.asarray(arrays["node_coords"], dtype=np.float64)
    deck_mask = np.asarray(arrays["deck_mask"], dtype=bool).reshape(-1)
    usable = deck_mask & np.all(np.isfinite(coords[:, :2]), axis=1)
    deck_indices = np.flatnonzero(usable)
    if deck_indices.size == 0:
        raise ValueError("No usable deck nodes for fuzzy-range geometry")
    xy = coords[deck_indices, :2]
    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    s = ((xy[:, 0] - x_min) / x_span).astype(np.float32)
    t = (2.0 * (xy[:, 1] - y_min) / y_span - 1.0).astype(np.float32)
    return DeckGeometry(
        node_coords=coords,
        deck_indices=deck_indices,
        deck_xy=xy.astype(np.float32),
        deck_s=s,
        deck_t=t,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        los=_normalise_los(los_vector),
    )


def _state_sort_key(state_id: str) -> tuple[int, object]:
    try:
        return (0, int(state_id))
    except (TypeError, ValueError):
        return (1, str(state_id))


class FuzzyRangeSource:
    """Lazy clean LOS cache and fuzzy bag/target generator."""

    def __init__(
        self,
        index: FEMArchiveIndex,
        geometry: DeckGeometry,
        config: FuzzyConfig,
        *,
        seed: int = 0,
        cache_limit: int = 0,
    ):
        self.index = index
        self.geometry = geometry
        self.config = config
        self.seed = int(seed)
        self.cache_limit = int(cache_limit)
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def _record_seed(self, record: FEMArchiveRecord, realization: int) -> int:
        sample_seed = float(record.metadata.get("sample", {}).get("random_seed", 0.0))
        return int(self.seed + int(sample_seed) + 1009 * int(realization) + 7919)

    def _clean_arrays(
        self, record: FEMArchiveRecord
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(record.sample_id)
        if cached is not None:
            self._cache.move_to_end(record.sample_id)
            return cached
        arrays = read_fem_record(record)
        field = np.asarray(arrays["field"], dtype=np.float64)
        valid = np.asarray(arrays["valid"], dtype=bool)
        projected = np.einsum("tnc,c->tn", np.nan_to_num(field, nan=0.0), self.geometry.los)
        projected_valid = valid.all(axis=2) & np.isfinite(field).all(axis=2)
        deck_values = projected[:, self.geometry.deck_indices]
        deck_valid = projected_valid[:, self.geometry.deck_indices]
        reference = (deck_values * deck_valid).sum(axis=1) / np.maximum(
            deck_valid.sum(axis=1), 1
        )
        referenced = np.where(
            deck_valid,
            (deck_values - reference[:, None]) * 1000.0,
            0.0,
        ).astype(np.float32)
        temperature = np.nan_to_num(
            np.asarray(arrays["temperature"], dtype=np.float32), nan=0.0
        )
        delta_days = np.nan_to_num(
            np.asarray(arrays["delta_days"], dtype=np.float32), nan=0.0
        )
        result = (referenced, deck_valid.astype(bool), temperature, delta_days)
        if self.cache_limit <= 0:
            self._cache[record.sample_id] = result
        else:
            self._cache[record.sample_id] = result
            self._cache.move_to_end(record.sample_id)
            while len(self._cache) > self.cache_limit:
                self._cache.popitem(last=False)
        return result

    def _sample_true_range(self, rng: np.random.Generator) -> FuzzyRange:
        geometry = self.geometry
        deck_length = geometry.deck_length_m
        deck_width = geometry.deck_width_m
        length = float(
            rng.uniform(
                self.config.range_length_m_min,
                min(self.config.range_length_m_max, 0.85 * deck_length),
            )
        )
        width = float(
            rng.uniform(
                self.config.range_width_m_min,
                min(self.config.range_width_m_max, 0.95 * deck_width),
            )
        )
        half_s = (length / 2.0) / deck_length
        half_t = (width / 2.0) / max(deck_width / 2.0, 1e-6)
        margin = self.config.range_center_margin_fraction
        center_s = float(rng.uniform(margin + half_s, 1.0 - margin - half_s))
        center_t = float(rng.uniform(-1.0 + half_t, 1.0 - half_t))
        return FuzzyRange(center_s, center_t, half_s, half_t)

    def _interpolate(
        self, los_values: np.ndarray, los_valid: np.ndarray, true_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest-k inverse-distance interpolation for a bag of points."""
        geometry = self.geometry
        distances = np.linalg.norm(
            true_xy[:, None, :] - geometry.deck_xy[None, :, :], axis=2
        )
        k = min(int(self.config.point_interpolation_k), distances.shape[1])
        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        weights = 1.0 / (nearest_distances + 1.0e-3)
        weights = weights / weights.sum(axis=1, keepdims=True)
        values = los_values[:, nearest]  # [T,P,k]
        valid = los_valid[:, nearest]
        weighted_valid = valid * weights[None, :, :]
        denominator = weighted_valid.sum(axis=2)
        numerator = (values * weighted_valid).sum(axis=2)
        interpolated = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), 0.0)
        point_valid = denominator > 0
        return (
            interpolated.astype(np.float32),
            point_valid,
            nearest_distances[:, 0].astype(np.float32),
        )

    def _degrade_observations(
        self,
        interpolated: np.ndarray,
        point_valid: np.ndarray,
        true_s: np.ndarray,
        true_t: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply coordinate jitter, bias, noise and dropouts to interpolated points."""
        geometry = self.geometry
        point_count = int(true_s.size)
        jitter_s = self.config.coordinate_jitter_m / max(geometry.deck_length_m, 1e-6)
        jitter_t = self.config.coordinate_jitter_m / max(
            geometry.deck_width_m / 2.0, 1e-6
        )
        observed_s = np.clip(
            true_s + rng.normal(0.0, jitter_s, point_count), 0.0, 1.0
        )
        observed_t = np.clip(
            true_t + rng.normal(0.0, jitter_t, point_count), -1.0, 1.0
        )
        point_bias = rng.normal(
            0.0, self.config.point_bias_std_mm, size=(point_count,)
        ).astype(np.float32)
        noise = rng.normal(
            0.0, self.config.noise_std_mm, size=interpolated.shape
        ).astype(np.float32)
        valid = point_valid.copy()
        dropout = rng.random(valid.shape) < self.config.missing_probability
        valid &= ~dropout
        block_length = max(
            1, int(round(valid.shape[0] * self.config.block_dropout_fraction))
        )
        if valid.shape[0] > 1:
            for point in range(valid.shape[1]):
                if rng.random() < self.config.block_dropout_probability:
                    start = int(
                        rng.integers(0, max(1, valid.shape[0] - block_length + 1))
                    )
                    valid[start : min(valid.shape[0], start + block_length), point] = False
        observed = np.where(
            valid,
            interpolated + point_bias[None, :] + noise,
            0.0,
        ).astype(np.float32)
        quality = np.full(
            valid.shape,
            np.float32(
                max(self.config.min_quality, 1.0 - self.config.noise_std_mm / 10.0)
            ),
            dtype=np.float32,
        )
        quality[~valid] = 0.0
        return observed, valid, quality, observed_s, observed_t

    def _range_weights(self, ranges: list[FuzzyRange]) -> np.ndarray:
        geometry = self.geometry
        centers_s = np.asarray([r.center_s for r in ranges], dtype=np.float32)[:, None]
        centers_t = np.asarray([r.center_t for r in ranges], dtype=np.float32)[:, None]
        half_s = np.asarray([max(r.half_s, 1e-5) for r in ranges], dtype=np.float32)[:, None]
        half_t = np.asarray([max(r.half_t, 1e-5) for r in ranges], dtype=np.float32)[:, None]
        ds = (geometry.deck_s[None, :] - centers_s) / half_s
        dt = (geometry.deck_t[None, :] - centers_t) / half_t
        weights = np.exp(-0.5 * (ds * ds + dt * dt)).astype(np.float64)
        totals = weights.sum(axis=1, keepdims=True)
        weights = weights / np.maximum(totals, 1e-12)
        return weights.astype(np.float32)

    def _targets(
        self,
        bag_s: np.ndarray,
        bag_t: np.ndarray,
        los_values: np.ndarray,
        los_valid: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[FuzzyTarget, list[FuzzyRange]]:
        geometry = self.geometry
        deck_length = geometry.deck_length_m
        deck_width = geometry.deck_width_m
        center_s = float(np.median(bag_s))
        center_t = float(np.median(bag_t))
        q25_s, q75_s = np.quantile(bag_s, [0.25, 0.75])
        q25_t, q75_t = np.quantile(bag_t, [0.25, 0.75])
        base_half_s = max(float(q75_s - q25_s), 0.5 * self.config.range_length_m_min / deck_length)
        base_half_t = max(
            float(q75_t - q25_t),
            0.5 * self.config.range_width_m_min / max(deck_width / 2.0, 1e-6),
        )
        min_half_s = 0.5 * self.config.range_length_m_min / deck_length
        max_half_s = 0.5 * self.config.range_length_m_max / deck_length
        min_half_t = 0.5 * self.config.range_width_m_min / max(deck_width / 2.0, 1e-6)
        max_half_t = 0.5 * self.config.range_width_m_max / max(deck_width / 2.0, 1e-6)
        ranges: list[FuzzyRange] = []
        attempts = 0
        while len(ranges) < int(self.config.target_range_count):
            attempts += 1
            if attempts > int(self.config.target_range_count) * 10:
                break
            candidate_s = float(
                center_s
                + rng.normal(0.0, self.config.target_center_sigma_fraction * base_half_s)
            )
            candidate_t = float(
                center_t
                + rng.normal(0.0, self.config.target_center_sigma_fraction * base_half_t)
            )
            half_s = float(
                np.clip(
                    base_half_s * np.exp(rng.normal(0.0, self.config.target_extent_log_sigma)),
                    min_half_s,
                    max_half_s,
                )
            )
            half_t = float(
                np.clip(
                    base_half_t * np.exp(rng.normal(0.0, self.config.target_extent_log_sigma)),
                    min_half_t,
                    max_half_t,
                )
            )
            candidate_s = float(np.clip(candidate_s, half_s, 1.0 - half_s))
            candidate_t = float(np.clip(candidate_t, -1.0 + half_t, 1.0 - half_t))
            candidate = FuzzyRange(candidate_s, candidate_t, half_s, half_t)
            weights = self._range_weights([candidate])[0]
            # Node support must use weights relative to the footprint peak; the
            # absolute weights are mass-normalized, so large ranges would
            # otherwise fall below any fixed threshold on a dense deck.
            peak = float(weights.max())
            if peak <= 0.0:
                continue
            if np.count_nonzero(weights >= peak * 1e-3) < self.config.target_min_nodes:
                continue
            ranges.append(candidate)
        if len(ranges) != int(self.config.target_range_count):
            raise RuntimeError(
                "Could not sample the requested number of fuzzy target ranges: "
                f"{len(ranges)}/{self.config.target_range_count}"
            )
        weights = self._range_weights(ranges)
        weighted_values = los_values * los_valid
        numerator = weights @ weighted_values.T  # [K,T]
        denominator = weights @ los_valid.T  # [K,T]
        sample_valid = denominator > 0
        responses = np.where(
            sample_valid,
            numerator / np.maximum(denominator, 1.0e-12),
            0.0,
        ).astype(np.float32)  # [K,T]
        counts = sample_valid.sum(axis=0)
        time_valid = counts > 0
        mean = np.where(
            time_valid,
            (responses * sample_valid).sum(axis=0) / np.maximum(counts, 1),
            0.0,
        ).astype(np.float32)
        variance = np.where(
            time_valid,
            (((responses - mean[None, :]) ** 2) * sample_valid).sum(axis=0)
            / np.maximum(counts, 1),
            0.0,
        )
        std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        quantiles = np.zeros((responses.shape[1], 3), dtype=np.float32)
        for time_index in range(responses.shape[1]):
            values = responses[sample_valid[:, time_index], time_index]
            if values.size:
                quantiles[time_index] = np.quantile(values, [0.05, 0.5, 0.95])
        return (
            FuzzyTarget(
                samples=responses,
                mean=mean,
                std=std,
                quantiles=quantiles,
                sample_valid=sample_valid,
                time_valid=time_valid,
            ),
            ranges,
        )

    def sample_query(
        self,
        record: FEMArchiveRecord,
        realization: int,
        *,
        with_targets: bool = True,
    ) -> dict[str, object]:
        rng = np.random.default_rng(self._record_seed(record, realization))
        los_values, los_valid, temperature, delta_days = self._clean_arrays(record)
        geometry = self.geometry
        true_range = self._sample_true_range(rng)
        point_count = int(
            rng.integers(self.config.point_count_min, self.config.point_count_max + 1)
        )
        halo_count = int(round(point_count * self.config.bag_halo_fraction))
        inlier_count = max(point_count - halo_count, 1)
        inlier_s = rng.uniform(
            true_range.center_s - true_range.half_s,
            true_range.center_s + true_range.half_s,
            inlier_count,
        )
        inlier_t = rng.uniform(
            true_range.center_t - true_range.half_t,
            true_range.center_t + true_range.half_t,
            inlier_count,
        )
        if halo_count > 0:
            spread_s = max(2.5 * true_range.half_s, 0.03)
            spread_t = max(2.5 * true_range.half_t, 0.05)
            halo_s = rng.uniform(
                true_range.center_s - spread_s,
                true_range.center_s + spread_s,
                halo_count,
            )
            halo_t = rng.uniform(
                true_range.center_t - spread_t,
                true_range.center_t + spread_t,
                halo_count,
            )
            true_s = np.concatenate([inlier_s, halo_s])
            true_t = np.concatenate([inlier_t, halo_t])
        else:
            true_s, true_t = inlier_s, inlier_t
        true_s = np.clip(true_s, 0.0, 1.0)
        true_t = np.clip(true_t, -1.0, 1.0)
        order = rng.permutation(point_count)
        true_s, true_t = true_s[order], true_t[order]
        true_xy = np.c_[
            geometry.x_min + true_s * geometry.deck_length_m,
            geometry.y_min + (true_t + 1.0) / 2.0 * geometry.deck_width_m,
        ]
        interpolated, point_valid, mapping_distance = self._interpolate(
            los_values, los_valid, true_xy
        )
        observed, valid, quality, observed_s, observed_t = self._degrade_observations(
            interpolated, point_valid, true_s, true_t, rng
        )
        if with_targets:
            target, target_ranges = self._targets(
                observed_s, observed_t, los_values, los_valid, rng
            )
            true_weights = self._range_weights([true_range])
            true_numerator = true_weights @ (los_values * los_valid).T
            true_denominator = true_weights @ los_valid.T
            true_response = np.where(
                true_denominator > 0,
                true_numerator / np.maximum(true_denominator, 1.0e-12),
                0.0,
            ).astype(np.float32)[0]
            target_range_count = len(target_ranges)
            target_valid_fraction = float(target.time_valid.mean())
            target_mean_mm = float(target.mean.mean())
            target_std_mm = float(target.std.mean())
            true_range_mean_mm = true_response.tolist()
        else:
            time_length = observed.shape[0]
            range_count = int(self.config.target_range_count)
            target = FuzzyTarget(
                samples=np.zeros((range_count, time_length), dtype=np.float32),
                mean=np.zeros(time_length, dtype=np.float32),
                std=np.zeros(time_length, dtype=np.float32),
                quantiles=np.zeros((time_length, 3), dtype=np.float32),
                sample_valid=np.zeros((range_count, time_length), dtype=bool),
                time_valid=np.zeros(time_length, dtype=bool),
            )
            target_range_count = 0
            target_valid_fraction = 0.0
            target_mean_mm = 0.0
            target_std_mm = 0.0
            true_range_mean_mm = []
        q25_s, q75_s = np.quantile(observed_s, [0.25, 0.75])
        q25_t, q75_t = np.quantile(observed_t, [0.25, 0.75])
        extent_s = max(float(q75_s - q25_s), 1e-4)
        extent_t = max(float(q75_t - q25_t), 1e-4)
        density = point_count / max(extent_s * extent_t, 1e-6)
        valid_ratio = float(valid.mean())
        quality_mean = float(quality[valid].mean()) if valid.any() else 0.0
        bag_features = np.asarray(
            [
                float(np.median(observed_s)),
                float(np.median(observed_t)),
                float(np.log1p(extent_s * 100.0)),
                float(np.log1p(extent_t * 100.0)),
                float(np.log1p(density)),
                float(point_count / max(self.config.point_count_max, 1)),
                valid_ratio,
                quality_mean,
                float(mapping_distance.mean()),
            ],
            dtype=np.float32,
        )
        return {
            "values": observed,
            "valid_mask": valid,
            "quality": quality,
            "point_coords": np.c_[observed_s, observed_t].astype(np.float32),
            "point_present": np.ones(point_count, dtype=bool),
            "mapping_distance": mapping_distance,
            "mapping_valid": point_valid.any(axis=0),
            "time_features": np.c_[temperature, delta_days].astype(np.float32),
            "bag_features": bag_features,
            "target_samples": target.samples,
            "target_mean": target.mean,
            "target_std": target.std,
            "target_quantiles": target.quantiles,
            "target_sample_valid": target.sample_valid,
            "target_time_valid": target.time_valid,
            "metadata": {
                "sample_id": record.sample_id,
                "structural_state_id": record.structural_state_id,
                "trajectory_id": record.trajectory_id,
                "observation_realization_id": int(realization),
                "point_count": int(point_count),
                "bag_halo_fraction": float(self.config.bag_halo_fraction),
                "true_range": {
                    "center_s": true_range.center_s,
                    "center_t": true_range.center_t,
                    "half_s": true_range.half_s,
                    "half_t": true_range.half_t,
                },
                "target_range_count": int(target_range_count),
                "target_valid_fraction": float(target_valid_fraction),
                "target_mean_mm": float(target_mean_mm),
                "target_std_mm": float(target_std_mm),
                # Diagnostic only: the hidden generator range, never a model input.
                "true_range_mean_mm": true_range_mean_mm,
            },
        }


def fit_fuzzy_normalization(
    source: FuzzyRangeSource,
    records: Iterable[FEMArchiveRecord],
    config: FuzzyConfig,
    *,
    time_feature_mode: str = "disabled",
) -> FuzzyNormalization:
    by_state: dict[str, list[FEMArchiveRecord]] = {}
    for record in records:
        by_state.setdefault(record.structural_state_id, []).append(record)
    values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    delta_days: list[np.ndarray] = []
    state_ids: set[str] = set()
    for state_id in sorted(by_state, key=_state_sort_key):
        items = by_state[state_id]
        if config.normalization_records_per_state > 0:
            items = items[: int(config.normalization_records_per_state)]
        for record in items:
            los_values, los_valid, temp, delta = source._clean_arrays(record)
            finite = los_values[los_valid]
            if finite.size:
                values.append(np.asarray(finite, dtype=np.float64))
            temperatures.append(temp)
            delta_days.append(delta)
            state_ids.add(state_id)
            query = source.sample_query(record, realization=0)
            target_values.append(
                np.asarray(query["target_samples"], dtype=np.float64).reshape(-1)
            )
    if not values:
        raise ValueError("No clean FEM values available for fuzzy normalization")
    flattened = np.concatenate(values)
    target_flattened = (
        np.concatenate(target_values) if target_values else flattened
    )
    temp = np.concatenate(temperatures) if temperatures else np.zeros(1, dtype=np.float32)
    delta = np.concatenate(delta_days) if delta_days else np.zeros(1, dtype=np.float32)
    return FuzzyNormalization(
        value_mean=float(flattened.mean()),
        value_std=max(float(flattened.std()), 1e-6),
        target_mean=float(target_flattened.mean()),
        target_std=max(float(target_flattened.std()), 1e-6),
        temperature_mean=float(temp.mean()),
        temperature_std=max(float(temp.std()), 1e-6),
        delta_days_mean=float(delta.mean()),
        delta_days_std=max(float(delta.std()), 1e-6),
        time_feature_mode=time_feature_mode,
        fitted_state_ids=tuple(sorted(state_ids, key=_state_sort_key)),
    )


def make_fuzzy_query_refs(
    records: Iterable[FEMArchiveRecord], realizations: int
) -> list[FuzzyQueryRef]:
    return [
        FuzzyQueryRef(record, realization)
        for record in records
        for realization in range(max(1, int(realizations)))
    ]


class FuzzyQueryDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        refs: list[FuzzyQueryRef],
        source: FuzzyRangeSource,
        normalization: FuzzyNormalization,
        group_map: AdGroupMap | None = None,
        with_targets: bool = True,
    ):
        self.refs = refs
        self.source = source
        self.normalization = normalization
        self.group_map = group_map
        self.with_targets = bool(with_targets)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        result = self.source.sample_query(
            ref.record, ref.realization, with_targets=self.with_targets
        )
        values = self.normalization.transform_values(
            result["values"], result["valid_mask"]
        )
        samples = self.normalization.transform_targets(result["target_samples"])
        target_mean = (
            np.asarray(result["target_mean"], dtype=np.float32)
            - self.normalization.target_mean
        ) / self.normalization.target_std
        target_std = (
            np.asarray(result["target_std"], dtype=np.float32)
            / self.normalization.target_std
        )
        quantiles = (
            np.asarray(result["target_quantiles"], dtype=np.float32)
            - self.normalization.target_mean
        ) / self.normalization.target_std
        time_features = self.normalization.transform_time_features(
            result["time_features"][:, 0], result["time_features"][:, 1]
        )
        metadata = dict(result["metadata"])
        state_id = ref.record.structural_state_id
        group_id = (
            None if self.group_map is None
            else int(self.group_map.state_to_group[state_id])
        )
        ad_rel = (
            None if self.group_map is None
            else float(self.group_map.state_to_value[state_id])
        )
        id_rel = ref.record.parameters.get("Id_rel")
        if group_id is not None:
            metadata["ad_group_id"] = int(group_id)
            metadata["ad_rel"] = float(ad_rel)
        if id_rel is not None:
            metadata["id_rel"] = float(id_rel)
        metadata["target_mean_mm"] = float(
            np.asarray(result["target_mean"], dtype=np.float64).mean()
        )
        metadata["target_std_mm"] = float(
            np.asarray(result["target_std"], dtype=np.float64).mean()
        )
        raw_values = np.asarray(result["values"], dtype=np.float64)
        raw_valid = np.asarray(result["valid_mask"], dtype=bool)
        bag_mean_mm = (
            np.where(raw_valid, raw_values, 0.0).sum(axis=1)
            / np.maximum(raw_valid.sum(axis=1), 1)
        ).astype(np.float32)
        item = {
            "values": values,
            "valid_mask": result["valid_mask"],
            "quality": result["quality"],
            "point_coords": result["point_coords"],
            "point_present": result["point_present"],
            "mapping_distance": result["mapping_distance"],
            "mapping_valid": result["mapping_valid"],
            "time_features": time_features,
            "bag_features": result["bag_features"],
            "target_samples": samples,
            "target_mean": target_mean.astype(np.float32),
            "target_std": target_std.astype(np.float32),
            "target_quantiles": quantiles.astype(np.float32),
            "target_sample_valid": np.asarray(result["target_sample_valid"], dtype=bool),
            "target_time_valid": np.asarray(result["target_time_valid"], dtype=bool),
            "bag_mean_mm": bag_mean_mm,
            "metadata": metadata,
        }
        if group_id is not None:
            item["ad_group_id"] = np.int64(group_id)
            item["ad_rel"] = np.float32(ad_rel)
            item["id_rel"] = np.float32(id_rel if id_rel is not None else 0.0)
        return item


def collate_fuzzy_queries(items: list[dict[str, object]]) -> WindowBatch:
    batch_size = len(items)
    window_length = int(np.asarray(items[0]["values"]).shape[0])
    max_points = max(int(np.asarray(item["values"]).shape[1]) for item in items)

    def pad_matrix(key: str, fill: float, dtype: np.dtype) -> np.ndarray:
        padded = np.full((batch_size, window_length, max_points), fill, dtype=dtype)
        for row, item in enumerate(items):
            value = np.asarray(item[key])
            padded[row, :, : value.shape[1]] = value
        return padded

    def pad_points(key: str, fill: float, dtype: np.dtype) -> np.ndarray:
        width = np.asarray(items[0][key]).shape[-1]
        padded = np.full((batch_size, max_points, width), fill, dtype=dtype)
        for row, item in enumerate(items):
            value = np.asarray(item[key])
            padded[row, : value.shape[0], :] = value
        return padded

    point_present = np.zeros((batch_size, max_points), dtype=bool)
    mapping_distance = np.zeros((batch_size, max_points), dtype=np.float32)
    mapping_valid = np.zeros((batch_size, max_points), dtype=bool)
    for row, item in enumerate(items):
        count = int(np.asarray(item["point_present"]).shape[0])
        point_present[row, :count] = np.asarray(item["point_present"], dtype=bool)
        mapping_distance[row, :count] = np.asarray(item["mapping_distance"], dtype=np.float32)
        mapping_valid[row, :count] = np.asarray(item["mapping_valid"], dtype=bool)

    targets = {
        "response_samples": torch.from_numpy(
            np.stack([np.asarray(item["target_samples"], dtype=np.float32) for item in items])
        ).float(),
        "response_mean": torch.from_numpy(
            np.stack([np.asarray(item["target_mean"], dtype=np.float32) for item in items])
        ).float(),
        "response_std": torch.from_numpy(
            np.stack([np.asarray(item["target_std"], dtype=np.float32) for item in items])
        ).float(),
        "response_quantiles": torch.from_numpy(
            np.stack([np.asarray(item["target_quantiles"], dtype=np.float32) for item in items])
        ).float(),
        "response_sample_valid": torch.from_numpy(
            np.stack(
                [np.asarray(item["target_sample_valid"], dtype=bool) for item in items]
            )
        ).bool(),
        "response_valid": torch.from_numpy(
            np.stack(
                [np.asarray(item["target_time_valid"], dtype=bool) for item in items]
            )
        ).bool(),
        "bag_mean_mm": torch.from_numpy(
            np.stack([np.asarray(item["bag_mean_mm"], dtype=np.float32) for item in items])
        ).float(),
    }
    if "ad_group_id" in items[0]:
        targets["ad_group_id"] = torch.as_tensor(
            [int(item["ad_group_id"]) for item in items], dtype=torch.long
        )
        targets["ad_rel"] = torch.as_tensor(
            [float(item["ad_rel"]) for item in items], dtype=torch.float32
        )
        targets["id_rel"] = torch.as_tensor(
            [float(item["id_rel"]) for item in items], dtype=torch.float32
        )
    return WindowBatch(
        values=torch.from_numpy(pad_matrix("values", 0.0, np.float32)).float(),
        valid_mask=torch.from_numpy(pad_matrix("valid_mask", False, bool)).bool(),
        quality=torch.from_numpy(pad_matrix("quality", 0.0, np.float32)).float(),
        point_features=torch.from_numpy(pad_points("point_coords", 0.0, np.float32)).float(),
        time_features=torch.from_numpy(
            np.stack([np.asarray(item["time_features"], dtype=np.float32) for item in items])
        ).float(),
        point_present_mask=torch.from_numpy(point_present).bool(),
        mapping_distance=torch.from_numpy(mapping_distance).float(),
        mapping_valid_mask=torch.from_numpy(mapping_valid).bool(),
        temperature_valid_mask=torch.ones(
            (batch_size, window_length, 1), dtype=torch.bool
        ),
        teacher_features={},
        targets=targets,
        metadata=[dict(item["metadata"]) for item in items],
        bag_features=torch.from_numpy(
            np.stack([np.asarray(item["bag_features"], dtype=np.float32) for item in items])
        ).float(),
        bag_feature_mask=torch.ones((batch_size,), dtype=torch.bool),
    )
