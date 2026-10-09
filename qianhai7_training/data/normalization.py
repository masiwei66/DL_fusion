from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .hdf5_source import HDF5StateSource
from .windows import WindowRef


def _safe_std(value: np.ndarray, floor: float = 1e-6) -> float:
    return max(float(np.std(value)), floor)


class _RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        count = int(values.size)
        mean = float(values.mean())
        m2 = float(((values - mean) ** 2).sum())
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean = (self.mean * self.count + mean * count) / total
        self.count = total

    def result(self) -> tuple[float, float]:
        if self.count == 0:
            raise ValueError("No values were provided")
        return self.mean, max(float(np.sqrt(self.m2 / self.count)), 1e-6)


@dataclass(frozen=True)
class NormalizationStats:
    """Train-only standardization statistics for the shared observation contract."""

    values_mean: float
    values_std: float
    temperature_mean: float
    temperature_std: float
    delta_days_mean: float
    delta_days_std: float
    coordinates_mean: tuple[float, float]
    coordinates_std: tuple[float, float]
    fitted_state_ids: tuple[str, ...]
    version: str = "standard_v1"

    def transform_values(self, values: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        value = np.asarray(values, dtype=np.float32).copy()
        if valid_mask is None:
            return ((value - self.values_mean) / self.values_std).astype(np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        value[valid] = (value[valid] - self.values_mean) / self.values_std
        value[~valid] = 0.0
        return value.astype(np.float32)

    def transform_temperature(self, temperature: np.ndarray) -> np.ndarray:
        return ((np.asarray(temperature, dtype=np.float32) - self.temperature_mean) / self.temperature_std).astype(np.float32)

    def transform_delta_days(self, delta_days: np.ndarray) -> np.ndarray:
        return ((np.asarray(delta_days, dtype=np.float32) - self.delta_days_mean) / self.delta_days_std).astype(np.float32)

    def transform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        value = np.asarray(coordinates, dtype=np.float32)
        return ((value - np.asarray(self.coordinates_mean, dtype=np.float32)) / np.asarray(self.coordinates_std, dtype=np.float32)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["coordinates_mean"] = tuple(payload["coordinates_mean"])
        payload["coordinates_std"] = tuple(payload["coordinates_std"])
        payload["fitted_state_ids"] = tuple(payload["fitted_state_ids"])
        return cls(**payload)


def fit_normalization_stats(refs: list[WindowRef], *, progress: bool = False) -> NormalizationStats:
    """Fit statistics from unique training states only.

    The full 26-step FEM trajectory is used for stable statistics. Derived rolling
    windows are intentionally not counted repeatedly, and validation/test refs are
    never touched by this function.
    """
    unique: dict[str, WindowRef] = {}
    for ref in refs:
        unique.setdefault(ref.record.structural_state_id, ref)
    if not unique:
        raise ValueError("Cannot fit normalization statistics from an empty training split")

    value_stats = _RunningStats()
    temperature_stats = _RunningStats()
    delta_stats = _RunningStats()
    coordinate_stats = [_RunningStats(), _RunningStats()]
    items = list(unique.values())
    total = len(items)
    report_every = max(1, total // 10)
    started = time.perf_counter()
    for index, ref in enumerate(items, start=1):
        arrays = HDF5StateSource(ref.record.path).read(cache=False)
        deck = arrays["deck_node_mask"]
        displacement = arrays["displacement_uz"][:, deck]
        displacement = displacement - displacement[0:1]
        value_stats.update(displacement)
        temperature_stats.update(arrays["temperature_C"])
        delta_stats.update(arrays["delta_days"])
        coordinate_array = arrays["node_coords"][deck, :2]
        coordinate_stats[0].update(coordinate_array[:, 0])
        coordinate_stats[1].update(coordinate_array[:, 1])
        if progress and (index % report_every == 0 or index == total):
            elapsed = time.perf_counter() - started
            print(
                f"normalization progress={index}/{total} "
                f"rate={index / max(elapsed, 1e-6):.1f} states_per_sec elapsed_sec={elapsed:.0f}",
                flush=True,
            )
    values_mean, values_std = value_stats.result()
    temperature_mean, temperature_std = temperature_stats.result()
    delta_days_mean, delta_days_std = delta_stats.result()
    coordinates_mean = (coordinate_stats[0].result()[0], coordinate_stats[1].result()[0])
    coordinates_std = (coordinate_stats[0].result()[1], coordinate_stats[1].result()[1])
    return NormalizationStats(
        values_mean=values_mean,
        values_std=values_std,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
        delta_days_mean=delta_days_mean,
        delta_days_std=delta_days_std,
        coordinates_mean=coordinates_mean,
        coordinates_std=coordinates_std,
        fitted_state_ids=tuple(sorted(unique)),
    )
