from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import WindowBatch
from .hdf5_source import HDF5StateSource
from .normalization import NormalizationStats
from .observation import ObservationConfig, sample_virtual_observation
from .windows import WindowRef


class QianhaiWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        refs: list[WindowRef],
        observation: ObservationConfig,
        seed: int = 42,
        normalization: NormalizationStats | None = None,
    ):
        self.refs = refs
        self.observation = observation
        self.seed = seed
        self.normalization = normalization
        self._observation_cache: dict[str, dict[str, np.ndarray]] = {}

    def _get_observation(self, ref: WindowRef, arrays: dict[str, Any]) -> dict[str, np.ndarray]:
        key = ref.record.structural_state_id
        cached = self._observation_cache.get(key)
        if cached is not None:
            return cached
        seed = self.seed + sum(ord(char) for char in key)
        rng = np.random.default_rng(seed)
        point_count = int(rng.integers(self.observation.point_count_min, self.observation.point_count_max + 1))
        cached = sample_virtual_observation(
            arrays["node_coords"], arrays["deck_node_mask"], arrays["displacement"],
            self.observation, rng, node_ids=arrays["node_ids"],
            deck_element_connectivity=arrays["deck_element_connectivity"], point_count=point_count,
        )
        cached["point_count"] = np.asarray(point_count, dtype=np.int64)
        self._observation_cache[key] = cached
        return cached

    def __len__(self) -> int:
        return len(self.refs)

    @staticmethod
    def _at_step(array: np.ndarray, index: int, default: Any) -> Any:
        value = np.asarray(array)
        if value.size == 0 or value.shape[0] <= index:
            return default
        return value[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        arrays = HDF5StateSource(ref.record.path).read()
        view = self._get_observation(ref, arrays)
        sl = slice(ref.start, ref.start + ref.length)
        values = view["values"][sl]
        point_coords = view["point_coords"][:, :2]
        temperature = arrays["temperature_C"][sl]
        delta_days = arrays["delta_days"][sl]
        valid_mask = view["valid_mask"][sl]
        temperature_valid_mask = np.isfinite(temperature)
        delta_days_valid_mask = np.isfinite(delta_days)
        temperature = np.nan_to_num(temperature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        delta_days = np.nan_to_num(delta_days, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.normalization is not None:
            values = self.normalization.transform_values(values, valid_mask)
            point_coords = self.normalization.transform_coordinates(point_coords)
            temperature = self.normalization.transform_temperature(temperature)
            delta_days = self.normalization.transform_delta_days(delta_days)
        support_mm = self._at_step(arrays["support_settlement_mm"], ref.target_index, np.zeros(0, dtype=np.float32))
        support_ratio = self._at_step(arrays["support_settlement_ratio"], ref.target_index, np.zeros(0, dtype=np.float32))
        local_deck = self._at_step(arrays["local_deck_settlement_mm"], ref.target_index, np.zeros(0, dtype=np.float32))
        ratio_value = float(np.nanmax(np.asarray(support_ratio, dtype=np.float32))) if np.asarray(support_ratio).size else 0.0
        material = np.asarray(arrays["material_severity"], dtype=np.float32)
        material_value = float(np.nanmax(material)) if material.size else 0.0
        return {
            "values": values,
            "valid_mask": valid_mask,
            "quality": view["quality"][sl],
            "point_coords": point_coords,
            "mapping_distance": view["mapping_distance"],
            "mapping_valid_mask": view["mapping_valid_mask"],
            "temperature_C": temperature,
            "temperature_valid_mask": temperature_valid_mask,
            "delta_days_valid_mask": delta_days_valid_mask,
            "delta_days": delta_days,
            "target_state_family": arrays["state_family_code"][ref.target_index],
            "target_global_severity": arrays["global_severity"][ref.target_index],
            "target_global_severity_value": ratio_value,
            "target_support_settlement_mm": np.asarray(support_mm, dtype=np.float32),
            "target_support_settlement_ratio": np.asarray(support_ratio, dtype=np.float32),
            "target_material_severity": material_value,
            "target_local_deck_settlement_mm": np.asarray(local_deck, dtype=np.float32),
            "metadata": {
                "sample_id": ref.record.sample_id,
                "structural_state_id": ref.record.structural_state_id,
                "window_start": ref.start,
                "window_length": ref.length,
                "mapping_valid_fraction": float(view["mapping_valid"].mean()),
                "mapping_distance_max": float(view["mapping_distance"].max()),
                "point_count": int(view["point_count"]),
                "observation_realization_id": 0,
                "point_count_seed": self.seed + sum(ord(char) for char in ref.record.structural_state_id),
            },
            "teacher_features": {
                "modal_frequencies": arrays.get("modal_frequencies", np.zeros(0, dtype=np.float32)),
            },
        }


def collate_window_batches(items: list[dict[str, Any]]) -> WindowBatch:
    max_points = max(item["values"].shape[1] for item in items)

    def pad_target_vectors(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        arrays = [np.asarray(item[key], dtype=np.float32).reshape(-1) for item in items]
        width = max((array.size for array in arrays), default=0)
        if width == 0:
            return torch.zeros((len(items), 0), dtype=torch.float32), torch.zeros((len(items), 0), dtype=torch.bool)
        values = np.zeros((len(items), width), dtype=np.float32)
        valid = np.zeros((len(items), width), dtype=bool)
        for row, array in enumerate(arrays):
            if array.size:
                values[row, : array.size] = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                valid[row, : array.size] = True
        return torch.from_numpy(values), torch.from_numpy(valid)

    support_mm, support_mm_valid = pad_target_vectors("target_support_settlement_mm")
    support_ratio, support_ratio_valid = pad_target_vectors("target_support_settlement_ratio")
    local_deck, local_deck_valid = pad_target_vectors("target_local_deck_settlement_mm")

    def pad(item: dict[str, Any], key: str, fill: float, dtype: Any) -> np.ndarray:
        value = np.asarray(item[key])
        padded = np.full((value.shape[0], max_points), fill, dtype=dtype) if value.ndim == 2 else np.full((max_points,), fill, dtype=dtype)
        if value.ndim == 2:
            padded[:, : value.shape[1]] = value
        else:
            padded[: value.shape[0]] = value
        return padded

    return WindowBatch(
        values=torch.from_numpy(np.stack([pad(item, "values", 0.0, np.float32) for item in items])).float(),
        valid_mask=torch.from_numpy(np.stack([pad(item, "valid_mask", False, bool) for item in items])).bool(),
        quality=torch.from_numpy(np.stack([pad(item, "quality", 0.0, np.float32) for item in items])).float(),
        point_features=torch.from_numpy(np.stack([np.pad(item["point_coords"][:, :2], ((0, max_points-item["point_coords"].shape[0]), (0,0))) for item in items])).float(),
        time_features=torch.from_numpy(np.stack([np.c_[item["temperature_C"], item["delta_days"]] for item in items])).float(),
        point_present_mask=torch.from_numpy(np.stack([np.r_[np.ones(item["values"].shape[1], dtype=bool), np.zeros(max_points-item["values"].shape[1], dtype=bool)] for item in items])).bool(),
        mapping_distance=torch.from_numpy(np.stack([pad(item, "mapping_distance", 0.0, np.float32) for item in items])).float(),
        mapping_valid_mask=torch.from_numpy(np.stack([np.r_[item["mapping_valid_mask"], np.zeros(max_points-item["mapping_valid_mask"].shape[0], dtype=bool)] for item in items])).bool(),
        temperature_valid_mask=torch.from_numpy(np.stack([item["temperature_valid_mask"][:, None] for item in items])).bool(),
        teacher_features={"modal_frequencies": torch.nn.utils.rnn.pad_sequence([torch.from_numpy(np.asarray(item["teacher_features"]["modal_frequencies"], dtype=np.float32)) for item in items], batch_first=True)},
        targets={
            "state_family": torch.tensor([item["target_state_family"] for item in items], dtype=torch.long),
            "global_severity": torch.tensor([item["target_global_severity"] for item in items], dtype=torch.long),
            "global_severity_value": torch.tensor([item["target_global_severity_value"] for item in items], dtype=torch.float32),
            "support_settlement_mm": support_mm,
            "support_settlement_mm_valid": support_mm_valid,
            "support_settlement_ratio": support_ratio,
            "support_settlement_ratio_valid": support_ratio_valid,
            "material_severity": torch.tensor([item["target_material_severity"] for item in items], dtype=torch.float32),
            "local_deck_settlement_mm": local_deck,
            "local_deck_settlement_mm_valid": local_deck_valid,
        },
        metadata=[item["metadata"] for item in items],
    )
