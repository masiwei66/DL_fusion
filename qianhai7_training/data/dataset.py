from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import WindowBatch
from .hdf5_source import HDF5StateSource
from .normalization import NormalizationStats
from .observation import ObservationConfig, sample_virtual_observation
from ..task_labels import TASK_LABEL_LEVEL_NAMES, TASK_LABEL_NAMES, task_targets_at_index
from .windows import WindowRef

# Large enough to dominate `sum(ord(char) for char in state_id)`, so that
# realization 0 reproduces the original single-realization seeds exactly.
_REALIZATION_STRIDE = 10**6


class QianhaiWindowDataset(Dataset[dict[str, Any]]):
    # Everything a window needs from a state other than the observation view. The
    # displacement field is deliberately absent: it only feeds the observation, so
    # keeping it would defeat the cache with a multi-megabyte read per window.
    # Keep the v3 ordinal labels with the cached state so every overlapping
    # window reads the same acquisition-aligned target without reopening HDF5.
    _STATE_ARRAYS = (
        "temperature_C",
        "delta_days",
        "support_active_by_acquisition",
        "support_active_by_acquisition_by_group",
        "support_settlement_mm",
        "support_settlement_ratio",
        "support_settlement_limit_mm",
        "local_deck_settlement_mm",
        "modal_frequencies",
        "material_ids",
        "material_scale_by_material",
        "support_group_names",
        "support_trans_scale_by_group",
        "support_rot_scale_by_group",
        "task_label_names",
        "task_label_level_names",
        "task_level_by_acquisition",
    )

    def __init__(
        self,
        refs: list[WindowRef],
        observation: ObservationConfig,
        seed: int,
        normalization: NormalizationStats | None,
        realizations: int,
        regional_target_config,
    ):
        self.refs = refs
        self.observation = observation
        self.seed = seed
        self.normalization = normalization
        self.realizations = max(int(realizations), 1)
        # Kept in the constructor for existing experiment configs; v3 labels are
        # read directly from the archive and are not threshold-derived here.
        self._state_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def _realization_seed(self, state_id: str, realization: int) -> int:
        return self.seed + sum(ord(char) for char in state_id) + realization * _REALIZATION_STRIDE

    def _get_state(self, ref: WindowRef, realization: int) -> dict[str, Any]:
        """Read one (state, observation realization) once per process.

        All windows of a state reuse the same label arrays and the same virtual
        observation, so the trajectory is reduced to a small bundle on first use.
        Repeats then cost a dictionary lookup instead of an HDF5 read. Keying on
        the realization as well keeps `realizations > 1` cheap: every variant is
        built once and then reused for the rest of the run.
        """
        key = (ref.record.structural_state_id, realization)
        bundle = self._state_cache.get(key)
        if bundle is not None:
            return bundle
        arrays = HDF5StateSource(ref.record.path).read()
        rng = np.random.default_rng(self._realization_seed(ref.record.structural_state_id, realization))
        point_count = int(rng.integers(self.observation.point_count_min, self.observation.point_count_max + 1))
        view = sample_virtual_observation(
            arrays["node_coords"], arrays["deck_node_mask"], arrays["displacement_uz"],
            self.observation, rng, node_ids=arrays["node_ids"],
            deck_element_connectivity=arrays["deck_element_connectivity"], point_count=point_count,
        )
        view["point_count"] = np.asarray(point_count, dtype=np.int64)
        bundle = {"view": view, "arrays": {name: arrays[name] for name in self._STATE_ARRAYS if name in arrays}}
        self._state_cache[key] = bundle
        return bundle

    def __len__(self) -> int:
        return len(self.refs) * self.realizations

    @staticmethod
    def _at_step(array: np.ndarray, index: int, default: Any) -> Any:
        value = np.asarray(array)
        if value.size == 0 or value.shape[0] <= index:
            return default
        return value[index]

    @staticmethod
    def _mechanism_targets(arrays: dict[str, Any], index: int) -> dict[str, int]:
        """Derive first-stage mechanism labels while preserving fine labels."""
        def step_bool(key: str) -> np.ndarray:
            value = np.asarray(arrays.get(key, np.asarray([], dtype=bool)))
            if value.ndim == 0 or value.size == 0 or value.shape[0] <= index:
                return np.asarray([], dtype=bool)
            return np.asarray(value[index], dtype=bool).reshape(-1)

        return {}

    def __getitem__(self, index: int) -> dict[str, Any]:
        window_count = len(self.refs)
        ref = self.refs[index % window_count]
        realization = index // window_count
        state = self._get_state(ref, realization)
        view = state["view"]
        arrays = state["arrays"]
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
        mechanisms = self._mechanism_targets(arrays, ref.target_index)
        task_targets = task_targets_at_index(arrays, ref.target_index)
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
            **{f"target_{key}": value for key, value in mechanisms.items()},
            **{f"target_{key}": np.int64(value) for key, value in task_targets.items()},
            "target_support_settlement_mm": np.asarray(support_mm, dtype=np.float32),
            "target_support_settlement_ratio": np.asarray(support_ratio, dtype=np.float32),
            "target_local_deck_settlement_mm": np.asarray(local_deck, dtype=np.float32),
            "metadata": {
                "sample_id": ref.record.sample_id,
                "structural_state_id": ref.record.structural_state_id,
                "window_start": ref.start,
                "window_length": ref.length,
                "target_index": ref.target_index,
                "target_task_level_names": {
                    name: TASK_LABEL_LEVEL_NAMES[value]
                    for name, value in task_targets.items()
                },
                "mapping_valid_fraction": float(view["mapping_valid"].mean()),
                "mapping_distance_max": float(view["mapping_distance"].max()),
                "point_count": int(view["point_count"]),
                "observation_realization_id": realization,
                "point_count_seed": self._realization_seed(ref.record.structural_state_id, realization),
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
            "support_settlement_mm": support_mm,
            "support_settlement_mm_valid": support_mm_valid,
            "support_settlement_ratio": support_ratio,
            "support_settlement_ratio_valid": support_ratio_valid,
            "local_deck_settlement_mm": local_deck,
            "local_deck_settlement_mm_valid": local_deck_valid,
            **{
                f"target_{name}": torch.as_tensor(
                    [int(item[f"target_{name}"]) for item in items], dtype=torch.long
                )
                for name in TASK_LABEL_NAMES
            },
            **{
                key: torch.from_numpy(np.stack([np.asarray(item[key], dtype=np.float32) for item in items]))
                for key in (
                    "target_physical_deck_material", "target_physical_pier_settlement", "target_physical_pier_material", "target_physical_bearing_stiffness",
                    "target_engineering_deck_material", "target_engineering_pier_settlement", "target_engineering_pier_material", "target_engineering_bearing_stiffness",
                )
                if key in items[0]
            },
        },
        metadata=[item["metadata"] for item in items],
    )
