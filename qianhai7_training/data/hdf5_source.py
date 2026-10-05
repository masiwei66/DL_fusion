from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..task_labels import TASK_LABEL_LEVEL_NAMES, TASK_LABEL_NAMES, TASK_CLASS_COUNT


class HDF5StateSource:
    """Lazy adapter from one v3 ordinal-label archive file to loader arrays."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    _cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
    _cache_limit = 4

    def read(self, *, cache: bool = True) -> dict[str, Any]:
        cached = self._cache.get(self.path)
        if cache and cached is not None:
            self._cache.move_to_end(self.path)
            return cached
        with h5py.File(self.path, "r") as handle:
            raw_manifest = handle["manifest_json"][()]
            if isinstance(raw_manifest, bytes):
                raw_manifest = raw_manifest.decode("utf-8")
            task_names = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["labels/task_label_names"][()]
            )
            task_level_names = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["labels/task_label_level_names"][()]
            )
            task_levels = np.asarray(
                handle["labels/task_level_by_acquisition"], dtype=np.int64
            )
            if task_names != TASK_LABEL_NAMES:
                raise ValueError(
                    f"{self.path}: task label order {task_names} does not match "
                    f"the training contract {TASK_LABEL_NAMES}"
                )
            if task_level_names != TASK_LABEL_LEVEL_NAMES:
                raise ValueError(
                    f"{self.path}: task level vocabulary {task_level_names} does not "
                    f"match {TASK_LABEL_LEVEL_NAMES}"
                )
            if task_levels.ndim != 2 or task_levels.shape[1] != len(TASK_LABEL_NAMES):
                raise ValueError(
                    f"{self.path}: task_level_by_acquisition must have shape (steps, 4); "
                    f"got {task_levels.shape}"
                )
            if np.any((task_levels < 0) | (task_levels >= TASK_CLASS_COUNT)):
                raise ValueError(f"{self.path}: task labels contain codes outside 0..3")
            result = {
                "manifest": json.loads(raw_manifest),
                "task_label_names": np.asarray(task_names, dtype=str),
                "task_label_level_names": np.asarray(task_level_names, dtype=str),
                "task_level_by_acquisition": task_levels,
                "node_coords": np.asarray(handle["mesh/node_coords"], dtype=np.float32),
                "node_ids": np.asarray(handle["mesh/node_ids"], dtype=np.int64),
                "deck_node_mask": np.asarray(handle["mesh/deck_node_mask"], dtype=bool),
                "deck_element_connectivity": np.asarray(handle["mesh/deck_element_connectivity"], dtype=np.int64),
                # Only UZ is ever consumed, and the dataset is chunked along the
                # component axis, so this slice reads a third of the bytes.
                "displacement_uz": np.asarray(handle["responses/fem/disp_at_acquisition"][:, :, 2], dtype=np.float32),
                "temperature_C": np.asarray(handle["environment/temperature_by_acquisition_C"], dtype=np.float32),
                "delta_days": np.asarray(handle["environment/delta_days"], dtype=np.float32),
                "state_family_code": np.asarray(handle["labels/state_family_code_by_acquisition"], dtype=np.int64),
                "state_family_vocabulary": np.asarray(handle["labels/state_family_vocabulary"], dtype=str) if "labels/state_family_vocabulary" in handle else np.asarray([], dtype=str),
                "global_severity": np.asarray(handle["labels/global_severity_level_by_acquisition"], dtype=np.int64),
                "material_active_by_acquisition": np.asarray(handle["labels/material_active_by_acquisition"], dtype=bool) if "labels/material_active_by_acquisition" in handle else np.asarray([], dtype=bool),
                "support_active_by_acquisition": np.asarray(handle["labels/support_active_by_acquisition"], dtype=bool) if "labels/support_active_by_acquisition" in handle else np.asarray([], dtype=bool),
                "support_active_by_acquisition_by_group": np.asarray(handle["labels/support_active_by_acquisition_by_group"], dtype=bool) if "labels/support_active_by_acquisition_by_group" in handle else np.asarray([], dtype=bool),
                "support_stiffness_active_by_acquisition": np.asarray(handle["labels/support_stiffness_active_by_acquisition"], dtype=bool) if "labels/support_stiffness_active_by_acquisition" in handle else np.asarray([], dtype=bool),
                "local_deck_active_by_acquisition": np.asarray(handle["labels/local_deck_active_by_acquisition"], dtype=bool) if "labels/local_deck_active_by_acquisition" in handle else np.asarray([], dtype=bool),
                "modal_frequencies": np.asarray(handle["responses/fem/modal_frequencies"], dtype=np.float32)
                if "responses/fem/modal_frequencies" in handle else np.zeros(0, dtype=np.float32),
                "support_settlement_mm": np.asarray(handle["state/support_settlement_by_acquisition_mm"], dtype=np.float32)
                if "state/support_settlement_by_acquisition_mm" in handle else np.zeros((0, 0), dtype=np.float32),
                "support_settlement_ratio": np.asarray(handle["labels/support_settlement_ratio_to_limit_by_acquisition"], dtype=np.float32)
                if "labels/support_settlement_ratio_to_limit_by_acquisition" in handle else np.zeros((0, 0), dtype=np.float32),
                # Per-period normative limit, needed to build per-support ratios.
                "support_settlement_limit_mm": np.asarray(handle["labels/support_settlement_limit_by_acquisition_mm"], dtype=np.float32)
                if "labels/support_settlement_limit_by_acquisition_mm" in handle else np.zeros(0, dtype=np.float32),
                "material_severity": np.asarray(handle["labels/material_severity_by_material"], dtype=np.float32)
                if "labels/material_severity_by_material" in handle else np.zeros(0, dtype=np.float32),
                "material_ids": np.asarray(handle["state/material_ids"], dtype=np.int64)
                if "state/material_ids" in handle else np.asarray([], dtype=np.int64),
                "material_scale_by_material": np.asarray(handle["state/material_scale_by_material"], dtype=np.float32)
                if "state/material_scale_by_material" in handle else np.asarray([], dtype=np.float32),
                "support_group_names": np.asarray(handle["mesh/support_group_names"]).astype(str)
                if "mesh/support_group_names" in handle else np.asarray([], dtype=str),
                "bearing_group_names": np.asarray(handle["state/support_stiffness_group_ids"], dtype=np.int64)
                if "state/support_stiffness_group_ids" in handle else np.arange(1, 9, dtype=np.int64),
                "support_trans_scale_by_group": np.asarray(handle["state/support_trans_scale_by_group"], dtype=np.float32)
                if "state/support_trans_scale_by_group" in handle else np.ones(8, dtype=np.float32),
                "support_rot_scale_by_group": np.asarray(handle["state/support_rot_scale_by_group"], dtype=np.float32)
                if "state/support_rot_scale_by_group" in handle else np.ones(8, dtype=np.float32),
                "local_deck_settlement_mm": np.asarray(handle["state/local_deck_settlement_by_acquisition_mm"], dtype=np.float32)
                if "state/local_deck_settlement_by_acquisition_mm" in handle else np.zeros((0, 0), dtype=np.float32),
                "global_severity_value": np.asarray(handle["labels/support_settlement_ratio_to_limit_by_acquisition"], dtype=np.float32)
                if "labels/support_settlement_ratio_to_limit_by_acquisition" in handle else np.asarray(handle["labels/global_severity_level_by_acquisition"], dtype=np.float32),
            }
        if cache:
            self._cache[self.path] = result
            self._cache.move_to_end(self.path)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return result
