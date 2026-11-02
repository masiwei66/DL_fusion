from __future__ import annotations

from typing import Any

import numpy as np

from ..config import RegionalTargetConfig


DECK_MATERIAL_IDS = (2, 3, 4)
PIER_MATERIAL_IDS = (5, 6, 7)
PIER_SETTLEMENT_NAMES = (
    "pier_1_foundation_a",
    "pier_1_foundation_b",
    "pier_2_foundation",
)
BEARING_GROUP_NAMES = (
    "pier_1_left",
    "pier_1_right",
    "pier_2_left",
    "pier_2_right",
    "abutment_0_left",
    "abutment_0_right",
    "abutment_3_left",
    "abutment_3_right",
)

# Every state array the lookups below read. They are accessed with `.get`, so a
# key dropped upstream (for example by the loader's state-array whitelist) would
# be replaced by a default and turn the whole target set into zeros - training
# then drives the loss toward zero while learning nothing. Fail loudly instead.
REQUIRED_ARRAY_KEYS = (
    "material_ids",
    "material_scale_by_material",
    "support_group_names",
    "support_active_by_acquisition",
    "support_settlement_mm",
    "support_settlement_limit_mm",
    "support_trans_scale_by_group",
    "support_rot_scale_by_group",
)


def _require_arrays(arrays: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_ARRAY_KEYS if key not in arrays]
    if missing:
        raise KeyError(f"regional_targets is missing required state arrays: {missing}")


def _as_float(value: Any, default: float = 1.0) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(array[0]) if array.size else default


def _decode_names(values: Any) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return result


def _active_settlement(arrays: dict[str, Any], index: int) -> np.ndarray:
    values = np.asarray(arrays["support_active_by_acquisition"], dtype=bool)
    if values.ndim != 2 or values.shape[0] <= index:
        return np.zeros(len(PIER_SETTLEMENT_NAMES), dtype=bool)
    name_to_column = {name: position for position, name in enumerate(_decode_names(arrays["support_group_names"]))}
    return np.asarray([
        values[index, name_to_column[name]] if name in name_to_column and name_to_column[name] < values.shape[1] else False
        for name in PIER_SETTLEMENT_NAMES
    ], dtype=bool)


def _material_scales(arrays: dict[str, Any]) -> dict[int, float]:
    ids = np.asarray(arrays["material_ids"], dtype=np.int64).reshape(-1)
    scales = np.asarray(arrays["material_scale_by_material"], dtype=np.float32).reshape(-1)
    return {int(material_id): float(scales[position]) if position < scales.size else 1.0 for position, material_id in enumerate(ids)}


def _settlement_ratios(arrays: dict[str, Any], index: int) -> np.ndarray:
    """Per-support settlement as a fraction of the per-period normative limit.

    The archive only stores a scalar ratio per acquisition (the maximum over all
    supports), so each support's ratio is rebuilt from its own settlement.
    """
    settlement_mm = np.asarray(arrays["support_settlement_mm"], dtype=np.float32)
    limit = np.asarray(arrays["support_settlement_limit_mm"], dtype=np.float32).reshape(-1)
    if settlement_mm.ndim != 2 or settlement_mm.shape[0] <= index or limit.size <= index:
        return np.zeros(len(PIER_SETTLEMENT_NAMES), dtype=np.float32)
    limit_value = float(limit[index])
    if limit_value <= 0.0:
        return np.zeros(len(PIER_SETTLEMENT_NAMES), dtype=np.float32)
    column_of = {name: position for position, name in enumerate(_decode_names(arrays["support_group_names"]))}
    return np.asarray([
        settlement_mm[index, column_of[name]] / limit_value
        if name in column_of and column_of[name] < settlement_mm.shape[1] else 0.0
        for name in PIER_SETTLEMENT_NAMES
    ], dtype=np.float32)


def regional_targets(arrays: dict[str, Any], index: int, config: RegionalTargetConfig) -> dict[str, np.ndarray]:
    """Derive physical and engineering regional labels without changing HDF5."""
    _require_arrays(arrays)
    scales = _material_scales(arrays)
    material_physical = np.asarray([scales.get(material_id, 1.0) < 1.0 for material_id in DECK_MATERIAL_IDS], dtype=np.float32)
    pier_material_physical = np.asarray([scales.get(material_id, 1.0) < 1.0 for material_id in PIER_MATERIAL_IDS], dtype=np.float32)

    trans = np.asarray(arrays.get("support_trans_scale_by_group", []), dtype=np.float32).reshape(-1)
    rot = np.asarray(arrays.get("support_rot_scale_by_group", []), dtype=np.float32).reshape(-1)
    bearing_physical = np.asarray([
        (position < trans.size and trans[position] < 1.0) or (position < rot.size and rot[position] < 1.0)
        for position in range(8)
    ], dtype=np.float32)
    settlement_physical = _active_settlement(arrays, index).astype(np.float32)

    settlement_ratios = _settlement_ratios(arrays, index)
    settlement_engineering = settlement_physical.astype(bool) & (settlement_ratios > config.settlement_ratio_threshold)
    material_engineering = material_physical.astype(bool) & np.asarray([scales.get(material_id, 1.0) <= config.material_scale_threshold for material_id in DECK_MATERIAL_IDS])
    pier_material_engineering = pier_material_physical.astype(bool) & np.asarray([scales.get(material_id, 1.0) <= config.material_scale_threshold for material_id in PIER_MATERIAL_IDS])
    bearing_engineering = bearing_physical.astype(bool) & np.asarray([
        (position < trans.size and trans[position] <= config.bearing_scale_threshold)
        or (position < rot.size and rot[position] <= config.bearing_scale_threshold)
        for position in range(8)
    ])

    physical = {
        "deck_material": material_physical,
        "pier_settlement": settlement_physical,
        "pier_material": pier_material_physical,
        "bearing_stiffness": bearing_physical,
    }
    engineering = {
        "deck_material": material_engineering.astype(np.float32),
        "pier_settlement": settlement_engineering.astype(np.float32),
        "pier_material": pier_material_engineering.astype(np.float32),
        "bearing_stiffness": bearing_engineering.astype(np.float32),
    }
    selected = engineering if config.label_mode == "engineering_anomaly" else physical
    return {
        **{f"physical_{key}": value for key, value in physical.items()},
        **{f"engineering_{key}": value for key, value in engineering.items()},
        **selected,
    }
