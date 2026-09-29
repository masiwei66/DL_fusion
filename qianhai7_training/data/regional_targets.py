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


def _support_column_names(arrays: dict[str, Any]) -> list[str]:
    names = _decode_names(arrays.get("support_group_names", []))
    return names


def _active_settlement(arrays: dict[str, Any], index: int) -> np.ndarray:
    values = np.asarray(arrays.get("support_active_by_acquisition", []), dtype=bool)
    names = arrays.get("support_group_names", [])
    if values.ndim != 2 or values.shape[0] <= index:
        return np.zeros(3, dtype=bool)
    name_to_column = {name: position for position, name in enumerate(_decode_names(names))}
    result = np.asarray([
        values[index, name_to_column[name]] if name in name_to_column and name_to_column[name] < values.shape[1] else False
        for name in PIER_SETTLEMENT_NAMES
    ], dtype=bool)
    return result


def _material_scales(arrays: dict[str, Any]) -> dict[int, float]:
    ids = np.asarray(arrays.get("material_ids", DECK_MATERIAL_IDS + PIER_MATERIAL_IDS), dtype=np.int64).reshape(-1)
    scales = np.asarray(arrays.get("material_scale_by_material", []), dtype=np.float32).reshape(-1)
    return {int(material_id): float(scales[position]) if position < scales.size else 1.0 for position, material_id in enumerate(ids)}


def regional_targets(arrays: dict[str, Any], index: int, config: RegionalTargetConfig) -> dict[str, np.ndarray]:
    """Derive physical and engineering regional labels without changing HDF5."""
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

    ratio = np.asarray(arrays.get("support_settlement_ratio", []), dtype=np.float32)
    support_names = _decode_names(arrays.get("support_group_names", []))
    support_name_to_column = {name: position for position, name in enumerate(support_names)}
    settlement_ratios = np.asarray([
        ratio[index, support_name_to_column[name]]
        if ratio.ndim == 2 and ratio.shape[0] > index and name in support_name_to_column
        and support_name_to_column[name] < ratio.shape[1] else 0.0
        for name in PIER_SETTLEMENT_NAMES
    ], dtype=np.float32)
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
