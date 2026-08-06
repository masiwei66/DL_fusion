"""Engineering safety rules for multi-condition bridge state assessment.

The neural network predicts response-derived condition variables.  These rules
turn material stiffness loss and support settlement into region and global
safety labels that can be used both during dataset generation and training.
All lengths are in millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


CANDIDATE_MATERIAL_IDS = [8, 9, 19, 20, 24, 41]
SUPPORT_SETTLEMENT_NODES = [221, 513, 395, 687]
SUPPORT_SETTLEMENT_DOF = "UZ"
SUPPORT_SETTLEMENT_MIN_MM = 0.0
SUPPORT_SETTLEMENT_MAX_MM = 8.0

RISK_LEVELS = ["safe", "warning", "risk", "danger"]
SAFE = 0
WARNING = 1
RISK = 2
DANGER = 3


@dataclass(frozen=True)
class SafetyThresholds:
    """Initial engineering thresholds for generated bridge states.

    Material scaling factor uses lower-is-worse convention:
    1.0 means undamaged, 0.8 means 20 percent stiffness loss.
    Settlement uses positive downward magnitude in mm.
    """

    material_warning: float = 0.90
    material_risk: float = 0.80
    material_danger: float = 0.70
    support_warning_mm: float = 2.0
    support_risk_mm: float = 5.0
    support_danger_mm: float = 7.0


DEFAULT_THRESHOLDS = SafetyThresholds()


# Initial region design.  The names are intentionally engineering-facing rather
# than model-facing; adjust these lists after confirming the bridge partition.
REGION_DEFINITIONS = {
    "left_support1_zone": {
        "material_ids": [],
        "support_nodes": [221],
        "description": "Left support1",
    },
    "left_support2_zone": {
        "material_ids": [],
        "support_nodes": [513],
        "description": "Left support2",
    },
    "right_support2_zone": {
        "material_ids": [],
        "support_nodes": [395],
        "description": "Right support2",
    },
    "right_zone": {
        "material_ids": [9,20],
        "support_nodes": [],
        "description": "Right structure",
    },
    "mid_zone": {
        "material_ids": [24,41],
        "support_nodes": [],
        "description": "Mid structure",
    },
    "left_zone": {
        "material_ids": [8,19],
        "support_nodes": [],
        "description": "Left structure",
    },
    "right_support1_zone": {
        "material_ids": [],
        "support_nodes": [687],
        "description": "Right support1",
    }
}


def risk_level_name(level: int) -> str:
    return RISK_LEVELS[int(np.clip(level, 0, len(RISK_LEVELS) - 1))]


def material_risk_level(scaling_factor: float, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS) -> int:
    if scaling_factor <= thresholds.material_danger:
        return DANGER
    if scaling_factor <= thresholds.material_risk:
        return RISK
    if scaling_factor <= thresholds.material_warning:
        return WARNING
    return SAFE


def support_risk_level(settlement_mm: float, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS) -> int:
    value = abs(float(settlement_mm))
    if value >= thresholds.support_danger_mm:
        return DANGER
    if value >= thresholds.support_risk_mm:
        return RISK
    if value >= thresholds.support_warning_mm:
        return WARNING
    return SAFE


def material_binary_label(scaling_factor: float, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS) -> int:
    return int(material_risk_level(scaling_factor, thresholds) >= WARNING)


def support_binary_label(settlement_mm: float, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS) -> int:
    return int(support_risk_level(settlement_mm, thresholds) >= WARNING)


def _as_mapping(ids: Sequence[int], values: Sequence[float]) -> Dict[int, float]:
    return {int(i): float(v) for i, v in zip(ids, values)}


def make_support_settlement_case(
    rng=None,
    min_mm: float = SUPPORT_SETTLEMENT_MIN_MM,
    max_mm: float = SUPPORT_SETTLEMENT_MAX_MM,
) -> Dict:
    """Sample one settlement condition over 1-4 support nodes.

    Returned values are positive settlement magnitudes.  `load_step` uses the
    negative sign for UZ because downward settlement is normally negative in the
    bridge model coordinate system.
    """

    rng = rng or np.random.default_rng()
    n_active = int(rng.integers(1, len(SUPPORT_SETTLEMENT_NODES) + 1))
    active_nodes = sorted(rng.choice(SUPPORT_SETTLEMENT_NODES, size=n_active, replace=False).tolist())
    values = {node: float(rng.uniform(min_mm, max_mm)) for node in active_nodes}
    vector = [values.get(node, 0.0) for node in SUPPORT_SETTLEMENT_NODES]
    load_step = {
        node: (SUPPORT_SETTLEMENT_DOF, -values.get(node, 0.0))
        for node in SUPPORT_SETTLEMENT_NODES
    }

    return {
        "node_ids": SUPPORT_SETTLEMENT_NODES,
        "direction": SUPPORT_SETTLEMENT_DOF,
        "values_mm": vector,
        "active": [int(node in active_nodes) for node in SUPPORT_SETTLEMENT_NODES],
        "active_node_ids": active_nodes,
        "load_step": load_step,
    }


def build_safety_labels(
    material_ids: Sequence[int],
    material_scaling_factors: Sequence[float],
    support_settlement_mm: Sequence[float] | Mapping[int, float] | None,
    thresholds: SafetyThresholds = DEFAULT_THRESHOLDS,
    region_definitions: Mapping[str, Mapping[str, Iterable[int]]] = REGION_DEFINITIONS,
) -> Dict:
    """Create material, support, region, and global safety labels."""

    material_map = _as_mapping(material_ids, material_scaling_factors)
    if support_settlement_mm is None:
        support_map = {node: 0.0 for node in SUPPORT_SETTLEMENT_NODES}
    elif isinstance(support_settlement_mm, Mapping):
        support_map = {node: float(support_settlement_mm.get(node, 0.0)) for node in SUPPORT_SETTLEMENT_NODES}
    else:
        support_map = _as_mapping(SUPPORT_SETTLEMENT_NODES, support_settlement_mm)

    material_levels = {
        mid: material_risk_level(material_map.get(mid, 1.0), thresholds)
        for mid in CANDIDATE_MATERIAL_IDS
    }
    support_levels = {
        node: support_risk_level(support_map.get(node, 0.0), thresholds)
        for node in SUPPORT_SETTLEMENT_NODES
    }

    regions = {}
    for name, definition in region_definitions.items():
        mat_ids = [int(v) for v in definition.get("material_ids", [])]
        support_nodes = [int(v) for v in definition.get("support_nodes", [])]
        mat_level = max([material_levels.get(mid, SAFE) for mid in mat_ids] or [SAFE])
        sup_level = max([support_levels.get(node, SAFE) for node in support_nodes] or [SAFE])
        level = max(mat_level, sup_level)
        regions[name] = {
            "level": int(level),
            "state": risk_level_name(level),
            "material_ids": mat_ids,
            "support_nodes": support_nodes,
        }

    global_level = max([item["level"] for item in regions.values()] or [SAFE])
    return {
        "thresholds": thresholds.__dict__,
        "risk_levels": RISK_LEVELS,
        "material_labels": [material_binary_label(material_map.get(mid, 1.0), thresholds) for mid in CANDIDATE_MATERIAL_IDS],
        "material_risk_levels": [material_levels[mid] for mid in CANDIDATE_MATERIAL_IDS],
        "support_labels": [support_binary_label(support_map.get(node, 0.0), thresholds) for node in SUPPORT_SETTLEMENT_NODES],
        "support_risk_levels": [support_levels[node] for node in SUPPORT_SETTLEMENT_NODES],
        "region_names": list(region_definitions.keys()),
        "region_risk_levels": [regions[name]["level"] for name in region_definitions],
        "regions": regions,
        "global_level": int(global_level),
        "global_state": risk_level_name(global_level),
    }
