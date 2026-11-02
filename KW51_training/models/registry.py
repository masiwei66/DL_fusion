"""KW51 registry delegates neural architecture construction to dl_fusion."""
from __future__ import annotations

from dl_fusion.models.registry import BACKBONE_REGISTRY, build_backbone

MODEL_REGISTRY = {name: {"implemented": True} for name in BACKBONE_REGISTRY}


def available_model_names() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def build_model(name: str, **kwargs):
    return build_backbone(name, **kwargs)
