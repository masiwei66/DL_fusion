"""Shared model contracts, backbones and construction."""

from .base import StudentModel
from .registry import BACKBONE_REGISTRY, build_backbone

__all__ = ["BACKBONE_REGISTRY", "StudentModel", "build_backbone"]
