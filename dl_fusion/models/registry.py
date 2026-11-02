"""Factory for task-independent student backbones."""
from __future__ import annotations

from .mlp import PointSetMLP
from .spatiotemporal import GCNLSTMModel
from .temporal import GRUModel, LSTMModel, ResNet1DModel, ResNet50_1DModel, TemporalTransformerModel


BACKBONE_REGISTRY = {
    "mlp": PointSetMLP,
    "resnet1d": ResNet1DModel,
    "resnet50_1d": ResNet50_1DModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "transformer": TemporalTransformerModel,
    "gnn_lstm": GCNLSTMModel,
}


def build_backbone(name: str, **kwargs):
    try:
        backbone = BACKBONE_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(BACKBONE_REGISTRY)
        raise ValueError(f"Unknown backbone {name!r}; choose one of: {choices}") from exc
    return backbone(**kwargs)
