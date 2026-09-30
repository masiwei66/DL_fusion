from __future__ import annotations

from .mlp import PointSetMLP
from .temporal import GRUModel, LSTMModel, ResNet1DModel, ResNet50_1DModel, TemporalTransformerModel
from .xgboost_model import XGBoostModel


MODEL_REGISTRY = {
    "mlp": {"implemented": True, "family": "point_set"},
    "xgboost": {"implemented": True, "family": "tabular"},
    "resnet1d": {"implemented": True, "family": "temporal"},
    "resnet50_1d": {"implemented": True, "family": "temporal"},
    "lstm": {"implemented": True, "family": "temporal"},
    "gru": {"implemented": True, "family": "temporal"},
    "transformer": {"implemented": True, "family": "temporal"},
    "gnn_lstm": {"implemented": False, "family": "spatiotemporal"},
}


def available_model_names() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def build_model(name: str, **kwargs):
    if name == "mlp":
        return PointSetMLP(**kwargs)
    if name == "resnet1d":
        return ResNet1DModel(**kwargs)
    if name == "resnet50_1d":
        return ResNet50_1DModel(**kwargs)
    if name == "lstm":
        return LSTMModel(**kwargs)
    if name == "gru":
        return GRUModel(**kwargs)
    if name == "transformer":
        return TemporalTransformerModel(**kwargs)
    if name == "xgboost":
        return XGBoostModel(**kwargs)
    if name in MODEL_REGISTRY and not MODEL_REGISTRY[name]["implemented"]:
        raise NotImplementedError(
            f"Model {name!r} is registered but not implemented yet. "
            "Keep the config name, then add its adapter/model implementation."
        )
    choices = ", ".join(available_model_names())
    raise ValueError(f"Unknown model {name!r}; choose one of: {choices}")
