from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from dl_fusion.metrics import (
    CLASSIFICATION_METRIC_PROTOCOL,
    _observed_macro_auc,
    classification_metric_metadata,
    classification_metrics,
    multiclass_cross_entropy,
)

__all__ = [
    "CLASSIFICATION_METRIC_PROTOCOL",
    "Stopwatch",
    "_observed_macro_auc",
    "classification_metric_metadata",
    "classification_metrics",
    "count_parameters",
    "multiclass_cross_entropy",
    "write_report",
]


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class Stopwatch:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started
