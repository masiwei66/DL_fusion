from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def classification_metrics(y_true, y_pred, y_prob=None, class_values=None) -> dict[str, object]:
    """Return stable multiclass metrics, including absent-class-safe ROC-AUC."""
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    labels = np.asarray(class_values if class_values is not None else np.unique(np.concatenate([y_true, y_pred])))
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_values": labels.tolist(),
    }
    if y_prob is not None:
        probability = np.asarray(y_prob)
        try:
            if len(np.unique(y_true)) < 2:
                result["roc_auc"] = None
                return result
            if len(labels) == 2 and probability.shape[1] == 2:
                result["roc_auc"] = float(roc_auc_score(y_true, probability[:, 1]))
            elif len(labels) > 2 and probability.shape[1] == len(labels):
                result["roc_auc"] = float(roc_auc_score(y_true, probability, labels=labels, multi_class="ovr", average="macro"))
            else:
                result["roc_auc"] = None
        except (ValueError, IndexError):
            result["roc_auc"] = None
    else:
        result["roc_auc"] = None
    return result


def multiclass_cross_entropy(y_true, y_prob, class_values) -> float | None:
    import numpy as np
    labels = np.asarray(class_values)
    indices = {int(label): index for index, label in enumerate(labels)}
    if any(int(label) not in indices for label in y_true):
        return None
    target = np.asarray([indices[int(label)] for label in y_true], dtype=np.int64)
    probability = np.asarray(y_prob, dtype=np.float64)
    return float(-np.log(np.clip(probability[np.arange(len(target)), target], 1e-12, 1.0)).mean())


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
