from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def _observed_macro_auc(y_true, probability, labels) -> float | None:
    """Macro one-vs-rest AUC over the classes that actually occur in ``y_true``.

    Absent classes are dropped instead of being handed to sklearn: sklearn warns
    once per missing class and then returns NaN, which is not valid JSON. The
    remaining columns are renormalized because multiclass ROC-AUC requires the
    scores to sum to one; when every class is present this is a no-op.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    index_of = {int(label): index for index, label in enumerate(labels)}
    if any(int(value) not in index_of for value in y_true):
        return None
    if probability.ndim != 2 or probability.shape[1] != len(index_of):
        return None
    target = np.fromiter((index_of[int(value)] for value in y_true), dtype=np.int64, count=len(y_true))
    present = np.unique(target)
    if present.size < 2:
        return None
    try:
        if present.size == 2:
            positive = np.isin(target, present[1]).astype(np.int64)
            value = roc_auc_score(positive, probability[:, present[1]])
        else:
            scores = probability[:, present]
            totals = scores.sum(axis=1, keepdims=True)
            if not np.all(totals > 0):
                return None
            value = roc_auc_score(target, scores / totals, labels=present, multi_class="ovr", average="macro")
    except (ValueError, IndexError):
        return None
    return float(value) if np.isfinite(value) else None


def classification_metrics(y_true, y_pred, y_prob=None, class_values=None) -> dict[str, object]:
    """Return stable multiclass metrics, including absent-class-safe ROC-AUC."""
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    labels = np.asarray(class_values if class_values is not None else np.unique(np.concatenate([y_true, y_pred]))).reshape(-1)
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_values": labels.tolist(),
    }
    result["roc_auc"] = None if y_prob is None else _observed_macro_auc(y_true, np.asarray(y_prob), labels)
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
