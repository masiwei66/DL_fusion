"""Shared classification metrics and reporting metadata.

The private Qianhai and public KW51 pipelines have different task semantics,
but their categorical summary metrics should use one implementation.  Keeping
the implementation here prevents small changes in absent-class or probability
handling from making the two reports drift apart.
"""
from __future__ import annotations

from typing import Any

import numpy as np


CLASSIFICATION_METRIC_PROTOCOL = "macro_p_r_f1_observed_ovr_auc_v1"


def _observed_macro_auc(y_true, probability, labels) -> float | None:
    """Return macro OVR AUC over classes observed in ``y_true``.

    Classes absent from the evaluation sample are excluded.  The returned
    value is always JSON-safe: invalid shapes, degenerate labels, failed
    sklearn calculations, and non-finite results become ``None``.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    index_of = {int(label): index for index, label in enumerate(labels)}
    if any(int(value) not in index_of for value in y_true):
        return None
    if probability.ndim != 2 or probability.shape[1] != len(index_of):
        return None
    target = np.fromiter(
        (index_of[int(value)] for value in y_true),
        dtype=np.int64,
        count=len(y_true),
    )
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
            if not np.all(totals > 0.0):
                return None
            value = roc_auc_score(
                target,
                scores / totals,
                labels=present,
                multi_class="ovr",
                average="macro",
            )
    except (ValueError, IndexError):
        return None
    return float(value) if np.isfinite(value) else None


def multiclass_cross_entropy(y_true, y_prob, class_values) -> float | None:
    """Compute mean multiclass NLL using explicit class values."""
    y_true = np.asarray(y_true).reshape(-1)
    labels = np.asarray(class_values).reshape(-1)
    probability = np.asarray(y_prob, dtype=np.float64)
    indices = {int(label): index for index, label in enumerate(labels)}
    if probability.ndim != 2 or probability.shape[1] != len(labels):
        return None
    if any(int(label) not in indices for label in y_true):
        return None
    if y_true.size == 0:
        return None
    target = np.asarray([indices[int(label)] for label in y_true], dtype=np.int64)
    value = float(
        -np.log(np.clip(probability[np.arange(len(target)), target], 1e-12, 1.0)).mean()
    )
    return value if np.isfinite(value) else None


def classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
    class_values=None,
) -> dict[str, Any]:
    """Return the shared categorical metric set.

    ``class_values`` fixes the class order and ensures absent classes still
    contribute zero to macro precision/recall/F1 and to the confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if class_values is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(class_values).reshape(-1)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_values": labels.tolist(),
        "f1_per_class": f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        "num_samples": int(y_true.size),
    }
    if y_prob is None:
        result["roc_auc"] = None
        result["nll"] = None
    else:
        probability = np.asarray(y_prob)
        result["roc_auc"] = _observed_macro_auc(y_true, probability, labels)
        result["nll"] = multiclass_cross_entropy(y_true, probability, labels)
    return result


def classification_metric_metadata(
    *,
    evaluation_unit: str,
    num_classes: int,
    aggregation: str,
) -> dict[str, Any]:
    """Build the common report envelope for a categorical evaluation."""
    return {
        "metric_protocol": CLASSIFICATION_METRIC_PROTOCOL,
        # Keep the legacy KW51 key while new consumers use metric_protocol.
        "classification_metric_protocol": CLASSIFICATION_METRIC_PROTOCOL,
        "evaluation_unit": evaluation_unit,
        "num_classes": int(num_classes),
        "metric_aggregation": aggregation,
    }
