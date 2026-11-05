from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from dl_fusion.metrics import (
    CLASSIFICATION_METRIC_PROTOCOL,
    _observed_macro_auc,
    classification_metrics as shared_classification_metrics,
)
from KW51_training.training.fuzzy_metrics import group_metrics
from KW51_training.training.fuzzy_region_runner import (
    _training_monitor_policy,
    _write_classification_plot,
)
from qianhai7_training.training.report import classification_metrics


def test_ad_group_metrics_match_qianhai_macro_classification_metrics() -> None:
    probability = torch.tensor(
        [
            [0.70, 0.10, 0.10, 0.05, 0.05],
            [0.15, 0.50, 0.15, 0.10, 0.10],
            [0.10, 0.15, 0.20, 0.45, 0.10],
            [0.05, 0.15, 0.20, 0.50, 0.10],
            [0.05, 0.10, 0.15, 0.20, 0.50],
        ],
        dtype=torch.float32,
    )
    y_true = np.arange(5, dtype=np.int64)
    logits = probability.log()
    result = group_metrics(logits, torch.from_numpy(y_true), [0.780, 0.805, 0.830, 0.855, 0.880])
    model_probability = torch.softmax(logits, dim=-1).numpy()
    y_pred = model_probability.argmax(axis=1)
    expected = classification_metrics(
        y_true,
        y_pred,
        model_probability,
        class_values=np.arange(5),
    )

    assert result["accuracy"] == pytest.approx(expected["accuracy"])
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["precision_macro"] == expected["precision_macro"]
    assert result["recall_macro"] == expected["recall_macro"]
    assert result["f1_macro"] == expected["f1_macro"]
    assert result["roc_auc"] == expected["roc_auc"]
    assert result["nll"] == pytest.approx(expected["nll"])
    assert result["confusion_matrix"] == expected["confusion_matrix"]
    assert result["f1_per_class"] == f1_score(
        y_true, y_pred, labels=np.arange(5), average=None, zero_division=0
    ).tolist()


def test_macro_auc_is_none_when_test_has_only_one_observed_class() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = group_metrics(logits, torch.tensor([0, 0]), [0.780, 0.830, 0.880])

    assert result["roc_auc"] is None
    assert result["f1_macro"] >= 0.0


def test_shared_metric_entrypoint_is_the_private_report_implementation() -> None:
    y_true = np.array([0, 1, 2, 0], dtype=np.int64)
    probability = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.7, 0.2, 0.1]],
        dtype=np.float64,
    )
    y_pred = probability.argmax(axis=1)
    shared = shared_classification_metrics(y_true, y_pred, probability, class_values=[0, 1, 2])
    private = classification_metrics(y_true, y_pred, probability, class_values=[0, 1, 2])

    assert shared == private
    assert CLASSIFICATION_METRIC_PROTOCOL == "macro_p_r_f1_observed_ovr_auc_v1"


def test_shared_auc_rejects_non_finite_result() -> None:
    value = _observed_macro_auc(
        np.array([0, 1, 0, 1]),
        np.array([[0.8, np.nan], [0.3, 0.7], [0.8, 0.2], [0.4, 0.6]]),
        np.array([0, 1]),
    )
    assert value is None


def test_f1_checkpoint_selection_uses_smooth_nll_training_monitor() -> None:
    assert _training_monitor_policy("ad_rel_classification", "f1_macro") == (
        "nll",
        "min",
    )
    assert _training_monitor_policy("candidate_compatibility", "mrr") == (
        "mrr",
        "max",
    )


def test_classification_history_plot_uses_epoch_metrics(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    path = tmp_path / "training_history.png"
    history = [
        {
            "epoch": 1,
            "train_group_ce": 1.6,
            "validation_state_nll": 1.5,
            "validation_state_f1_macro": 0.2,
            "validation_state_roc_auc": 0.6,
        },
        {
            "epoch": 2,
            "train_group_ce": 1.4,
            "validation_state_nll": 1.3,
            "validation_state_f1_macro": 0.4,
            "validation_state_roc_auc": 0.7,
        },
    ]

    assert _write_classification_plot(history, path)
    assert path.is_file() and path.stat().st_size > 0
