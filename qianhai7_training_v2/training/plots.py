"""Small, self-contained plots for the current qianhai7 training reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_training_history(history: list[dict[str, Any]], save_path: str | Path) -> bool:
    """Plot total train/validation loss and per-task validation Macro F1."""
    if not history:
        return False
    plt = _load_matplotlib()
    if plt is None:
        return False
    epochs = [row["epoch"] for row in history]
    tasks = ["deck_material", "pier_settlement", "pier_material", "bearing_stiffness"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train loss")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sum of four task cross-entropy losses")
    axes[0].set_title("Training and validation loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for task in tasks:
        key = f"validation_{task}_macro_f1"
        axes[1].plot(epochs, [row.get(key, 0.0) for row in history], label=task)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation Macro F1 by task")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_final_evaluation(metrics: dict[str, Any], save_path: str | Path) -> bool:
    """Plot final confusion matrices and summary metrics for all tasks."""
    plt = _load_matplotlib()
    if plt is None:
        return False
    tasks = [task for task in ("deck_material", "pier_settlement", "pier_material", "bearing_stiffness") if task in metrics]
    if not tasks:
        return False
    columns = 3
    rows = (len(tasks) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), squeeze=False)
    for index, task in enumerate(tasks):
        metric = metrics[task]
        ax = axes[index // columns][index % columns]
        f1 = metric.get("macro_f1", float("nan"))
        values = metric.get("f1_per_class", [])
        if values:
            ax.bar(range(len(values)), values, color="#2f6f9f")
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(metric.get("class_names", range(len(values))))
            ax.set_xlabel("Severity class")
            ax.set_ylabel("F1")
        else:
            ax.text(0.5, 0.5, "No metric values", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title(f"{task}\nMacro F1={f1:.3f}")
    for index in range(len(tasks), rows * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle("Final test-set evaluation", y=1.02)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
