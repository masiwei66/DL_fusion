"""Shared optimization and checkpoint helpers for task-specific runners."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter


@dataclass
class BestMetricTracker:
    """Track a maximized validation metric and patience-based early stopping."""

    best_score: float = float("-inf")
    best_epoch: int = 0
    stale_epochs: int = 0

    def update(self, epoch: int, score: float) -> bool:
        if epoch < 1:
            raise ValueError("epoch must be one-based and >= 1")
        if not math.isfinite(score):
            raise ValueError(f"validation score must be finite, got {score!r}")
        if score > self.best_score:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
            return True
        self.stale_epochs += 1
        return False

    def should_stop(self, patience: int) -> bool:
        return patience > 0 and self.stale_epochs >= patience


def build_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str | None,
    epochs: int,
    factor: float,
    patience: int,
    min_learning_rate: float,
    threshold: float | None = None,
) -> (
    torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None
):
    if name in (None, "none"):
        return None
    if name == "plateau":
        options: dict[str, Any] = {
            "mode": "max",
            "factor": factor,
            "patience": patience,
            "min_lr": min_learning_rate,
        }
        if threshold is not None:
            options["threshold"] = threshold
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **options)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=min_learning_rate
        )
    raise ValueError(f"unsupported learning-rate scheduler: {name!r}")


def step_scheduler(
    scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None
    ),
    validation_score: float,
) -> None:
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_score)
    elif scheduler is not None:
        scheduler.step()


def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    epoch: int,
    metadata: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save a common model payload while leaving task metadata to its runner."""
    payload = dict(metadata)
    payload["epoch"] = int(epoch)
    payload["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def restore_model_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def materialize_lazy_parameters(
    model: nn.Module,
    batch: Any | Callable[[], Any],
    forward: Callable[[Any], Any],
) -> bool:
    """Initialize lazy layers from a real batch without changing train-time state."""
    has_uninitialized = any(
        isinstance(parameter, UninitializedParameter)
        for parameter in model.parameters()
    )
    if not has_uninitialized:
        return False
    batch = batch() if callable(batch) else batch
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            forward(batch)
    finally:
        model.train(was_training)
    return True
