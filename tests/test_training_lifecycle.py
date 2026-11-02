from __future__ import annotations

import torch
from torch import nn

from dl_fusion.training.lifecycle import (
    BestMetricTracker,
    build_adamw,
    build_scheduler,
    materialize_lazy_parameters,
    restore_model_checkpoint,
    save_model_checkpoint,
    step_scheduler,
)


def test_best_metric_tracker_resets_patience_on_improvement() -> None:
    tracker = BestMetricTracker()

    assert tracker.update(1, 0.5)
    assert not tracker.update(2, 0.5)
    assert tracker.stale_epochs == 1
    assert tracker.should_stop(1)
    assert tracker.update(3, 0.6)
    assert tracker.best_epoch == 3
    assert tracker.stale_epochs == 0
    assert not tracker.should_stop(1)


def test_lazy_materialization_preserves_training_state_and_batch_norm_stats() -> None:
    model = nn.Sequential(nn.BatchNorm1d(2), nn.LazyLinear(1))
    model.train()
    running_mean = model[0].running_mean.detach().clone()
    batch = torch.ones(4, 2)

    materialized = materialize_lazy_parameters(model, batch, model)

    assert materialized
    assert model.training
    assert torch.equal(model[0].running_mean, running_mean)
    assert not isinstance(model[1].weight, torch.nn.parameter.UninitializedParameter)


def test_checkpoint_helpers_round_trip_model_and_task_metadata(tmp_path) -> None:
    model = nn.Linear(2, 1)
    optimizer = build_adamw(model, learning_rate=1e-2, weight_decay=1e-3)
    path = tmp_path / "models" / "best.pt"
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}

    save_model_checkpoint(
        path,
        model,
        epoch=4,
        metadata={"task": "example"},
        optimizer=optimizer,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    checkpoint = restore_model_checkpoint(path, model, device="cpu")

    assert checkpoint["epoch"] == 4
    assert checkpoint["task"] == "example"
    assert "optimizer_state_dict" in checkpoint
    for key, value in expected.items():
        assert torch.equal(model.state_dict()[key], value)


def test_scheduler_factory_supports_task_configurations() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_adamw(model, learning_rate=1e-2, weight_decay=0.0)
    plateau = build_scheduler(
        optimizer,
        name="plateau",
        epochs=4,
        factor=0.5,
        patience=1,
        min_learning_rate=1e-5,
    )
    cosine = build_scheduler(
        optimizer,
        name="cosine",
        epochs=4,
        factor=0.5,
        patience=1,
        min_learning_rate=1e-5,
    )

    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    step_scheduler(plateau, 0.5)
    step_scheduler(cosine, 0.5)
    assert isinstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert isinstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
