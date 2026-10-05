from __future__ import annotations

import torch
import time
from typing import Any

from ..contracts import WindowBatch
from .losses import supervised_loss
from .progress import make_progress


def train_one_epoch(model, loader, optimizer, device: torch.device, *, desc: str | None = None,
                    progress: dict[str, Any] | None = None,
                    task_weights: dict[str, float] | None = None) -> float:
    model.train()
    total = 0.0
    count = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    with make_progress(desc, total=total_batches, options=progress) as tracker:
        for batch_index, batch in enumerate(loader, start=1):
            assert isinstance(batch, WindowBatch)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model.forward_batch(batch), batch, task_weights=task_weights)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * batch.batch_size
            count += batch.batch_size
            tracker.update(batch_index, average_loss=total / max(count, 1))
    return total / max(count, 1)


def train_one_epoch_timed(model, loader, optimizer, device: torch.device) -> tuple[float, float]:
    started = time.perf_counter()
    loss = train_one_epoch(model, loader, optimizer, device)
    return loss, time.perf_counter() - started
