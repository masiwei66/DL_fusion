from __future__ import annotations

import torch
import time

from ..contracts import WindowBatch
from .losses import supervised_loss


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        assert isinstance(batch, WindowBatch)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(model.forward_batch(batch), batch)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch.batch_size
        count += batch.batch_size
    return total / max(count, 1)


def train_one_epoch_timed(model, loader, optimizer, device: torch.device) -> tuple[float, float]:
    started = time.perf_counter()
    loss = train_one_epoch(model, loader, optimizer, device)
    return loss, time.perf_counter() - started
