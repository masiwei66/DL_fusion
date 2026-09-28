"""Epoch-aware sampling for the multi-realization observation dataset."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class RotatingRealizationSampler(Sampler[int]):
    """One pass over every window per epoch, rotating the observation realization.

    `QianhaiWindowDataset` stores its windows in a flat index space of
    ``[realization, window]``. Worker processes hold a *copy* of the dataset, so
    no dataset attribute set here in the main process can reach them - but the
    index stream is produced in the main process and re-iterated every epoch
    (the same reason ``DistributedSampler.set_epoch`` works). Folding the
    realization into the index is therefore the only channel that survives
    ``persistent_workers=True``.
    """

    def __init__(self, window_count: int, realizations: int, seed: int = 42, shuffle: bool = True):
        if int(realizations) < 1:
            raise ValueError("RotatingRealizationSampler needs realizations >= 1")
        self.window_count = int(window_count)
        self.realizations = int(realizations)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # DataLoader falls back to len(dataset) (windows * realizations) without this.
        return self.window_count

    def __iter__(self) -> Iterator[int]:
        order = np.arange(self.window_count)
        if self.shuffle:
            np.random.default_rng([self.seed, self.epoch]).shuffle(order)
        offset = (self.epoch % self.realizations) * self.window_count
        return iter((int(position) + offset for position in order))
