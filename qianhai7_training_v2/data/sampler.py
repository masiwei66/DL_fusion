"""Epoch-aware sampling for the multi-realization observation dataset."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class RotatingRealizationSampler(Sampler[int]):
    """Sample window/observation-realization pairs for one epoch.

    `QianhaiWindowDataset` stores its windows in a flat index space of
    ``[realization, window]``. Worker processes hold a *copy* of the dataset, so
    no dataset attribute set here in the main process can reach them - but the
    index stream is produced in the main process and re-iterated every epoch
    (the same reason ``DistributedSampler.set_epoch`` works). Folding the
    realization into the index is therefore the only channel that survives
    ``persistent_workers=True``.

    ``strategy="balanced_random"`` is the training default. It visits every
    window once per epoch and assigns realizations in a shuffled, exactly
    balanced way. This keeps an epoch the same size as the old rotating sampler
    while preventing the model from seeing one complete epoch through a single
    point/noise realization.

    ``strategy="rotate"`` preserves the old one-realization-per-epoch behavior
    for controlled comparisons. ``strategy="all"`` visits every window for
    every realization and is used for deterministic multi-realization
    validation/test evaluation.
    """

    _STRATEGIES = frozenset(("balanced_random", "rotate", "all"))

    def __init__(
        self,
        window_count: int,
        realizations: int,
        seed: int = 42,
        shuffle: bool = True,
        strategy: str = "balanced_random",
    ):
        if int(realizations) < 1:
            raise ValueError("RotatingRealizationSampler needs realizations >= 1")
        if int(window_count) < 0:
            raise ValueError("RotatingRealizationSampler needs window_count >= 0")
        if strategy not in self._STRATEGIES:
            choices = ", ".join(sorted(self._STRATEGIES))
            raise ValueError(f"Unknown realization sampling strategy {strategy!r}; choose one of: {choices}")
        self.window_count = int(window_count)
        self.realizations = int(realizations)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.strategy = strategy
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.strategy == "all":
            return self.window_count * self.realizations
        return self.window_count

    def __iter__(self) -> Iterator[int]:
        order = np.arange(self.window_count)
        rng = np.random.default_rng([self.seed, self.epoch])
        if self.shuffle:
            rng.shuffle(order)

        if self.strategy == "rotate":
            offset = (self.epoch % self.realizations) * self.window_count
            return iter((int(position) + offset for position in order))

        if self.strategy == "all":
            def all_indices() -> Iterator[int]:
                for realization in range(self.realizations):
                    for position in order:
                        yield realization * self.window_count + int(position)

            return all_indices()

        # Assign each realization the same number of windows, with at most one
        # extra window when the count is not divisible by the realization count.
        realization_ids = np.arange(self.window_count, dtype=np.int64) % self.realizations
        if self.shuffle:
            rng.shuffle(realization_ids)
        return iter(
            int(realization_ids[position_index]) * self.window_count + int(position)
            for position_index, position in enumerate(order)
        )
