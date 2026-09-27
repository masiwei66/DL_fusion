from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from ..contracts import WindowBatch


class StudentModel(nn.Module, ABC):
    @abstractmethod
    def forward_batch(self, batch: WindowBatch):
        raise NotImplementedError
