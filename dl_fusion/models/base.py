from __future__ import annotations

from torch import nn

from ..contracts import WindowBatch


def validate_input_mode(batch: WindowBatch, input_mode: str) -> None:
    """Reject a data adapter/backbone mode mismatch at the model boundary."""
    if batch.input_mode is not None and batch.input_mode != input_mode:
        raise ValueError(
            f"batch input_mode={batch.input_mode!r} does not match backbone "
            f"input_mode={input_mode!r}"
        )


class StudentModel(nn.Module):
    """Backbone/head interface shared by Qianhai and KW51.

    ``forward_representation`` is the small seam used by task-specific
    runners.  ``forward_batch`` remains the compatibility entry point for
    single-dataset heads such as Qianhai's four-task classifier.
    """

    def forward_representation(self, batch: WindowBatch):
        raise NotImplementedError

    def forward_batch(self, batch: WindowBatch):
        """Compatibility entry point for tasks without a dedicated head."""
        return {"representation": self.forward_representation(batch)}
