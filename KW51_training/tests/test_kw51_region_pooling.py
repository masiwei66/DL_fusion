import pytest
import torch

from KW51_training.contracts import WindowBatch
from dl_fusion.models.temporal import LSTMModel


def _batch(batch_size: int = 2, window: int = 3, points: int = 4) -> WindowBatch:
    return WindowBatch(
        values=torch.randn(batch_size, window, points),
        valid_mask=torch.ones(batch_size, window, points, dtype=torch.bool),
        quality=torch.ones(batch_size, window, points),
        point_features=torch.randn(batch_size, points, 2),
        time_features=torch.zeros(batch_size, window, 2),
        point_present_mask=torch.ones(batch_size, points, dtype=torch.bool),
        mapping_distance=torch.zeros(batch_size, points),
        mapping_valid_mask=torch.ones(batch_size, points, dtype=torch.bool),
        temperature_valid_mask=torch.ones(batch_size, window, 1, dtype=torch.bool),
        teacher_features={},
        targets={},
        metadata=[],
    )


def test_flatten_region_pooling_keeps_the_region_axis() -> None:
    model = LSTMModel(
        input_dim=6,
        input_mode="region",
        region_pooling="flatten",
        hidden_size=8,
        num_layers=1,
    )

    representation = model.forward_representation(_batch())

    assert representation.shape == (2, 4 * 8)


def test_mean_region_pooling_is_unchanged() -> None:
    model = LSTMModel(
        input_dim=6,
        input_mode="region",
        hidden_size=8,
        num_layers=1,
    )

    representation = model.forward_representation(_batch())
    outputs = model.forward_batch(_batch())

    assert representation.shape == (2, 8)
    assert outputs["representation"].shape == (2, 8)


def test_flatten_pooling_requires_region_mode() -> None:
    with pytest.raises(ValueError, match="input_mode=region"):
        LSTMModel(input_dim=6, input_mode="node", region_pooling="flatten")
