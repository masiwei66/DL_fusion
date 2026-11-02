from __future__ import annotations

import math

import pytest
import torch

from KW51_training.contracts import WindowBatch
from KW51_training.models.fuzzy_region import FuzzyRegionModel
from KW51_training.models.fuzzy_region_summary import FuzzyRegionSummaryModel
from KW51_training.models.fuzzy_region_mlp import FuzzyRegionMLPModel
from KW51_training.training.fuzzy_losses import fuzzy_region_multitask_loss
from KW51_training.training.fuzzy_metrics import fuzzy_region_multitask_metrics


def _batch(*, point_mode: bool) -> WindowBatch:
    batch_size, time_count, region_count = 2, 4, 3
    point_count = 6 if point_mode else region_count
    values = torch.arange(
        batch_size * time_count * point_count, dtype=torch.float32
    ).reshape(batch_size, time_count, point_count) / 10.0
    region_index = torch.tensor([[0, 0, 1, 1, 2, 2]] * batch_size) if point_mode else None
    point_present = torch.ones((batch_size, point_count), dtype=torch.bool)
    region_present = torch.tensor([[True, True, False], [True, True, True]])
    valid = torch.ones_like(values, dtype=torch.bool)
    if point_mode:
        valid[0, :, 4:] = False
    else:
        valid[0, :, 2] = False
    target_valid = torch.tensor(
        [
            [[True] * time_count, [True] * time_count, [False] * time_count],
            [[True] * time_count, [True] * time_count, [True] * time_count],
        ]
    )
    return WindowBatch(
        values=values,
        valid_mask=valid,
        quality=valid.float(),
        point_features=torch.rand(batch_size, point_count, 2),
        time_features=torch.rand(batch_size, time_count, 2),
        point_present_mask=point_present,
        mapping_distance=torch.zeros(batch_size, point_count),
        mapping_valid_mask=point_present,
        temperature_valid_mask=torch.ones(batch_size, time_count, 1, dtype=torch.bool),
        teacher_features={},
        targets={
            "ad_group_id": torch.tensor([0, 1]),
            "region_response_target": torch.zeros(batch_size, region_count, time_count),
            "region_response_valid_mask": target_valid,
            "region_area_coverage": target_valid.float(),
        },
        metadata=[
            {"structural_state_id": "s1", "point_count": 40},
            {"structural_state_id": "s2", "point_count": 160},
        ],
        region_index=region_index,
        region_present=region_present,
        region_coverage=region_present.float(),
        bag_features=torch.zeros(batch_size, 8),
        bag_feature_mask=torch.ones(batch_size, 1, dtype=torch.bool),
    )


def _model_params(input_mode: str = "node") -> dict[str, object]:
    return {
        "input_dim": 6,
        "input_mode": input_mode,
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
        **({"region_pooling": "flatten"} if input_mode == "region" else {}),
    }


@pytest.mark.parametrize("summary", [False, True])
def test_region_models_keep_response_region_and_time_axes(summary: bool) -> None:
    batch = _batch(point_mode=not summary)
    if summary:
        model = FuzzyRegionSummaryModel(
            "lstm", _model_params("region"), region_count=3, group_count=2, head_hidden=8,
            dropout=0.0, bag_feature_dropout=0.0,
        )
    else:
        model = FuzzyRegionModel(
            "lstm", _model_params(), region_count=3, group_count=2,
            head_hidden=8, dropout=0.0, bag_feature_dropout=0.0,
        )
    output = model(batch)
    assert output["region_response_mean"].shape == (2, 3, 4)
    assert output["region_response_scale"].shape == (2, 3, 4)
    assert torch.all(output["region_response_scale"] > 0)
    assert output["state_logits"].shape == (2, 2)
    assert output["group_logits"] is output["state_logits"]
    assert output["region_state_logits"].shape == (2, 3, 2)


def test_region_mean_mlp_baseline_keeps_region_time_axes() -> None:
    model = FuzzyRegionMLPModel(region_count=3, group_count=2, hidden_dim=8, dropout=0.0)
    output = model(_batch(point_mode=False))
    assert output["region_response_mean"].shape == (2, 3, 4)
    assert output["region_response_scale"].shape == (2, 3, 4)
    assert output["region_state_logits"].shape == (2, 3, 2)


def test_dense_region_loss_ignores_invalid_targets_and_trains_both_tasks() -> None:
    batch = _batch(point_mode=False)
    outputs = {
        "region_response_mean": torch.zeros(2, 3, 4, requires_grad=True),
        "region_response_scale": torch.ones(2, 3, 4, requires_grad=True),
        "state_logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True),
        "group_logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True),
        "region_state_logits": torch.zeros(2, 3, 2, requires_grad=True),
    }
    batch.targets["region_response_target"][0, 2] = 1.0e6
    loss, parts = fuzzy_region_multitask_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert parts["l1_nll"] == pytest.approx(0.5 * math.log(2.0 * math.pi), rel=1e-5)
    assert parts["l2_global_nll"] < 0.03
    loss.backward()
    assert outputs["region_response_mean"].grad is not None
    assert outputs["state_logits"].grad is not None


def test_multitask_metrics_include_required_strata() -> None:
    batch = _batch(point_mode=False)
    output = {
        "region_response_mean": torch.zeros(2, 3, 4),
        "region_response_scale": torch.ones(2, 3, 4),
        "state_logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        "region_state_logits": torch.tensor(
            [[[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]],
             [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]]]
        ),
    }
    metrics = fuzzy_region_multitask_metrics(output, batch)
    for task in ("l1", "l2"):
        assert set(metrics[task]) == {
            "overall", "per_state", "per_region", "per_density"
        }
    assert metrics["l1"]["overall"]["mae"] == pytest.approx(0.0)
    assert metrics["l1"]["overall"]["coverage_50"] == pytest.approx(1.0)
    assert metrics["l1"]["overall"]["coverage_90"] == pytest.approx(1.0)
    assert metrics["l2"]["overall"]["top1"] == pytest.approx(1.0)
    assert metrics["l2"]["overall"]["mrr"] == pytest.approx(1.0)
    assert metrics["l2"]["overall"]["nll"] < 0.03
    assert "ece" in metrics["l2"]["overall"]
    assert "s2" in metrics["l1"]["per_state"]
    assert "40" in metrics["l2"]["per_density"]
