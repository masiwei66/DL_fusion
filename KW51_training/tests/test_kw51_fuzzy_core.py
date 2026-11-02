import math

import numpy as np
import pytest
import torch

from KW51_training.config import FuzzyConfig, load_experiment_config
from KW51_training.contracts import WindowBatch
from KW51_training.data.fuzzy_range import (
    DeckGeometry,
    FuzzyRangeSource,
    collate_fuzzy_queries,
)
from KW51_training.models.fuzzy_response import FuzzyResponseModel
from KW51_training.training.fuzzy_losses import (
    fuzzy_response_loss,
    gaussian_crps,
    gaussian_nll,
)
from KW51_training.training.fuzzy_metrics import distribution_metrics


def _geometry() -> DeckGeometry:
    xs = np.linspace(0.0, 100.0, 11)
    ys = np.linspace(0.0, 10.0, 5)
    xy = np.asarray([[x, y] for x in xs for y in ys], dtype=np.float32)
    s = xy[:, 0] / 100.0
    t = 2.0 * xy[:, 1] / 10.0 - 1.0
    return DeckGeometry(
        node_coords=np.c_[xy, np.zeros(len(xy), dtype=np.float32)],
        deck_indices=np.arange(len(xy)),
        deck_xy=xy,
        deck_s=s.astype(np.float32),
        deck_t=t.astype(np.float32),
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=10.0,
        los=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )


def _range_source(config: FuzzyConfig) -> FuzzyRangeSource:
    source = FuzzyRangeSource.__new__(FuzzyRangeSource)
    source.geometry = _geometry()
    source.config = config
    source.seed = 0
    source._cache = {}
    return source


def test_latent_range_target_distribution_shapes() -> None:
    config = FuzzyConfig(
        point_count_min=10,
        point_count_max=20,
        range_length_m_min=10.0,
        range_length_m_max=40.0,
        range_width_m_min=4.0,
        range_width_m_max=10.0,
        target_range_count=8,
        target_min_nodes=3,
    )
    source = _range_source(config)
    rng = np.random.default_rng(0)
    true_range = source._sample_true_range(rng)
    assert 0.0 <= true_range.center_s <= 1.0
    assert -1.0 <= true_range.center_t <= 1.0
    weights = source._range_weights([true_range])
    assert weights.shape == (1, len(source.geometry.deck_s))
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    los_values = rng.normal(size=(26, len(source.geometry.deck_s))).astype(np.float32)
    los_valid = np.ones_like(los_values, dtype=bool)
    target, ranges = source._targets(
        np.linspace(0.2, 0.8, 20),
        np.zeros(20, dtype=np.float32),
        los_values,
        los_valid,
        np.random.default_rng(1),
    )
    assert len(ranges) == 8
    assert target.samples.shape == (8, 26)
    assert target.mean.shape == (26,)
    assert target.std.shape == (26,)
    assert target.quantiles.shape == (26, 3)
    assert target.sample_valid.shape == (8, 26)
    assert target.time_valid.shape == (26,)


def _item(point_count: int, window: int = 4, target_count: int = 3) -> dict[str, object]:
    rng = np.random.default_rng(point_count)
    metadata = {
        "sample_id": f"sample-{point_count}",
        "structural_state_id": "1",
        "trajectory_id": "traj",
        "observation_realization_id": 0,
        "point_count": point_count,
        "target_mean_mm": 0.0,
        "target_std_mm": 1.0,
    }
    return {
        "values": rng.normal(size=(window, point_count)).astype(np.float32),
        "valid_mask": np.ones((window, point_count), dtype=bool),
        "quality": np.ones((window, point_count), dtype=np.float32),
        "point_coords": rng.normal(size=(point_count, 2)).astype(np.float32),
        "point_present": np.ones(point_count, dtype=bool),
        "mapping_distance": np.zeros(point_count, dtype=np.float32),
        "mapping_valid": np.ones(point_count, dtype=bool),
        "time_features": np.zeros((window, 2), dtype=np.float32),
        "bag_features": rng.normal(size=9).astype(np.float32),
        "target_samples": rng.normal(size=(target_count, window)).astype(np.float32),
        "target_mean": rng.normal(size=window).astype(np.float32),
        "target_std": np.ones(window, dtype=np.float32),
        "target_quantiles": rng.normal(size=(window, 3)).astype(np.float32),
        "target_sample_valid": np.ones((target_count, window), dtype=bool),
        "target_time_valid": np.ones(window, dtype=bool),
        "bag_mean_mm": rng.normal(size=window).astype(np.float32),
        "metadata": metadata,
    }


def test_collate_pads_variable_point_bags() -> None:
    batch = collate_fuzzy_queries([_item(3), _item(5)])
    assert batch.values.shape == (2, 4, 5)
    assert batch.point_present_mask.sum().item() == 8
    assert batch.bag_features.shape == (2, 9)
    assert batch.targets["response_samples"].shape == (2, 3, 4)
    assert batch.targets["response_mean"].shape == (2, 4)
    assert batch.targets["response_quantiles"].shape == (2, 4, 3)
    assert batch.targets["response_sample_valid"].shape == (2, 3, 4)
    assert batch.targets["response_valid"].shape == (2, 4)
    assert batch.targets["bag_mean_mm"].shape == (2, 4)
    assert batch.bag_feature_mask.all()


def _batch() -> WindowBatch:
    batch = collate_fuzzy_queries([_item(3), _item(5)])
    return batch


def test_fuzzy_model_forward_and_loss() -> None:
    model = FuzzyResponseModel(
        "lstm",
        {
            "input_dim": 6,
            "input_mode": "node",
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False,
        },
        4,
        head_hidden=8,
        dropout=0.0,
        bag_feature_dropout=0.0,
    )
    batch = _batch()
    outputs = model(batch)
    assert outputs["mean"].shape == (2, 4)
    assert outputs["logvar"].shape == (2, 4)
    loss, components = fuzzy_response_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert set(components) == {"nll", "crps", "mean_mse"}


def test_gaussian_losses_for_zero_residual_unit_variance() -> None:
    mean = torch.zeros(1, 4)
    logvar = torch.zeros(1, 4)
    samples = torch.zeros(1, 5, 4)
    nll = gaussian_nll(mean, logvar, samples)
    crps = gaussian_crps(mean, logvar, samples)
    assert float(nll) == pytest.approx(0.5 * math.log(2.0 * math.pi), rel=1e-5)
    expected_crps = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    assert float(crps) == pytest.approx(expected_crps, rel=1e-4)


def test_distribution_metrics_and_coverage() -> None:
    mean = torch.zeros(1, 4)
    logvar = torch.zeros(1, 4)
    samples = torch.zeros(1, 6, 4)
    target_mean = torch.zeros(1, 4)
    target_std = torch.ones(1, 4)
    valid = torch.ones(1, 4, dtype=torch.bool)
    metrics = distribution_metrics(
        mean,
        logvar,
        samples,
        target_mean,
        target_std,
        valid,
        target_scale_mm=2.0,
    )
    assert metrics["coverage"]["95"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["sharpness_mm"] == pytest.approx(2.0, rel=1e-5)
    assert metrics["num_acquisitions"] == 4


def test_fuzzy_config_validation_and_load() -> None:
    with pytest.raises(ValueError):
        FuzzyConfig(point_count_min=0)
    with pytest.raises(ValueError):
        FuzzyConfig(target_range_count=1)
    with pytest.raises(ValueError):
        FuzzyConfig(time_feature_mode="bad")
    config = load_experiment_config(
        "KW51_training/configs/kw51_fuzzy_lstm_smoke.yaml"
    )
    assert config.task == "kw51_fuzzy_response"
    assert config.fuzzy is not None
    assert config.fuzzy.target_range_count == 8


def test_target_time_mask_masks_missing_fem_times() -> None:
    config = FuzzyConfig(
        point_count_min=10,
        point_count_max=20,
        range_length_m_min=10.0,
        range_length_m_max=40.0,
        range_width_m_min=4.0,
        range_width_m_max=10.0,
        target_range_count=8,
        target_min_nodes=3,
    )
    source = _range_source(config)
    rng = np.random.default_rng(4)
    los_values = rng.normal(size=(26, len(source.geometry.deck_s))).astype(np.float32)
    los_valid = np.ones_like(los_values, dtype=bool)
    los_valid[:3, :] = False
    target, _ = source._targets(
        np.linspace(0.2, 0.8, 20),
        np.zeros(20, dtype=np.float32),
        los_values,
        los_valid,
        np.random.default_rng(5),
    )
    assert not target.time_valid[:3].any()
    assert not target.sample_valid[:, :3].any()
    assert np.allclose(target.samples[:, :3], 0.0)
    assert target.time_valid[3:].all()


def test_large_target_ranges_are_not_rejected_on_dense_deck() -> None:
    xs = np.linspace(0.0, 100.0, 200)
    ys = np.linspace(0.0, 10.0, 40)
    xy = np.asarray([[x, y] for x in xs for y in ys], dtype=np.float32)
    dense_geometry = DeckGeometry(
        node_coords=np.c_[xy, np.zeros(len(xy), dtype=np.float32)],
        deck_indices=np.arange(len(xy)),
        deck_xy=xy,
        deck_s=(xy[:, 0] / 100.0).astype(np.float32),
        deck_t=(2.0 * xy[:, 1] / 10.0 - 1.0).astype(np.float32),
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=10.0,
        los=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    config = FuzzyConfig(
        point_count_min=10,
        point_count_max=20,
        range_length_m_min=10.0,
        range_length_m_max=40.0,
        range_width_m_min=4.0,
        range_width_m_max=10.0,
        target_range_count=8,
        target_min_nodes=20,
    )
    source = _range_source(config)
    source.geometry = dense_geometry
    rng = np.random.default_rng(0)
    los_values = rng.normal(size=(26, len(xy))).astype(np.float32)
    los_valid = np.ones_like(los_values, dtype=bool)
    target, ranges = source._targets(
        np.linspace(0.2, 0.8, 20),
        np.zeros(20, dtype=np.float32),
        los_values,
        los_valid,
        rng,
    )
    assert len(ranges) == 8
    assert target.samples.shape == (8, 26)


def test_gaussian_nll_ignores_invalid_target_samples() -> None:
    mean = torch.zeros(1, 2)
    logvar = torch.zeros(1, 2)
    samples = torch.zeros(1, 3, 2)
    valid = torch.ones(1, 2, dtype=torch.bool)
    sample_valid = torch.tensor(
        [[[True, True], [False, True], [False, True]]], dtype=torch.bool
    )
    nll = gaussian_nll(mean, logvar, samples, valid, sample_valid)
    assert float(nll) == pytest.approx(0.5 * math.log(2.0 * math.pi), rel=1e-5)
