from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from KW51_training.config import FuzzyConfig
from KW51_training.data.fem_insar import build_fem_archive_index
from KW51_training.data.fuzzy_range import (
    FuzzyNormalization,
    FuzzyQueryRef,
    build_deck_geometry,
)
from KW51_training.data.fuzzy_region import FuzzyRegionSource, build_region_layout
from KW51_training.data.support_tokens import (
    FuzzySupportTokenDataset,
    SupportTokenLayout,
    aggregate_area_weighted_targets,
    aggregate_observation_tokens,
    build_support_token_layout,
    collate_fuzzy_support_tokens,
    fit_support_token_value_normalization,
    support_token_inputs_from_window_batch,
    support_token_targets_from_window_batch,
)
from KW51_training.models.support_response import (
    SupportTokenResponseModel,
    masked_gaussian_nll,
)
from KW51_training.training.public_benchmark_runner import (
    _assert_dense_targets_not_model_inputs,
    _build_public_model,
)


SMOKE = Path("KW51_training/runs/_fuzzy_smoke_archive")
SKIP = pytest.mark.skipif(not SMOKE.exists(), reason="local smoke archive is absent")


def _layout() -> SupportTokenLayout:
    centers = np.asarray([[0.25, 0.5], [0.75, 0.5]], dtype=np.float32)
    bandwidths = np.full((2, 2), 0.3, dtype=np.float32)
    features = np.asarray(
        [[-0.5, 0.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]], dtype=np.float32
    )
    return SupportTokenLayout(centers, bandwidths, features)


def test_soft_support_observations_preserve_constants_and_mask_missing_values():
    layout = _layout()
    coordinates = np.asarray([[0.2, 0.0], [0.8, 0.0]], dtype=np.float32)
    values = np.full((3, 2), 7.0, dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    quality = np.ones_like(values, dtype=np.float32)
    valid[1:, 0] = False
    quality[1:, 0] = 0.0

    tokens, token_valid, token_quality, coverage = aggregate_observation_tokens(
        values, valid, quality, coordinates, layout, minimum_coverage=0.5
    )

    assert tokens.shape == token_valid.shape == token_quality.shape == coverage.shape == (3, 2)
    assert np.allclose(tokens[0], 7.0)
    assert np.all(token_valid[0])
    assert np.isfinite(tokens).all()
    assert np.allclose(tokens[1:, 0], 0.0)
    assert not token_valid[1:, 0].any()
    assert token_valid[1:, 1].all()
    assert np.allclose(tokens[1:, 1], 7.0)


def test_area_weighted_soft_targets_preserve_constant_response():
    layout = _layout()
    coordinates = np.asarray(
        [[0.0, -1.0], [0.25, -0.5], [0.5, 0.0], [0.75, 0.5], [1.0, 1.0]],
        dtype=np.float32,
    )
    values = np.full((4, len(coordinates)), -2.5, dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    areas = np.asarray([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)

    target, mask, coverage, support_area = aggregate_area_weighted_targets(
        values, valid, coordinates, areas, layout
    )

    assert target.shape == mask.shape == coverage.shape == (4, 2)
    assert np.allclose(target, -2.5)
    assert mask.all()
    assert np.allclose(coverage, 1.0)
    assert np.all(support_area > 0.0)


def test_target_aggregation_does_not_mutate_fem_validity_mask():
    layout = _layout()
    coordinates = np.asarray(
        [[0.0, -1.0], [0.25, -0.5], [0.5, 0.0], [0.75, 0.5], [1.0, 1.0]],
        dtype=np.float32,
    )
    valid = np.ones((2, len(coordinates)), dtype=bool)
    valid[0, 1] = False
    original = valid.copy()
    aggregate_area_weighted_targets(
        np.ones_like(valid, dtype=np.float32),
        valid,
        coordinates,
        np.ones(len(coordinates), dtype=np.float32),
        layout,
        minimum_area_coverage=0.5,
    )
    assert np.array_equal(valid, original)


@SKIP
def test_smoke_archive_adapter_hides_exact_points_and_trains_response_model():
    index = build_fem_archive_index(SMOKE)
    record = index.records[0]
    geometry = build_deck_geometry(record, [0.0, 0.0, 1.0])
    region_layout = build_region_layout(
        geometry, mode="longitudinal", rows=4, cols=1
    )
    support_layout = build_support_token_layout(region_layout)
    config = FuzzyConfig(
        region_point_budget=32,
        region_missing_probability=0.0,
        region_min_valid_points=1,
    )
    source = FuzzyRegionSource(index, geometry, config, seed=41)
    support_value_normalization = fit_support_token_value_normalization(
        source, [record], region_layout, support_layout
    )
    normalization = FuzzyNormalization(
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, "disabled", ()
    )
    dataset = FuzzySupportTokenDataset(
        [FuzzyQueryRef(record, 0)],
        source,
        normalization,
        region_layout,
        support_layout,
        support_value_normalization,
    )
    item = dataset[0]
    batch = collate_fuzzy_support_tokens([item, item])
    fold_data = {
        "base_config": SimpleNamespace(
            model=SimpleNamespace(
                name="lstm",
                kwargs={"input_dim": 6, "hidden_size": 16, "num_layers": 1},
            )
        ),
        "layout": region_layout,
        "support_layout": support_layout,
        "group_map": SimpleNamespace(group_count=5),
    }
    model = _build_public_model("support_token_gru", fold_data)
    assert isinstance(model, SupportTokenResponseModel)
    model_inputs = support_token_inputs_from_window_batch(batch)
    model_targets = support_token_targets_from_window_batch(batch)
    output = model(model_inputs)
    loss = masked_gaussian_nll(output, model_targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert "point_coords" not in item
    assert "dense_region_target" not in item
    assert not hasattr(model_inputs, "targets")
    assert batch.values.shape == (2, 26, region_layout.region_count)
    assert batch.support_features.shape == (2, region_layout.region_count, 4)
    assert batch.support_coverage.shape == batch.values.shape
    assert model_targets.response.shape == batch.values.shape
    assert output["mean"].shape == batch.values.shape
    assert batch.targets["region_response_target"].shape == (
        2,
        region_layout.region_count,
        26,
    )
    assert torch.isfinite(output["mean"]).all()
    assert torch.isfinite(output["scale"]).all()
    assert torch.isfinite(loss)
    assert _assert_dense_targets_not_model_inputs(model, batch) == "PASS"
