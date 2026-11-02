from __future__ import annotations

import numpy as np
import pytest
import torch

from qianhai7_training_v2.data.support_tokens import (
    build_spatial_support_layout,
    build_support_token_sample,
    collate_support_token_samples,
)
from qianhai7_training_v2.models.support_response import (
    SpatialSupportResponseModel,
    masked_gaussian_nll,
)


@pytest.fixture
def mesh():
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    node_ids = np.arange(1, 7, dtype=np.int64)
    elements = np.asarray([[1, 2, 5, 4], [2, 3, 6, 5]], dtype=np.int64)
    deck_mask = np.ones(6, dtype=bool)
    return coords, node_ids, elements, deck_mask


def make_sample(mesh, *, point_coords=None, all_missing_at_second_step=False):
    coords, node_ids, elements, deck_mask = mesh
    layout = build_spatial_support_layout(
        coords,
        deck_mask,
        node_ids,
        elements,
        rows=1,
        cols=2,
        bandwidth_scale=0.8,
    )
    dense = np.asarray(
        [
            [0.0, 1.0, 3.0, 0.0, 1.0, 3.0],
            [0.0, 2.0, 6.0, 0.0, 2.0, 6.0],
        ],
        dtype=np.float32,
    )
    values = dense.copy()
    valid = np.ones_like(values, dtype=bool)
    if all_missing_at_second_step:
        valid[1] = False
        values[1] = 0.0
    quality = valid.astype(np.float32)
    sample = build_support_token_sample(
        point_values=values,
        point_coords=coords[:, :2] if point_coords is None else point_coords,
        valid_mask=valid,
        quality=quality,
        time_features=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        node_coords=coords,
        dense_target=dense,
        layout=layout,
    )
    return sample, layout, dense


def test_support_targets_are_area_weighted_and_inputs_hide_exact_points(mesh):
    coords, node_ids, elements, deck_mask = mesh
    layout = build_spatial_support_layout(
        coords,
        deck_mask,
        node_ids,
        elements,
        rows=1,
        cols=1,
    )
    nodal = layout.nodal_area_weights
    assert nodal.sum() == pytest.approx(3.0)
    assert np.average([0.0, 1.0, 3.0, 0.0, 1.0, 3.0], weights=nodal) == pytest.approx(1.5)

    sample = build_support_token_sample(
        point_values=np.ones((2, 6), dtype=np.float32),
        point_coords=coords[:, :2],
        valid_mask=np.ones((2, 6), dtype=bool),
        quality=np.ones((2, 6), dtype=np.float32),
        time_features=np.zeros((2, 1), dtype=np.float32),
        node_coords=coords,
        dense_target=np.tile([0.0, 1.0, 3.0, 0.0, 1.0, 3.0], (2, 1)),
        layout=layout,
    )
    assert sample.target_response[:, 0] == pytest.approx([1.5, 1.5])
    assert sample.target_area_coverage.min() == pytest.approx(1.0)
    assert not hasattr(sample, "point_coords")
    assert sample.support_features.shape == (1, 4)


def test_fuzzy_memberships_preserve_constants_and_report_missing_supports(mesh):
    sample, layout, _ = make_sample(mesh, all_missing_at_second_step=True)
    assert layout.support_count == 2
    assert sample.values.shape == (2, 2)
    assert np.isfinite(sample.values).all()
    assert sample.coverage[0] == pytest.approx([1.0, 1.0])
    assert not sample.valid_mask[1].any()
    assert sample.values[1] == pytest.approx([0.0, 0.0])

    coords, node_ids, elements, deck_mask = mesh
    one_support = build_spatial_support_layout(
        coords, deck_mask, node_ids, elements, rows=1, cols=1
    )
    constant = build_support_token_sample(
        point_values=np.full((2, 6), 7.0, dtype=np.float32),
        point_coords=coords[:, :2],
        valid_mask=np.ones((2, 6), dtype=bool),
        quality=np.ones((2, 6), dtype=np.float32),
        time_features=np.zeros((2, 1), dtype=np.float32),
        node_coords=coords,
        dense_target=np.full((2, 6), 7.0, dtype=np.float32),
        layout=one_support,
    )
    assert constant.values == pytest.approx(np.full((2, 1), 7.0))
    assert constant.target_response == pytest.approx(np.full((2, 1), 7.0))


def test_building_tokens_does_not_mutate_observation_mask(mesh):
    coords, node_ids, elements, deck_mask = mesh
    layout = build_spatial_support_layout(
        coords, deck_mask, node_ids, elements, rows=1, cols=2
    )
    valid = np.ones((2, 6), dtype=bool)
    original = valid.copy()
    build_support_token_sample(
        point_values=np.ones((2, 6), dtype=np.float32),
        point_coords=coords[:, :2],
        valid_mask=valid,
        quality=np.ones((2, 6), dtype=np.float32),
        time_features=np.zeros((2, 1), dtype=np.float32),
        node_coords=coords,
        dense_target=np.ones((2, 6), dtype=np.float32),
        layout=layout,
    )
    assert np.array_equal(valid, original)


def test_coarse_support_features_do_not_change_with_point_jitter(mesh):
    sample, layout, dense = make_sample(mesh)
    coords, _, _, _ = mesh
    jittered = coords[:, :2].copy()
    jittered[:, 0] += np.asarray([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    jittered_sample = build_support_token_sample(
        point_values=dense,
        point_coords=jittered,
        valid_mask=np.ones_like(dense, dtype=bool),
        quality=np.ones_like(dense, dtype=np.float32),
        time_features=np.zeros((2, 1), dtype=np.float32),
        node_coords=coords,
        dense_target=dense,
        layout=layout,
    )
    assert jittered_sample.support_features == pytest.approx(sample.support_features)
    assert np.max(np.abs(jittered_sample.values - sample.values)) < 0.05


def test_support_response_model_has_finite_train_step(mesh):
    first, _, _ = make_sample(mesh)
    second, _, _ = make_sample(mesh)
    inputs, targets = collate_support_token_samples([first, second])
    model = SpatialSupportResponseModel(
        time_feature_count=inputs.time_features.shape[-1],
        hidden_size=16,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    predictions = model(inputs)
    loss = masked_gaussian_nll(predictions, targets)
    assert predictions["mean"].shape == targets.response.shape
    assert torch.isfinite(predictions["mean"]).all()
    assert torch.isfinite(predictions["scale"]).all()
    assert torch.all(predictions["scale"] > 0.0)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters())
    )
