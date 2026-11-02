from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from KW51_training.data.fuzzy_range import DeckGeometry
from KW51_training.data.fuzzy_region import RegionLayout
from KW51_training.data.support_tokens import (
    SupportTokenLayout as KW51SupportLayout,
    _soft_memberships as kw51_memberships,
    aggregate_observation_tokens as kw51_aggregate,
    build_support_token_layout as build_kw51_layout,
)
from KW51_training.models.support_response import (
    SupportTokenResponseModel as KW51SupportResponseModel,
)
from qianhai7_training_v2.data.support_tokens import (
    QianhaiSupportTokenDataset,
    _soft_memberships as qianhai_memberships,
    aggregate_observation_tokens as qianhai_aggregate,
    build_spatial_support_layout,
    collate_support_token_samples,
)
from qianhai7_training_v2.data import support_tokens as qh_support_module
from qianhai7_training_v2.data.normalization import NormalizationStats
from qianhai7_training_v2.models.support_response import (
    SpatialSupportResponseModel,
)
from spatial_support_contract import (
    SUPPORT_FEATURE_ORDER,
    SUPPORT_NORMALIZATION_POLICY,
    SUPPORT_TOKEN_SCHEMA,
    SupportTokenInputs,
    SupportTokenTargets,
    SupportTokenValueNormalization,
)


def _equivalent_layouts():
    coords = np.asarray(
        [[x, y, 0.0] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)],
        dtype=np.float32,
    )
    node_ids = np.arange(1, 10, dtype=np.int64)
    elements = np.asarray(
        [[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]],
        dtype=np.int64,
    )
    qh = build_spatial_support_layout(
        coords,
        np.ones(9, dtype=bool),
        node_ids,
        elements,
        rows=2,
        cols=2,
        bandwidth_scale=0.75,
    )
    deck_geometry = DeckGeometry(
        node_coords=coords,
        deck_indices=np.arange(9),
        deck_xy=coords[:, :2],
        deck_s=coords[:, 0],
        deck_t=2.0 * coords[:, 1] - 1.0,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        los=np.asarray([0.0, 0.0, 1.0]),
    )
    region_layout = RegionLayout(
        mode="grid",
        rows=2,
        cols=2,
        s_edges=np.asarray([0.0, 0.5, 1.0]),
        t_edges=np.asarray([-1.0, 0.0, 1.0]),
        centers_s=np.asarray([0.25, 0.25, 0.75, 0.75]),
        centers_t=np.asarray([-0.5, 0.5, -0.5, 0.5]),
    )
    kw = build_kw51_layout(region_layout, bandwidth_scale=0.75)
    return qh, kw


def test_qianhai_and_kw51_support_coordinates_and_pooling_are_equivalent():
    qh, kw = _equivalent_layouts()
    assert SUPPORT_FEATURE_ORDER == (
        "center_s_unit_minus1_to1",
        "center_t_unit_minus1_to1",
        "extent_s_fraction_of_domain",
        "extent_t_fraction_of_domain",
    )
    assert np.allclose(qh.support_features, kw.features)
    assert np.allclose(qh.bandwidths_unit, kw.bandwidths_unit)

    point_values = np.asarray([[1.0, 3.0, 5.0]], dtype=np.float32)
    valid = np.asarray([[True, False, True]])
    quality = np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32)
    qh_xy = np.asarray([[0.2, 0.2], [0.8, 0.8], [0.2, 0.8]], dtype=np.float32)
    kw_st = np.c_[qh_xy[:, 0], 2.0 * qh_xy[:, 1] - 1.0]
    qh_membership = qianhai_memberships(qh_xy, qh)
    kw_membership = kw51_memberships(kw_st, kw)
    assert np.allclose(qh_membership, kw_membership)

    qh_tokens = qianhai_aggregate(
        point_values, qh_xy, valid, quality, qh, minimum_coverage=0.0
    )
    kw_tokens = kw51_aggregate(
        point_values, valid, quality, kw_st, kw, minimum_coverage=0.0
    )
    for qh_value, kw_value in zip(qh_tokens, kw_tokens):
        assert np.allclose(qh_value, kw_value)


def test_both_projects_use_the_same_model_contract_and_normalization_policy():
    assert SpatialSupportResponseModel is KW51SupportResponseModel
    assert SUPPORT_TOKEN_SCHEMA == "spatial_support_tokens_v1"
    assert SUPPORT_NORMALIZATION_POLICY == "fit_training_states_only_apply_after_support_pooling"
    inputs = SupportTokenInputs(
        values=torch.zeros((2, 3, 2)),
        valid_mask=torch.ones((2, 3, 2), dtype=torch.bool),
        quality=torch.ones((2, 3, 2)),
        coverage=torch.ones((2, 3, 2)),
        support_features=torch.tensor(
            [[[-0.5, 0.0, 0.5, 1.0], [0.5, 0.0, 0.5, 1.0]]]
        ).expand(2, -1, -1),
        time_features=torch.zeros((2, 3, 2)),
    )
    targets = SupportTokenTargets(
        response=torch.zeros((2, 3, 2)),
        valid_mask=torch.ones((2, 3, 2), dtype=torch.bool),
        area_coverage=torch.ones((2, 3, 2)),
    )
    model = SpatialSupportResponseModel(time_feature_count=2, hidden_size=8)
    output = model(inputs)
    assert output["mean"].shape == inputs.values.shape
    assert output["scale"].shape == targets.response.shape
    assert torch.isfinite(output["mean"]).all()
    assert torch.isfinite(output["scale"]).all()
    with pytest.raises(ValueError, match="training-state provenance"):
        SupportTokenValueNormalization(0.0, 1.0, ())
    normalized = SupportTokenValueNormalization(2.0, 4.0, ("train-state",))
    values = normalized.transform_numpy(
        np.asarray([[6.0, 10.0]], dtype=np.float32),
        np.asarray([[True, False]]),
    )
    np.testing.assert_allclose(values, [[1.0, 0.0]])


def test_qianhai_support_dataset_emits_isolated_canonical_batch(monkeypatch):
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
    ids = np.arange(1, 7, dtype=np.int64)
    elements = np.asarray([[1, 2, 5, 4], [2, 3, 6, 5]], dtype=np.int64)
    dense = np.asarray(
        [[0, 1, 3, 0, 1, 3], [0, 2, 6, 0, 2, 6], [0, 3, 9, 0, 3, 9]],
        dtype=np.float32,
    )
    arrays = {
        "node_coords": coords,
        "node_ids": ids,
        "deck_node_mask": np.ones(6, dtype=bool),
        "deck_element_connectivity": elements,
        "displacement_uz": dense,
        "temperature_C": np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
        "delta_days": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
    }
    view = {
        "values": dense.copy(),
        "valid_mask": np.ones_like(dense, dtype=bool),
        "quality": np.ones_like(dense, dtype=np.float32),
        "point_coords": coords.copy(),
    }
    monkeypatch.setattr(
        qh_support_module,
        "_sample_qianhai_state",
        lambda *args: (arrays, view, dense.copy()),
    )
    record = SimpleNamespace(path=Path("unused.h5"), structural_state_id="train-1")
    ref = SimpleNamespace(record=record, start=0, length=2)
    layout = build_spatial_support_layout(
        coords,
        arrays["deck_node_mask"],
        ids,
        elements,
        rows=1,
        cols=2,
    )
    time_stats = NormalizationStats(
        values_mean=0.0,
        values_std=1.0,
        temperature_mean=10.0,
        temperature_std=1.0,
        delta_days_mean=0.0,
        delta_days_std=1.0,
        coordinates_mean=(0.0, 0.0),
        coordinates_std=(1.0, 1.0),
        fitted_state_ids=("train-1",),
    )
    value_stats = SupportTokenValueNormalization(0.0, 1.0, ("train-1",))
    dataset = QianhaiSupportTokenDataset(
        [ref],
        observation=SimpleNamespace(point_count_min=6, point_count_max=6),
        seed=1,
        time_normalization=time_stats,
        value_normalization=value_stats,
        layout=layout,
    )
    inputs, targets = collate_support_token_samples([dataset[0]])
    assert inputs.values.shape == (1, 2, 2)
    assert inputs.support_features.shape == (1, 2, 4)
    assert targets.response.shape == inputs.values.shape
    assert torch.isfinite(inputs.values).all()
    assert torch.isfinite(targets.response).all()
    assert not hasattr(inputs, "target_response")
