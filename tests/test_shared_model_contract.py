from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_fusion.contracts import WindowBatch
from KW51_training.config import load_experiment_config as load_kw51_config
from qianhai7_training_v2.config import load_experiment_config as load_qianhai_config
from KW51_training.data.fem_insar import RegionConfig, build_region_map
from qianhai7_training_v2.config import load_experiment_config
from qianhai7_training_v2.data.observation import build_observation_grid
from qianhai7_training_v2.models.registry import build_model
from qianhai7_training_v2.cli.train import materialize_lazy_model
from qianhai7_training_v2.training.report import count_parameters


def _batch(mode: str, *, points: int = 5) -> WindowBatch:
    values = torch.randn(2, 4, points)
    return WindowBatch.from_observations(
        values=values,
        valid_mask=torch.ones_like(values, dtype=torch.bool),
        quality=torch.ones_like(values),
        coordinates=torch.randn(2, points, 2),
        time_features=torch.randn(2, 4, 2),
        point_present_mask=torch.ones(2, points, dtype=torch.bool),
        coverage=torch.ones_like(values),
        input_mode=mode,
    )


@pytest.mark.parametrize("mode,points", [("node", 5), ("region", 7)])
def test_qianhai_four_task_head_uses_shared_representation_interface(mode: str, points: int) -> None:
    model = build_model(
        "lstm",
        input_mode=mode,
        region_pooling="flatten" if mode == "region" else "mean",
        hidden_size=8,
        num_layers=1,
    )
    batch = _batch(mode, points=points)
    outputs = model.forward_batch(batch)
    loss = sum(value.square().mean() for key, value in outputs.items() if key != "representation")
    loss.backward()

    assert set(outputs) == {
        "representation", "deck_material", "pier_material", "pier_settlement", "bearing_stiffness"
    }
    assert outputs["representation"].shape[0] == 2
    assert torch.isfinite(loss)


def test_region_lazy_heads_are_materialized_before_parameter_counting() -> None:
    model = build_model(
        "lstm", input_mode="region", region_pooling="flatten", hidden_size=8, num_layers=1
    )

    assert materialize_lazy_model(model, [_batch("region", points=7)], torch.device("cpu"))
    counts = count_parameters(model)

    assert model.training
    assert counts["trainable_parameters"] == counts["total_parameters"]
    assert counts["total_parameters"] > 0


def test_shared_backbone_rejects_adapter_mode_mismatch() -> None:
    model = build_model("lstm", input_mode="node", hidden_size=8, num_layers=1)
    with pytest.raises(ValueError, match="does not match backbone"):
        model.forward_representation(_batch("region"))


def test_shared_observation_batch_preserves_coverage_on_indexing() -> None:
    batch = _batch("region")
    selected = batch.index_select(torch.tensor([1]))

    assert selected.input_mode == "region"
    assert selected.coverage is not None
    assert selected.region_coverage is not None
    assert selected.coverage.shape == (1, 4, 5)


def test_qianhai_and_kw51_build_metric_grids_from_millimeter_mesh_coordinates() -> None:
    millimeter_xy = np.asarray(
        [[0, 0], [50000, 0], [100000, 0], [0, 20000], [100000, 20000]],
        dtype=np.float32,
    )
    deck = np.ones(len(millimeter_xy), dtype=bool)

    qianhai_grid = build_observation_grid(millimeter_xy, deck, 10.0, 0.001)
    kw51_map = build_region_map(
        millimeter_xy,
        deck,
        RegionConfig(region_grid_size_m=10.0, coordinate_scale_to_m=0.001),
    )

    assert qianhai_grid.cell_size_m == 10.0
    assert qianhai_grid.region_count == 20
    assert kw51_map.grid.region_count == 20


@pytest.mark.parametrize("model_name", ["mlp", "gnn_lstm"])
def test_region_config_does_not_pass_temporal_pooling_to_other_backbones(
    tmp_path, model_name: str
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "input_mode: region\n"
        "data:\n  data_root: ./fixture\n"
        f"model:\n  name: {model_name}\n  params: {{}}\n",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.input_mode == "region"
    assert config.model.kwargs["input_mode"] == "region"
    assert "region_pooling" not in config.model.kwargs


def test_conflicting_qianhai_modes_fail_during_config_loading(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "input_mode: node\n"
        "data:\n  data_root: ./fixture\n"
        "model:\n  name: lstm\n  params:\n    input_mode: region\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must agree"):
        load_experiment_config(config_path)


@pytest.mark.parametrize("model_name", ["mlp", "gnn_lstm"])
def test_kw51_config_uses_shared_mode_without_unused_pooling_kwargs(
    tmp_path, model_name: str
) -> None:
    config_path = tmp_path / "kw51.yaml"
    config_path.write_text(
        "task: kw51_candidate_ranking\n"
        "input_mode: region\n"
        "data:\n  data_root: ./fixture\n  dataset_kind: kw51\n"
        f"model:\n  name: {model_name}\n  params: {{}}\n",
        encoding="utf-8",
    )

    config = load_kw51_config(config_path)

    assert config.input_mode == "region"
    assert config.data.input_mode == "region"
    assert config.model.kwargs["input_mode"] == "region"
    assert "region_pooling" not in config.model.kwargs


def test_root_training_configs_resolve_task_model_kwargs_and_runtime() -> None:
    qianhai = load_qianhai_config("configs/qianhai7_v12_baseline.yaml")
    kw51 = load_kw51_config("configs/kw51_candidate_ranking_region.yaml")

    assert qianhai.task == "qianhai_four_task"
    assert qianhai.input_mode == "node"
    assert qianhai.train.seed == 20261031
    assert qianhai.train.deterministic
    assert qianhai.model.kwargs["hidden_size"] == 128
    assert kw51.task == "kw51_candidate_ranking"
    assert kw51.input_mode == kw51.data.input_mode == "region"
    assert kw51.train.device == "auto"
    assert kw51.model.kwargs["hidden_size"] == 128

