import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from qianhai7_training.contracts import WindowBatch
from qianhai7_training.data.dataset import collate_window_batches
from qianhai7_training.data.hdf5_source import HDF5StateSource
from qianhai7_training.task_labels import (
    TASK_LABEL_LEVEL_NAMES,
    TASK_LABEL_NAMES,
    task_targets_at_index,
)
from qianhai7_training.training.losses import supervised_loss
from qianhai7_training.models.registry import build_model


def _label_arrays():
    labels = np.zeros((26, 4), dtype=np.uint8)
    labels[7] = [0, 1, 2, 3]
    return {
        "task_label_names": np.asarray(TASK_LABEL_NAMES),
        "task_label_level_names": np.asarray(TASK_LABEL_LEVEL_NAMES),
        "task_level_by_acquisition": labels,
    }


def _window_item(levels):
    width, points = 6, 5
    return {
        "values": np.zeros((width, points), dtype=np.float32),
        "valid_mask": np.ones((width, points), dtype=bool),
        "quality": np.ones((width, points), dtype=np.float32),
        "point_coords": np.zeros((points, 2), dtype=np.float32),
        "mapping_distance": np.zeros(points, dtype=np.float32),
        "mapping_valid_mask": np.ones(points, dtype=bool),
        "temperature_C": np.full(width, 20.0, dtype=np.float32),
        "delta_days": np.full(width, 14.0, dtype=np.float32),
        "temperature_valid_mask": np.ones(width, dtype=bool),
        "teacher_features": {"modal_frequencies": np.zeros(20, dtype=np.float32)},
        "target_task_levels": dict(zip(TASK_LABEL_NAMES, levels)),
        **{f"target_{name}": np.int64(level) for name, level in zip(TASK_LABEL_NAMES, levels)},
        "target_support_settlement_mm": np.zeros(0, dtype=np.float32),
        "target_support_settlement_ratio": np.zeros(0, dtype=np.float32),
        "target_local_deck_settlement_mm": np.zeros(0, dtype=np.float32),
        "metadata": {},
    }


def _batch(targets):
    batch_size, window, points = targets.shape[0], 6, 5
    return WindowBatch(
        values=torch.zeros(batch_size, window, points),
        valid_mask=torch.ones(batch_size, window, points, dtype=torch.bool),
        quality=torch.ones(batch_size, window, points),
        point_features=torch.zeros(batch_size, points, 2),
        time_features=torch.zeros(batch_size, window, 2),
        point_present_mask=torch.ones(batch_size, points, dtype=torch.bool),
        mapping_distance=torch.zeros(batch_size, points),
        mapping_valid_mask=torch.ones(batch_size, points, dtype=torch.bool),
        temperature_valid_mask=torch.ones(batch_size, window, 1, dtype=torch.bool),
        teacher_features={"modal_frequencies": torch.zeros(batch_size, 20)},
        targets={f"target_{name}": targets[:, index] for index, name in enumerate(TASK_LABEL_NAMES)},
        metadata=[{} for _ in range(batch_size)],
    )


class QianhaiOrdinalTargetTests(unittest.TestCase):
    def test_v3_label_reader_uses_window_terminal_acquisition(self):
        arrays = _label_arrays()
        self.assertEqual(
            task_targets_at_index(arrays, 7),
            dict(zip(TASK_LABEL_NAMES, (0, 1, 2, 3))),
        )
        with self.assertRaises(IndexError):
            task_targets_at_index(arrays, 26)
        arrays["task_level_by_acquisition"][7, 2] = 4
        with self.assertRaises(ValueError):
            task_targets_at_index(arrays, 7)

    def test_hdf5_source_reads_v3_task_label_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            with h5py.File(path, "w") as handle:
                for group in (
                    "mesh", "responses/fem", "environment", "labels", "state",
                ):
                    handle.require_group(group)
                handle["manifest_json"] = "{}"
                handle["mesh/node_coords"] = np.zeros((2, 3), dtype=np.float32)
                handle["mesh/node_ids"] = np.asarray([1, 2], dtype=np.int64)
                handle["mesh/deck_node_mask"] = np.asarray([1, 1], dtype=np.uint8)
                handle["mesh/deck_element_connectivity"] = np.asarray([[1, 2]], dtype=np.int64)
                handle["responses/fem/disp_at_acquisition"] = np.zeros((26, 2, 3), dtype=np.float32)
                handle["environment/temperature_by_acquisition_C"] = np.zeros(26, dtype=np.float32)
                handle["environment/delta_days"] = np.ones(26, dtype=np.int16)
                handle["labels/state_family_code_by_acquisition"] = np.zeros(26, dtype=np.int64)
                handle["labels/state_family_vocabulary"] = np.asarray(["healthy"], dtype="S")
                handle["labels/global_severity_level_by_acquisition"] = np.zeros(26, dtype=np.int64)
                handle["labels/task_label_names"] = np.asarray(TASK_LABEL_NAMES, dtype="S")
                handle["labels/task_label_level_names"] = np.asarray(TASK_LABEL_LEVEL_NAMES, dtype="S")
                handle["labels/task_level_by_acquisition"] = _label_arrays()["task_level_by_acquisition"]

            arrays = HDF5StateSource(path).read(cache=False)
            self.assertEqual(
                task_targets_at_index(arrays, 7),
                dict(zip(TASK_LABEL_NAMES, (0, 1, 2, 3))),
            )

    def test_collate_and_loss_use_four_scalar_class_targets(self):
        batch = collate_window_batches([
            _window_item((0, 1, 2, 3)),
            _window_item((3, 2, 1, 0)),
        ])
        for name in TASK_LABEL_NAMES:
            self.assertEqual(batch.targets[f"target_{name}"].shape, (2,))
            self.assertEqual(batch.targets[f"target_{name}"].dtype, torch.long)
        outputs = {name: torch.randn(2, 4, requires_grad=True) for name in TASK_LABEL_NAMES}
        loss = supervised_loss(outputs, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(output.grad is not None for output in outputs.values()))

    def test_all_implemented_neural_models_emit_four_logits_per_task(self):
        batch = _batch(torch.zeros(2, 4, dtype=torch.long))
        model_options = {
            "mlp": {},
            "resnet1d": {"base_width": 2, "layers": (1, 1, 1, 1)},
            "resnet50_1d": {"base_width": 2, "layers": (1, 1, 1, 1)},
            "lstm": {"hidden_size": 8, "num_layers": 1},
            "gru": {"hidden_size": 8, "num_layers": 1},
            "transformer": {"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16},
            "gnn_lstm": {"graph_hidden_dim": 8, "lstm_hidden_size": 8, "graph_layers": 1, "lstm_layers": 1, "knn_k": 2},
        }
        for model_name, options in model_options.items():
            with self.subTest(model=model_name):
                model = build_model(model_name, **options).eval()
                with torch.no_grad():
                    outputs = model.forward_batch(batch)
                for task in TASK_LABEL_NAMES:
                    self.assertEqual(outputs[task].shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
