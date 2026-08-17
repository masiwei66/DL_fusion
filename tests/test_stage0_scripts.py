"""阶段 0 数据准备脚本的行为测试。

这些测试只构造轻量 JSON，不依赖真实响应张量；模型 forward/训练由实际阶段 0 运行验证。
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STAGE0_SCRIPTS = SCRIPTS / "stage0"
sys.path.insert(0, str(STAGE0_SCRIPTS))

from stage0_common import deranged_donors, prepare_new_output_dir, replace_supervision
from check_smoke_batch import check_input_boundary, check_models


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def sample(index):
    return {
        "sample_index": {"sample_id": f"sample_{index}", "structural_state_id": f"state_{index}"},
        "responses": {"marker": index},
        "environment": {"temperature": {"value_C": index}},
        "safety_labels": {
            "material_labels": [index % 2, (index + 1) % 2],
            "support_labels": [index % 2],
            "region_risk_levels": [index % 4],
            "global_level": index % 4,
        },
        "material_scaling_factors": [1.0 - index * 0.01],
        "support_settlement": {"values_mm": [float(index)]},
    }


class Stage0CommonTests(unittest.TestCase):
    def test_derangement_has_no_fixed_points(self):
        import random

        donors = deranged_donors(12, random.Random(42))
        self.assertEqual(sorted(donors), list(range(12)))
        self.assertTrue(all(index != donor for index, donor in enumerate(donors)))

    def test_replace_all_supervision_keeps_inputs(self):
        recipient = sample(1)
        donor = sample(2)
        changed = replace_supervision(recipient, donor, mode="all")
        self.assertEqual(changed["responses"], recipient["responses"])
        self.assertEqual(changed["environment"], recipient["environment"])
        self.assertEqual(changed["safety_labels"], donor["safety_labels"])
        self.assertEqual(changed["support_settlement"], donor["support_settlement"])

    def test_nonempty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output"
            path.mkdir()
            (path / "old.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_new_output_dir(path)

    def test_sanitized_inputs_support_all_model_backward_paths(self):
        batch_size = 2
        batch = {
            "disp": torch.randn(batch_size, 6, 6, 3),
            "strain": torch.randn(batch_size, 6, 6, 3),
            "ace": torch.randn(batch_size, 64, 6, 3),
            "condition": torch.randn(batch_size, 1),
            "temperature_C": torch.randn(batch_size),
            "temperature_steps_C": torch.randn(batch_size, 6),
            "quality_metrics": torch.randn(batch_size, 3),
            "static_pos": torch.randn(6, 3),
            "dynamic_pos": torch.randn(6, 3),
            "target": torch.randint(0, 2, (batch_size, 6)).float(),
            "raw_sf": torch.rand(batch_size, 6),
            "support_disp": torch.rand(batch_size, 4),
            "support_target": torch.randint(0, 2, (batch_size, 4)).float(),
            "region_target": torch.randint(0, 4, (batch_size, 5)),
            "global_target": torch.randint(0, 4, (batch_size,)),
            "metadata": [{"sample_id": "a"}, {"sample_id": "b"}],
        }
        boundary_problems, boundary = check_input_boundary(batch)
        self.assertEqual(boundary_problems, [])
        self.assertNotIn("target", boundary["model_input_keys"])
        problems, results = check_models(batch, check_backward=True)
        self.assertEqual(problems, [])
        self.assertEqual(set(results), {"static_only", "dynamic_only", "fusion"})
        self.assertTrue(all(item["updated_parameter_count"] > 0 for item in results.values()))


class ShuffleLabelsCliTests(unittest.TestCase):
    def test_only_train_files_change_and_manifest_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_dir = base / "data"
            output_dir = base / "negative"
            data_dir.mkdir()
            records = []
            splits = ["train", "train", "train", "train", "val", "test"]
            original_bytes = {}
            for index, split in enumerate(splits):
                name = f"result_{index:06d}.json"
                path = data_dir / name
                write_json(path, sample(index))
                original_bytes[name] = path.read_bytes()
                records.append({
                    "filename": name,
                    "sample_id": f"sample_{index}",
                    "split": split,
                    "sha256": "source-hash-not-used-by-preparer",
                })
            manifest_path = base / "split.json"
            write_json(manifest_path, {"records": records, "counts": {"train": 4, "val": 1, "test": 1}})

            completed = subprocess.run(
                [
                    sys.executable,
                    str(STAGE0_SCRIPTS / "shuffle_labels.py"),
                    str(data_dir),
                    "--split-manifest", str(manifest_path),
                    "--output", str(output_dir),
                    "--seed", "42",
                ],
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))

            for index, split in enumerate(splits):
                name = f"result_{index:06d}.json"
                current = (output_dir / name).read_bytes()
                if split in ("val", "test"):
                    self.assertEqual(current, original_bytes[name])
                else:
                    original = json.loads(original_bytes[name].decode("utf-8"))
                    changed = json.loads(current.decode("utf-8"))
                    self.assertEqual(changed["responses"], original["responses"])
                    self.assertNotEqual(changed["support_settlement"], original["support_settlement"])

            audit = json.loads((output_dir / "stage0_permutation_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["train_donor_derangement"])
            self.assertTrue(audit["validation_and_test_byte_identical"])
            generated = json.loads((output_dir / "stage0_negative_split.json").read_text(encoding="utf-8"))
            for record in generated["records"]:
                path = output_dir / record["filename"]
                import hashlib

                self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
