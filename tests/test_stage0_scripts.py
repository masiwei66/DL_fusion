"""阶段 0 数据准备脚本的行为测试。

这些测试只构造轻量 JSON，不依赖真实响应张量；模型 forward/训练由实际阶段 0 运行验证。
"""

import hashlib
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
from dataset import _ensure_unique_sample_ids
from main import load_split_manifest


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

    def test_duplicate_source_ids_are_disambiguated_without_changing_group(self):
        metadata = [
            {
                "filename": "result_000000.json",
                "sample_id": "state_sample",
                "structural_state_id": "state_0",
                "excitation_id": "exc_0",
            },
            {
                "filename": "result_000001.json",
                "sample_id": "state_sample",
                "structural_state_id": "state_0",
                "excitation_id": "exc_1",
            },
        ]

        result = _ensure_unique_sample_ids(metadata)

        self.assertEqual(len({item["sample_id"] for item in result}), 2)
        self.assertEqual({item["source_sample_id"] for item in result}, {"state_sample"})
        self.assertEqual({item["structural_state_id"] for item in result}, {"state_0"})

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


class SplitManifestTests(unittest.TestCase):
    class DatasetStub:
        def __init__(self, files, sample_metadata):
            self.files = files
            self.sample_metadata = sample_metadata

        def __len__(self):
            return len(self.files)

    def test_duplicate_sample_ids_are_resolved_by_unique_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            files = []
            metadata = []
            records = []
            for index, split in enumerate(("train", "val", "test")):
                filename = f"result_{index:06d}.json"
                path = base / filename
                write_json(path, {"index": index})
                files.append(str(path))
                sample_id = "duplicate" if index < 2 else "unique"
                metadata.append({"filename": filename, "sample_id": sample_id})
                records.append({
                    "filename": filename,
                    "sample_id": sample_id,
                    "split": split,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })

            manifest_path = base / "split.json"
            write_json(manifest_path, {"records": records})
            dataset = self.DatasetStub(files, metadata)

            train_idx, val_idx, test_idx = load_split_manifest(manifest_path, dataset)

            self.assertEqual(train_idx, [0])
            self.assertEqual(val_idx, [1])
            self.assertEqual(test_idx, [2])

    def test_duplicate_sample_id_without_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            files = []
            metadata = []
            for index in range(2):
                filename = f"result_{index:06d}.json"
                path = base / filename
                write_json(path, {"index": index})
                files.append(str(path))
                metadata.append({"filename": filename, "sample_id": "duplicate"})

            manifest_path = base / "split.json"
            write_json(manifest_path, {
                "records": [
                    {"sample_id": "duplicate", "split": "train"},
                    {"sample_id": "duplicate", "split": "val"},
                ]
            })
            dataset = self.DatasetStub(files, metadata)

            with self.assertRaisesRegex(ValueError, "non-unique sample_id"):
                load_split_manifest(manifest_path, dataset, verify_hashes=False)


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
                self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_group_level_permutation_keeps_excitations_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_dir = base / "data"
            output_dir = base / "negative"
            data_dir.mkdir()
            records = []
            original_bytes = {}
            split_by_group = {
                "state_0": "train",
                "state_1": "train",
                "state_2": "train",
                "state_3": "train",
                "state_4": "val",
                "state_5": "test",
            }
            index = 0
            for group_index, (group, split) in enumerate(split_by_group.items()):
                group_sample = sample(group_index)
                for excitation_index in range(2):
                    name = f"result_{index:06d}.json"
                    payload = json.loads(json.dumps(group_sample))
                    payload["sample_index"] = {
                        "sample_id": f"sample_{group_index}",
                        "structural_state_id": group,
                    }
                    payload["dynamic_excitation"] = {
                        "excitation_id": f"exc_{excitation_index}"
                    }
                    payload["responses"] = {
                        "marker": [group_index, excitation_index]
                    }
                    path = data_dir / name
                    write_json(path, payload)
                    original_bytes[name] = path.read_bytes()
                    records.append({
                        "filename": name,
                        "sample_id": f"sample_{group_index}__exc_{excitation_index}",
                        "source_sample_id": f"sample_{group_index}",
                        "structural_state_id": group,
                        "group_by": "structural_state_id",
                        "group_value": group,
                        "split": split,
                    })
                    index += 1

            manifest_path = base / "split.json"
            write_json(manifest_path, {
                "group_by": "structural_state_id",
                "records": records,
            })
            completed = subprocess.run(
                [
                    sys.executable,
                    str(STAGE0_SCRIPTS / "shuffle_labels.py"),
                    str(data_dir),
                    "--split-manifest", str(manifest_path),
                    "--output", str(output_dir),
                    "--seed", "29",
                ],
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))

            audit = json.loads((output_dir / "stage0_permutation_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["permutation_unit"], "group")
            self.assertTrue(audit["train_group_donor_derangement"])
            self.assertTrue(audit["validation_and_test_byte_identical"])
            for group in ("state_0", "state_1", "state_2", "state_3"):
                names = [
                    record["filename"] for record in records
                    if record["group_value"] == group
                ]
                labels = [
                    json.loads((output_dir / name).read_text(encoding="utf-8"))["safety_labels"]
                    for name in names
                ]
                self.assertEqual(labels[0], labels[1])
            for record in records:
                if record["split"] in ("val", "test"):
                    self.assertEqual(
                        (output_dir / record["filename"]).read_bytes(),
                        original_bytes[record["filename"]],
                    )


class MakeSplitsCliTests(unittest.TestCase):
    def test_grouped_multilabel_split_has_unique_ids_and_no_group_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_dir = base / "data"
            data_dir.mkdir()
            output = base / "split.json"
            index = 0
            for group_index in range(12):
                for excitation_index in range(2):
                    payload = sample(group_index)
                    payload["sample_index"] = {
                        "sample_id": f"sample_{group_index}",
                        "structural_state_id": f"state_{group_index}",
                    }
                    payload["dynamic_excitation"] = {
                        "excitation_id": f"exc_{excitation_index}"
                    }
                    write_json(data_dir / f"result_{index:06d}.json", payload)
                    index += 1

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "make_splits.py"),
                    str(data_dir),
                    "--output", str(output),
                    "--group-by", "structural_state_id",
                    "--seed", "42",
                ],
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(manifest["effective_sample_ids_unique"])
            self.assertGreater(manifest["source_sample_id_duplicate_count"], 0)
            self.assertTrue(all(not values for values in manifest["split_group_overlap"].values()))
            self.assertEqual(set(manifest["material_label_prevalence"]), {"train", "val", "test"})
            self.assertTrue(all(manifest["actual_counts"][name] > 0 for name in ("train", "val", "test")))


if __name__ == "__main__":
    unittest.main()
