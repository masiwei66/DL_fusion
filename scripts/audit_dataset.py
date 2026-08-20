"""审计 V2 JSON 数据集的结构、标签与质量元数据。"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def iter_samples(data_dir):
    for name in sorted(os.listdir(data_dir)):
        if name.startswith("result_") and name.endswith(".json"):
            path = os.path.join(data_dir, name)
            with open(path, encoding="utf-8") as file:
                yield path, json.load(file)


def main():
    parser = argparse.ArgumentParser(description="Audit V2 JSON dataset.")
    parser.add_argument("data_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Only print the first N samples in detail.")
    args = parser.parse_args()

    samples = list(iter_samples(args.data_dir))
    if not samples:
        raise FileNotFoundError(f"no result_*.json found in {args.data_dir}")

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "audit_created_at": datetime.now().isoformat(timespec="seconds"),
        "data_version": os.path.basename(os.path.abspath(args.data_dir)),
        "data_dir": os.path.abspath(args.data_dir),
        "sample_count": len(samples),
        "sample_ids": [],
        "structural_state_ids": [],
        "excitation_ids": [],
        "condition_groups": [],
        "temperature_steps": Counter(),
        "quality_metric_keys": Counter(),
        "array_integrity_keys": Counter(),
        "label_distribution": {
            "material": Counter(),
            "support": Counter(),
            "region": Counter(),
            "global": Counter(),
        },
        "samples": [],
    }

    for index, (path, sample) in enumerate(samples):
        sample_index = sample.get("sample_index", {}) or {}
        structural_state = sample.get("structural_state", {}) or {}
        excitation = sample.get("dynamic_excitation", {}) or {}
        temperature = sample.get("environment", {}).get("temperature", {}) or {}
        safety = sample.get("safety_labels", {}) or {}
        quality = sample.get("quality_metrics", {}) or {}
        integrity = sample.get("array_integrity", {}) or {}

        sample_id = sample_index.get("sample_id", os.path.basename(path))
        structural_state_id = sample_index.get("structural_state_id", sample_index.get("group_id"))
        excitation_id = excitation.get("excitation_id", structural_state.get("excitation_id"))
        condition_group = sample_index.get("condition_group")

        summary["sample_ids"].append(sample_id)
        if structural_state_id is not None:
            summary["structural_state_ids"].append(structural_state_id)
        if excitation_id is not None:
            summary["excitation_ids"].append(excitation_id)
        if condition_group is not None:
            summary["condition_groups"].append(condition_group)

        steps = tuple(temperature.get("temperature_steps_C", []))
        summary["temperature_steps"][str(len(steps))] += 1
        for key in quality:
            summary["quality_metric_keys"][key] += 1
        for key in integrity:
            summary["array_integrity_keys"][key] += 1

        for label in safety.get("material_labels", []):
            summary["label_distribution"]["material"][str(label)] += 1
        for label in safety.get("support_labels", []):
            summary["label_distribution"]["support"][str(label)] += 1
        for label in safety.get("region_risk_levels", []):
            summary["label_distribution"]["region"][str(label)] += 1
        summary["label_distribution"]["global"][str(safety.get("global_level", 0))] += 1

        record = {
            "index": index,
            "sample_id": sample_id,
            "filename": os.path.basename(path),
            "structural_state_id": structural_state_id,
            "excitation_id": excitation_id,
            "condition_group": condition_group,
            "temperature_steps_count": len(steps),
            "quality_keys": sorted(quality.keys()),
            "array_integrity_keys": sorted(integrity.keys()),
            "sha256": file_sha256(path),
        }
        summary["samples"].append(record)

    summary["unique_structural_state_ids"] = len(set(summary["structural_state_ids"]))
    summary["unique_excitation_ids"] = len(set(summary["excitation_ids"]))
    summary["unique_condition_groups"] = len(set(summary["condition_groups"]))
    summary["scenario_distribution"] = dict(
        Counter(
            (sample.get("structural_state", {}) or {}).get("scenario", "unknown")
            for _path, sample in samples
        )
    )
    summary["excitation_distribution"] = dict(Counter(summary["excitation_ids"]))
    summary["structural_state_distribution"] = dict(Counter(summary["structural_state_ids"]))
    summary["file_manifest_sha256"] = hashlib.sha256(
        "\n".join(f"{record['filename']}:{record['sha256']}" for record in summary["samples"]).encode("utf-8")
    ).hexdigest()

    summary_path = os.path.join(output_dir, "dataset_audit.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    txt_path = os.path.join(output_dir, "dataset_audit.txt")
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(f"data_dir: {summary['data_dir']}\n")
        file.write(f"sample_count: {summary['sample_count']}\n")
        file.write(f"unique_structural_state_ids: {summary['unique_structural_state_ids']}\n")
        file.write(f"unique_excitation_ids: {summary['unique_excitation_ids']}\n")
        file.write(f"unique_condition_groups: {summary['unique_condition_groups']}\n")
        file.write(f"data_version: {summary['data_version']}\n")
        file.write(f"file_manifest_sha256: {summary['file_manifest_sha256']}\n")
        file.write(f"scenario_distribution: {summary['scenario_distribution']}\n")
        file.write(f"excitation_distribution: {summary['excitation_distribution']}\n")
        file.write(f"temperature_step_counts: {dict(summary['temperature_steps'])}\n")
        file.write(f"quality_metric_keys: {dict(summary['quality_metric_keys'])}\n")
        file.write(f"array_integrity_keys: {dict(summary['array_integrity_keys'])}\n")

    print(f"saved: {summary_path}")
    print(f"saved: {txt_path}")
    print(f"samples: {summary['sample_count']}")


if __name__ == "__main__":
    main()
