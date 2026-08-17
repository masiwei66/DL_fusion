"""为 V2 JSON 数据集生成分组隔离、多标签平衡的划分清单。"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np


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


def group_value(sample, group_by, fallback):
    sample_index = sample.get("sample_index", {}) or {}
    structural_state = sample.get("structural_state", {}) or {}
    excitation = sample.get("dynamic_excitation", {}) or {}
    mapping = {
        "sample_id": sample_index.get("sample_id"),
        "structural_state_id": sample_index.get("structural_state_id", sample_index.get("group_id")),
        "state_family_id": sample_index.get("state_family_id", sample_index.get("condition_group")),
        "condition_group": sample_index.get("condition_group"),
        "excitation_id": excitation.get("excitation_id", structural_state.get("excitation_id")),
        "scenario": structural_state.get("scenario"),
        "file_block": (
            str(int(os.path.splitext(fallback)[0].split("_")[-1]) // 50)
            if fallback.startswith("result_") and os.path.splitext(fallback)[0].split("_")[-1].isdigit()
            else fallback
        ),
    }
    if group_by not in mapping:
        raise ValueError(f"unknown group_by: {group_by}")
    value = mapping[group_by]
    if value in (None, ""):
        raise ValueError(f"{fallback} missing required group field: {group_by}")
    return str(value)


def _material_labels(sample, filename):
    labels = (sample.get("safety_labels") or {}).get("material_labels")
    if labels is None:
        raise ValueError(f"{filename} missing safety_labels.material_labels")
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{filename} has invalid material_labels")
    return values


def _ensure_unique_record_ids(records):
    source_ids = [
        "" if record.get("sample_id") in (None, "") else str(record["sample_id"])
        for record in records
    ]
    counts = Counter(source_ids)
    used = set()
    for record, source_id in zip(records, source_ids):
        record["source_sample_id"] = source_id
        candidate = source_id
        if not source_id or counts[source_id] > 1:
            excitation_id = record.get("excitation_id")
            suffix = str(excitation_id) if excitation_id not in (None, "") else ""
            if suffix:
                candidate = f"{source_id or 'sample'}__{suffix}"
            if not suffix or candidate in used:
                stem = os.path.splitext(record["filename"])[0]
                candidate = f"{source_id or 'sample'}__{stem}"
        if candidate in used:
            stem = os.path.splitext(record["filename"])[0]
            candidate = f"{source_id or 'sample'}__{stem}"
        if candidate in used:
            raise ValueError(f"cannot construct a unique sample_id for {record['filename']}")
        record["sample_id"] = candidate
        used.add(candidate)


def greedy_group_split(groups, ratios, seed, sample_targets):
    targets = np.asarray(ratios, dtype=np.float64)
    targets = targets / targets.sum()
    rng = np.random.default_rng(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    total = sum(len(indices) for _, indices in items)
    desired = np.rint(targets * total).astype(int)
    desired[-1] = total - desired[:-1].sum()
    splits = [[], [], []]
    counts = [0, 0, 0]
    label_counts = [np.zeros(sample_targets.shape[1], dtype=np.float64) for _ in range(3)]
    total_pos = sample_targets.sum(axis=0)
    label_targets = [total_pos * ratio for ratio in targets]

    for _, indices in items:
        group_pos = sample_targets[indices].sum(axis=0)

        def assignment_cost(split_id):
            size_after = counts[split_id] + len(indices)
            size_cost = abs(size_after - desired[split_id]) / max(1, desired[split_id])
            pos_after = label_counts[split_id] + group_pos
            label_denom = np.maximum(label_targets[split_id], 1.0)
            label_cost = np.mean(np.abs(pos_after - label_targets[split_id]) / label_denom)
            overshoot = max(0, size_after - desired[split_id]) / max(1, desired[split_id])
            return size_cost + 0.75 * label_cost + 0.25 * overshoot

        split_id = min(range(3), key=lambda i: (assignment_cost(i), counts[i], i))
        splits[split_id].extend(indices)
        counts[split_id] += len(indices)
        label_counts[split_id] += group_pos

    return splits, desired.tolist(), counts, label_counts


def main():
    parser = argparse.ArgumentParser(description="Generate a sample-level split manifest.")
    parser.add_argument("data_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-by", default="structural_state_id")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = list(iter_samples(args.data_dir))
    if not samples:
        raise FileNotFoundError(f"no result_*.json found in {args.data_dir}")

    groups = defaultdict(list)
    records = []
    targets = []
    for index, (path, sample) in enumerate(samples):
        fallback = os.path.basename(path)
        value = group_value(sample, args.group_by, fallback)
        groups[value].append(index)
        targets.append(_material_labels(sample, fallback))
        sample_index = sample.get("sample_index", {}) or {}
        structural_state = sample.get("structural_state", {}) or {}
        excitation = sample.get("dynamic_excitation", {}) or {}
        records.append({
            "index": index,
            "sample_id": sample_index.get("sample_id", fallback),
            "filename": fallback,
            "path": os.path.abspath(path),
            "group_by": args.group_by,
            "group_value": value,
            "structural_state_id": sample_index.get("structural_state_id", sample_index.get("group_id")),
            "state_family_id": sample_index.get("state_family_id", sample_index.get("condition_group")),
            "condition_group": sample_index.get("condition_group"),
            "excitation_id": excitation.get("excitation_id", structural_state.get("excitation_id")),
            "sha256": file_sha256(path),
        })

    if len(groups) < 3:
        raise ValueError("group split requires at least 3 distinct groups")
    _ensure_unique_record_ids(records)
    sample_targets = np.stack(targets)

    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    if min(ratios) <= 0:
        raise ValueError("train/val/test ratios must all be positive")
    splits, desired, counts, label_counts = greedy_group_split(
        groups, ratios, args.seed, sample_targets
    )
    split_names = ("train", "val", "test")
    assigned = {}
    for split_name, indices in zip(split_names, splits):
        for index in indices:
            assigned[index] = split_name

    for record in records:
        record["split"] = assigned[record["index"]]

    split_groups = {
        name: {records[index]["group_value"] for index in indices}
        for name, indices in zip(split_names, splits)
    }
    overlaps = {
        "train_val": sorted(split_groups["train"] & split_groups["val"]),
        "train_test": sorted(split_groups["train"] & split_groups["test"]),
        "val_test": sorted(split_groups["val"] & split_groups["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"group leakage detected: {overlaps}")

    label_count_payload = {
        name: values.astype(int).tolist()
        for name, values in zip(split_names, label_counts)
    }
    label_prevalence = {
        name: (values / max(1, len(indices))).astype(float).tolist()
        for name, values, indices in zip(split_names, label_counts, splits)
    }

    payload = {
        "data_dir": os.path.abspath(args.data_dir),
        "group_by": args.group_by,
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": ratios[2]},
        "desired_counts": {"train": desired[0], "val": desired[1], "test": desired[2]},
        "actual_counts": {"train": len(splits[0]), "val": len(splits[1]), "test": len(splits[2])},
        "group_count": len(groups),
        "split_group_counts": {name: len(split_groups[name]) for name in split_names},
        "split_group_overlap": overlaps,
        "sample_count": len(records),
        "source_sample_id_duplicate_count": sum(
            count - 1 for count in Counter(record["source_sample_id"] for record in records).values()
            if count > 1
        ),
        "effective_sample_ids_unique": len({record["sample_id"] for record in records}) == len(records),
        "material_label_counts": label_count_payload,
        "material_label_prevalence": label_prevalence,
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "records": records,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"saved: {args.output}")
    print(f"groups: {len(groups)}")
    print(f"counts: {payload['actual_counts']}")
    print(f"material prevalence: {payload['material_label_prevalence']}")


if __name__ == "__main__":
    main()
