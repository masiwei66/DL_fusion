"""为 V2 JSON 数据集生成样本级划分清单。"""

import argparse
import hashlib
import json
import os
from collections import defaultdict

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
        "sample_id": sample_index.get("sample_id", fallback),
        "structural_state_id": sample_index.get("structural_state_id", sample_index.get("group_id", fallback)),
        "state_family_id": sample_index.get("state_family_id", sample_index.get("condition_group", fallback)),
        "condition_group": sample_index.get("condition_group", fallback),
        "excitation_id": excitation.get("excitation_id", structural_state.get("excitation_id", fallback)),
        "scenario": structural_state.get("scenario", fallback),
        "file_block": (
            str(int(os.path.splitext(fallback)[0].split("_")[-1]) // 50)
            if fallback.startswith("result_") and os.path.splitext(fallback)[0].split("_")[-1].isdigit()
            else fallback
        ),
    }
    if group_by not in mapping:
        raise ValueError(f"unknown group_by: {group_by}")
    return str(mapping[group_by])


def greedy_group_split(groups, ratios, seed):
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

    for _, indices in items:
        split_id = min(
            range(3),
            key=lambda i: (
                counts[i] / max(1, desired[i]),
                counts[i],
                i,
            ),
        )
        splits[split_id].extend(indices)
        counts[split_id] += len(indices)

    return splits, desired.tolist(), counts


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
    for index, (path, sample) in enumerate(samples):
        fallback = os.path.basename(path)
        value = group_value(sample, args.group_by, fallback)
        groups[value].append(index)
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

    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    splits, desired, counts = greedy_group_split(groups, ratios, args.seed)
    split_names = ("train", "val", "test")
    assigned = {}
    for split_name, indices in zip(split_names, splits):
        for index in indices:
            assigned[index] = split_name

    for record in records:
        record["split"] = assigned[record["index"]]

    payload = {
        "data_dir": os.path.abspath(args.data_dir),
        "group_by": args.group_by,
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": ratios[2]},
        "desired_counts": {"train": desired[0], "val": desired[1], "test": desired[2]},
        "actual_counts": {"train": len(splits[0]), "val": len(splits[1]), "test": len(splits[2])},
        "group_count": len(groups),
        "sample_count": len(records),
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


if __name__ == "__main__":
    main()
