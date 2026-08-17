"""阶段 0.3：按固定 split 仅置换训练集监督，保留验证/测试真实标签。

重要备注：默认 ``--supervision all`` 会把材料、支座、区域、全桥及相关回归真值
作为同一监督束置换。多任务训练时不能只打乱 material_labels，否则未打乱的辅助真值
仍可向模型提供原始结构状态。输出会生成哈希已更新的新 split manifest。
"""

import argparse
import os
import random
import shutil
from collections import defaultdict

import numpy as np

from stage0_common import (
    SUPERVISION_JSON_KEYS,
    deranged_donors,
    file_sha256,
    json_sha256,
    load_json,
    manifest_train_filenames,
    materialize_shared_assets,
    prepare_new_output_dir,
    replace_supervision,
    result_files,
    save_json,
)


def _input_fingerprint(sample):
    return json_sha256({key: value for key, value in sample.items() if key not in SUPERVISION_JSON_KEYS})


def _material_labels(sample):
    labels = (sample.get("safety_labels") or {}).get("material_labels")
    if labels is None:
        raise ValueError("sample missing safety_labels.material_labels")
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isin(values, (0, 1)).all():
        raise ValueError("invalid safety_labels.material_labels")
    return values


def _record_filename(record):
    return record.get("filename") or os.path.basename(record.get("path", ""))


def _sample_group_value(sample, group_by, filename):
    sample_index = sample.get("sample_index", {}) or {}
    structural_state = sample.get("structural_state", {}) or {}
    excitation = sample.get("dynamic_excitation", {}) or {}
    values = {
        "sample_id": sample_index.get("sample_id"),
        "structural_state_id": sample_index.get("structural_state_id", sample_index.get("group_id")),
        "state_family_id": sample_index.get("state_family_id", sample_index.get("condition_group")),
        "condition_group": sample_index.get("condition_group"),
        "excitation_id": excitation.get("excitation_id", structural_state.get("excitation_id")),
        "scenario": structural_state.get("scenario"),
    }
    value = values.get(group_by)
    if value in (None, ""):
        raise ValueError(f"{filename} missing required group field: {group_by}")
    return str(value)


def _manifest_groups(manifest, samples, train_names, group_by):
    records = {_record_filename(record): record for record in manifest.get("records", [])}
    groups = defaultdict(list)
    for name in train_names:
        record = records.get(name, {})
        value = None
        if record.get("group_by") == group_by:
            value = record.get("group_value")
        if value in (None, ""):
            value = record.get(group_by)
        if value in (None, ""):
            value = _sample_group_value(samples[name], group_by, name)
        groups[str(value)].append(name)
    return {key: sorted(value) for key, value in groups.items()}


def _supervision_payload(sample):
    return {key: sample.get(key) for key in SUPERVISION_JSON_KEYS}


def _validate_group_supervision(groups, samples):
    inconsistent = []
    canonical = {}
    for group, names in groups.items():
        first = names[0]
        signature = json_sha256(_supervision_payload(samples[first]))
        canonical[group] = first
        if any(json_sha256(_supervision_payload(samples[name])) != signature for name in names[1:]):
            inconsistent.append(group)
    if inconsistent:
        preview = ", ".join(inconsistent[:5])
        raise ValueError(f"同一结构状态内监督不一致，不能执行组级置换: {preview}")
    return canonical


def _label_correlation(original, shuffled):
    correlations = []
    for index in range(original.shape[1]):
        x = original[:, index]
        y = shuffled[:, index]
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return correlations


def main():
    parser = argparse.ArgumentParser(description="阶段0.3：只置换训练集监督的负控数据")
    parser.add_argument("input_dir", help="完整数据目录，不建议使用只有训练样本的 tiny-set")
    parser.add_argument("--split-manifest", required=True, help="正常实验使用的固定划分清单")
    parser.add_argument("--output", required=True, help="必须为空或不存在的新目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--supervision", choices=("all", "material_labels"), default="all")
    parser.add_argument("--group-by", default=None, help="置换单位；默认沿用 split manifest 的 group_by")
    parser.add_argument("--asset-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()

    if args.supervision == "material_labels":
        print("警告: material_labels 模式仅可配合关闭全部辅助损失的训练脚本使用。")

    files = result_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"no result_*.json found in {args.input_dir}")
    manifest, by_split = manifest_train_filenames(args.split_manifest)
    manifest_files = set().union(*by_split.values())
    if set(files) != manifest_files:
        missing = sorted(set(files) - manifest_files)[:5]
        extra = sorted(manifest_files - set(files))[:5]
        raise ValueError(f"数据目录与 split manifest 不一致: unassigned={missing}, missing_files={extra}")

    samples = {name: load_json(os.path.join(args.input_dir, name)) for name in files}
    train_names = sorted(by_split["train"])
    group_by = args.group_by or manifest.get("group_by") or "structural_state_id"
    train_groups = _manifest_groups(manifest, samples, train_names, group_by)
    if len(train_groups) < 2:
        raise ValueError("组级标签置换至少需要 2 个训练组")
    canonical_by_group = _validate_group_supervision(train_groups, samples)
    group_names = sorted(train_groups)
    rng = random.Random(args.seed)
    donor_indices = deranged_donors(len(group_names), rng)
    donor_by_group = {
        group: group_names[donor_indices[index]]
        for index, group in enumerate(group_names)
    }
    group_by_name = {
        name: group for group, names in train_groups.items() for name in names
    }
    output = prepare_new_output_dir(args.output)
    permutation_records = []
    changed_content = 0
    original_group_labels = []
    shuffled_group_labels = []

    for group in group_names:
        donor_group = donor_by_group[group]
        original_group_labels.append(_material_labels(samples[canonical_by_group[group]]))
        shuffled_group_labels.append(_material_labels(samples[canonical_by_group[donor_group]]))

    for name in files:
        source = os.path.join(args.input_dir, name)
        destination = os.path.join(output, name)
        if name not in by_split["train"]:
            shutil.copy2(source, destination)
            continue
        recipient_group = group_by_name[name]
        donor_group = donor_by_group[recipient_group]
        donor_name = canonical_by_group[donor_group]
        original = samples[name]
        changed = replace_supervision(original, samples[donor_name], mode=args.supervision)
        if _input_fingerprint(original) != _input_fingerprint(changed):
            raise AssertionError(f"置换意外修改了模型输入字段: {name}")
        save_json(destination, changed)
        original_supervision = {key: original.get(key) for key in SUPERVISION_JSON_KEYS}
        changed_supervision = {key: changed.get(key) for key in SUPERVISION_JSON_KEYS}
        content_changed = original_supervision != changed_supervision
        material_labels_changed = not np.array_equal(
            _material_labels(original), _material_labels(changed)
        )
        changed_content += int(content_changed)
        permutation_records.append({
            "recipient": name,
            "recipient_group": recipient_group,
            "donor": donor_name,
            "donor_group": donor_group,
            "content_changed": content_changed,
            "material_labels_changed": material_labels_changed,
            "input_fingerprint": _input_fingerprint(original),
        })

    assets = materialize_shared_assets(args.input_dir, output, list(samples.values()), mode=args.asset_mode)

    # main.py 默认核验样本哈希，因此必须为置换后的目录生成同一划分的新 manifest。
    new_manifest = dict(manifest)
    new_records = []
    for record in manifest.get("records", []):
        updated = dict(record)
        name = updated.get("filename") or os.path.basename(updated.get("path", ""))
        path = os.path.join(output, name)
        updated["path"] = os.path.abspath(path)
        updated["sha256"] = file_sha256(path)
        new_records.append(updated)
    new_manifest.update({
        "data_dir": output,
        "source_manifest": os.path.abspath(args.split_manifest),
        "negative_control_seed": args.seed,
        "supervision_mode": args.supervision,
        "negative_control_group_by": group_by,
        "negative_control_permutation_unit": "group",
        "records": new_records,
    })
    new_manifest.pop("manifest_sha256", None)
    new_manifest["manifest_sha256"] = json_sha256(new_manifest)
    output_manifest = os.path.join(output, "stage0_negative_split.json")
    save_json(output_manifest, new_manifest)

    original_group_labels = np.stack(original_group_labels)
    shuffled_group_labels = np.stack(shuffled_group_labels)
    per_label_match_rate = (original_group_labels == shuffled_group_labels).mean(axis=0)
    exact_label_match = np.all(original_group_labels == shuffled_group_labels, axis=1)
    copied_names = sorted(by_split["val"] | by_split["test"])
    byte_mismatches = [
        name for name in copied_names
        if file_sha256(os.path.join(args.input_dir, name)) != file_sha256(os.path.join(output, name))
    ]
    validation_and_test_byte_identical = not byte_mismatches
    if not validation_and_test_byte_identical:
        raise AssertionError(f"val/test 文件复制后哈希变化: {byte_mismatches[:5]}")

    audit = {
        "experiment": "stage0.3-permutation",
        "source_dir": os.path.abspath(args.input_dir),
        "output_dir": output,
        "source_manifest": os.path.abspath(args.split_manifest),
        "output_manifest": output_manifest,
        "seed": args.seed,
        "supervision_mode": args.supervision,
        "group_by": group_by,
        "permutation_unit": "group",
        "split_counts": {key: len(value) for key, value in by_split.items()},
        "train_group_count": len(group_names),
        "train_group_donor_derangement": all(group != donor_by_group[group] for group in group_names),
        "train_donor_derangement": all(
            item["recipient_group"] != item["donor_group"] for item in permutation_records
        ),
        "train_supervision_content_changed": changed_content,
        "train_material_label_changed_count": int(sum(
            item["material_labels_changed"] for item in permutation_records
        )),
        "group_material_label_exact_match_count": int(exact_label_match.sum()),
        "group_material_label_exact_match_rate": float(exact_label_match.mean()),
        "group_material_label_per_label_match_rate": per_label_match_rate.astype(float).tolist(),
        "group_material_label_per_label_correlation": _label_correlation(
            original_group_labels, shuffled_group_labels
        ),
        "validation_and_test_byte_identical": validation_and_test_byte_identical,
        "validation_and_test_hash_mismatches": byte_mismatches,
        "group_permutation": [
            {"recipient_group": group, "donor_group": donor_by_group[group]}
            for group in group_names
        ],
        "permutation": permutation_records,
        "shared_assets": assets,
        "notes": [
            "只有 train 结构状态组的监督被置换；同组全部激励共享同一 donor 监督。",
            "val/test JSON 使用 copy2，并逐文件验证 SHA-256 与原文件一致。",
            "相同类别组合可能导致 donor 组不同但材料标签相同，审计中记录匹配率和相关系数。",
            "训练时必须使用 output_manifest，确保划分不变且哈希校验通过。",
        ],
    }
    save_json(os.path.join(output, "stage0_permutation_audit.json"), audit)
    print(f"负控目录: {output}")
    print(
        f"train={len(train_names)}, groups={len(group_names)}, "
        f"内容变化={changed_content}, seed={args.seed}"
    )
    print(f"训练清单: {output_manifest}")


if __name__ == "__main__":
    main()
