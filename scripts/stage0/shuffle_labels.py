"""阶段 0.3：按固定 split 仅置换训练集监督，保留验证/测试真实标签。

重要备注：默认 ``--supervision all`` 会把材料、支座、区域、全桥及相关回归真值
作为同一监督束置换。多任务训练时不能只打乱 material_labels，否则未打乱的辅助真值
仍可向模型提供原始结构状态。输出会生成哈希已更新的新 split manifest。
"""

import argparse
import os
import random
import shutil

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


def main():
    parser = argparse.ArgumentParser(description="阶段0.3：只置换训练集监督的负控数据")
    parser.add_argument("input_dir", help="完整数据目录，不建议使用只有训练样本的 tiny-set")
    parser.add_argument("--split-manifest", required=True, help="正常实验使用的固定划分清单")
    parser.add_argument("--output", required=True, help="必须为空或不存在的新目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--supervision", choices=("all", "material_labels"), default="all")
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
    rng = random.Random(args.seed)
    donor_indices = deranged_donors(len(train_names), rng)
    donor_by_name = {name: train_names[donor_indices[index]] for index, name in enumerate(train_names)}
    output = prepare_new_output_dir(args.output)
    permutation_records = []
    changed_content = 0

    for name in files:
        source = os.path.join(args.input_dir, name)
        destination = os.path.join(output, name)
        if name not in by_split["train"]:
            shutil.copy2(source, destination)
            continue
        donor_name = donor_by_name[name]
        original = samples[name]
        changed = replace_supervision(original, samples[donor_name], mode=args.supervision)
        if _input_fingerprint(original) != _input_fingerprint(changed):
            raise AssertionError(f"置换意外修改了模型输入字段: {name}")
        save_json(destination, changed)
        original_supervision = {key: original.get(key) for key in SUPERVISION_JSON_KEYS}
        changed_supervision = {key: changed.get(key) for key in SUPERVISION_JSON_KEYS}
        content_changed = original_supervision != changed_supervision
        changed_content += int(content_changed)
        permutation_records.append({
            "recipient": name,
            "donor": donor_name,
            "content_changed": content_changed,
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
        "records": new_records,
    })
    new_manifest.pop("manifest_sha256", None)
    new_manifest["manifest_sha256"] = json_sha256(new_manifest)
    output_manifest = os.path.join(output, "stage0_negative_split.json")
    save_json(output_manifest, new_manifest)

    audit = {
        "experiment": "stage0.3-permutation",
        "source_dir": os.path.abspath(args.input_dir),
        "output_dir": output,
        "source_manifest": os.path.abspath(args.split_manifest),
        "output_manifest": output_manifest,
        "seed": args.seed,
        "supervision_mode": args.supervision,
        "split_counts": {key: len(value) for key, value in by_split.items()},
        "train_donor_derangement": all(item["recipient"] != item["donor"] for item in permutation_records),
        "train_supervision_content_changed": changed_content,
        "validation_and_test_byte_identical": True,
        "permutation": permutation_records,
        "shared_assets": assets,
        "notes": [
            "只有 train 样本的监督被置换；val/test JSON 使用 copy2 保持原始真值。",
            "相同类别组合可能导致 donor 不同但监督内容相同，审计中单独记录数量。",
            "训练时必须使用 output_manifest，确保划分不变且哈希校验通过。",
        ],
    }
    save_json(os.path.join(output, "stage0_permutation_audit.json"), audit)
    print(f"负控目录: {output}")
    print(f"train={len(train_names)}, 内容变化={changed_content}, seed={args.seed}")
    print(f"训练清单: {output_manifest}")


if __name__ == "__main__":
    main()
