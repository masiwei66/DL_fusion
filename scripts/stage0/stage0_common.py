"""阶段 0 实验的共享工具。

这里集中维护输入/监督边界、文件哈希、固定划分读取和小样本目录依赖复制。
阶段 0 的脚本应复用这些定义，避免各脚本对“输入”和“真值”采用不同口径。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from datetime import datetime


RESULT_PREFIX = "result_"
RESULT_SUFFIX = ".json"

# 只允许这些字段传入模型 forward。监督字段即使存在于 batch，也必须先剥离。
MODEL_INPUT_KEYS = {
    "disp",
    "strain",
    "ace",
    "condition",
    "temperature_C",
    "temperature_steps_C",
    "quality_metrics",
    "static_mask",
    "dynamic_mask",
    "temperature_mask",
    "static_pos",
    "dynamic_pos",
}

TARGET_BATCH_KEYS = {
    "target",
    "raw_sf",
    "support_disp",
    "support_target",
    "region_target",
    "global_target",
}

# 完整负控必须把同一 donor 的这些监督字段作为一个整体置换。
# 它们不属于模型输入；responses、坐标、温度和质量字段始终保留在 recipient 样本中。
SUPERVISION_JSON_KEYS = (
    "safety_labels",
    "material_scaling_factors",
    "scaling_factors",
    "support_settlement",
)


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_sha256(payload):
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def result_files(data_dir):
    return sorted(
        name
        for name in os.listdir(data_dir)
        if name.startswith(RESULT_PREFIX) and name.endswith(RESULT_SUFFIX)
    )


def prepare_new_output_dir(path):
    """创建新的实验目录；拒绝复用非空目录，避免旧样本污染结果。"""
    absolute = os.path.abspath(path)
    if os.path.isdir(absolute) and os.listdir(absolute):
        raise FileExistsError(
            f"输出目录非空，拒绝混入旧实验文件: {absolute}。请使用新的输出目录。"
        )
    if os.path.exists(absolute) and not os.path.isdir(absolute):
        raise FileExistsError(f"输出路径不是目录: {absolute}")
    os.makedirs(absolute, exist_ok=True)
    return absolute


def copy_or_link(source, destination, mode="hardlink"):
    """复制依赖文件；hardlink 失败时回退到 copy，避免共享 HDF5 被重复占用空间。"""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def materialize_shared_assets(source_dir, output_dir, samples, mode="hardlink"):
    """复制 manifest，并为紧凑 JSON 的相对 array_store 创建依赖。"""
    assets = []
    manifest = os.path.join(source_dir, "dataset_manifest.json")
    if os.path.isfile(manifest):
        destination = os.path.join(output_dir, "dataset_manifest.json")
        shutil.copy2(manifest, destination)
        assets.append({"path": "dataset_manifest.json", "mode": "copy"})

    relative_files = set()
    for sample in samples:
        store_file = (sample.get("array_store") or {}).get("file")
        if store_file and not os.path.isabs(store_file):
            relative_files.add(os.path.normpath(store_file))

    for relative in sorted(relative_files):
        source = os.path.abspath(os.path.join(source_dir, relative))
        if os.path.commonpath([os.path.abspath(source_dir), source]) != os.path.abspath(source_dir):
            raise ValueError(f"array_store 相对路径越出数据目录: {relative}")
        if not os.path.isfile(source):
            raise FileNotFoundError(f"array_store 依赖不存在: {source}")
        destination = os.path.join(output_dir, relative)
        used_mode = copy_or_link(source, destination, mode=mode)
        assets.append({"path": relative, "mode": used_mode})
    return assets


def manifest_train_filenames(manifest_path):
    """返回 train 文件名集合，并验证 manifest 三个集合互斥且均非空。"""
    manifest = load_json(manifest_path)
    by_split = {name: set() for name in ("train", "val", "test")}
    for record in manifest.get("records", []):
        split = record.get("split")
        if split not in by_split:
            continue
        filename = record.get("filename")
        if not filename and record.get("path"):
            filename = os.path.basename(record["path"])
        if not filename:
            raise ValueError("split manifest 记录缺少 filename/path")
        if filename in set().union(*by_split.values()):
            raise ValueError(f"split manifest 中样本重复分配: {filename}")
        by_split[split].add(filename)

    missing = [name for name, values in by_split.items() if not values]
    if missing:
        raise ValueError(f"split manifest 存在空集合: {missing}")
    return manifest, by_split


def deranged_donors(size, rng):
    """生成无固定点的置换，确保每个训练样本确实获得另一条监督。"""
    if size < 2:
        raise ValueError("标签置换至少需要 2 个训练样本")
    donors = list(range(size))
    for _ in range(100):
        rng.shuffle(donors)
        if all(index != donor for index, donor in enumerate(donors)):
            return donors
    shift = rng.randrange(1, size)
    return [(index + shift) % size for index in range(size)]


def replace_supervision(recipient, donor, mode="all"):
    """返回置换监督后的样本，不修改输入响应。

    ``all`` 是多任务训练的安全默认值。``material_labels`` 只适用于显式关闭所有
    辅助损失的材料分类负控，否则未置换的辅助真值会泄露原监督关系。
    """
    output = deepcopy(recipient)
    if mode == "all":
        for key in SUPERVISION_JSON_KEYS:
            if key in donor:
                output[key] = deepcopy(donor[key])
            else:
                output.pop(key, None)
        return output
    if mode == "material_labels":
        donor_safety = donor.get("safety_labels") or {}
        if "material_labels" not in donor_safety:
            raise ValueError("donor 样本缺少 safety_labels.material_labels")
        safety = deepcopy(output.get("safety_labels") or {})
        safety["material_labels"] = deepcopy(donor_safety["material_labels"])
        output["safety_labels"] = safety
        return output
    raise ValueError(f"未知监督置换模式: {mode}")


def sanitized_model_input(batch):
    return {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS}


def report_envelope(experiment, passed, details, notes=None):
    return {
        "experiment": experiment,
        "passed": bool(passed),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "details": details,
        "notes": list(notes or []),
    }
