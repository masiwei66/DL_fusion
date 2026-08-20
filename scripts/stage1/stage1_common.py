"""阶段 1 共用的校验、指标与汇总工具。

阶段 1 刻意把编排逻辑放在模型代码之外，让"划分清单 + 预测文件"成为唯一
事实来源，保证所有基线（先验 / 传统 / 深度）都在完全相同的样本上可比。
任何基线都不能私自改划分或换数据。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


# 论文统一随机种子：13, 29, 42, 71, 101（研究计划第 1 节规定，5 种子报告均值±标准差）。
DEFAULT_SEEDS = (13, 29, 42, 71, 101)
SPLIT_NAMES = ("train", "val", "test")


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload):
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain one or more unique integers")
    return seeds


def result_files(data_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(data_dir)
        if name.startswith("result_") and name.endswith(".json")
    )


def validate_data_and_manifest(data_dir: str, manifest_path: str):
    """核验划分清单与 data_dir 里的文件严格对应，返回（manifest, 划分名→文件索引）。

    校验是硬性的：文件缺失 / 哈希变化 / 清单外多余文件 / 空划分 / 组重叠，
    任一不满足直接抛错。因为划分冻结后只要有一个文件被改过，
    阶段 1 的结果就不可比，必须显式失败而不是默默放行。
    """

    data_dir = os.path.abspath(data_dir)
    manifest_path = os.path.abspath(manifest_path)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"split manifest does not exist: {manifest_path}")

    files = result_files(data_dir)
    if not files:
        raise FileNotFoundError(f"no result_*.json files found in {data_dir}")
    manifest = load_json(manifest_path)
    records = manifest.get("records", [])
    if not records:
        raise ValueError("split manifest has no records")

    by_name = {name: index for index, name in enumerate(files)}
    seen = set()
    splits = {name: [] for name in SPLIT_NAMES}
    missing = []
    changed = []
    for record in records:
        name = record.get("filename") or os.path.basename(record.get("path", ""))
        split = record.get("split")
        if name not in by_name:
            missing.append(name)
            continue
        if split not in splits:
            raise ValueError(f"invalid split {split!r} for {name}")
        if name in seen:
            raise ValueError(f"duplicate manifest record: {name}")
        seen.add(name)
        actual_hash = file_sha256(os.path.join(data_dir, name))
        expected_hash = record.get("sha256")
        if expected_hash and actual_hash != expected_hash:
            changed.append({"filename": name, "expected": expected_hash, "actual": actual_hash})
        splits[split].append(by_name[name])

    if missing:
        raise ValueError(f"manifest files missing from data directory: {missing[:5]}")
    if changed:
        preview = ", ".join(item["filename"] for item in changed[:5])
        raise ValueError(f"manifest SHA-256 mismatch for {len(changed)} files: {preview}")
    if seen != set(files):
        extra = sorted(set(files) - seen)
        raise ValueError(f"data directory contains files absent from manifest: {extra[:5]}")
    if any(not splits[name] for name in SPLIT_NAMES):
        raise ValueError(f"manifest has an empty split: {splits}")

    split_groups = {name: set() for name in SPLIT_NAMES}
    for record in records:
        split_groups[record["split"]].add(
            str(record.get("group_value") or record.get("structural_state_id") or "")
        )
    overlaps = {
        "train_val": sorted(split_groups["train"] & split_groups["val"]),
        "train_test": sorted(split_groups["train"] & split_groups["test"]),
        "val_test": sorted(split_groups["val"] & split_groups["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"split groups overlap: {overlaps}")

    return manifest, splits


def _require_sklearn():
    try:
        from sklearn.metrics import average_precision_score, f1_score
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("stage 1 metrics require scikit-learn") from exc
    return average_precision_score, f1_score


def multilabel_metrics(y_true, y_prob, threshold=0.5, y_pred=None):
    """计算材料多标签通用指标：宏/微 F1、Exact Match、宏 AUPRC 与逐标签 AUPRC。

    某个标签在数据里只有一种取值时无法算 AUPRC，记为 NaN 并从宏平均中剔除，
    避免缺类标签把整体指标拖成 0 而失真。
    """

    average_precision_score, f1_score = _require_sklearn()
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(np.float32)
    else:
        y_pred = np.asarray(y_pred, dtype=np.float32)
    per_label = []
    for index in range(y_true.shape[1]):
        if len(np.unique(y_true[:, index])) < 2:
            per_label.append(float("nan"))
        else:
            per_label.append(float(average_precision_score(y_true[:, index], y_prob[:, index])))
    valid_ap = [value for value in per_label if np.isfinite(value)]
    return {
        "sample_count": int(y_true.shape[0]),
        "macro_auprc": float(np.mean(valid_ap)) if valid_ap else 0.0,
        "per_label_auprc": per_label,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(np.all(y_true == y_pred, axis=1).mean()),
        "positive_prevalence": y_true.mean(axis=0).astype(float).tolist(),
    }


def find_best_thresholds(y_true, y_prob, grid=None):
    """只用验证集，为每个标签在 0.02–0.95 网格上挑选 F1 最优阈值。

    纪律要求：阈值必须在看到测试结果之前固定；禁止用测试集调阈值。
    """

    _average_precision_score, f1_score = _require_sklearn()
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    grid = np.linspace(0.02, 0.95, 94) if grid is None else np.asarray(grid)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for label_index in range(y_true.shape[1]):
        best_score = -1.0
        for threshold in grid:
            prediction = (y_prob[:, label_index] >= threshold).astype(np.float32)
            score = f1_score(y_true[:, label_index], prediction, zero_division=0)
            if score > best_score:
                best_score = score
                thresholds[label_index] = float(threshold)
    return thresholds


def aggregate_seed_metrics(rows: Iterable[dict]) -> dict:
    """跨种子聚合标量指标（均值 / 标准差 / 最小最大 / 各值列表）。

    缺失值保留为 None 而不是填 0，避免"某些种子没跑出来"被悄悄掩盖。
    """

    rows = list(rows)
    if not rows:
        return {"seed_count": 0, "metrics": {}}
    scalar_names = (
        "macro_auprc", "macro_f1", "micro_f1", "exact_match",
        "support_macro_auprc", "support_macro_f1", "support_micro_f1",
        "support_disp_mae_mm", "region_macro_f1", "global_macro_f1", "global_mae",
    )
    summary = {}
    for name in scalar_names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
            "values": values,
        }
    return {"seed_count": len(rows), "metrics": summary}


def prediction_records_metrics_from_records(records: list[dict]) -> dict:
    """从 trainer 导出的预测 records 计算各任务指标。

    预测记录包含材料、支座、区域和全桥任务的真值与预测值；该函数只依赖
    records 本身，便于结构状态级 bootstrap 在内存中重复计算。
    """
    if not records:
        raise ValueError("prediction records are empty")
    label_ids = list(records[0]["material_true"].keys())
    y_true = np.asarray(
        [[record["material_true"][label] for label in label_ids] for record in records],
        dtype=np.float32,
    )
    y_prob = np.asarray(
        [[record["material_prob"][label] for label in label_ids] for record in records],
        dtype=np.float32,
    )
    y_pred = np.asarray(
        [[record["material_pred"][label] for label in label_ids] for record in records],
        dtype=np.float32,
    )
    result = multilabel_metrics(y_true, y_prob, y_pred=y_pred)
    if "support_true" in records[0]:
        support_ids = list(records[0]["support_true"].keys())
        support_true = np.asarray(
            [[record["support_true"][label] for label in support_ids] for record in records],
            dtype=np.float32,
        )
        support_prob = np.asarray(
            [[record["support_prob"][label] for label in support_ids] for record in records],
            dtype=np.float32,
        )
        support_pred = np.asarray(
            [[record["support_pred"][label] for label in support_ids] for record in records],
            dtype=np.float32,
        )
        support_metrics = multilabel_metrics(support_true, support_prob, y_pred=support_pred)
        for key in ("macro_auprc", "macro_f1", "micro_f1"):
            result[f"support_{key}"] = support_metrics[key]
        true_disp = np.asarray(
            [[record["support_disp_true_mm"][label] for label in support_ids] for record in records],
            dtype=np.float32,
        )
        pred_disp = np.asarray(
            [[record["support_disp_pred_mm"][label] for label in support_ids] for record in records],
            dtype=np.float32,
        )
        result["support_disp_mae_mm"] = float(np.abs(true_disp - pred_disp).mean())
    _average_precision_score, f1_score = _require_sklearn()
    if "region_true" in records[0]:
        region_ids = list(records[0]["region_true"].keys())
        region_true = np.asarray(
            [[record["region_true"][label] for label in region_ids] for record in records],
            dtype=np.int64,
        )
        region_pred = np.asarray(
            [[record["region_pred"][label] for label in region_ids] for record in records],
            dtype=np.int64,
        )
        result["region_macro_f1"] = float(
            f1_score(region_true.reshape(-1), region_pred.reshape(-1), average="macro", zero_division=0)
        )
    if "global_true" in records[0]:
        global_true = np.asarray([record["global_true"] for record in records], dtype=np.int64)
        global_pred = np.asarray([record["global_pred"] for record in records], dtype=np.int64)
        result["global_macro_f1"] = float(
            f1_score(global_true, global_pred, average="macro", zero_division=0)
        )
        result["global_mae"] = float(np.abs(global_true - global_pred).mean())
    result["label_ids"] = [int(label) if str(label).isdigit() else str(label) for label in label_ids]
    return result


def prediction_records_metrics(prediction_path: str) -> dict:
    """从模型预测文件解析并计算各任务指标。"""
    payload = load_json(prediction_path)
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"prediction file has no records: {prediction_path}")
    result = prediction_records_metrics_from_records(records)
    result["prediction_path"] = os.path.abspath(prediction_path)
    return result


def find_prediction_file(run_dir: str, model_type: str) -> str | None:
    """在 run 目录里定位某模型类型的测试预测文件（兼容新旧两种目录布局）。"""
    candidates = [
        Path(run_dir) / "logs" / model_type / f"{model_type}_test_predictions.json",
        Path(run_dir) / "logs" / f"{model_type}_test_predictions.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None
