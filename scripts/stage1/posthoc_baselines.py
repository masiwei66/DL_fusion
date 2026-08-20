"""阶段 1 后处理基线：固定分数融合与任务路由。

该脚本只使用已经保存的验证/测试概率：阈值在验证集选择，随后一次性评估测试集。
它不重新训练模型，因此可以低成本补齐阶段 1 的简单融合对照。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from stage1.stage1_common import (  # noqa: E402
    aggregate_seed_metrics,
    find_best_thresholds,
    multilabel_metrics,
    save_json,
)


def _load(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _index(records):
    return {str(row["sample_id"]): row for row in records}


def _material_arrays(records, ids):
    true = np.asarray([[row["material_true"][i] for i in ids] for row in records], dtype=np.float32)
    prob = np.asarray([[row["material_prob"][i] for i in ids] for row in records], dtype=np.float32)
    return true, prob


def _metrics_from_records(records, material_prob, thresholds, safety_records=None):
    ids = list(records[0]["material_true"].keys())
    true, _ = _material_arrays(records, ids)
    pred = (material_prob >= np.asarray(thresholds).reshape(1, -1)).astype(np.float32)
    result = multilabel_metrics(true, material_prob, y_pred=pred)
    # 路由基线的辅助任务来自融合模型，保证比较的是同一套安全输出。
    safety_records = safety_records or records
    if "support_true" in safety_records[0]:
        support_ids = list(safety_records[0]["support_true"].keys())
        support_true = np.asarray([[r["support_true"][i] for i in support_ids] for r in safety_records], dtype=np.float32)
        support_prob = np.asarray([[r["support_prob"][i] for i in support_ids] for r in safety_records], dtype=np.float32)
        support_pred = np.asarray([[r["support_pred"][i] for i in support_ids] for r in safety_records], dtype=np.float32)
        support_metrics = multilabel_metrics(support_true, support_prob, y_pred=support_pred)
        result.update({f"support_{k}": support_metrics[k] for k in ("macro_auprc", "macro_f1", "micro_f1")})
        true_disp = np.asarray([[r["support_disp_true_mm"][i] for i in support_ids] for r in safety_records], dtype=np.float32)
        pred_disp = np.asarray([[r["support_disp_pred_mm"][i] for i in support_ids] for r in safety_records], dtype=np.float32)
        result["support_disp_mae_mm"] = float(np.abs(true_disp - pred_disp).mean())
    if "region_true" in safety_records[0]:
        from sklearn.metrics import f1_score
        region_ids = list(safety_records[0]["region_true"].keys())
        rt = np.asarray([[r["region_true"][i] for i in region_ids] for r in safety_records])
        rp = np.asarray([[r["region_pred"][i] for i in region_ids] for r in safety_records])
        result["region_macro_f1"] = float(f1_score(rt.reshape(-1), rp.reshape(-1), average="macro", zero_division=0))
    if "global_true" in safety_records[0]:
        gt = np.asarray([r["global_true"] for r in safety_records])
        gp = np.asarray([r["global_pred"] for r in safety_records])
        from sklearn.metrics import f1_score
        result["global_macro_f1"] = float(f1_score(gt, gp, average="macro", zero_division=0))
        result["global_mae"] = float(np.abs(gt - gp).mean())
    return result


def run_seed(seed_dir, output_dir, test_seed_dir=None):
    seed_name = Path(seed_dir).name.removeprefix("seed_")
    test_seed_dir = test_seed_dir or seed_dir
    paths = {
        (model, split): Path(seed_dir) / "logs" / model / f"{model}_{split}_predictions.json"
        for model in ("static_only", "dynamic_only", "fusion")
        for split in ("val", "test")
    }
    # 兼容手工放置的预测文件名；测试文件是正式阶段 1 默认产物。
    for model in ("static_only", "dynamic_only", "fusion"):
        paths[model, "test"] = Path(test_seed_dir) / "logs" / model / f"{model}_test_predictions.json"
    if not all(paths[model, "val"].is_file() for model in ("static_only", "dynamic_only", "fusion")):
        raise FileNotFoundError(f"missing validation predictions under {seed_dir}; run main.py --mode test --eval-split val first")
    payloads = {(m, s): _load(paths[m, s]) for m in ("static_only", "dynamic_only", "fusion") for s in ("val", "test")}
    val = {m: payloads[m, "val"]["records"] for m in ("static_only", "dynamic_only", "fusion")}
    test = {m: payloads[m, "test"]["records"] for m in ("static_only", "dynamic_only", "fusion")}
    by_val = {m: _index(val[m]) for m in ("static_only", "dynamic_only", "fusion")}
    by_test = {m: _index(test[m]) for m in ("static_only", "dynamic_only", "fusion")}
    ids_val = sorted(set.intersection(*(set(by_val[m]) for m in by_val)))
    ids_test = sorted(set.intersection(*(set(by_test[m]) for m in by_test)))
    val_rows = {m: [by_val[m][i] for i in ids_val] for m in by_val}
    test_rows = {m: [by_test[m][i] for i in ids_test] for m in by_test}
    label_ids = list(val_rows["fusion"][0]["material_true"].keys())

    results = []
    # 固定 0.5 分数融合：每个材料概率为 static/dynamic 的算术平均。
    val_static_true, val_static_prob = _material_arrays(val_rows["static_only"], label_ids)
    val_dynamic_true, val_dynamic_prob = _material_arrays(val_rows["dynamic_only"], label_ids)
    test_static_true, test_static_prob = _material_arrays(test_rows["static_only"], label_ids)
    test_dynamic_true, test_dynamic_prob = _material_arrays(test_rows["dynamic_only"], label_ids)
    blend_val_prob = 0.5 * val_static_prob + 0.5 * val_dynamic_prob
    blend_test_prob = 0.5 * test_static_prob + 0.5 * test_dynamic_prob
    thresholds = find_best_thresholds(val_static_true, blend_val_prob)
    blend_metrics = _metrics_from_records(test_rows["fusion"], blend_test_prob, thresholds, safety_records=test_rows["fusion"])
    results.append({"name": "posthoc_fixed_score_fusion", "seed": int(seed_name), "thresholds": thresholds.tolist(), **blend_metrics})

    # 任务路由：材料使用 dynamic-only；支座/区域/全桥使用 fusion 的输出。
    dyn_thresholds = find_best_thresholds(val_dynamic_true, val_dynamic_prob)
    route_metrics = _metrics_from_records(test_rows["dynamic_only"], test_dynamic_prob, dyn_thresholds, safety_records=test_rows["fusion"])
    results.append({"name": "posthoc_task_routed_dynamic_material_fusion_safety", "seed": int(seed_name), "thresholds": dyn_thresholds.tolist(), **route_metrics})
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute stage 1 post-hoc fusion baselines.")
    parser.add_argument("stage1_dir")
    parser.add_argument(
        "--test-stage1-dir", default=None,
        help="测试预测所在的另一份 stage1 目录；用于 val 导出目录与只读正式报告分离",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = os.path.abspath(args.output or os.path.join(args.stage1_dir, "posthoc_baselines.json"))
    rows = []
    for seed_dir in sorted(Path(args.stage1_dir).glob("deep/seed_*")):
        test_seed_dir = None
        if args.test_stage1_dir:
            test_seed_dir = Path(args.test_stage1_dir) / "deep" / seed_dir.name
        rows.extend(run_seed(str(seed_dir), os.path.dirname(output), test_seed_dir=str(test_seed_dir) if test_seed_dir else None))
    methods = {}
    for row in rows:
        clean = {k: v for k, v in row.items() if k != "thresholds"}
        methods.setdefault(row["name"], []).append(clean)
    payload = {
        "experiment": "stage1.posthoc",
        "source_dir": os.path.abspath(args.stage1_dir),
        "methods": {
            name: {"runs": values, "aggregate": aggregate_seed_metrics(values)}
            for name, values in sorted(methods.items())
        },
    }
    save_json(output, payload)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
