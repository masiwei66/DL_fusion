"""跨种子汇总阶段 1 的基线 / 深度结果 JSON，生成 stage1_results.json。

stage1_results.json 是"论文结果唯一来源"：每张结果表、每张图都应从这里取数。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from stage1.stage1_common import (  # noqa: E402
    aggregate_seed_metrics,
    find_prediction_file,
    load_json,
    prediction_records_metrics,
    save_json,
)


def _add_result(results, name, seed, metrics, source):
    """把一次运行（某方法 × 某 seed 的测试指标）收进结果表。

    source 记录来源文件路径，保证每个数字都能追溯到原始运行产物。
    """
    test_metrics = metrics.get("test", metrics)
    row = {"name": name, "seed": seed, "source": os.path.abspath(source)}
    row.update({
        key: test_metrics.get(key)
        for key in (
            "macro_auprc", "macro_f1", "micro_f1", "exact_match",
            "support_macro_auprc", "support_macro_f1", "support_micro_f1",
            "support_disp_mae_mm", "region_macro_f1", "global_macro_f1", "global_mae",
        )
    })
    results.setdefault(name, []).append(row)


def aggregate(output_dir):
    """扫描输出目录，汇总先验 / 传统 / 深度三类结果并按方法分组。"""
    results = {}
    prior_path = os.path.join(output_dir, "baseline", "prior_baseline.json")
    if os.path.isfile(prior_path):
        _add_result(results, "prior", "fixed", load_json(prior_path)["metrics"]["test"], prior_path)

    for path in glob.glob(os.path.join(output_dir, "traditional", "*", "seed_*", "report.json")):
        payload = load_json(path)
        name = f"traditional_{payload['mode']}_{payload['estimator']}"
        _add_result(results, name, payload["seed"], payload["metrics"], path)

    for seed_dir in glob.glob(os.path.join(output_dir, "deep", "seed_*")):
        seed_name = os.path.basename(seed_dir).removeprefix("seed_")
        for model_type in ("static_only", "dynamic_only", "fusion", "concat_fusion"):
            prediction_path = find_prediction_file(seed_dir, model_type)
            if prediction_path:
                _add_result(
                    results,
                    f"deep_{model_type}",
                    int(seed_name),
                    prediction_records_metrics(prediction_path),
                    prediction_path,
                )

    summary = {
        "experiment": "stage1.aggregate",
        "output_dir": os.path.abspath(output_dir),
        "methods": {
            name: {"runs": rows, "aggregate": aggregate_seed_metrics(rows)}
            for name, rows in sorted(results.items())
        },
    }
    save_json(os.path.join(output_dir, "stage1_results.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate stage 1 results.")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    summary = aggregate(args.output_dir)
    print(f"saved: {os.path.join(args.output_dir, 'stage1_results.json')}")
    for name, item in summary["methods"].items():
        aggregate_metrics = item["aggregate"]["metrics"]
        mean = aggregate_metrics.get("macro_auprc", {}).get("mean")
        print(f"{name:32s} test_macro_auprc_mean={mean}")


if __name__ == "__main__":
    main()
