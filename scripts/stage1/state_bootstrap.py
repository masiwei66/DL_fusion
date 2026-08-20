"""按结构状态重采样的阶段 1 配对 bootstrap 比较。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from stage1.stage1_common import prediction_records_metrics_from_records, save_json  # noqa: E402


METRICS = (
    "macro_auprc", "macro_f1", "micro_f1", "exact_match",
    "support_macro_f1", "support_disp_mae_mm", "region_macro_f1",
    "global_macro_f1", "global_mae",
)
LOWER_IS_BETTER = {"support_disp_mae_mm", "global_mae"}


def _load_records(path):
    with open(path, encoding="utf-8") as file:
        records = json.load(file).get("records", [])
    if not records:
        raise ValueError(f"no prediction records: {path}")
    by_state = {}
    for row in records:
        state = str(row.get("structural_state_id") or row.get("sample_id"))
        by_state.setdefault(state, []).append(row)
    return by_state


def _metrics(records):
    return prediction_records_metrics_from_records(records)


def compare(left_path, right_path, iterations, rng):
    left = _load_records(left_path)
    right = _load_records(right_path)
    states = sorted(set(left) & set(right))
    if len(states) < 3:
        raise ValueError("paired bootstrap requires at least three shared structural states")
    if set(left) != set(right):
        raise ValueError("prediction files do not contain the same structural states")
    point_left = _metrics([row for state in states for row in left[state]])
    point_right = _metrics([row for state in states for row in right[state]])
    deltas = {metric: [] for metric in METRICS if point_left.get(metric) is not None}
    for _ in range(iterations):
        sampled = rng.choice(states, size=len(states), replace=True)
        sample_left = [row for state in sampled for row in left[state]]
        sample_right = [row for state in sampled for row in right[state]]
        metrics_left = _metrics(sample_left)
        metrics_right = _metrics(sample_right)
        for metric in deltas:
            sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
            deltas[metric].append(sign * (metrics_right[metric] - metrics_left[metric]))
    summary = {}
    for metric, values in deltas.items():
        values = np.asarray(values, dtype=float)
        sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
        observed = sign * (point_right[metric] - point_left[metric])
        summary[metric] = {
            "left": float(point_left[metric]),
            "right": float(point_right[metric]),
            "right_minus_left": float(point_right[metric] - point_left[metric]),
            "higher_is_better_delta": float(observed),
            "ci95_higher_is_better_delta": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "probability_right_better": float((values > 0).mean()),
        }
    return {"state_count": len(states), "record_count": sum(len(left[s]) for s in states), "metrics": summary}


def main():
    parser = argparse.ArgumentParser(description="State-level paired bootstrap for stage 1 predictions.")
    parser.add_argument("left_predictions")
    parser.add_argument("right_predictions")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.iterations < 100:
        raise ValueError("--iterations must be at least 100")
    payload = {
        "experiment": "stage1.state_paired_bootstrap",
        "left_predictions": os.path.abspath(args.left_predictions),
        "right_predictions": os.path.abspath(args.right_predictions),
        "iterations": args.iterations,
        "seed": args.seed,
        **compare(args.left_predictions, args.right_predictions, args.iterations, np.random.default_rng(args.seed)),
    }
    save_json(args.output, payload)
    print(f"saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
