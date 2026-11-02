"""Summarize KW51 state-level cross-validation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# Two-sided 95% Student-t critical values. df>=30 falls back to the normal
# approximation, which is accurate enough for the fold counts used here.
_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_critical(df: int) -> float:
    return _T975.get(int(df), 1.96)


def _mean_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    mean = float(array.mean()) if count else 0.0
    if count <= 1:
        return {"n": count, "mean": mean, "std": 0.0, "ci95": [mean, mean]}
    std = float(array.std(ddof=1))
    half = _t_critical(count - 1) * std / float(np.sqrt(count))
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
    }


def _bootstrap_ci(values: list[float], *, repeats: int = 5000, seed: int = 0) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        means[index] = rng.choice(array, size=array.size, replace=True).mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _state_sort_key(state_id: str) -> tuple[int, Any]:
    try:
        return (0, int(state_id))
    except (TypeError, ValueError):
        return (1, str(state_id))


def load_folds(base: Path, folds: int) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        report_path = base.parent / f"{base.name}_fold{fold}" / "training_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing fold report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        test = report["metrics"]["test"]
        runs.append(
            {
                "fold": fold,
                "report": str(report_path),
                "best_epoch": report.get("best_epoch"),
                "run_status": report.get("run_status"),
                "diagnostic_only": report.get("diagnostic_only"),
                "evaluation_setting": report.get("evaluation_setting"),
                "gallery_exclusion": report.get("gallery_exclusion"),
                "split_states": report.get("split_states"),
                "test": test,
            }
        )
    return runs


def summarize(base: Path, folds: int, *, repeats: int = 5000) -> dict[str, Any]:
    runs = load_folds(base, folds)
    fold_rows: list[dict[str, Any]] = []
    state_metrics: dict[str, dict[str, float]] = {}
    for run in runs:
        test = run["test"]
        fold_rows.append(
            {
                "fold": run["fold"],
                "best_epoch": run["best_epoch"],
                "run_status": run["run_status"],
                "diagnostic_only": run["diagnostic_only"],
                "evaluation_setting": run["evaluation_setting"],
                "gallery_exclusion": run["gallery_exclusion"],
                "top1_hit_rate": float(test["top1_hit_rate"]),
                "mrr": float(test["mrr"]),
                "pairwise_accuracy": float(test["pairwise_accuracy"]),
                "ndcg": float(test["ndcg"]),
                "split_states": run["split_states"],
            }
        )
        for state_id, metrics in test.get("per_query_state", {}).items():
            if state_id in state_metrics:
                raise ValueError(
                    f"state {state_id} appears as a test state in more than one fold"
                )
            state_metrics[state_id] = {
                "top1_hit_rate": float(metrics["top1_hit_rate"]),
                "mrr": float(metrics["mrr"]),
                "query_groups": float(metrics["query_groups"]),
            }
    ordered_states = sorted(state_metrics, key=_state_sort_key)
    state_top1 = [state_metrics[state]["top1_hit_rate"] for state in ordered_states]
    state_mrr = [state_metrics[state]["mrr"] for state in ordered_states]
    fold_top1 = [row["top1_hit_rate"] for row in fold_rows]
    fold_mrr = [row["mrr"] for row in fold_rows]
    fold_pairwise = [row["pairwise_accuracy"] for row in fold_rows]
    fold_ndcg = [row["ndcg"] for row in fold_rows]
    return {
        "schema": "kw51_state_level_cv_summary_v1",
        "base": str(base),
        "evaluation_setting": (
            fold_rows[0]["evaluation_setting"]
            if fold_rows
            and len({row["evaluation_setting"] for row in fold_rows}) == 1
            else "mixed"
        ),
        "gallery_exclusion": (
            fold_rows[0]["gallery_exclusion"]
            if fold_rows
            and len({row["gallery_exclusion"] for row in fold_rows}) == 1
            else "mixed"
        ),
        "all_runs_formal": all(row["run_status"] == "FORMAL" for row in fold_rows),
        "any_diagnostic_only": any(bool(row["diagnostic_only"]) for row in fold_rows),
        "folds": fold_rows,
        "fold_level": {
            "top1_hit_rate": _mean_ci(fold_top1),
            "mrr": _mean_ci(fold_mrr),
            "pairwise_accuracy": _mean_ci(fold_pairwise),
            "ndcg": _mean_ci(fold_ndcg),
        },
        "state_level": {
            "states": {
                state: state_metrics[state] for state in ordered_states
            },
            "top1_hit_rate": {
                "mean": float(np.mean(state_top1)) if state_top1 else 0.0,
                "ci95_bootstrap": _bootstrap_ci(
                    state_top1, repeats=repeats, seed=0
                ),
            },
            "mrr": {
                "mean": float(np.mean(state_mrr)) if state_mrr else 0.0,
                "ci95_bootstrap": _bootstrap_ci(
                    state_mrr, repeats=repeats, seed=0
                ),
            },
        },
        "note": (
            "fold_level CI uses a t interval over folds; state_level CI "
            "bootstraps the 25 held-out structural states. Query groups within "
            "a state are not independent."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="CV output_root without the _fold suffix")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    base = Path(args.base)
    summary = summarize(base, args.folds, repeats=args.bootstrap)
    output = Path(args.output) if args.output else base.parent / f"{base.name}_cv_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("fold  top1    mrr     pairwise  best_epoch  run_status")
    for row in summary["folds"]:
        print(
            f"{row['fold']:>4}  {row['top1_hit_rate']:.4f}  {row['mrr']:.4f}  "
            f"{row['pairwise_accuracy']:.4f}  {row['best_epoch']:>9}  "
            f"{row['run_status']}"
        )
    fold = summary["fold_level"]
    state = summary["state_level"]
    print(
        "fold mean top1=%.4f 95%%CI [%.4f, %.4f]" % (
            fold["top1_hit_rate"]["mean"],
            *fold["top1_hit_rate"]["ci95"],
        )
    )
    print(
        "fold mean mrr=%.4f 95%%CI [%.4f, %.4f]" % (
            fold["mrr"]["mean"],
            *fold["mrr"]["ci95"],
        )
    )
    print(
        "state mean top1=%.4f bootstrap 95%%CI [%.4f, %.4f]" % (
            state["top1_hit_rate"]["mean"],
            *state["top1_hit_rate"]["ci95_bootstrap"],
        )
    )
    print(
        "state mean mrr=%.4f bootstrap 95%%CI [%.4f, %.4f]" % (
            state["mrr"]["mean"],
            *state["mrr"]["ci95_bootstrap"],
        )
    )
    print(f"cv_summary={output}")


if __name__ == "__main__":
    main()
