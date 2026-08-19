"""阶段 1 统一执行器。

职责链：校验冻结划分 → 执行先验 / 传统特征基线 → 每个种子启动一次现有深度
训练入口（DL_model/main.py --model all）→ 每 seed 独立输出目录 → 汇总结果。
单个任务失败只记录、不清除其他结果；只要有任何请求的任务失败，最终报告的
passed 即为 false，进程以非零码退出。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from stage1.aggregate_results import aggregate  # noqa: E402
from stage1.stage1_common import (  # noqa: E402
    DEFAULT_SEEDS,
    parse_seeds,
    save_json,
    validate_data_and_manifest,
)


def _run(command, cwd, log_path, dry_run=False, expected_path=None):
    """执行一个子进程并记录结果；dry_run 时只返回计划，不真正运行。

    expected_path 用于判断任务是否真正完成：命令返回 0 且预期产物存在才算通过。
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    result = {
        "command": command,
        "cwd": cwd,
        "log_path": os.path.abspath(log_path),
    }
    if dry_run:
        result.update({"returncode": None, "passed": True, "dry_run": True})
        return result
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    output_exists = expected_path is None or os.path.isfile(expected_path)
    result.update({
        "returncode": completed.returncode,
        "passed": completed.returncode == 0 and output_exists,
        "expected_path": os.path.abspath(expected_path) if expected_path else None,
        "output_exists": output_exists,
    })
    return result


def _python():
    return sys.executable


def _script(path):
    return os.path.join(SCRIPT_DIR, path)


def main():
    parser = argparse.ArgumentParser(description="运行可复现的阶段 1 基线矩阵")
    parser.add_argument("data_dir", help="数据目录（含 result_*.json）")
    parser.add_argument("--split-manifest", required=True, help="冻结的划分清单 JSON，必须与 data_dir 严格对应")
    parser.add_argument("--output-dir", required=True, help="输出根目录（必须为空；每 seed 独立子目录）")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS),
                        help="随机种子列表，逗号分隔（默认 13,29,42,71,101）")
    parser.add_argument("--data-version", default="stage1_v2", help="数据版本名，写入 plan 与 run 元数据")
    parser.add_argument("--deep-epochs", type=int, default=None, help="深度模型训练轮数覆盖（小规模试跑用）")
    parser.add_argument("--batch", type=int, default=None, help="深度模型批大小覆盖")
    parser.add_argument("--lr", type=float, default=None, help="深度模型学习率覆盖")
    parser.add_argument("--estimator", choices=("random_forest", "logistic"), default="random_forest",
                        help="传统基线使用的分类器")
    parser.add_argument("--skip-prior", action="store_true", help="跳过先验基线")
    parser.add_argument("--skip-traditional", action="store_true", help="跳过传统特征基线")
    parser.add_argument("--skip-deep", action="store_true", help="跳过深度基线")
    parser.add_argument("--dry-run", action="store_true", help="只校验并生成执行计划，不真正运行")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    output_dir = os.path.abspath(args.output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir) and not args.dry_run:
        raise FileExistsError(f"stage 1 output directory must be empty: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # 冻结校验：文件哈希 / 组互斥 / 划分完整性，任何不符直接报错，避免在脏数据上白跑。
    manifest, splits = validate_data_and_manifest(args.data_dir, args.split_manifest)

    # 先落盘执行计划（含 manifest 哈希），保证"这份结果对应哪份数据/哪种配置"可追溯。
    plan = {
        "experiment": "stage1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": REPO_ROOT,
        "data_dir": os.path.abspath(args.data_dir),
        "split_manifest": os.path.abspath(args.split_manifest),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "sample_count": sum(len(values) for values in splits.values()),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "seeds": seeds,
        "data_version": args.data_version,
        "deep_models": ["static_only", "dynamic_only", "fusion"],
        "traditional_estimator": args.estimator,
        "deep_epochs": args.deep_epochs,
        "batch": args.batch,
        "lr": args.lr,
        "dry_run": args.dry_run,
    }
    save_json(os.path.join(output_dir, "stage1_plan.json"), plan)

    # 三类任务：先验（一次）、传统特征（static/dynamic × 每 seed）、深度（每 seed 一个 main.py --model all）。
    jobs = []
    if not args.skip_prior:
        prior_output = os.path.join(output_dir, "baseline", "prior_baseline.json")
        command = [
            _python(), _script("prior_baseline.py"), args.data_dir,
            "--split-manifest", args.split_manifest, "--output", prior_output,
        ]
        jobs.append({"name": "prior", "result": _run(
            command, REPO_ROOT, os.path.join(output_dir, "baseline", "prior.log"),
            args.dry_run, expected_path=prior_output,
        )})

    if not args.skip_traditional:
        for mode in ("static", "dynamic"):
            for seed in seeds:
                seed_dir = os.path.join(output_dir, "traditional", mode, f"seed_{seed}")
                report_path = os.path.join(seed_dir, "report.json")
                command = [
                    _python(), _script("traditional_baseline.py"), args.data_dir,
                    "--split-manifest", args.split_manifest, "--output", report_path,
                    "--mode", mode, "--estimator", args.estimator, "--seed", str(seed),
                ]
                jobs.append({
                    "name": f"traditional_{mode}_seed_{seed}",
                    "result": _run(
                        command, REPO_ROOT, os.path.join(seed_dir, "run.log"),
                        args.dry_run, expected_path=report_path,
                    ),
                })

    if not args.skip_deep:
        for seed in seeds:
            seed_dir = os.path.join(output_dir, "deep", f"seed_{seed}")
            command = [
                _python(), os.path.join(REPO_ROOT, "DL_model", "main.py"),
                "--model", "all", "--seed", str(seed),
                "--data-dir", args.data_dir, "--split-manifest", args.split_manifest,
                "--data-version", args.data_version, "--run-id", f"stage1_seed_{seed}",
                "--output-dir", seed_dir,
            ]
            if args.deep_epochs is not None:
                command.extend(["--epochs", str(args.deep_epochs)])
            if args.batch is not None:
                command.extend(["--batch", str(args.batch)])
            if args.lr is not None:
                command.extend(["--lr", str(args.lr)])
            jobs.append({
                "name": f"deep_all_seed_{seed}",
                "result": _run(
                    command, REPO_ROOT, os.path.join(seed_dir, "run.log"), args.dry_run,
                    # 以"static_only 测试预测文件是否生成"作为该 seed 训练完成的判据。
                    expected_path=os.path.join(seed_dir, "logs", "static_only", "static_only_test_predictions.json"),
                ),
            })

    report = {
        "experiment": "stage1",
        "passed": all(job["result"]["passed"] for job in jobs),
        "plan": plan,
        "jobs": jobs,
    }
    save_json(os.path.join(output_dir, "stage1_job_report.json"), report)
    if not args.dry_run:
        aggregate(output_dir)
    print(json.dumps({"passed": report["passed"], "output_dir": output_dir}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
