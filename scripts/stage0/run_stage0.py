"""统一执行阶段 0.1-0.4，并生成 ``stage0_report.json``。

推荐先用 ``scripts/make_splits.py`` 为完整数据生成固定 split manifest，再运行本脚本。
统一 runner 会建立新的工作目录，因此不会污染原始数据；负控默认执行 3 个置换种子。
子实验未通过时 runner 仍会继续收集其余报告，最后以非零退出码表示阶段 0 未通过。
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from stage0_common import load_json, prepare_new_output_dir, save_json


def run_command(name, command, report_path=None):
    print(f"\n=== {name} ===")
    print(subprocess.list2cmdline(command))
    completed = subprocess.run(command, check=False)
    result = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
    }
    if report_path and os.path.isfile(report_path):
        result["report_path"] = os.path.abspath(report_path)
        result["report"] = load_json(report_path)
        result["passed"] = result["passed"] and bool(result["report"].get("passed", False))
    return result


def main():
    parser = argparse.ArgumentParser(description="阶段0统一执行与验收")
    parser.add_argument("data_dir")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--work-dir", required=True, help="必须为空或不存在")
    parser.add_argument("--tiny-count", type=int, default=16)
    parser.add_argument("--overfit-epochs", type=int, default=50)
    parser.add_argument("--negative-epochs", type=int, default=40)
    parser.add_argument("--seeds", default="13,29,42", help="负控置换种子，逗号分隔；建议至少 3 个")
    parser.add_argument("--model", choices=("static_only", "dynamic_only", "fusion"), default="static_only")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("至少提供一个负控种子")
    if len(seeds) < 3:
        print("备注: 正式负控结论建议至少 3 个置换种子。")

    work_dir = prepare_new_output_dir(args.work_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable
    results = []

    smoke_report = os.path.join(work_dir, "smoke_contract_report.json")
    results.append(run_command(
        "0.1+0.4 batch contract / leakage",
        [python, os.path.join(script_dir, "check_smoke_batch.py"), args.data_dir,
         "--samples", "8", "--batch-size", "2", "--check-backward", "--report", smoke_report],
        smoke_report,
    ))

    tiny_dir = os.path.join(work_dir, "tiny_data")
    selection = run_command(
        "0.2 select tiny-set",
        [python, os.path.join(script_dir, "make_smoke_dir.py"), args.data_dir,
         "--output", tiny_dir, "--count", str(args.tiny_count), "--seed", "42"],
    )
    results.append(selection)
    if selection["passed"]:
        overfit_report = os.path.join(work_dir, "overfit_report.json")
        results.append(run_command(
            "0.2 tiny-set overfit",
            [python, os.path.join(script_dir, "overfit_tiny.py"), tiny_dir,
             "--output", overfit_report, "--model", args.model,
             "--epochs", str(args.overfit_epochs), "--batch", str(args.batch), "--seed", "42"],
            overfit_report,
        ))

    for seed in seeds:
        negative_dir = os.path.join(work_dir, f"negative_seed_{seed}")
        permutation = run_command(
            f"0.3 prepare permutation seed={seed}",
            [python, os.path.join(script_dir, "shuffle_labels.py"), args.data_dir,
             "--split-manifest", args.split_manifest, "--output", negative_dir,
             "--seed", str(seed), "--supervision", "all"],
        )
        results.append(permutation)
        if permutation["passed"]:
            negative_report = os.path.join(work_dir, f"negative_control_seed_{seed}.json")
            results.append(run_command(
                f"0.3 evaluate permutation seed={seed}",
                [python, os.path.join(script_dir, "run_negative_control.py"), negative_dir,
                 "--split-manifest", os.path.join(negative_dir, "stage0_negative_split.json"),
                 "--output", negative_report, "--model", args.model,
                 "--epochs", str(args.negative_epochs), "--batch", str(args.batch),
                 "--seed", str(seed)],
                negative_report,
            ))

    report = {
        "experiment": "stage0",
        "passed": all(result["passed"] for result in results),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": os.path.abspath(args.data_dir),
        "split_manifest": os.path.abspath(args.split_manifest),
        "work_dir": work_dir,
        "model": args.model,
        "negative_control_seeds": seeds,
        "results": results,
        "notes": [
            "0.2 是训练链路诊断，不是泛化性能实验。",
            "0.3 的 val/test 使用真实标签，训练仅使用被置换的材料标签。",
            "所有通过阈值应在查看结果前固定；失败结果也必须保留。",
        ],
    }
    output = os.path.join(work_dir, "stage0_report.json")
    save_json(output, report)
    print(json.dumps({"passed": report["passed"], "report": output}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
