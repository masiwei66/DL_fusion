"""阶段 0.2：构建可复现且无旧文件污染的 tiny-set 数据目录。

备注：本脚本只负责固定抽样和物化数据依赖；真正的“全部样本用于训练、关闭正则、
自动判断是否过拟合”由 ``overfit_tiny.py`` 完成。
"""

import argparse
import os
import random
import shutil

from stage0_common import (
    file_sha256,
    json_sha256,
    load_json,
    materialize_shared_assets,
    prepare_new_output_dir,
    result_files,
    save_json,
)


def main():
    parser = argparse.ArgumentParser(description="构建阶段0.2 tiny-set 目录")
    parser.add_argument("data_dir", help="完整数据目录")
    parser.add_argument("--output", required=True, help="必须为空或不存在的新目录")
    parser.add_argument("--count", type=int, default=16, help="建议 8-16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset-mode", choices=("hardlink", "copy"), default="hardlink", help="紧凑数据的相对 HDF5 依赖处理方式")
    args = parser.parse_args()

    files = result_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"no result_*.json found in {args.data_dir}")
    if args.count < 2:
        raise ValueError("tiny-set 至少需要 2 个样本")
    count = min(args.count, len(files))
    if not 8 <= count <= 16:
        print(f"备注: 阶段0计划建议 8-16 个样本，当前 count={count}")

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(files, count))
    output = prepare_new_output_dir(args.output)
    samples = []
    records = []
    for name in selected:
        source = os.path.join(args.data_dir, name)
        destination = os.path.join(output, name)
        shutil.copy2(source, destination)
        sample = load_json(source)
        samples.append(sample)
        records.append({
            "filename": name,
            "sample_id": (sample.get("sample_index") or {}).get("sample_id"),
            "source_sha256": file_sha256(source),
        })

    assets = materialize_shared_assets(args.data_dir, output, samples, mode=args.asset_mode)
    manifest = {
        "experiment": "stage0.2-selection",
        "source_dir": os.path.abspath(args.data_dir),
        "output_dir": output,
        "seed": args.seed,
        "count": count,
        "records": records,
        "shared_assets": assets,
        "notes": [
            "输出目录必须为空，防止旧 result_*.json 混入。",
            "若 JSON 通过相对 array_store 引用 HDF5，依赖会被硬链接或复制。",
            "此目录不得按 train/val/test 切分；请使用 overfit_tiny.py 将全部样本用于训练。",
        ],
    }
    manifest["manifest_sha256"] = json_sha256(manifest)
    save_json(os.path.join(output, "stage0_selection.json"), manifest)
    print(f"已创建 tiny-set: {output} ({count} samples, seed={args.seed})")
    print(f"下一步: python scripts/stage0/overfit_tiny.py {output} --output {os.path.join(output, 'overfit_report.json')}")


if __name__ == "__main__":
    main()
