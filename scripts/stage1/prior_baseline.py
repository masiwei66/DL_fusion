"""先验（prevalence）基线：用训练集正样本比例做预测。

这是最弱的参考线。数据不平衡时，任何模型至少要明显超过它，才能说明
"学到了输入里的信息"，否则只是在拟合类别先验。
所有统计量只从训练集估计，val/test 完全不参与拟合。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from stage1.stage1_common import (  # noqa: E402
    SPLIT_NAMES,
    load_json,
    multilabel_metrics,
    save_json,
    validate_data_and_manifest,
)


def load_material_labels(data_dir, files):
    """按传入的文件名顺序读取每个样本的材料标签（多标签 0/1 数组）。"""
    labels = []
    for name in files:
        sample = load_json(os.path.join(data_dir, name))
        values = (sample.get("safety_labels", {}) or {}).get("material_labels")
        if values is None:
            raise ValueError(f"{name} has no safety_labels.material_labels")
        labels.append(np.asarray(values, dtype=np.float32))
    return np.stack(labels)


def main():
    parser = argparse.ArgumentParser(description="Compute the prevalence baseline for stage 1.")
    parser.add_argument("data_dir")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    _manifest, splits = validate_data_and_manifest(args.data_dir, args.split_manifest)
    files = sorted(
        name for name in os.listdir(args.data_dir)
        if name.startswith("result_") and name.endswith(".json")
    )
    y = load_material_labels(args.data_dir, files)
    train = np.asarray(splits["train"], dtype=int)
    # 先验 = 训练集正样本占比；val/test 不参与估计，防止信息泄漏。
    prevalence = y[train].mean(axis=0)
    payload = {
        "experiment": "stage1.prior",
        "data_dir": os.path.abspath(args.data_dir),
        "split_manifest": os.path.abspath(args.split_manifest),
        "train_prevalence": prevalence.astype(float).tolist(),
        "metrics": {},
    }
    # 对每个划分输出同一套指标，便于 stage1_results 跨方法对比。
    for split_name in SPLIT_NAMES:
        indices = np.asarray(splits[split_name], dtype=int)
        # 先验模型对所有样本预测同一个先验概率，不依赖任何输入。
        probabilities = np.broadcast_to(prevalence, (len(indices), y.shape[1]))
        payload["metrics"][split_name] = multilabel_metrics(y[indices], probabilities)
    save_json(args.output, payload)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
