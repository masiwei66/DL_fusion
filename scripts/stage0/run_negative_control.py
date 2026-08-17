"""阶段 0.3：在置换训练监督上训练，并用真实 val/test 与类别先验比较。

输入目录应由 ``shuffle_labels.py`` 生成。该诊断只优化材料分类 BCE，固定 0.5 阈值，
不在测试集调阈值。通过条件基于 macro-AUPRC 与训练集 prevalence baseline 的差值；
阈值容差应在实验前固定，不能观察测试结果后修改。
"""

import argparse
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "DL_model"))

import numpy as np
import torch

from DL_config import Config
from dataset import StructuralDataset
from main import load_split_manifest
from model import build_model
from trainer import make_loader
from stage0_common import report_envelope, save_json
from stage0_train_utils import evaluate_material, prevalence_baseline, train_material_epoch


def main():
    parser = argparse.ArgumentParser(description="阶段0.3：标签置换负控自动验收")
    parser.add_argument("data_dir", help="shuffle_labels.py 的输出目录")
    parser.add_argument("--split-manifest", required=True, help="置换脚本生成的 stage0_negative_split.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", choices=("static_only", "dynamic_only", "fusion"), default="static_only")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-auprc-over-prior", type=float, default=0.10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = Config()
    cfg.set_model_type(args.model)
    cfg.batch_size = args.batch
    cfg.num_workers = 0
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = StructuralDataset(args.data_dir, normalize=True, fit_normalizer=False)
    dataset.apply_to_config(cfg)
    train_idx, val_idx, test_idx = load_split_manifest(args.split_manifest, dataset)
    if not train_idx or not val_idx or not test_idx:
        raise ValueError("负控要求非空 train/val/test")
    dataset.fit_normalizer(train_idx)
    train_loader = make_loader(dataset, train_idx, dataset.static_pos, dataset.dynamic_pos, True, cfg)
    train_eval_loader = make_loader(dataset, train_idx, dataset.static_pos, dataset.dynamic_pos, False, cfg)
    val_loader = make_loader(dataset, val_idx, dataset.static_pos, dataset.dynamic_pos, False, cfg)
    test_loader = make_loader(dataset, test_idx, dataset.static_pos, dataset.dynamic_pos, False, cfg)
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    curve = []
    bad_gradient = False
    for epoch in range(args.epochs):
        train_loss, bad = train_material_epoch(model, train_loader, optimizer, device)
        bad_gradient |= bad
        curve.append({"epoch": epoch + 1, "train_loss": train_loss})
        print(f"E{epoch + 1:03d} train_loss={train_loss:.6f}")

    train_loss, train_metrics, train_true, _ = evaluate_material(model, train_eval_loader, device)
    val_loss, val_metrics, val_true, _ = evaluate_material(model, val_loader, device)
    test_loss, test_metrics, test_true, _ = evaluate_material(model, test_loader, device)
    val_prior = prevalence_baseline(train_true, val_true)
    test_prior = prevalence_baseline(train_true, test_true)
    val_gap = val_metrics["macro_auprc"] - val_prior["macro_auprc"]
    test_gap = test_metrics["macro_auprc"] - test_prior["macro_auprc"]
    checks = {
        "finite_gradients": not bad_gradient,
        "validation_near_prior": val_gap <= args.max_auprc_over_prior,
        "test_near_prior": test_gap <= args.max_auprc_over_prior,
    }
    report = report_envelope(
        "stage0.3",
        all(checks.values()),
        {
            "data_dir": os.path.abspath(args.data_dir), "split_manifest": os.path.abspath(args.split_manifest),
            "model": args.model, "epochs": args.epochs, "seed": args.seed,
            "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "train": {"loss": train_loss, "metrics": train_metrics},
            "validation": {"loss": val_loss, "metrics": val_metrics, "prior": val_prior, "macro_auprc_gap": val_gap},
            "test": {"loss": test_loss, "metrics": test_metrics, "prior": test_prior, "macro_auprc_gap": test_gap},
            "max_auprc_over_prior": args.max_auprc_over_prior, "checks": checks, "curve": curve,
        },
        notes=[
            "val/test 保留真实标签；只有训练监督被置换。",
            "只训练材料分类，未置换的辅助监督不会参与损失。",
            "正式结论建议至少运行 3 个置换种子，并报告 gap 的均值和范围。",
        ],
    )
    output = args.output or os.path.join(args.data_dir, "negative_control_report.json")
    save_json(output, report)
    print(json.dumps({"passed": report["passed"], "report": os.path.abspath(output), "checks": checks}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
