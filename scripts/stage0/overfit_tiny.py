"""阶段 0.2：把全部 tiny-set 样本训练到近似记忆并自动验收。

备注：该诊断不切分验证/测试集，关闭数据增强、EMA、label smoothing、weight decay、
早停和全部辅助任务，只优化材料分类 BCE。它不是正式性能实验，结果不得写入主结果表。
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
from model import build_model
from trainer import make_loader
from stage0_common import report_envelope, save_json
from stage0_train_utils import evaluate_material, train_material_epoch


def main():
    parser = argparse.ArgumentParser(description="阶段0.2：tiny-set 自动过拟合验收")
    parser.add_argument("data_dir")
    parser.add_argument("--output", default=None, help="报告 JSON；默认 data_dir/overfit_report.json")
    parser.add_argument("--model", choices=("static_only", "dynamic_only", "fusion"), default="static_only")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-loss-reduction", type=float, default=0.50, help="(初始loss-最终loss)/初始loss 的最低值")
    parser.add_argument("--min-macro-f1", type=float, default=0.90)
    parser.add_argument("--min-exact-match", type=float, default=0.75)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("epochs 和 batch 必须为正数")

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
    indices = list(range(len(dataset)))
    if not 2 <= len(indices) <= 32:
        print(f"备注: tiny overfit 通常使用 8-16 个样本，当前为 {len(indices)}")
    dataset.fit_normalizer(indices)
    dataset.apply_to_config(cfg)
    loader = make_loader(dataset, indices, dataset.static_pos, dataset.dynamic_pos, True, cfg)
    eval_loader = make_loader(dataset, indices, dataset.static_pos, dataset.dynamic_pos, False, cfg)
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    initial_loss, initial_metrics, _, _ = evaluate_material(model, eval_loader, device)
    curve = []
    bad_gradient = False
    for epoch in range(args.epochs):
        train_loss, bad = train_material_epoch(model, loader, optimizer, device)
        bad_gradient |= bad
        eval_loss, metrics, _, _ = evaluate_material(model, eval_loader, device)
        curve.append({"epoch": epoch + 1, "train_loss": train_loss, "eval_loss": eval_loss, **metrics})
        print(f"E{epoch + 1:03d} loss={eval_loss:.6f} macro_f1={metrics['macro_f1']:.4f} exact={metrics['exact_match']:.4f}")

    final = curve[-1]
    reduction = (initial_loss - final["eval_loss"]) / max(initial_loss, 1e-12)
    checks = {
        "finite_gradients": not bad_gradient,
        "loss_reduction": reduction >= args.min_loss_reduction,
        "macro_f1": final["macro_f1"] >= args.min_macro_f1,
        "exact_match": final["exact_match"] >= args.min_exact_match,
    }
    report = report_envelope(
        "stage0.2",
        all(checks.values()),
        {
            "data_dir": os.path.abspath(args.data_dir), "model": args.model,
            "sample_count": len(dataset), "epochs": args.epochs, "seed": args.seed,
            "initial_loss": initial_loss, "initial_metrics": initial_metrics,
            "final_loss": final["eval_loss"], "final_metrics": {key: value for key, value in final.items() if key not in ("epoch", "train_loss", "eval_loss")},
            "loss_reduction": reduction, "thresholds": {
                "min_loss_reduction": args.min_loss_reduction,
                "min_macro_f1": args.min_macro_f1,
                "min_exact_match": args.min_exact_match,
            }, "checks": checks, "curve": curve,
        },
        notes=[
            "全部样本只作训练集；不执行模型选择或测试集评估。",
            "forward 使用净化后的观测输入；训练只读取 material target。",
            "未通过时先检查标签多样性、学习率和模型容量，不得直接放宽阈值掩盖链路错误。",
        ],
    )
    output = args.output or os.path.join(args.data_dir, "overfit_report.json")
    save_json(output, report)
    print(json.dumps({"passed": report["passed"], "report": os.path.abspath(output), "checks": checks}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
