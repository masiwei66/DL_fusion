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

from DL_config import CANDIDATE_IDS, Config
from dataset import StructuralDataset
from main import load_split_manifest
from model import build_model
from trainer import make_loader
from stage0_common import load_json, report_envelope, save_json
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

    manifest = load_json(args.split_manifest)
    group_by = manifest.get("group_by") or manifest.get("negative_control_group_by")
    split_groups = {name: set() for name in ("train", "val", "test")}
    for record in manifest.get("records", []):
        split = record.get("split")
        if split not in split_groups:
            continue
        group = record.get("group_value")
        if group in (None, "") and group_by:
            group = record.get(group_by)
        if group in (None, ""):
            raise ValueError("split manifest 缺少 group_value，无法验证分组隔离")
        split_groups[split].add(str(group))
    group_overlap = {
        "train_val": sorted(split_groups["train"] & split_groups["val"]),
        "train_test": sorted(split_groups["train"] & split_groups["test"]),
        "val_test": sorted(split_groups["val"] & split_groups["test"]),
    }
    split_groups_disjoint = not any(group_overlap.values())
    if not split_groups_disjoint:
        raise ValueError(f"split manifest 存在分组泄漏: {group_overlap}")

    audit_path = os.path.join(args.data_dir, "stage0_permutation_audit.json")
    if not os.path.isfile(audit_path):
        raise FileNotFoundError(f"负控置换审计不存在: {audit_path}")
    permutation_audit = load_json(audit_path)
    permutation_audit_valid = bool(
        permutation_audit.get("permutation_unit") == "group"
        and permutation_audit.get("train_group_donor_derangement")
        and permutation_audit.get("validation_and_test_byte_identical")
    )
    if not permutation_audit_valid:
        raise ValueError("负控置换审计未通过组级错排或 val/test 哈希检查")
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
    val_per_label_gap = [
        float(metric - prior)
        for metric, prior in zip(val_metrics["per_label_auprc"], val_prior["per_label_auprc"])
    ]
    test_per_label_gap = [
        float(metric - prior)
        for metric, prior in zip(test_metrics["per_label_auprc"], test_prior["per_label_auprc"])
    ]
    checks = {
        "finite_gradients": not bad_gradient,
        "split_groups_disjoint": split_groups_disjoint,
        "permutation_audit_valid": permutation_audit_valid,
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
            "group_by": group_by,
            "group_counts": {key: len(value) for key, value in split_groups.items()},
            "group_overlap": group_overlap,
            "permutation_audit": {
                "path": os.path.abspath(audit_path),
                "permutation_unit": permutation_audit.get("permutation_unit"),
                "train_group_count": permutation_audit.get("train_group_count"),
                "group_material_label_exact_match_rate": permutation_audit.get(
                    "group_material_label_exact_match_rate"
                ),
                "group_material_label_per_label_match_rate": permutation_audit.get(
                    "group_material_label_per_label_match_rate"
                ),
                "group_material_label_per_label_correlation": permutation_audit.get(
                    "group_material_label_per_label_correlation"
                ),
            },
            "material_label_ids": list(CANDIDATE_IDS),
            "train": {"loss": train_loss, "metrics": train_metrics},
            "validation": {
                "loss": val_loss, "metrics": val_metrics, "prior": val_prior,
                "macro_auprc_gap": val_gap, "per_label_auprc_gap": val_per_label_gap,
            },
            "test": {
                "loss": test_loss, "metrics": test_metrics, "prior": test_prior,
                "macro_auprc_gap": test_gap, "per_label_auprc_gap": test_per_label_gap,
            },
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
