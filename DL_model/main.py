"""Training, evaluation, inference, and plotting entry point."""

import argparse
import json
import os
import sys
import time
from copy import deepcopy

# 将项目根目录加入模块搜索路径，确保绘图包 plot 可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split

from DL_config import Config
from dataset import StructuralDataset
from model import build_model
from plot import (
    extract_reference_summaries,
    plot_evaluation,
    plot_history,
    plot_domain_comparison,
    plot_method_comparison,
    plot_paper_figures,
    save_gate_diagnostics,
)
from trainer import EMAModel, make_loader, print_metrics, train_epoch, validate


MODEL_TYPES = ("static_only", "dynamic_only", "fusion")


def config_to_dict(cfg):
    """Return checkpoint-friendly config values."""
    keys = [
        "data_root", "corrected_data_dir", "reference_data_dir",
        "data_dir", "save_root", "log_root", "save_dir", "log_dir",
        "model_type", "run_name", "pos_dim", "static_dim", "dynamic_dim",
        "fusion_dim", "n_materials", "n_supports", "n_regions",
        "n_safety_levels", "support_disp_scale_mm",
        "static_steps", "static_channels", "n_static_nodes",
        "dynamic_channels", "n_dynamic_nodes", "static_node_ids",
        "dynamic_node_ids", "support_nodes", "region_names",
        "temperature_condition_scale",
        "batch_size", "lr", "weight_decay",
        "n_epochs", "early_stop", "min_epochs", "dynamic_only_min_epochs",
        "fusion_min_epochs", "selection_metric", "threshold_tuning",
        "threshold_recall_constraints",
        "train_ratio", "val_ratio", "pos_weight", "aux_weight",
        "support_cls_weight", "support_reg_weight", "region_weight",
        "global_state_weight", "rule_consistency_weight",
        "fusion_branch_weight", "fusion_gate_weight", "fusion_warm_start",
        "fusion_train_alpha_only", "fusion_anchor", "fusion_alpha_init",
        "fusion_alpha_l2_weight", "fusion_gate_bias", "fusion_margin_weight",
        "fusion_margin", "fusion_static_margin_weight",
        "fusion_static_margin", "fusion_modality_dropout",
        "fusion_per_class_gate_bias", "fusion_strong_dynamic_classes",
        "fusion_per_class_gate_weight", "fusion_safety_source",
        "fusion_freeze_dynamic_epochs", "fusion_use_legacy_schedule",
        "split_strategy", "group_by",
        "dynamic_augment", "aug_prob", "aug_noise_std", "aug_amp_scale",
        "aug_time_crop_ratio", "aug_time_shift_ratio", "aug_sensor_dropout",
        "aug_channel_dropout",
        "static_augment", "static_aug_prob", "static_aug_noise_std",
        "static_aug_scale", "static_aug_sensor_dropout",
        "static_only_aug_prob", "static_only_aug_noise_std",
        "static_only_aug_scale", "static_only_aug_sensor_dropout",
        "fusion_static_aug_prob", "fusion_static_aug_noise_std",
        "fusion_static_aug_scale", "fusion_static_aug_sensor_dropout",
        "dynamic_only_aug_prob",
        "dynamic_only_aug_noise_std", "dynamic_only_aug_amp_scale",
        "dynamic_only_aug_time_crop_ratio", "dynamic_only_aug_time_shift_ratio",
        "dynamic_only_aug_sensor_dropout", "dynamic_only_aug_channel_dropout",
        "fusion_aug_prob", "fusion_aug_noise_std", "fusion_aug_amp_scale",
        "fusion_aug_time_crop_ratio", "fusion_aug_time_shift_ratio",
        "fusion_aug_sensor_dropout", "fusion_aug_channel_dropout",
        "label_smoothing", "ema_decay", "seed", "num_workers", "device",
    ]
    return {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}


def split_indices(dataset_or_n, cfg):
    """Split train/val/test indices, optionally isolating sample groups."""
    if isinstance(dataset_or_n, int):
        n_samples = dataset_or_n
        group_labels = None
        sample_targets = None
    else:
        n_samples = len(dataset_or_n)
        group_labels = dataset_or_n.get_group_labels(
            getattr(cfg, "group_by", "damage_pattern")
        )
        sample_targets = dataset_or_n.targets.cpu().numpy()

    if getattr(cfg, "split_strategy", "random") == "group" and group_labels is not None:
        return split_indices_by_group(group_labels, cfg, sample_targets=sample_targets)

    n_train = int(n_samples * cfg.train_ratio)
    n_val = int(n_samples * cfg.val_ratio)
    n_test = n_samples - n_train - n_val
    parts = random_split(
        range(n_samples),
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    return [list(p) for p in parts]


def split_indices_by_group(group_labels, cfg, sample_targets=None):
    """Group-aware split with approximate per-label positive balance."""
    groups = {}
    for idx, group in enumerate(group_labels):
        groups.setdefault(group, []).append(idx)

    if len(groups) < 3:
        print("Warning: too few groups, falling back to random split.")
        return split_indices(len(group_labels), cfg)

    rng = np.random.default_rng(cfg.seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    n_samples = len(group_labels)
    n_train = int(n_samples * cfg.train_ratio)
    n_val = int(n_samples * cfg.val_ratio)
    targets = [n_train, n_val, n_samples - n_train - n_val]
    splits = [[], [], []]
    counts = [0, 0, 0]

    label_counts = None
    label_targets = None
    if sample_targets is not None:
        sample_targets = np.asarray(sample_targets, dtype=np.float32)
        label_counts = [
            np.zeros(sample_targets.shape[1], dtype=np.float32) for _ in range(3)
        ]
        total_pos = sample_targets.sum(axis=0)
        ratios = np.asarray(targets, dtype=np.float32) / max(1, n_samples)
        label_targets = [total_pos * ratio for ratio in ratios]

    for _, indices in items:
        group_pos = sample_targets[indices].sum(axis=0) if sample_targets is not None else None

        def assignment_cost(split_id):
            size_after = counts[split_id] + len(indices)
            size_cost = abs(size_after - targets[split_id]) / max(1, targets[split_id])
            if label_counts is None:
                return size_cost
            pos_after = label_counts[split_id] + group_pos
            denom = max(1.0, float(label_targets[split_id].mean()))
            label_cost = np.abs(pos_after - label_targets[split_id]).mean() / denom
            return size_cost + 0.5 * label_cost

        split_id = min(range(3), key=assignment_cost)
        splits[split_id].extend(indices)
        counts[split_id] += len(indices)
        if label_counts is not None:
            label_counts[split_id] += group_pos

    for split in splits:
        rng.shuffle(split)

    print(
        f"Group split: strategy=group group_by={cfg.group_by} "
        f"groups={len(groups)} sizes={counts}"
    )
    if label_counts is not None:
        for name, pos in zip(("train", "val", "test"), label_counts):
            print(f"  {name} positives: {[int(x) for x in pos.tolist()]}")
    return splits


def make_criterion(dataset, train_idx, cfg, device):
    """Build BCE and regression losses from train-set class balance."""
    targets = dataset.targets[train_idx]
    pos = targets.sum(dim=0)
    neg = targets.size(0) - pos
    weights = (neg / pos.clamp(min=1.0)).clamp(min=1.0, max=float(cfg.pos_weight))
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    return criterion_cls, nn.MSELoss(), weights


def checkpoint_path(cfg):
    return os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt")


def load_checkpoint(model, optimizer, dataset, path, device):
    """Restore model, optimizer, and normalizer state from a checkpoint."""
    ckpt = torch.load(path, map_location=device)
    if "model_ema" in ckpt:
        model.load_state_dict(ckpt["model"])
        model.load_state_dict(
            {k: ckpt["model_ema"][k].to(device) for k in ckpt["model_ema"]},
            strict=False,
        )
    else:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    dataset.load_normalizer_state_dict(ckpt.get("normalizer"))
    return ckpt


def evaluate_model_on_dataset(model, dataset, indices, criterion_cls, criterion_reg,
                              cfg, desc, plot_path=None):
    """Evaluate a loaded model on a dataset without refitting normalizers."""
    device = torch.device(cfg.device)
    static_pos = dataset.static_pos.to(device)
    dynamic_pos = dataset.dynamic_pos.to(device)
    loader = make_loader(dataset, indices, static_pos, dynamic_pos, False, cfg)
    loss, metrics, results = validate(
        model, loader, criterion_cls, criterion_reg, cfg, desc=desc
    )
    if plot_path:
        plot_evaluation(*results, plot_path)
    return loss, metrics, results


def build_external_dataset(cfg, ckpt, data_dir):
    """Load an external test dataset using checkpoint normalizer statistics."""
    if not data_dir or not os.path.isdir(data_dir):
        return None
    dataset = StructuralDataset(data_dir, normalize=True, fit_normalizer=False)
    dataset.load_normalizer_state_dict(ckpt.get("normalizer"))
    dataset.apply_to_config(cfg)
    return dataset


def apply_checkpoint_config(cfg, ckpt_config):
    """Restore checkpoint-time input/output contract into Config."""
    for key, value in (ckpt_config or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def selection_score(metrics, cfg):
    """Score used for checkpointing and early stopping."""
    metric = getattr(cfg, "selection_metric", "f1")
    f1 = metrics.get("f1", 0.0)
    auc = metrics.get("auc", 0.0)
    if metric == "auc":
        return auc
    if metric == "f1_auc":
        return 0.7 * f1 + 0.3 * auc
    return f1


def warm_start_fusion_branches(model, cfg, device):
    """Initialize fusion branches and heads from single-branch checkpoints."""
    if cfg.model_type != "fusion" or not getattr(cfg, "fusion_warm_start", False):
        return False

    loaded_branches = set()
    candidates = [
        (
            "static",
            model.static_branch,
            model.fusion.static_head,
            Config.method_dir(cfg.save_root, "static_only"),
            "static_only_best.pt",
        ),
        (
            "dynamic",
            model.dynamic_branch,
            model.fusion.dynamic_head,
            Config.method_dir(cfg.save_root, "dynamic_only"),
            "dynamic_only_best.pt",
        ),
    ]

    for name, branch, head, folder, filename in candidates:
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            print(f"  -> skip warm start {name}: checkpoint not found: {path}")
            continue
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("model_ema") or ckpt.get("model")

        prefix = f"{name}_branch."
        branch_state = {
            k[len(prefix):]: v.to(device)
            for k, v in state.items()
            if k.startswith(prefix)
        }
        current_state = branch.state_dict()
        branch_state = {
            k: v
            for k, v in branch_state.items()
            if k in current_state and current_state[k].shape == v.shape
        }
        if branch_state:
            missing, unexpected = branch.load_state_dict(branch_state, strict=False)
            print(f"  -> warm start {name}_branch from {path}")
            loaded_branches.add(name)
            if missing or unexpected:
                print(f"     missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"  -> skip warm start {name}_branch: no compatible weights")

        head_prefix = "head."
        head_state = {
            k[len(head_prefix):]: v.to(device)
            for k, v in state.items()
            if k.startswith(head_prefix)
        }
        head_state = {
            k.replace("material_classifier", "classifier")
             .replace("material_regressor", "regressor"): v
            for k, v in head_state.items()
            if k.startswith("material_classifier") or k.startswith("material_regressor")
        }
        current_head = head.state_dict()
        head_state = {
            k: v
            for k, v in head_state.items()
            if k in current_head and current_head[k].shape == v.shape
        }
        if head_state:
            missing, unexpected = head.load_state_dict(head_state, strict=False)
            print(f"  -> warm start fusion.{name}_head from {path}")
            if missing or unexpected:
                print(f"     head missing={len(missing)} unexpected={len(unexpected)}")

        if name == "static" and hasattr(model, "safety_head"):
            safety_state = {
                k[len("head."):]: v.to(device)
                for k, v in state.items()
                if k.startswith("head.")
            }
            current_safety = model.safety_head.state_dict()
            safety_state = {
                k: v
                for k, v in safety_state.items()
                if k in current_safety and current_safety[k].shape == v.shape
            }
            if safety_state:
                missing, unexpected = model.safety_head.load_state_dict(safety_state, strict=False)
                print(f"  -> warm start fusion.safety_head from {path}")
                if missing or unexpected:
                    print(f"     safety head missing={len(missing)} unexpected={len(unexpected)}")

    return loaded_branches == {"static", "dynamic"}


def set_fusion_alpha_only_trainable(model):
    """Freeze the fusion model except the class-wise alpha logits."""
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "fusion") and hasattr(model.fusion, "alpha_logit"):
        model.fusion.alpha_logit.requires_grad = True
        print("  -> trainable parameters: fusion.alpha_logit only")


def set_fusion_dynamic_trainable(model, trainable):
    """Freeze or unfreeze the dynamic branch and its auxiliary head."""
    if not hasattr(model, "dynamic_branch") or not hasattr(model, "fusion"):
        return
    for param in model.dynamic_branch.parameters():
        param.requires_grad = trainable
    for param in model.fusion.dynamic_head.parameters():
        param.requires_grad = trainable


def _save_checkpoint(cfg, ckpt_data, dataset, pos_weight, train_idx, val_idx, test_idx,
                     model, optimizer, history, best_val_f1, best_score, ema):
    """Persist a checkpoint to disk."""
    ckpt_data.update({
        "model": deepcopy(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "best_f1": best_val_f1,
        "best_score": best_score,
        "config": config_to_dict(cfg),
        "history": history,
        "normalizer": dataset.normalizer_state_dict(),
        "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
        "pos_weight": pos_weight.cpu(),
    })
    if hasattr(cfg, "eval_thresholds"):
        ckpt_data["eval_thresholds"] = np.asarray(cfg.eval_thresholds, dtype=np.float32).tolist()
    if ema is not None:
        ckpt_data["model_ema"] = {k: v.cpu().clone() for k, v in ema.shadow.items()}
    path = checkpoint_path(cfg)
    if os.path.exists(path):
        os.remove(path)
    torch.save(ckpt_data, path)


def _run_training_stage(model, stage_name, total_epochs, lr, train_loader, val_loader,
                        criterion_cls, criterion_reg, cfg, device, history,
                        best_val_f1, best_score, patience, pos_weight,
                        dataset, train_idx, val_idx, test_idx,
                        ema=None, effective_min_epochs=0):
    """Run one training stage and return updated tracking state."""
    print("\n" + "=" * 72)
    print(f"Stage: {stage_name}  ({total_epochs} epochs, lr={lr:.1e})")
    print("=" * 72)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    best_state = deepcopy(model.state_dict())
    epoch_offset = len(history)
    stage_patience = 0

    for epoch in range(total_epochs):
        global_epoch = epoch_offset + epoch
        t0 = time.time()

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_reg, cfg, global_epoch, ema=ema,
        )

        if ema is not None:
            ema.apply_shadow(model)
        val_loss, val_metrics, _ = validate(
            model, val_loader, criterion_cls, criterion_reg, cfg,
            desc=f"Val {stage_name}", tune_thresholds=True,
        )
        if ema is not None:
            ema.restore(model)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        val_f1 = val_metrics.get("f1", 0.0)
        val_score = selection_score(val_metrics, cfg)

        print(
            f"[{stage_name}] E{global_epoch:3d} | "
            f"tr_loss={train_loss:.4f} tr_f1={train_metrics.get('f1', 0):.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
            f"score={val_score:.4f} | lr={lr_now:.2e} | {elapsed:.0f}s"
        )

        history.append({
            "epoch": global_epoch,
            "stage": stage_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_f1": train_metrics.get("f1", 0.0),
            "val_f1": val_f1,
            "train_acc": train_metrics.get("accuracy", 0.0),
            "val_acc": val_metrics.get("accuracy", 0.0),
            "train_precision": train_metrics.get("precision", 0.0),
            "val_precision": val_metrics.get("precision", 0.0),
            "train_recall": train_metrics.get("recall", 0.0),
            "val_recall": val_metrics.get("recall", 0.0),
            "train_auc": train_metrics.get("auc", 0.0),
            "val_auc": val_metrics.get("auc", 0.0),
            "val_score": val_score,
            "lr": lr_now,
        })

        if val_score > best_score:
            best_val_f1 = val_f1
            best_score = val_score
            best_state = deepcopy(model.state_dict())
            stage_patience = 0
            _save_checkpoint(cfg, {"epoch": global_epoch}, dataset, pos_weight,
                             train_idx, val_idx, test_idx, model, optimizer,
                             history, best_val_f1, best_score, ema)
            print(f"  -> saved best model (f1={best_val_f1:.4f}, score={best_score:.4f})")
        else:
            if epoch + 1 >= effective_min_epochs:
                stage_patience += 1
            if epoch + 1 >= effective_min_epochs and stage_patience >= cfg.early_stop:
                print(f"  early stopping at epoch {global_epoch}")
                break

    model.load_state_dict(best_state)
    print(f"\n[{stage_name}] Best val F1: {best_val_f1:.4f}  score: {best_score:.4f}")
    return best_val_f1, best_score, history


def train_one_model(model_type, base_cfg, dataset, train_idx, val_idx, test_idx, args,
                    reference_dataset=None):
    """Train or test one model type."""
    cfg = deepcopy(base_cfg)
    cfg.set_model_type(model_type)

    device = torch.device(cfg.device)
    dataset.fit_normalizer(train_idx)
    static_pos = dataset.static_pos.to(device)
    dynamic_pos = dataset.dynamic_pos.to(device)

    train_loader = make_loader(dataset, train_idx, static_pos, dynamic_pos, True, cfg)
    val_loader = make_loader(dataset, val_idx, static_pos, dynamic_pos, False, cfg)
    test_loader = make_loader(dataset, test_idx, static_pos, dynamic_pos, False, cfg)

    model = build_model(cfg).to(device)
    criterion_cls, criterion_reg, pos_weight = make_criterion(dataset, train_idx, cfg, device)

    print("\n" + "=" * 72)
    print(f"Model type: {model_type}")
    print(f"Log dir: {cfg.log_dir}")
    print(f"Train pos_weight: {[round(x, 3) for x in pos_weight.tolist()]}")
    if cfg.ema_decay > 0:
        print(f"EMA: decay={cfg.ema_decay}  Label smoothing: {cfg.label_smoothing}")
    print("=" * 72)

    ckpt_path = args.resume or checkpoint_path(cfg)

    if args.mode == "test":
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return None
        ckpt = load_checkpoint(model, None, dataset, ckpt_path, device)
        if hasattr(model, 'set_stage'):
            model.set_stage('finetune')
        if "eval_thresholds" in ckpt:
            cfg.eval_thresholds = np.asarray(ckpt["eval_thresholds"], dtype=np.float32)
        print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")
        test_loss, test_metrics, test_results = validate(
            model, test_loader, criterion_cls, criterion_reg, cfg, desc=f"Test {model_type}"
        )
        print_metrics(test_metrics)
        plot_evaluation(*test_results, os.path.join(cfg.log_dir, f"{cfg.run_name}_evaluation.png"))
        summary = {"model_type": model_type, "test_loss": test_loss, **test_metrics}
        if reference_dataset is not None:
            reference_dataset.load_normalizer_state_dict(ckpt.get("normalizer"))
            reference_dataset.apply_to_config(cfg)
            ref_idx = list(range(len(reference_dataset)))
            ref_loss, ref_metrics, _ = evaluate_model_on_dataset(
                model,
                reference_dataset,
                ref_idx,
                criterion_cls,
                criterion_reg,
                cfg,
                desc=f"Reference Test {model_type}",
                plot_path=os.path.join(cfg.log_dir, f"{cfg.run_name}_reference_evaluation.png"),
            )
            print(f"\n[{model_type}] Reference-model test results")
            print_metrics(ref_metrics)
            summary["reference_test_loss"] = ref_loss
            summary.update({f"reference_{k}": v for k, v in ref_metrics.items()})
        return summary

    history = []
    best_val_f1 = float("-inf")
    best_score = float("-inf")

    if model_type == "fusion" and getattr(cfg, "fusion_use_legacy_schedule", False):
        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            best_val_f1 = ckpt.get("best_f1", 0.0)
            best_score = ckpt.get("best_score", best_val_f1)
            history = ckpt.get("history", [])
            print(f"Resumed from {args.resume}")
            # jump directly to stage 3 (finetune) for final evaluation
            model.set_stage("finetune")
        else:
            # ── Stage 1: Joint branch pretraining ──
            model.set_stage("pretrain")
            best_val_f1, best_score, history = _run_training_stage(
                model, "S1-pretrain",
                total_epochs=cfg.fusion_stage1_epochs,
                lr=cfg.fusion_stage1_lr,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                cfg=cfg, device=device,
                history=history,
                best_val_f1=best_val_f1,
                best_score=best_score,
                patience=0,
                pos_weight=pos_weight,
                dataset=dataset, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                ema=None,
                effective_min_epochs=cfg.fusion_stage1_epochs // 3,
            )

            # ── Stage 2: Fusion head training (branches frozen) ──
            model.set_stage("fusion_head")
            for p in model.static_branch.parameters():
                p.requires_grad = False
            for p in model.dynamic_branch.parameters():
                p.requires_grad = False
            best_val_f1, best_score, history = _run_training_stage(
                model, "S2-fusion_head",
                total_epochs=cfg.fusion_stage2_epochs,
                lr=cfg.fusion_stage2_lr,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                cfg=cfg, device=device,
                history=history,
                best_val_f1=best_val_f1,
                best_score=best_score,
                patience=0,
                pos_weight=pos_weight,
                dataset=dataset, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                ema=None,
                effective_min_epochs=cfg.fusion_stage2_epochs // 3,
            )

            # ── Stage 3: End-to-end fine-tuning ──
            model.set_stage("finetune")
            for p in model.static_branch.parameters():
                p.requires_grad = True
            for p in model.dynamic_branch.parameters():
                p.requires_grad = True
            ema = EMAModel(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None
            best_val_f1, best_score, history = _run_training_stage(
                model, "S3-finetune",
                total_epochs=cfg.fusion_stage3_epochs,
                lr=cfg.fusion_stage3_lr,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                cfg=cfg, device=device,
                history=history,
                best_val_f1=best_val_f1,
                best_score=best_score,
                patience=0,
                pos_weight=pos_weight,
                dataset=dataset, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                ema=ema,
                effective_min_epochs=cfg.fusion_stage3_epochs // 3,
            )

    else:
        # static_only / dynamic_only: single-stage training
        if hasattr(model, "set_stage"):
            model.set_stage("finetune")
        fusion_warm_started = False
        if model_type == "fusion":
            fusion_warm_started = warm_start_fusion_branches(model, cfg, device)
            if fusion_warm_started and getattr(cfg, "fusion_train_alpha_only", False):
                set_fusion_alpha_only_trainable(model)
        effective_min_epochs = getattr(cfg, f"{model_type}_min_epochs", getattr(cfg, "min_epochs", 0))
        use_ema = cfg.ema_decay > 0 and not (
            model_type == "fusion"
            and fusion_warm_started
            and getattr(cfg, "fusion_train_alpha_only", False)
        )
        ema = EMAModel(model, decay=cfg.ema_decay) if use_ema else None
        if args.resume:
            ckpt = load_checkpoint(model, None, dataset, args.resume, device)
            best_val_f1 = ckpt.get("best_f1", 0.0)
            best_score = ckpt.get("best_score", best_val_f1)
            history = ckpt.get("history", [])
            print(f"Resumed from {args.resume}")
        best_val_f1, best_score, history = _run_training_stage(
            model, model_type,
            total_epochs=cfg.n_epochs,
            lr=cfg.lr,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            cfg=cfg, device=device,
            history=history,
            best_val_f1=best_val_f1,
            best_score=best_score,
            patience=0,
            pos_weight=pos_weight,
            dataset=dataset, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            ema=ema,
            effective_min_epochs=effective_min_epochs,
        )

    print(f"\n[{model_type}] Best validation F1: {best_val_f1:.4f}")
    plot_history(history, os.path.join(cfg.log_dir, f"{cfg.run_name}_history.png"))

    # ── Final test evaluation ──
    best_ckpt = torch.load(checkpoint_path(cfg), map_location=device)
    if "eval_thresholds" in best_ckpt:
        cfg.eval_thresholds = np.asarray(best_ckpt["eval_thresholds"], dtype=np.float32)
    if "model_ema" in best_ckpt:
        model.load_state_dict(best_ckpt["model"])
        model.load_state_dict(
            {k: best_ckpt["model_ema"][k].to(device) for k in best_ckpt["model_ema"]},
            strict=False,
        )
        print(f"[{model_type}] Testing with EMA weights")
    else:
        model.load_state_dict(best_ckpt["model"])
    if hasattr(model, 'set_stage'):
        model.set_stage('finetune')

    test_loss, test_metrics, test_results = validate(
        model, test_loader, criterion_cls, criterion_reg, cfg, desc=f"Test {model_type}"
    )

    print(f"\n[{model_type}] Test results")
    print_metrics(test_metrics)
    plot_evaluation(*test_results, os.path.join(cfg.log_dir, f"{cfg.run_name}_evaluation.png"))
    if model_type == "fusion":
        save_gate_diagnostics(
            test_metrics,
            os.path.join(cfg.log_dir, f"{cfg.run_name}_fusion_diagnostics.json"),
        )

    summary = {
        "model_type": model_type,
        "best_val_f1": best_val_f1,
        "test_loss": test_loss,
        **test_metrics,
    }
    if reference_dataset is not None:
        reference_dataset.load_normalizer_state_dict(best_ckpt.get("normalizer"))
        reference_dataset.apply_to_config(cfg)
        ref_idx = list(range(len(reference_dataset)))
        ref_loss, ref_metrics, _ = evaluate_model_on_dataset(
            model,
            reference_dataset,
            ref_idx,
            criterion_cls,
            criterion_reg,
            cfg,
            desc=f"Reference Test {model_type}",
            plot_path=os.path.join(cfg.log_dir, f"{cfg.run_name}_reference_evaluation.png"),
        )
        print(f"\n[{model_type}] Reference-model test results")
        print_metrics(ref_metrics)
        summary["reference_test_loss"] = ref_loss
        summary.update({f"reference_{k}": v for k, v in ref_metrics.items()})
        if model_type == "fusion":
            save_gate_diagnostics(
                ref_metrics,
                os.path.join(cfg.log_dir, f"{cfg.run_name}_reference_fusion_diagnostics.json"),
            )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", choices=["train", "test", "infer"])
    parser.add_argument(
        "--model",
        default="all",
        choices=["fusion", "static_only", "dynamic_only", "all"],
        help="Model type to train/test. all runs static_only, dynamic_only, fusion.",
    )
    parser.add_argument("--input", default=None, help="JSON file for inference")
    parser.add_argument("--resume", default=None, help="Checkpoint path for resume/test/infer")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--corrected-data-dir",
        default=None,
        help="Corrected-model dataset used for train/val/same-domain test.",
    )
    parser.add_argument(
        "--reference-data-dir",
        default=None,
        help="Reference-model dataset used only for cross-domain testing.",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.corrected_data_dir:
        cfg.corrected_data_dir = args.corrected_data_dir
    if args.reference_data_dir:
        cfg.reference_data_dir = args.reference_data_dir
    cfg.data_dir = cfg.corrected_data_dir
    if args.epochs:
        cfg.n_epochs = args.epochs
    if args.batch:
        cfg.batch_size = args.batch
    if args.lr:
        cfg.lr = args.lr
    cfg.seed = args.seed
    if args.model != "all":
        cfg.set_model_type(args.model)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Device: {device}")
    print(f"Config: {config_to_dict(cfg)}")
    print()

    if args.mode == "infer":
        if not args.input:
            print("Please provide --input path.json for inference.")
            return
        ckpt_path = args.resume or checkpoint_path(cfg)
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location=device)
        apply_checkpoint_config(cfg, ckpt.get("config"))
        model_type = ckpt.get("config", {}).get("model_type", cfg.model_type)
        cfg.set_model_type(model_type)
        dataset = StructuralDataset(cfg.data_dir, normalize=True, fit_normalizer=False)
        dataset.apply_to_config(cfg)
        model = build_model(cfg).to(device)
        if "model_ema" in ckpt:
            model.load_state_dict(ckpt["model"])
            model.load_state_dict(
                {k: ckpt["model_ema"][k].to(device) for k in ckpt["model_ema"]},
                strict=False,
            )
        else:
            model.load_state_dict(ckpt["model"])
        thresholds = ckpt.get("eval_thresholds")
        model.eval()

        dataset.load_normalizer_state_dict(ckpt.get("normalizer"))
        with open(args.input, encoding="utf-8") as f:
            raw = json.load(f)

        from predict import predict_one, print_result

        result = predict_one(
            model, raw, dataset, device, thresholds=thresholds, return_full=True
        )
        print_result(args.input, result)
        return

    print("Loading corrected-model dataset...")
    dataset = StructuralDataset(cfg.corrected_data_dir, normalize=True, fit_normalizer=False)
    dataset.apply_to_config(cfg)
    train_idx, val_idx, test_idx = split_indices(dataset, cfg)
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    reference_dataset = None
    if os.path.isdir(getattr(cfg, "reference_data_dir", "")):
        print("Loading reference-model dataset for cross-domain testing...")
        reference_dataset = StructuralDataset(
            cfg.reference_data_dir, normalize=True, fit_normalizer=False
        )
        print(f"Reference test samples: {len(reference_dataset)}")
    else:
        print(f"Reference dataset not found, skipped: {getattr(cfg, 'reference_data_dir', '')}")

    model_types = MODEL_TYPES if args.model == "all" else (cfg.model_type,)
    if args.resume and args.model == "all":
        print("Warning: --resume is ignored in --model all mode.")
        args.resume = None

    summaries = []
    for model_type in model_types:
        summary = train_one_model(
            model_type,
            cfg,
            dataset,
            train_idx,
            val_idx,
            test_idx,
            args,
            reference_dataset=reference_dataset,
        )
        if summary is not None:
            summaries.append(summary)

    if len(summaries) > 1:
        print("\n" + "=" * 72)
        print("Method comparison")
        print("=" * 72)
        for item in summaries:
            print(
                f"{item['model_type']:>12} | "
                f"val_f1={item.get('best_val_f1', 0):.4f} | "
                f"test_f1={item.get('f1', 0):.4f} | "
                f"auc={item.get('auc', 0):.4f} | "
                f"recall={item.get('recall', 0):.4f}"
            )

        summary_path = os.path.join(cfg.log_root, "method_comparison.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        plot_method_comparison(summaries, os.path.join(cfg.log_root, "method_comparison.png"))
        plot_paper_figures(summaries, cfg.log_root)
        reference_summaries = extract_reference_summaries(summaries)
        if reference_summaries:
            plot_paper_figures(reference_summaries, cfg.log_root, prefix="reference_paper")
        plot_domain_comparison(
            summaries,
            os.path.join(cfg.log_root, "domain_comparison.png"),
        )


if __name__ == "__main__":
    main()
