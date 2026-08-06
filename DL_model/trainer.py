"""Training and evaluation utilities for multi-condition safety assessment."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from DL_config import CANDIDATE_IDS, SUPPORT_NODES


class EMAModel:
    """Exponential moving average of trainable weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.detach(), alpha=1.0 - self.decay
                )

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()


def find_best_thresholds(y_true, y_prob, grid=None, recall_constraints=None):
    """Find per-label thresholds that maximize validation F1."""

    if grid is None:
        grid = np.linspace(0.02, 0.95, 94)
    recall_constraints = recall_constraints or {}
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for i in range(y_true.shape[1]):
        best_f1 = -1.0
        best_t = 0.5
        recall_floor = recall_constraints.get(i, 0.0)
        if recall_floor > 0.0:
            for threshold in grid:
                pred = (y_prob[:, i] > threshold).astype("float32")
                rec = recall_score(y_true[:, i], pred, zero_division=0)
                if rec < recall_floor:
                    continue
                score = f1_score(y_true[:, i], pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = threshold
            if best_f1 < 0.0:
                recall_floor = 0.0
                best_f1 = -1.0
        if recall_floor <= 0.0:
            for threshold in grid:
                pred = (y_prob[:, i] > threshold).astype("float32")
                score = f1_score(y_true[:, i], pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = threshold
        thresholds[i] = best_t
    return thresholds


def compute_multilabel_metrics(y_true, y_pred, y_prob, label_ids, prefix=""):
    results = {}
    p = f"{prefix}_" if prefix else ""
    results[f"{p}exact_match"] = accuracy_score(y_true, y_pred)
    results[f"{p}accuracy"] = float((y_true == y_pred).mean())
    results[f"{p}f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results[f"{p}f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    results[f"{p}f1"] = results[f"{p}f1_macro"]
    results[f"{p}precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    results[f"{p}recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    for i, label_id in enumerate(label_ids):
        results[f"{p}f1_{label_id}"] = f1_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        )

    cols = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
    if cols:
        try:
            results[f"{p}auc"] = roc_auc_score(
                y_true[:, cols], y_prob[:, cols], average="macro"
            )
        except Exception:
            results[f"{p}auc"] = 0.0
    else:
        results[f"{p}auc"] = 0.0
    return results


def compute_metrics(y_true, y_pred, y_prob):
    """Backward-compatible material metrics."""

    return compute_multilabel_metrics(y_true, y_pred, y_prob, CANDIDATE_IDS)


def compute_full_metrics(
    mat_true,
    mat_pred,
    mat_prob,
    support_true=None,
    support_pred=None,
    support_prob=None,
    region_true=None,
    region_pred=None,
    support_disp_true=None,
    support_disp_pred=None,
):
    metrics = compute_multilabel_metrics(mat_true, mat_pred, mat_prob, CANDIDATE_IDS)
    if support_true is not None:
        metrics.update(
            compute_multilabel_metrics(
                support_true, support_pred, support_prob, SUPPORT_NODES, prefix="support"
            )
        )
    if region_true is not None:
        metrics["region_accuracy"] = float((region_true == region_pred).mean())
        metrics["region_f1_macro"] = f1_score(
            region_true.reshape(-1),
            region_pred.reshape(-1),
            average="macro",
            zero_division=0,
        )
    if support_disp_true is not None:
        err = np.abs(support_disp_pred - support_disp_true)
        metrics["support_disp_mae_mm"] = float(err.mean())
        metrics["support_disp_maxe_mm"] = float(err.max())
    return metrics


def print_metrics(metrics, prefix=""):
    for k, v in metrics.items():
        print(f"  {prefix}{k}: {v:.4f}" if prefix else f"  {k}: {v:.4f}")


def collate_fn(batch, static_pos, dynamic_pos):
    keys = [
        "disp",
        "strain",
        "ace",
        "target",
        "raw_sf",
        "support_disp",
        "support_target",
        "region_target",
        "global_target",
        "condition",
        "temperature_C",
    ]
    out = {key: torch.stack([b[key] for b in batch]) for key in keys}
    out["static_pos"] = static_pos
    out["dynamic_pos"] = dynamic_pos
    return out


def make_loader(dataset, indices, static_pos, dynamic_pos, shuffle, config):
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, static_pos, dynamic_pos),
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )


def _maybe_apply(prob):
    return torch.rand(()) < float(prob)


def _aug_value(config, name, default):
    model_type = getattr(config, "model_type", "")
    specific = f"{model_type}_{name}"
    if hasattr(config, specific):
        return getattr(config, specific)
    return getattr(config, name, default)


def augment_dynamic_response(ace, config):
    if not getattr(config, "dynamic_augment", False):
        return ace
    if not _maybe_apply(_aug_value(config, "aug_prob", 1.0)):
        return ace

    out = ace
    batch_size, n_steps, n_sensors, n_channels = out.shape

    amp_scale = _aug_value(config, "aug_amp_scale", 0.0)
    if amp_scale > 0:
        scale = 1.0 + (
            torch.rand(batch_size, 1, 1, 1, device=out.device) * 2.0 - 1.0
        ) * amp_scale
        out = out * scale

    shift_ratio = _aug_value(config, "aug_time_shift_ratio", 0.0)
    if shift_ratio > 0 and n_steps > 1:
        max_shift = max(1, int(n_steps * shift_ratio))
        shifts = torch.randint(-max_shift, max_shift + 1, (batch_size,), device=out.device)
        out = torch.stack(
            [torch.roll(out[i], shifts=int(shift.item()), dims=0) for i, shift in enumerate(shifts)],
            dim=0,
        )

    crop_ratio = _aug_value(config, "aug_time_crop_ratio", 1.0)
    if 0 < crop_ratio < 1.0 and n_steps > 4:
        crop_len = max(4, int(n_steps * crop_ratio))
        starts = torch.randint(0, n_steps - crop_len + 1, (batch_size,), device=out.device)
        cropped = []
        for i, start in enumerate(starts.tolist()):
            segment = out[i, start:start + crop_len].permute(1, 2, 0)
            segment = segment.reshape(1, n_sensors * n_channels, crop_len)
            segment = F.interpolate(segment, size=n_steps, mode="linear", align_corners=False)
            cropped.append(segment.reshape(n_sensors, n_channels, n_steps).permute(2, 0, 1))
        out = torch.stack(cropped, dim=0)

    sensor_drop = _aug_value(config, "aug_sensor_dropout", 0.0)
    if sensor_drop > 0:
        mask = (torch.rand(batch_size, 1, n_sensors, 1, device=out.device) > sensor_drop).float()
        keep_any = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        out = out * mask * (n_sensors / keep_any)

    channel_drop = _aug_value(config, "aug_channel_dropout", 0.0)
    if channel_drop > 0:
        mask = (torch.rand(batch_size, 1, 1, n_channels, device=out.device) > channel_drop).float()
        keep_any = mask.sum(dim=3, keepdim=True).clamp(min=1.0)
        out = out * mask * (n_channels / keep_any)

    noise_std = _aug_value(config, "aug_noise_std", 0.0)
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std

    return out


def augment_static_response(disp, strain, config):
    if not getattr(config, "static_augment", False):
        return disp, strain
    if not _maybe_apply(_aug_value(config, "static_aug_prob", 1.0)):
        return disp, strain

    batch_size, _n_steps, n_nodes, _n_channels = disp.shape
    scale_val = _aug_value(config, "static_aug_scale", 0.0)
    if scale_val > 0:
        scale = 1.0 + (
            torch.rand(batch_size, 1, 1, 1, device=disp.device) * 2.0 - 1.0
        ) * scale_val
        disp = disp * scale
        strain = strain * scale

    sensor_drop = _aug_value(config, "static_aug_sensor_dropout", 0.0)
    if sensor_drop > 0:
        mask = (torch.rand(batch_size, 1, n_nodes, 1, device=disp.device) > sensor_drop).float()
        keep_any = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        disp = disp * mask * (n_nodes / keep_any)
        strain = strain * mask * (n_nodes / keep_any)

    noise_std = _aug_value(config, "static_aug_noise_std", 0.0)
    if noise_std > 0:
        disp = disp + torch.randn_like(disp) * noise_std
        strain = strain + torch.randn_like(strain) * noise_std
    return disp, strain


def apply_fusion_modality_dropout(batch, config):
    batch["drop_static_aux"] = False
    batch["drop_dynamic_aux"] = False
    prob = getattr(config, "fusion_modality_dropout", 0.0)
    if getattr(config, "model_type", "") != "fusion" or prob <= 0:
        return
    if not _maybe_apply(prob):
        return
    if torch.rand(()) < 0.5:
        batch["ace"] = batch["ace"] * 0.0
        batch["drop_dynamic_aux"] = True
    else:
        batch["disp"] = batch["disp"] * 0.0
        batch["strain"] = batch["strain"] * 0.0
        batch["drop_static_aux"] = True


def contrastive_loss(feat_a, feat_b, temperature=0.07):
    batch_size = feat_a.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=feat_a.device, requires_grad=True)
    a = F.normalize(feat_a, dim=1)
    b = F.normalize(feat_b, dim=1)
    logits = torch.matmul(a, b.T) / temperature
    labels = torch.arange(batch_size, device=feat_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def _smooth_target(target, eps):
    if eps <= 0:
        return target
    return target * (1.0 - eps) + 0.5 * eps


def _move_batch(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)


def multitask_loss(output, batch, criterion_cls, criterion_reg, config, target_smooth):
    material_logits = output["material_logits"]
    material_reg = output["material_reg"]
    support_logits = output["support_logits"]
    support_reg = output["support_reg"]
    region_logits = output["region_logits"]
    global_logits = output["global_logits"]

    target = batch["target"]
    raw_sf = batch["raw_sf"]
    mask = (target > 0).float()

    loss_cls = criterion_cls(material_logits, target_smooth)
    loss_reg = criterion_reg(material_reg * mask, raw_sf * mask)
    support_loss = F.binary_cross_entropy_with_logits(
        support_logits, target=batch["support_target"]
    )
    support_target_scaled = batch["support_disp"] / float(config.support_disp_scale_mm)
    support_reg_loss = criterion_reg(torch.sigmoid(support_reg), support_target_scaled)

    region_loss = F.cross_entropy(
        region_logits.reshape(-1, config.n_safety_levels),
        batch["region_target"].reshape(-1),
    )
    global_loss = F.cross_entropy(global_logits, batch["global_target"])

    # Rule consistency: predicted global level should not be lower than the
    # maximum predicted region level in expectation.
    levels = torch.arange(config.n_safety_levels, device=global_logits.device).float()
    region_expected = torch.softmax(region_logits, dim=-1).matmul(levels).max(dim=1).values
    global_expected = torch.softmax(global_logits, dim=-1).matmul(levels)
    consistency = F.relu(region_expected - global_expected).mean()

    loss = (
        loss_cls
        + config.aux_weight * loss_reg
        + config.support_cls_weight * support_loss
        + config.support_reg_weight * support_reg_loss
        + config.region_weight * region_loss
        + config.global_state_weight * global_loss
        + config.rule_consistency_weight * consistency
    )
    return loss


def _material_logits_from_output(output):
    return output["material_logits"]


def train_epoch(model, loader, optimizer, criterion_cls, criterion_reg, config, epoch, ema=None):
    model.train()
    total_loss = 0.0
    all_targets, all_probs = [], []
    smoothing = getattr(config, "label_smoothing", 0.0)
    is_pretrain = getattr(model, "stage", None) == "pretrain"

    for batch in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        _move_batch(batch, config.device)

        if getattr(config, "model_type", "") in ("dynamic_only", "fusion"):
            batch["ace"] = augment_dynamic_response(batch["ace"], config)
        if getattr(config, "model_type", "") in ("static_only", "fusion"):
            batch["disp"], batch["strain"] = augment_static_response(
                batch["disp"], batch["strain"], config
            )
        if getattr(config, "model_type", "") == "fusion" and not is_pretrain:
            apply_fusion_modality_dropout(batch, config)

        target = batch["target"]
        raw_sf = batch["raw_sf"]
        target_smooth = _smooth_target(target, smoothing)
        mask = (target > 0).float()

        if is_pretrain:
            static_logits, static_reg, dynamic_logits, dynamic_reg, static_feat, dynamic_feat = model(batch)
            loss_static = criterion_cls(static_logits, target_smooth)
            loss_dynamic = criterion_cls(dynamic_logits, target_smooth)
            loss_reg_static = criterion_reg(static_reg * mask, raw_sf * mask)
            loss_reg_dynamic = criterion_reg(dynamic_reg * mask, raw_sf * mask)
            loss_contrast = contrastive_loss(
                static_feat,
                dynamic_feat,
                getattr(config, "fusion_contrastive_temperature", 0.07),
            )
            loss = (
                loss_static
                + loss_dynamic
                + config.aux_weight * (loss_reg_static + loss_reg_dynamic)
                + getattr(config, "fusion_contrastive_weight", 0.15) * loss_contrast
            )
            logits_for_metrics = static_logits
        else:
            output = model(batch)
            logits_for_metrics = _material_logits_from_output(output)
            loss = multitask_loss(output, batch, criterion_cls, criterion_reg, config, target_smooth)

            aux = getattr(model, "last_aux", None)
            if aux is not None:
                branch_weight = getattr(config, "fusion_branch_weight", 0.0)
                if branch_weight > 0:
                    branch_losses = []
                    if not batch.get("drop_static_aux", False):
                        branch_losses.append(criterion_cls(aux["static_logits"], target_smooth))
                    if not batch.get("drop_dynamic_aux", False):
                        branch_losses.append(criterion_cls(aux["dynamic_logits"], target_smooth))
                    if branch_losses:
                        loss = loss + branch_weight * sum(branch_losses)

                alpha = aux.get("fusion_alpha")
                alpha_l2_weight = getattr(config, "fusion_alpha_l2_weight", 0.0)
                if alpha is not None and alpha_l2_weight > 0:
                    target_alpha = torch.as_tensor(
                        getattr(config, "fusion_alpha_init", [0.7] * alpha.numel()),
                        device=alpha.device,
                        dtype=alpha.dtype,
                    )
                    loss = loss + alpha_l2_weight * F.mse_loss(alpha, target_alpha)

                gate_weight = getattr(config, "fusion_gate_weight", 0.0)
                if gate_weight > 0 and "gate" in aux:
                    disagreement = aux.get("branch_disagreement")
                    gate_penalty = (
                        aux["gate"] * (1.0 - disagreement.detach())
                    ).mean() if disagreement is not None else aux["gate"].mean()
                    loss = loss + gate_weight * gate_penalty

                per_class_gate_weight = getattr(config, "fusion_per_class_gate_weight", 0.0)
                strong_dynamic_classes = getattr(config, "fusion_strong_dynamic_classes", None)
                if per_class_gate_weight > 0 and strong_dynamic_classes and "gate" in aux:
                    loss = loss + per_class_gate_weight * aux["gate"][:, strong_dynamic_classes].mean()

                pos_weight = getattr(criterion_cls, "pos_weight", None)
                margin_weight = getattr(config, "fusion_margin_weight", 0.0)
                if margin_weight > 0 and not batch.get("drop_dynamic_aux", False):
                    fused_loss = F.binary_cross_entropy_with_logits(
                        logits_for_metrics, target_smooth, pos_weight=pos_weight, reduction="mean"
                    )
                    dynamic_loss = F.binary_cross_entropy_with_logits(
                        aux["dynamic_logits"], target_smooth, pos_weight=pos_weight, reduction="mean"
                    )
                    loss = loss + margin_weight * F.relu(
                        fused_loss - dynamic_loss + getattr(config, "fusion_margin", 0.0)
                    )
                static_margin_weight = getattr(config, "fusion_static_margin_weight", 0.0)
                if static_margin_weight > 0 and not batch.get("drop_static_aux", False):
                    fused_loss = F.binary_cross_entropy_with_logits(
                        logits_for_metrics, target_smooth, pos_weight=pos_weight, reduction="mean"
                    )
                    static_loss = F.binary_cross_entropy_with_logits(
                        aux["static_logits"], target_smooth, pos_weight=pos_weight, reduction="mean"
                    )
                    loss = loss + static_margin_weight * F.relu(
                        fused_loss - static_loss + getattr(config, "fusion_static_margin", 0.0)
                    )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        all_targets.append(target.detach().cpu())
        all_probs.append(torch.sigmoid(logits_for_metrics).detach().cpu())

    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_pred = (y_prob > 0.5).astype("float32")
    return total_loss / len(loader), compute_metrics(y_true, y_pred, y_prob)


@torch.no_grad()
def validate(model, loader, criterion_cls, criterion_reg, config, desc="Val", tune_thresholds=False):
    model.eval()
    total_loss = 0.0
    is_pretrain = getattr(model, "stage", None) == "pretrain"

    all_mat_true, all_mat_prob = [], []
    all_support_true, all_support_prob = [], []
    all_region_true, all_region_pred = [], []
    all_support_disp_true, all_support_disp_pred = [], []
    all_gate, all_disagreement = [], []
    fusion_alpha = None

    for batch in tqdm(loader, desc=desc, leave=False):
        _move_batch(batch, config.device)
        target = batch["target"]

        if is_pretrain:
            static_logits, _static_reg, _dynamic_logits, _dynamic_reg, _s_feat, _d_feat = model(batch)
            logits = static_logits
            loss = criterion_cls(logits, target)
            mat_prob = torch.sigmoid(logits)
        else:
            output = model(batch)
            logits = output["material_logits"]
            target_smooth = _smooth_target(target, 0.0)
            loss = multitask_loss(output, batch, criterion_cls, criterion_reg, config, target_smooth)
            mat_prob = torch.sigmoid(logits)

            all_support_true.append(batch["support_target"].cpu())
            all_support_prob.append(torch.sigmoid(output["support_logits"]).cpu())
            all_region_true.append(batch["region_target"].cpu())
            all_region_pred.append(output["region_logits"].argmax(dim=-1).cpu())
            all_support_disp_true.append(batch["support_disp"].cpu())
            all_support_disp_pred.append(
                (torch.sigmoid(output["support_reg"]) * float(config.support_disp_scale_mm)).cpu()
            )
            aux = getattr(model, "last_aux", None)
            if aux is not None and "fusion_alpha" in aux:
                fusion_alpha = aux["fusion_alpha"].detach().cpu()
            if aux is not None and "gate" in aux:
                all_gate.append(aux["gate"].detach().cpu())
                disagreement = aux.get("branch_disagreement")
                if disagreement is not None:
                    all_disagreement.append(disagreement.detach().cpu())

        total_loss += loss.item()
        all_mat_true.append(target.cpu())
        all_mat_prob.append(mat_prob.cpu())

    avg_loss = total_loss / len(loader)
    y_true = torch.cat(all_mat_true).numpy()
    y_prob = torch.cat(all_mat_prob).numpy()
    fixed_pred = (y_prob > 0.5).astype("float32")
    fixed_metrics = compute_metrics(y_true, fixed_pred, y_prob)

    thresholds = getattr(config, "eval_thresholds", None)
    if tune_thresholds and getattr(config, "threshold_tuning", False):
        thresholds = find_best_thresholds(
            y_true,
            y_prob,
            recall_constraints=getattr(config, "threshold_recall_constraints", None),
        )
        config.eval_thresholds = thresholds
    y_pred = fixed_pred if thresholds is None else (
        y_prob > np.asarray(thresholds).reshape(1, -1)
    ).astype("float32")

    if all_support_true:
        support_true = torch.cat(all_support_true).numpy()
        support_prob = torch.cat(all_support_prob).numpy()
        support_pred = (support_prob > 0.5).astype("float32")
        region_true = torch.cat(all_region_true).numpy()
        region_pred = torch.cat(all_region_pred).numpy()
        support_disp_true = torch.cat(all_support_disp_true).numpy()
        support_disp_pred = torch.cat(all_support_disp_pred).numpy()
    else:
        support_true = support_prob = support_pred = None
        region_true = region_pred = None
        support_disp_true = support_disp_pred = None

    metrics = compute_full_metrics(
        y_true,
        y_pred,
        y_prob,
        support_true=support_true,
        support_pred=support_pred,
        support_prob=support_prob,
        region_true=region_true,
        region_pred=region_pred,
        support_disp_true=support_disp_true,
        support_disp_pred=support_disp_pred,
    )
    if thresholds is not None:
        for key, value in fixed_metrics.items():
            metrics[f"{key}_at_05"] = value
    if fusion_alpha is not None:
        for i, label_id in enumerate(CANDIDATE_IDS):
            alpha_i = float(fusion_alpha[i])
            metrics[f"alpha_static_{label_id}"] = alpha_i
            metrics[f"alpha_dynamic_{label_id}"] = 1.0 - alpha_i
    if all_gate:
        gate = torch.cat(all_gate).numpy()
        disagreement = (
            torch.cat(all_disagreement).numpy()
            if all_disagreement else np.zeros_like(gate)
        )
        for i, label_id in enumerate(CANDIDATE_IDS):
            pos_mask = y_true[:, i] > 0.5
            neg_mask = ~pos_mask
            metrics[f"gate_mean_{label_id}"] = float(gate[:, i].mean())
            metrics[f"gate_pos_mean_{label_id}"] = (
                float(gate[pos_mask, i].mean()) if pos_mask.any() else 0.0
            )
            metrics[f"gate_neg_mean_{label_id}"] = (
                float(gate[neg_mask, i].mean()) if neg_mask.any() else 0.0
            )
            metrics[f"branch_disagreement_mean_{label_id}"] = float(
                disagreement[:, i].mean()
            )
    return avg_loss, metrics, (y_true, y_pred, y_prob)
