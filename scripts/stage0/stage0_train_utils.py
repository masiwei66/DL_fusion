"""阶段 0 诊断训练的轻量工具。

诊断训练只优化材料分类 BCE，不使用 raw_sf、支座、区域或全桥辅助监督。
这样可以明确回答“输入到材料标签的训练链路是否正常”，并避免辅助真值影响负控。
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

from stage0_common import sanitized_model_input


def move_inputs(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in sanitized_model_input(batch).items()
    }


def material_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)
    valid = [index for index in range(y_true.shape[1]) if np.unique(y_true[:, index]).size > 1]
    per_label_auprc = []
    for index in range(y_true.shape[1]):
        if y_true[:, index].sum() == 0:
            per_label_auprc.append(0.0)
        else:
            per_label_auprc.append(float(average_precision_score(y_true[:, index], y_prob[:, index])))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(np.all(y_true == y_pred, axis=1).mean()),
        "macro_auprc": float(np.mean([per_label_auprc[i] for i in valid])) if valid else 0.0,
        "per_label_auprc": per_label_auprc,
        "positive_prevalence": y_true.mean(axis=0).astype(float).tolist(),
    }


@torch.no_grad()
def evaluate_material(model, loader, device):
    model.eval()
    total_loss = 0.0
    targets = []
    probabilities = []
    for batch in loader:
        target = batch["target"].to(device)
        output = model(move_inputs(batch, device))
        loss = F.binary_cross_entropy_with_logits(output["material_logits"], target)
        total_loss += float(loss.item())
        targets.append(target.cpu())
        probabilities.append(torch.sigmoid(output["material_logits"]).cpu())
    y_true = torch.cat(targets).numpy()
    y_prob = torch.cat(probabilities).numpy()
    return total_loss / len(loader), material_metrics(y_true, y_prob), y_true, y_prob


def train_material_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    bad_gradient = False
    for batch in loader:
        target = batch["target"].to(device)
        output = model(move_inputs(batch, device))
        loss = F.binary_cross_entropy_with_logits(output["material_logits"], target)
        optimizer.zero_grad()
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                bad_gradient = True
                break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item())
    loss = total_loss / len(loader)
    if not math.isfinite(loss):
        bad_gradient = True
    return loss, bad_gradient


def prevalence_baseline(train_targets, evaluation_targets):
    train_targets = np.asarray(train_targets, dtype=np.float32)
    evaluation_targets = np.asarray(evaluation_targets, dtype=np.float32)
    prevalence = train_targets.mean(axis=0)
    probabilities = np.broadcast_to(prevalence, evaluation_targets.shape).copy()
    return material_metrics(evaluation_targets, probabilities)
