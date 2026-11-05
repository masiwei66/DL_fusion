"""Distributional metrics for fuzzy-range KW51 response prediction."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from dl_fusion.metrics import classification_metrics

from .fuzzy_losses import gaussian_crps, gaussian_nll

_COVERAGE_Z = {"50": 0.67448975, "80": 1.28155157, "95": 1.95996398}


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    repeats: int = 5000,
    seed: int = 0,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        means[index] = rng.choice(array, size=array.size, replace=True).mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def distribution_metrics(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    samples: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    target_scale_mm: float,
    sample_valid: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Overall metrics for normalized Gaussian predictions and K target samples."""
    mean = mean.detach().float()
    logvar = logvar.detach().float()
    samples = samples.detach().float()
    target_mean = target_mean.detach().float()
    target_std = target_std.detach().float()
    if valid is None:
        valid = torch.ones_like(mean, dtype=torch.bool)
    valid = valid.bool()
    sigma = torch.exp(0.5 * logvar).clamp_min(1.0e-6)
    mask = valid.to(mean.dtype)
    difference = (mean - target_mean).abs()
    mae_norm = float((difference * mask).sum() / mask.sum().clamp_min(1.0))
    rmse_norm = float(
        (((mean - target_mean).square() * mask).sum() / mask.sum().clamp_min(1.0)).sqrt()
    )
    z = (samples - mean[:, None, :]) / sigma[:, None, :]
    sample_mask = valid[:, None, :].expand_as(z)
    if sample_valid is not None:
        sample_mask = sample_mask & sample_valid.bool()
    coverage: dict[str, float] = {}
    for name, critical in _COVERAGE_Z.items():
        inside = (z.abs() <= critical) & sample_mask
        coverage[name] = float(inside.sum() / sample_mask.sum().clamp_min(1))
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pit_values = cdf[sample_mask]
    histogram = torch.histc(pit_values, bins=10, min=0.0, max=1.0)
    histogram = histogram / histogram.sum().clamp_min(1.0)
    pit_l1 = float((histogram - 0.1).abs().mean())
    sharpness_norm = float((sigma * mask).sum() / mask.sum().clamp_min(1.0))
    return {
        "nll": float(gaussian_nll(mean, logvar, samples, valid, sample_valid)),
        "crps": float(gaussian_crps(mean, logvar, samples, valid, sample_valid)),
        "mae": mae_norm * target_scale_mm,
        "rmse": rmse_norm * target_scale_mm,
        "mae_normalized": mae_norm,
        "rmse_normalized": rmse_norm,
        "coverage": coverage,
        "pit_l1": pit_l1,
        "pit_histogram": histogram.tolist(),
        "sharpness_mm": sharpness_norm * target_scale_mm,
        "target_std_mm": float(
            (target_std * mask).sum() / mask.sum().clamp_min(1.0)
        ) * target_scale_mm,
        "num_acquisitions": int(valid.sum()),
    }


def group_metrics(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    group_values: list[float] | np.ndarray,
) -> dict[str, Any]:
    logits = logits.detach().float()
    group_ids = group_ids.detach().long()
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    correct = predictions == group_ids
    accuracy = float(correct.float().mean())
    rows = torch.arange(group_ids.shape[0])
    true_probabilities = probabilities[rows, group_ids]
    ranks = (probabilities > true_probabilities[:, None]).sum(dim=1) + 1
    top3 = float((ranks <= 3).float().mean())
    mrr = float((1.0 / ranks.float()).mean())
    nll = float(
        (-torch.log(true_probabilities.clamp_min(1.0e-12))).mean()
    )
    one_hot = torch.nn.functional.one_hot(
        group_ids, num_classes=probabilities.shape[1]
    ).float()
    brier = float((probabilities - one_hot).square().sum(dim=1).mean())
    confidence = probabilities.max(dim=1).values
    correctness = correct.float()
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == 9:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if bool(mask.any()):
            bin_confidence = float(confidence[mask].mean())
            bin_accuracy = float(correctness[mask].mean())
            ece += abs(bin_confidence - bin_accuracy) * float(mask.float().mean())
    confusion = np.bincount(
        group_ids.numpy() * probabilities.shape[1] + predictions.numpy(),
        minlength=probabilities.shape[1] ** 2,
    ).reshape(probabilities.shape[1], probabilities.shape[1])
    y_true = group_ids.numpy().reshape(-1)
    y_pred = predictions.numpy().reshape(-1)
    y_prob = probabilities.numpy()
    labels = np.arange(probabilities.shape[1], dtype=np.int64)
    shared = classification_metrics(y_true, y_pred, y_prob, class_values=labels)
    result: dict[str, Any] = {
        **shared,
        "class_names": [f"Ad_rel={float(value):g}" for value in group_values],
        "top1_accuracy": accuracy,
        "top3_accuracy": top3,
        "mrr": mrr,
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "confusion_matrix": confusion.tolist(),
        "num_samples": int(group_ids.shape[0]),
    }
    result["macro_f1"] = result["f1_macro"]
    values = np.asarray(group_values, dtype=np.float64)
    if values.size == probabilities.shape[1]:
        probabilities_np = probabilities.numpy()
        expected = probabilities_np @ values
        variance = (
            probabilities_np
            * (values[None, :] - expected[:, None]) ** 2
        ).sum(axis=1)
        true_values = values[group_ids.numpy()]
        result["ad_rel_expected_mean"] = float(expected.mean())
        result["ad_rel_std_mean"] = float(np.sqrt(np.maximum(variance, 0.0)).mean())
        result["ad_rel_abs_error"] = float(np.mean(np.abs(expected - true_values)))
    return result


def dense_region_metrics(
    mean: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float | int]:
    valid = valid.bool()
    count = int(valid.sum())
    if count == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "crps": 0.0,
            "coverage_50": 0.0,
            "coverage_90": 0.0,
            "valid_region_times": 0,
        }
    mean = mean.float()
    scale = scale.float().clamp_min(1.0e-6)
    target = target.float()
    residual = (target - mean) / scale
    phi = torch.exp(-0.5 * residual.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(residual / math.sqrt(2.0)))
    crps = scale * (
        residual * (2.0 * cdf - 1.0)
        + 2.0 * phi
        - 1.0 / math.sqrt(math.pi)
    )
    absolute = (mean - target).abs()
    squared = (mean - target).square()
    return {
        "mae": float(absolute[valid].mean()),
        "rmse": float(squared[valid].mean().sqrt()),
        "crps": float(crps[valid].mean()),
        "coverage_50": float((residual.abs()[valid] <= 0.67448975).float().mean()),
        "coverage_90": float((residual.abs()[valid] <= 1.64485363).float().mean()),
        "valid_region_times": count,
    }


def _compatibility_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float | int]:
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    if labels.numel() == 0:
        return {"top1": 0.0, "mrr": 0.0, "nll": 0.0, "ece": 0.0, "queries": 0}
    raw = group_metrics(logits, labels, np.arange(logits.shape[-1]))
    return {
        "top1": float(raw["top1_accuracy"]),
        "mrr": float(raw["mrr"]),
        "nll": float(raw["nll"]),
        "ece": float(raw["ece"]),
        "queries": int(labels.numel()),
    }


def fuzzy_region_multitask_metrics(
    outputs: dict[str, torch.Tensor], batch: Any
) -> dict[str, dict[str, Any]]:
    """Report L1/L2 metrics overall and by state, region, and point density."""
    mean = outputs["region_response_mean"].detach().float().cpu()
    scale = outputs["region_response_scale"].detach().float().cpu()
    target = batch.targets["region_response_target"].detach().float().cpu()
    valid = batch.targets["region_response_valid_mask"].detach().bool().cpu()
    global_logits = outputs["state_logits"].detach().float().cpu()
    region_logits = outputs["region_state_logits"].detach().float().cpu()
    labels = batch.targets["ad_group_id"].detach().long().cpu()
    present = (
        batch.region_present.detach().bool().cpu()
        if batch.region_present is not None
        else valid.any(dim=-1)
    )
    states = [str(item.get("structural_state_id", index)) for index, item in enumerate(batch.metadata)]
    densities = [
        str(item.get("point_budget", item.get("point_count", "unknown")))
        for item in batch.metadata
    ]

    l1_per_state: dict[str, Any] = {}
    for state in sorted(set(states)):
        indices = [index for index, value in enumerate(states) if value == state]
        selector = torch.as_tensor(indices, dtype=torch.long)
        l1_per_state[state] = dense_region_metrics(
            mean.index_select(0, selector),
            scale.index_select(0, selector),
            target.index_select(0, selector),
            valid.index_select(0, selector),
        )
    l1_per_region = {
        str(region): dense_region_metrics(
            mean[:, region], scale[:, region], target[:, region], valid[:, region]
        )
        for region in range(mean.shape[1])
    }
    l1_per_density: dict[str, Any] = {}
    for density in sorted(set(densities)):
        indices = [index for index, value in enumerate(densities) if value == density]
        selector = torch.as_tensor(indices, dtype=torch.long)
        l1_per_density[density] = dense_region_metrics(
            mean.index_select(0, selector),
            scale.index_select(0, selector),
            target.index_select(0, selector),
            valid.index_select(0, selector),
        )

    l2_per_state: dict[str, Any] = {}
    for state in sorted(set(states)):
        indices = [index for index, value in enumerate(states) if value == state]
        selector = torch.as_tensor(indices, dtype=torch.long)
        l2_per_state[state] = _compatibility_metrics(
            global_logits.index_select(0, selector), labels.index_select(0, selector)
        )
    l2_per_region: dict[str, Any] = {}
    for region in range(region_logits.shape[1]):
        active = present[:, region]
        l2_per_region[str(region)] = _compatibility_metrics(
            region_logits[active, region], labels[active]
        )
    l2_per_density: dict[str, Any] = {}
    for density in sorted(set(densities)):
        indices = [index for index, value in enumerate(densities) if value == density]
        selector = torch.as_tensor(indices, dtype=torch.long)
        l2_per_density[density] = _compatibility_metrics(
            global_logits.index_select(0, selector), labels.index_select(0, selector)
        )
    return {
        "l1": {
            "overall": dense_region_metrics(mean, scale, target, valid),
            "per_state": l1_per_state,
            "per_region": l1_per_region,
            "per_density": l1_per_density,
        },
        "l2": {
            "overall": _compatibility_metrics(global_logits, labels),
            "per_state": l2_per_state,
            "per_region": l2_per_region,
            "per_density": l2_per_density,
        },
    }


def group_per_state(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["structural_state_id"]), []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for state, items in sorted(
        grouped.items(),
        key=lambda pair: (0, int(pair[0])) if pair[0].isdigit() else (1, pair[0]),
    ):
        accuracy = [float(item["correct"]) for item in items]
        nll = [float(item["group_nll"]) for item in items]
        result[state] = {
            "queries": float(len(items)),
            "accuracy": float(np.mean(accuracy)),
            "group_nll": float(np.mean(nll)),
            "accuracy_ci95": bootstrap_ci(accuracy),
            "group_nll_ci95": bootstrap_ci(nll),
        }
    return result


def per_state_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["structural_state_id"]), []).append(row)
    result: dict[str, dict[str, float]] = {}
    for state, items in sorted(grouped.items(), key=lambda pair: (0, int(pair[0])) if pair[0].isdigit() else (1, pair[0])):
        mae = float(np.mean([float(item["mae"]) for item in items]))
        nll = float(np.mean([float(item["nll"]) for item in items]))
        result[state] = {
            "queries": float(len(items)),
            "mae": mae,
            "nll": nll,
            "mae_ci95": bootstrap_ci([float(item["mae"]) for item in items]),
            "nll_ci95": bootstrap_ci([float(item["nll"]) for item in items]),
        }
    return result
