"""Distributional losses for fuzzy-range KW51 response prediction."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..contracts import WindowBatch

_LOG_2PI = math.log(2.0 * math.pi)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def _masked_mean(value: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return value.mean()
    mask = valid.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _sample_masked_mean(
    value: torch.Tensor,
    valid: torch.Tensor | None,
    sample_valid: torch.Tensor | None,
) -> torch.Tensor:
    if sample_valid is not None:
        mask = sample_valid.to(value.dtype)
        if valid is not None:
            mask = mask * valid[:, None, :].to(value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)
    return _masked_mean(value.mean(dim=1), valid)


def gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    samples: torch.Tensor,
    valid: torch.Tensor | None = None,
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean Gaussian NLL of K target samples under the predicted distribution."""
    variance = torch.exp(logvar).clamp_min(1.0e-6)
    errors = (samples - mean[:, None, :]).square() / variance[:, None, :]
    nll = 0.5 * (logvar[:, None, :] + errors + _LOG_2PI)
    return _sample_masked_mean(nll, valid, sample_valid)


def gaussian_crps(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    samples: torch.Tensor,
    valid: torch.Tensor | None = None,
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form CRPS for a Gaussian predictive distribution."""
    sigma = torch.exp(0.5 * logvar).clamp_min(1.0e-6)
    z = (samples - mean[:, None, :]) / sigma[:, None, :]
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma[:, None, :] * (
        z * (2.0 * cdf - 1.0) + 2.0 * phi - _INV_SQRT_PI
    )
    return _sample_masked_mean(crps, valid, sample_valid)


def group_cross_entropy(
    logits: torch.Tensor, group_ids: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(logits, group_ids.long())


def group_accuracy(logits: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == group_ids.long()).float().mean()


def fuzzy_state_loss(
    outputs: dict[str, torch.Tensor],
    batch: WindowBatch,
    *,
    response_auxiliary_weight: float = 0.0,
    mean_consistency_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if "group_logits" not in outputs:
        raise KeyError("fuzzy_state_loss requires group_logits")
    group_ids = batch.targets["ad_group_id"]
    group_loss = group_cross_entropy(outputs["group_logits"], group_ids)
    accuracy = group_accuracy(outputs["group_logits"], group_ids)
    components = {
        "group_ce": float(group_loss.detach()),
        "group_accuracy": float(accuracy.detach()),
    }
    loss = group_loss
    if response_auxiliary_weight > 0.0 and "mean" in outputs:
        response_loss, response_components = fuzzy_response_loss(
            outputs,
            batch,
            mean_consistency_weight=mean_consistency_weight,
        )
        loss = loss + float(response_auxiliary_weight) * response_loss
        components["response_nll"] = float(response_components["nll"])
        components["response_crps"] = float(response_components["crps"])
    return loss, components


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def dense_region_response_loss(
    mean: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    area_coverage: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mask-aware Gaussian NLL and CRPS for ``[batch, region, time]`` targets."""
    if mean.shape != target.shape or scale.shape != target.shape or valid.shape != target.shape:
        raise ValueError("dense regional response tensors must have matching [B,R,T] shapes")
    scale = scale.clamp_min(1.0e-6)
    residual = (target - mean) / scale
    nll = torch.log(scale) + 0.5 * residual.square() + 0.5 * _LOG_2PI
    z = residual
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = scale * (z * (2.0 * cdf - 1.0) + 2.0 * phi - _INV_SQRT_PI)
    weight = valid.to(mean.dtype)
    if area_coverage is not None:
        weight = weight * area_coverage.to(mean.dtype).clamp(0.0, 1.0)
    nll_mean = _weighted_mean(nll, weight)
    crps_mean = _weighted_mean(crps, weight)
    return nll_mean, {"nll": nll_mean, "crps": crps_mean}


def fuzzy_region_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: WindowBatch,
    *,
    l1_weight: float = 1.0,
    l2_weight: float = 1.0,
    region_l2_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train dense L1 response and L2 compatibility heads together."""
    required = {
        "region_response_mean",
        "region_response_scale",
        "state_logits",
        "region_state_logits",
    }
    missing = required - outputs.keys()
    if missing:
        raise KeyError(f"multi-task region model output missing {sorted(missing)}")
    target = batch.targets["region_response_target"].to(
        outputs["region_response_mean"].dtype
    )
    target_valid = batch.targets["region_response_valid_mask"].bool()
    l1_loss, l1_parts = dense_region_response_loss(
        outputs["region_response_mean"],
        outputs["region_response_scale"],
        target,
        target_valid,
        batch.targets.get("region_area_coverage"),
    )
    group_ids = batch.targets["ad_group_id"].long()
    global_ce = F.cross_entropy(outputs["state_logits"], group_ids)
    region_logits = outputs["region_state_logits"]
    region_valid = (
        batch.region_present.bool()
        if batch.region_present is not None
        else target_valid.any(dim=-1)
    )
    repeated_ids = group_ids[:, None].expand(-1, region_logits.shape[1])
    region_ce_values = F.cross_entropy(
        region_logits.reshape(-1, region_logits.shape[-1]),
        repeated_ids.reshape(-1),
        reduction="none",
    ).reshape_as(region_valid)
    region_ce = _weighted_mean(region_ce_values, region_valid)
    loss = (
        float(l1_weight) * l1_loss
        + float(l2_weight) * global_ce
        + float(region_l2_weight) * region_ce
    )
    return loss, {
        "l1_nll": float(l1_parts["nll"].detach()),
        "l1_crps": float(l1_parts["crps"].detach()),
        "l2_global_nll": float(global_ce.detach()),
        "l2_region_nll": float(region_ce.detach()),
        "total": float(loss.detach()),
    }


def fuzzy_response_loss(
    outputs: dict[str, torch.Tensor],
    batch: WindowBatch,
    *,
    mean_consistency_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    samples = batch.targets["response_samples"]
    valid = batch.targets.get("response_valid")
    sample_valid = batch.targets.get("response_sample_valid")
    nll = gaussian_nll(
        outputs["mean"], outputs["logvar"], samples, valid, sample_valid
    )
    crps = gaussian_crps(
        outputs["mean"], outputs["logvar"], samples, valid, sample_valid
    )
    loss = nll
    mean_mse = outputs["mean"].new_zeros(())
    if mean_consistency_weight > 0.0:
        target_mean = batch.targets["response_mean"]
        squared = (outputs["mean"] - target_mean).square()
        if valid is not None:
            mask = valid.to(squared.dtype)
            mean_mse = (squared * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            mean_mse = squared.mean()
        loss = loss + float(mean_consistency_weight) * mean_mse
    return loss, {
        "nll": float(nll.detach()),
        "crps": float(crps.detach()),
        "mean_mse": float(mean_mse.detach()),
    }
