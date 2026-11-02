"""Fuzzy-range multi-instance response-distribution training runner."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.fem_insar import (
    FEMArchiveIndex,
    build_fem_archive_index,
    derive_ad_groups,
    split_fem_records,
)
from ..data.fuzzy_range import (
    FuzzyNormalization,
    FuzzyQueryDataset,
    FuzzyRangeSource,
    build_deck_geometry,
    collate_fuzzy_queries,
    fit_fuzzy_normalization,
    make_fuzzy_query_refs,
)
from ..models.fuzzy_response import FuzzyResponseModel
from .fuzzy_audit import run_fuzzy_audit
from .fuzzy_losses import fuzzy_response_loss
from .fuzzy_metrics import bootstrap_ci, distribution_metrics, per_state_metrics
from .progress import make_progress


def _state_sort_key(state_id: str) -> tuple[int, Any]:
    try:
        return (0, int(state_id))
    except (TypeError, ValueError):
        return (1, str(state_id))


def _require_archive(index: FEMArchiveIndex, data) -> None:
    if not data.require_complete_archive:
        return
    run_status = str(index.run_manifest.get("status", "unknown"))
    audit = index.run_manifest.get("audit_report") or {}
    if audit.get("status") and str(audit.get("status")) != "PASS_FEM_ARCHIVE":
        raise RuntimeError(f"FEM archive audit is not passing: {audit.get('status')!r}")
    if audit.get("blockers"):
        raise RuntimeError(f"FEM archive audit has blockers: {audit['blockers']}")
    requested = int(
        index.run_manifest.get(
            "requested_samples",
            index.run_manifest.get(
                "sample_count",
                audit.get("sample_count", data.expected_samples),
            ),
        )
        or 0
    )
    expected_samples = max(int(data.expected_samples), requested)
    if run_status in {"active", "running", "in_progress"}:
        raise RuntimeError(
            f"FEM archive is still {run_status}: found {len(index.records)} files"
        )
    if expected_samples and len(index.records) < expected_samples:
        raise RuntimeError(
            f"FEM archive is incomplete: found {len(index.records)} files, "
            f"expected at least {expected_samples}"
        )
    if data.expected_structural_states and len(index.state_ids) < int(
        data.expected_structural_states
    ):
        raise RuntimeError(
            f"FEM archive has {len(index.state_ids)} structural states, expected "
            f"{data.expected_structural_states}"
        )
    audited_states = audit.get("structural_state_count")
    if audited_states is not None and int(audited_states) != len(index.state_ids):
        raise RuntimeError(
            "FEM archive audit state count does not match indexed files: "
            f"audit={audited_states}, indexed={len(index.state_ids)}"
        )


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    train: bool,
    progress: dict[str, Any] | None,
    desc: str | None,
    mean_consistency_weight: float,
    collect: bool = False,
):
    model.train(train)
    total_loss = 0.0
    total_count = 0
    component_sums = {"nll": 0.0, "crps": 0.0, "mean_mse": 0.0}
    collected: dict[str, list[Any]] = {
        "mean": [],
        "logvar": [],
        "samples": [],
        "target_mean": [],
        "target_std": [],
        "valid": [],
        "sample_valid": [],
        "bag_mean": [],
        "bag_centroid": [],
        "metadata": [],
    }
    with make_progress(desc, total=len(loader), options=progress) as tracker:
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            with torch.set_grad_enabled(train):
                outputs = model(batch)
                loss, components = fuzzy_response_loss(
                    outputs,
                    batch,
                    mean_consistency_weight=mean_consistency_weight,
                )
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            total_loss += float(loss.detach()) * batch.batch_size
            total_count += batch.batch_size
            for key, value in components.items():
                component_sums[key] += float(value) * batch.batch_size
            if collect:
                collected["mean"].append(outputs["mean"].detach().cpu())
                collected["logvar"].append(outputs["logvar"].detach().cpu())
                collected["samples"].append(
                    batch.targets["response_samples"].detach().cpu()
                )
                collected["target_mean"].append(
                    batch.targets["response_mean"].detach().cpu()
                )
                collected["target_std"].append(
                    batch.targets["response_std"].detach().cpu()
                )
                collected["valid"].append(
                    batch.targets["response_valid"].detach().cpu()
                )
                collected["sample_valid"].append(
                    batch.targets["response_sample_valid"].detach().cpu()
                )
                collected["bag_mean"].append(
                    batch.targets["bag_mean_mm"].detach().cpu()
                )
                if batch.bag_features is not None:
                    collected["bag_centroid"].append(
                        batch.bag_features[:, :2].detach().cpu()
                    )
                collected["metadata"].extend(batch.metadata)
            tracker.update(
                batch_index,
                average_loss=total_loss / max(total_count, 1),
            )
    average_components = {
        key: value / max(total_count, 1) for key, value in component_sums.items()
    }
    return total_loss / max(total_count, 1), average_components, collected


def _evaluate(
    model,
    loader,
    device,
    normalization: FuzzyNormalization,
    *,
    progress: dict[str, Any] | None,
    desc: str,
    mean_consistency_weight: float,
):
    loss, components, collected = _run_epoch(
        model,
        loader,
        None,
        device,
        train=False,
        progress=progress,
        desc=desc,
        mean_consistency_weight=mean_consistency_weight,
        collect=True,
    )
    mean = torch.cat(collected["mean"], dim=0)
    logvar = torch.cat(collected["logvar"], dim=0)
    samples = torch.cat(collected["samples"], dim=0)
    target_mean = torch.cat(collected["target_mean"], dim=0)
    target_std = torch.cat(collected["target_std"], dim=0)
    valid = torch.cat(collected["valid"], dim=0)
    sample_valid = torch.cat(collected["sample_valid"], dim=0)
    bag_mean = torch.cat(collected["bag_mean"], dim=0)
    bag_centroid = (
        torch.cat(collected["bag_centroid"], dim=0)
        if collected["bag_centroid"]
        else torch.zeros((len(collected["metadata"]), 2))
    )
    metadata = collected["metadata"]
    metrics = distribution_metrics(
        mean,
        logvar,
        samples,
        target_mean,
        target_std,
        valid,
        target_scale_mm=float(normalization.target_std),
        sample_valid=sample_valid,
    )
    metrics["loss"] = float(loss)
    metrics.update({f"component_{key}": float(value) for key, value in components.items()})
    sigma = torch.exp(0.5 * logvar).clamp_min(1.0e-6)
    variance = torch.exp(logvar).clamp_min(1.0e-6)
    nll_terms = 0.5 * (
        logvar[:, None, :]
        + (samples - mean[:, None, :]).square() / variance[:, None, :]
        + math.log(2.0 * math.pi)
    )
    sample_mask = valid[:, None, :] & sample_valid
    nll_per_query = (nll_terms * sample_mask).sum(dim=(1, 2)) / sample_mask.sum(
        dim=(1, 2)
    ).clamp_min(1)
    mae_per_query = (
        ((mean - target_mean).abs() * valid).sum(dim=1)
        / valid.sum(dim=1).clamp_min(1)
    ) * float(normalization.target_std)
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    scale = float(normalization.target_std)
    offset = float(normalization.target_mean)
    for index, item in enumerate(metadata):
        mean_mm = mean[index].numpy() * scale + offset
        std_mm = sigma[index].numpy() * scale
        target_mean_mm = target_mean[index].numpy() * scale + offset
        target_std_mm = target_std[index].numpy() * scale
        rows.append(
            {
                "structural_state_id": str(item["structural_state_id"]),
                "mae": float(mae_per_query[index]),
                "nll": float(nll_per_query[index]),
            }
        )
        predictions.append(
            {
                "sample_id": str(item["sample_id"]),
                "structural_state_id": str(item["structural_state_id"]),
                "trajectory_id": str(item.get("trajectory_id", "")),
                "observation_realization_id": int(
                    item.get("observation_realization_id", 0)
                ),
                "point_count": int(item.get("point_count", 0)),
                "mae_mm": float(mae_per_query[index]),
                "nll": float(nll_per_query[index]),
                "mean_mm": mean_mm.tolist(),
                "std_mm": std_mm.tolist(),
                "target_mean_mm": target_mean_mm.tolist(),
                "target_std_mm": target_std_mm.tolist(),
                "bag_mean_mm": bag_mean[index].numpy().tolist(),
                "bag_centroid_s": float(bag_centroid[index, 0]),
                "bag_centroid_t": float(bag_centroid[index, 1]),
                "true_range_mean_mm": item.get("true_range_mean_mm"),
            }
        )
    per_state = per_state_metrics(rows)
    return metrics, per_state, predictions


def _collect_train_reference(
    dataset, normalization: FuzzyNormalization, max_items: int = 300
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(min(int(max_items), len(dataset))):
        item = dataset[index]
        features = np.asarray(item["bag_features"], dtype=np.float64)
        target_mean_mm = (
            np.asarray(item["target_mean"], dtype=np.float64)
            * normalization.target_std
            + normalization.target_mean
        )
        rows.append(
            {
                "centroid": features[:2].copy(),
                "target_mean_mm": target_mean_mm,
            }
        )
    return rows


def _point_error(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "n": int(error.size),
    }


def _compute_baselines(
    train_dataset,
    test_predictions: list[dict[str, Any]],
    normalization: FuzzyNormalization,
    max_train_reference: int = 300,
) -> dict[str, Any] | None:
    if not test_predictions:
        return None
    target = np.stack(
        [np.asarray(row["target_mean_mm"], dtype=np.float64) for row in test_predictions]
    )
    bag = np.stack(
        [np.asarray(row["bag_mean_mm"], dtype=np.float64) for row in test_predictions]
    )
    model = np.stack(
        [np.asarray(row["mean_mm"], dtype=np.float64) for row in test_predictions]
    )
    global_prediction = np.full_like(target, float(normalization.target_mean))
    baselines: dict[str, Any] = {
        "bag_mean": _point_error(bag, target),
        "global_mean": _point_error(global_prediction, target),
    }
    reference = _collect_train_reference(
        train_dataset, normalization, max_train_reference
    )
    if reference:
        centers = np.stack([row["centroid"] for row in reference])
        reference_targets = np.stack(
            [row["target_mean_mm"] for row in reference]
        )
        test_centers = np.stack(
            [
                [row["bag_centroid_s"], row["bag_centroid_t"]]
                for row in test_predictions
            ]
        )
        neighbor_count = min(5, len(reference))
        position_predictions = []
        for center in test_centers:
            distances = np.linalg.norm(centers - center[None, :], axis=1)
            nearest = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
            position_predictions.append(reference_targets[nearest].mean(axis=0))
        baselines["position_only"] = _point_error(
            np.stack(position_predictions), target
        )
    true_rows = [row.get("true_range_mean_mm") for row in test_predictions]
    if all(row is not None and len(row) == target.shape[1] for row in true_rows):
        true_range = np.stack(
            [np.asarray(row, dtype=np.float64) for row in true_rows]
        )
        baselines["true_range_reference"] = {
            "model_vs_true_range_mae": float(
                np.mean(np.abs(model - true_range))
            ),
            "target_ab_vs_true_range_mae": float(
                np.mean(np.abs(target - true_range))
            ),
            "note": (
                "true_range_mean_mm is hidden generator information used only "
                "for diagnostics. It is not the same target as the A(B) "
                "ensemble and is not a universal upper bound."
            ),
        }
    return baselines


def _state_level_summary(per_state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = [value for value in per_state.values() if value.get("queries", 0) > 0]
    mae = [float(value["mae"]) for value in values]
    nll = [float(value["nll"]) for value in values]
    return {
        "states": len(values),
        "mae": {
            "mean": float(np.mean(mae)) if mae else 0.0,
            "ci95": bootstrap_ci(mae),
        },
        "nll": {
            "mean": float(np.mean(nll)) if nll else 0.0,
            "ci95": bootstrap_ci(nll),
        },
    }


def run_fuzzy_training(config) -> dict[str, Any]:
    data = config.data
    fuzzy = config.fuzzy
    if fuzzy is None:
        raise ValueError("kw51_fuzzy_response requires a fuzzy config section")
    output_root = Path(config.train.output_root)
    if data.split_fold is not None:
        output_root = output_root.parent / f"{output_root.name}_fold{int(data.split_fold)}"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(int(data.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(data.seed))
    index = build_fem_archive_index(data.data_root)
    _require_archive(index, data)
    geometry = build_deck_geometry(index.records[0], data.los_vector)
    group_map = derive_ad_groups(
        index,
        parameter=data.ad_group_parameter,
        expected_count=data.ad_group_count,
    )
    records = split_fem_records(
        index,
        data.split_seed,
        data.train_fraction,
        data.validation_fraction,
        group_map=group_map,
        split_mode=data.split_mode,
        fold=data.split_fold,
    )
    source = FuzzyRangeSource(index, geometry, fuzzy, seed=data.seed)
    normalization = fit_fuzzy_normalization(
        source,
        records["train"],
        fuzzy,
        time_feature_mode=fuzzy.time_feature_mode,
    )
    normalization_path = output_root / "normalization_stats.json"
    normalization_path.write_text(
        json.dumps(normalization.to_dict(), indent=2), encoding="utf-8"
    )
    target_length = int(source._clean_arrays(index.records[0])[0].shape[0])
    if int(data.window_length) != target_length:
        raise ValueError(
            f"data.window_length={data.window_length} does not match archive "
            f"acquisition count {target_length}"
        )
    train_refs = make_fuzzy_query_refs(records["train"], fuzzy.query_realizations)
    validation_refs = make_fuzzy_query_refs(records["validation"], 1)
    test_refs = make_fuzzy_query_refs(records["test"], 1)
    train_dataset = FuzzyQueryDataset(train_refs, source, normalization)
    validation_dataset = FuzzyQueryDataset(validation_refs, source, normalization)
    test_dataset = FuzzyQueryDataset(test_refs, source, normalization)
    loader_kwargs: dict[str, Any] = {
        "collate_fn": collate_fuzzy_queries,
        "num_workers": config.train.num_workers,
        "pin_memory": config.train.pin_memory,
    }
    if config.train.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.train.prefetch_factor
        loader_kwargs["persistent_workers"] = config.train.persistent_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    audit = run_fuzzy_audit(
        source,
        records["train"],
        normalization.target_mean,
        audit_records=fuzzy.audit_records,
        noise_std_mm=fuzzy.noise_std_mm,
        point_bias_std_mm=fuzzy.point_bias_std_mm,
        min_snr=fuzzy.audit_min_snr,
    )
    audit_path = output_root / "fuzzy_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit_failed = str(audit.get("status")) != "PASS"
    diagnostic_only = bool(data.diagnostic_only) or audit_failed
    print(
        f"fuzzy_audit status={audit['status']} bag_mean_mae={audit['bag_mean_mae_mm']:.4f} "
        f"global_mean_mae={audit['global_mean_mae_mm']:.4f} snr={audit['snr']:.3f}",
        flush=True,
    )
    if audit_failed and fuzzy.audit_abort_on_fail:
        raise RuntimeError(
            "KW51 fuzzy audit failed: " + "; ".join(audit.get("reasons", []))
        )
    validation_states = len(
        {record.structural_state_id for record in records["validation"]}
    )
    test_states = len({record.structural_state_id for record in records["test"]})
    archive_complete = (
        bool(data.require_complete_archive)
        and len(index.records) >= int(data.expected_samples)
        and len(index.state_ids) >= int(data.expected_structural_states)
    )
    eval_states_ok = (
        validation_states >= int(fuzzy.formal_min_eval_states)
        and test_states >= int(fuzzy.formal_min_eval_states)
    )
    formal_conditions = {
        "requested": bool(fuzzy.formal_run),
        "archive_complete": bool(archive_complete),
        "eval_states_ok": bool(eval_states_ok),
        "audit_pass": not bool(audit_failed),
        "not_diagnostic": not bool(data.diagnostic_only),
    }
    if fuzzy.formal_run:
        failed = [
            key for key, value in formal_conditions.items()
            if key != "requested" and not value
        ]
        if failed:
            raise RuntimeError(
                "formal_run requested but formal conditions failed: " + ", ".join(failed)
            )
    formal_ok = (
        bool(fuzzy.formal_run)
        and archive_complete
        and eval_states_ok
        and not audit_failed
        and not bool(data.diagnostic_only)
    )
    run_status = (
        "DIAGNOSTIC_ONLY"
        if diagnostic_only
        else "FORMAL" if formal_ok else "SMOKE"
    )
    print(
        f"run_status={run_status} formal_conditions={formal_conditions}",
        flush=True,
    )
    model_params = dict(config.model.kwargs)
    head_hidden = int(model_params.pop("head_hidden", 128))
    head_dropout = float(model_params.pop("head_dropout", 0.1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FuzzyResponseModel(
        config.model.name,
        model_params,
        target_length,
        head_hidden=head_hidden,
        dropout=head_dropout,
        bag_feature_dropout=fuzzy.bag_feature_dropout,
    ).to(device)
    print(
        "stage=first_batch (materializes bag pooling and response head)",
        flush=True,
    )
    first_batch = next(iter(train_loader)).to(device)
    with torch.no_grad():
        model(first_batch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = None
    if config.train.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.train.lr_factor,
            patience=config.train.lr_patience,
            min_lr=config.train.min_learning_rate,
        )
    progress_options = {
        "mode": config.train.progress,
        "log_every": config.train.log_every_batches,
    }
    checkpoint_path = output_root / "models" / f"{config.model.name}_fuzzy_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_nll = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.train.epochs + 1):
        train_loss, train_components, _ = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            progress=progress_options,
            desc=f"train epoch={epoch}",
            mean_consistency_weight=fuzzy.mean_consistency_weight,
        )
        validation_metrics, validation_per_state, validation_predictions = _evaluate(
            model,
            validation_loader,
            device,
            normalization,
            progress=progress_options,
            desc=f"validation epoch={epoch}",
            mean_consistency_weight=fuzzy.mean_consistency_weight,
        )
        selection = float(validation_metrics["nll"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_nll": train_components["nll"],
            "train_crps": train_components["crps"],
            "validation_loss": validation_metrics["loss"],
            "validation_nll": validation_metrics["nll"],
            "validation_crps": validation_metrics["crps"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_coverage_50": validation_metrics["coverage"]["50"],
            "validation_coverage_80": validation_metrics["coverage"]["80"],
            "validation_coverage_95": validation_metrics["coverage"]["95"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (output_root / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if selection < best_nll:
            best_nll = selection
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_name": config.model.name,
                    "model_params": model_params,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "task": "kw51_fuzzy_response",
                    "target_length": target_length,
                    "fuzzy": asdict(fuzzy),
                    "normalization": normalization.to_dict(),
                    "diagnostic_only": diagnostic_only,
                    "run_status": run_status,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(selection)
        print(
            f"epoch={epoch} train_nll={train_components['nll']:.4f} "
            f"validation_nll={validation_metrics['nll']:.4f} "
            f"validation_mae={validation_metrics['mae']:.4f} "
            f"coverage95={validation_metrics['coverage']['95']:.3f}",
            flush=True,
        )
        if config.train.early_stopping_patience and stale >= config.train.early_stopping_patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, validation_per_state, validation_predictions = _evaluate(
        model,
        validation_loader,
        device,
        normalization,
        progress=progress_options,
        desc="validation best-checkpoint",
        mean_consistency_weight=fuzzy.mean_consistency_weight,
    )
    test_metrics, test_per_state, test_predictions = _evaluate(
        model,
        test_loader,
        device,
        normalization,
        progress=progress_options,
        desc="test best-checkpoint",
        mean_consistency_weight=fuzzy.mean_consistency_weight,
    )
    test_baselines = _compute_baselines(
        train_dataset, test_predictions, normalization
    )
    validation_state_summary = _state_level_summary(validation_per_state)
    test_state_summary = _state_level_summary(test_per_state)
    if test_baselines is not None:
        test_baselines["model"] = {
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "nll": test_metrics["nll"],
        }
    validation_predictions_path = output_root / "validation_predictions.json"
    test_predictions_path = output_root / "test_predictions.json"
    validation_predictions_path.write_text(
        json.dumps(validation_predictions, indent=2), encoding="utf-8"
    )
    test_predictions_path.write_text(
        json.dumps(test_predictions, indent=2), encoding="utf-8"
    )
    split_states = {
        role: sorted(
            {record.structural_state_id for record in items},
            key=_state_sort_key,
        )
        for role, items in records.items()
    }
    report = {
        "task": "kw51_fuzzy_response",
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "data_root": str(data.data_root),
        "fuzzy": asdict(fuzzy),
        "split_mode": data.split_mode,
        "split_fold": data.split_fold,
        "split_states": split_states,
        "states": {role: len(items) for role, items in records.items()},
        "queries": {
            "train": len(train_refs),
            "validation": len(validation_refs),
            "test": len(test_refs),
        },
        "target_length": target_length,
        "normalization": normalization.to_dict(),
        "audit": audit,
        "diagnostic_only": bool(diagnostic_only),
        "run_status": run_status,
        "formal_run_requested": bool(fuzzy.formal_run),
        "formal_conditions": formal_conditions,
        "formal_ok": bool(formal_ok),
        "evaluation_state_counts": {
            "validation": validation_states,
            "test": test_states,
            "minimum_for_formal": int(fuzzy.formal_min_eval_states),
        },
        "audit_failed": bool(audit_failed),
        "target_semantics": (
            "gaussian_footprint_response_over_observed_bag_derived_ranges; "
            "not the posterior of the hidden true range; true_range_mean_mm is "
            "diagnostic-only and is never a model input"
        ),
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "per_state": {
            "validation": validation_per_state,
            "test": test_per_state,
        },
        "state_level": {
            "validation": validation_state_summary,
            "test": test_state_summary,
        },
        "baselines": {"test": test_baselines},
        "effective_n": {
            "validation_states": len(split_states["validation"]),
            "test_states": len(split_states["test"]),
            "note": "queries share structural states; state-level counts are the effective n",
        },
        "weights": {"checkpoint": str(checkpoint_path)},
        "history": str(output_root / "history.json"),
        "predictions": {
            "validation": str(validation_predictions_path),
            "test": str(test_predictions_path),
        },
        "elapsed_sec": time.perf_counter() - started,
    }
    report_path = output_root / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"training_report={report_path} run_status={report['run_status']}",
        flush=True,
    )
    return report
