"""Five-group fuzzy state inference (candidate FEM parameter-family ranking)."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.fem_insar import (
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
from .fuzzy_group_audit import run_fuzzy_group_audit
from .fuzzy_losses import fuzzy_state_loss
from .fuzzy_metrics import (
    bootstrap_ci,
    distribution_metrics,
    group_metrics,
    group_per_state,
)
from .fuzzy_runner import _require_archive
from .progress import make_progress


def _run_state_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    train: bool,
    progress: dict[str, Any] | None,
    desc: str | None,
    response_auxiliary_weight: float,
    mean_consistency_weight: float,
    collect: bool = False,
):
    model.train(train)
    total_loss = 0.0
    total_count = 0
    component_sums: dict[str, float] = defaultdict(float)
    collected: dict[str, list[Any]] = {
        "group_logits": [],
        "group_ids": [],
        "ad_rel": [],
        "id_rel": [],
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
                loss, components = fuzzy_state_loss(
                    outputs,
                    batch,
                    response_auxiliary_weight=response_auxiliary_weight,
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
                collected["group_logits"].append(
                    outputs["group_logits"].detach().cpu()
                )
                collected["group_ids"].append(
                    batch.targets["ad_group_id"].detach().cpu()
                )
                collected["ad_rel"].append(batch.targets["ad_rel"].detach().cpu())
                collected["id_rel"].append(batch.targets["id_rel"].detach().cpu())
                if "mean" in outputs:
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


def _evaluate_state(
    model,
    loader,
    device,
    normalization: FuzzyNormalization,
    group_map,
    *,
    progress: dict[str, Any] | None,
    desc: str,
    response_auxiliary_weight: float,
    mean_consistency_weight: float,
):
    loss, components, collected = _run_state_epoch(
        model,
        loader,
        None,
        device,
        train=False,
        progress=progress,
        desc=desc,
        response_auxiliary_weight=response_auxiliary_weight,
        mean_consistency_weight=mean_consistency_weight,
        collect=True,
    )
    logits = torch.cat(collected["group_logits"], dim=0)
    group_ids = torch.cat(collected["group_ids"], dim=0)
    ad_rel = torch.cat(collected["ad_rel"], dim=0)
    id_rel = torch.cat(collected["id_rel"], dim=0)
    bag_centroid = (
        torch.cat(collected["bag_centroid"], dim=0)
        if collected["bag_centroid"]
        else torch.zeros((len(collected["metadata"]), 2))
    )
    metadata = collected["metadata"]
    group_values = [group_map.group_value(index) for index in range(group_map.group_count)]
    group_result = group_metrics(logits, group_ids, group_values)
    metrics: dict[str, Any] = {
        "group": group_result,
        "loss": float(loss),
        **{f"component_{key}": float(value) for key, value in components.items()},
    }
    response_result = None
    if collected["mean"]:
        mean = torch.cat(collected["mean"], dim=0)
        logvar = torch.cat(collected["logvar"], dim=0)
        samples = torch.cat(collected["samples"], dim=0)
        target_mean = torch.cat(collected["target_mean"], dim=0)
        target_std = torch.cat(collected["target_std"], dim=0)
        valid = torch.cat(collected["valid"], dim=0)
        sample_valid = torch.cat(collected["sample_valid"], dim=0)
        response_result = distribution_metrics(
            mean,
            logvar,
            samples,
            target_mean,
            target_std,
            valid,
            target_scale_mm=float(normalization.target_std),
            sample_valid=sample_valid,
        )
        metrics["response"] = response_result
    else:
        mean = logvar = None
    probabilities = torch.softmax(logits, dim=-1)
    predicted = probabilities.argmax(dim=-1)
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    values = np.asarray(group_values, dtype=np.float64)
    for index, item in enumerate(metadata):
        probability = probabilities[index].numpy()
        true_group = int(group_ids[index])
        predicted_group = int(predicted[index])
        group_nll = float(-math.log(max(float(probability[true_group]), 1.0e-12)))
        expected_ad = float(probability @ values)
        std_ad = float(
            math.sqrt(max(float(probability @ ((values - expected_ad) ** 2)), 0.0))
        )
        rows.append(
            {
                "structural_state_id": str(item["structural_state_id"]),
                "correct": float(predicted_group == true_group),
                "group_nll": group_nll,
            }
        )
        prediction = {
            "sample_id": str(item["sample_id"]),
            "structural_state_id": str(item["structural_state_id"]),
            "trajectory_id": str(item.get("trajectory_id", "")),
            "observation_realization_id": int(
                item.get("observation_realization_id", 0)
            ),
            "point_count": int(item.get("point_count", 0)),
            "true_ad_group_id": true_group,
            "true_ad_rel": float(ad_rel[index]),
            "true_id_rel": float(id_rel[index]),
            "predicted_ad_group_id": predicted_group,
            "group_probabilities": probability.tolist(),
            "group_nll": group_nll,
            "correct": bool(predicted_group == true_group),
            "ad_rel_expected": expected_ad,
            "ad_rel_std": std_ad,
            "bag_centroid_s": float(bag_centroid[index, 0]),
            "bag_centroid_t": float(bag_centroid[index, 1]),
        }
        if response_result is not None and mean is not None:
            prediction["response_mean_mm"] = (
                mean[index].numpy() * float(normalization.target_std)
                + float(normalization.target_mean)
            ).tolist()
            prediction["response_std_mm"] = (
                torch.exp(0.5 * logvar[index]).numpy()
                * float(normalization.target_std)
            ).tolist()
        predictions.append(prediction)
    per_state = group_per_state(rows)
    return metrics, per_state, predictions


def _state_level_summary(per_state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = [value for value in per_state.values() if value.get("queries", 0) > 0]
    accuracy = [float(value["accuracy"]) for value in values]
    nll = [float(value["group_nll"]) for value in values]
    return {
        "states": len(values),
        "accuracy": {
            "mean": float(np.mean(accuracy)) if accuracy else 0.0,
            "ci95": bootstrap_ci(accuracy),
        },
        "group_nll": {
            "mean": float(np.mean(nll)) if nll else 0.0,
            "ci95": bootstrap_ci(nll),
        },
    }


def _position_only_group_baseline(
    train_dataset,
    test_predictions: list[dict[str, Any]],
    group_count: int,
    max_train_items: int = 300,
) -> dict[str, Any] | None:
    if not test_predictions:
        return None
    train_rows = []
    for index in range(min(int(max_train_items), len(train_dataset))):
        item = train_dataset[index]
        features = np.asarray(item["bag_features"], dtype=np.float64)
        train_rows.append(
            {
                "centroid": features[:2],
                "group": int(item["ad_group_id"]),
            }
        )
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        grouped[row["group"]].append(row["centroid"])
    centroids = {
        group: np.mean(np.stack(features, axis=0), axis=0)
        for group, features in grouped.items()
    }
    if not centroids:
        return None
    hits = 0
    total = 0
    for prediction in test_predictions:
        centroid = np.asarray(
            [prediction.get("bag_centroid_s", 0.0), prediction.get("bag_centroid_t", 0.0)]
        )
        distances = {
            group: float(np.linalg.norm(centroid - center))
            for group, center in centroids.items()
        }
        predicted = min(distances, key=distances.get)
        hits += int(predicted == int(prediction["true_ad_group_id"]))
        total += 1
    return {
        "accuracy": hits / max(total, 1),
        "n": total,
        "note": "Uses only the bag centroid, not the observed values.",
    }


def run_fuzzy_state_training(config) -> dict[str, Any]:
    data = config.data
    fuzzy = config.fuzzy
    if fuzzy is None:
        raise ValueError("kw51 fuzzy state training requires a fuzzy config")
    if fuzzy.primary_target not in ("state", "ad_group"):
        raise ValueError(
            "run_fuzzy_state_training requires primary_target state or ad_group"
        )
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
    with_targets = float(fuzzy.response_auxiliary_weight) > 0.0
    train_dataset = FuzzyQueryDataset(
        train_refs,
        source,
        normalization,
        group_map=group_map,
        with_targets=with_targets,
    )
    validation_dataset = FuzzyQueryDataset(
        validation_refs,
        source,
        normalization,
        group_map=group_map,
        with_targets=with_targets,
    )
    test_dataset = FuzzyQueryDataset(
        test_refs,
        source,
        normalization,
        group_map=group_map,
        with_targets=with_targets,
    )
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
    group_audit = None
    group_audit_failed = True
    if fuzzy.group_audit:
        group_audit = run_fuzzy_group_audit(
            source,
            group_map,
            train_refs,
            test_refs,
            max_refs_per_state=3,
            noise_std_mm=fuzzy.noise_std_mm,
            point_bias_std_mm=fuzzy.point_bias_std_mm,
            min_accuracy=fuzzy.group_audit_min_accuracy,
            min_snr=fuzzy.group_audit_min_snr,
        )
        group_audit_path = output_root / "fuzzy_group_audit.json"
        group_audit_path.write_text(
            json.dumps(group_audit, indent=2), encoding="utf-8"
        )
        group_audit_failed = str(group_audit.get("status")) != "PASS"
        print(
            f"fuzzy_group_audit status={group_audit['status']} "
            f"top1={group_audit['top1_accuracy']:.3f} "
            f"mrr={group_audit['mrr']:.3f} snr={group_audit['snr']:.3f}",
            flush=True,
        )
        if group_audit_failed and fuzzy.group_audit_abort_on_fail:
            raise RuntimeError(
                "KW51 fuzzy group audit failed: "
                + "; ".join(group_audit.get("reasons", []))
            )
    diagnostic_only = bool(data.diagnostic_only) or (
        bool(fuzzy.group_audit) and group_audit_failed
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
        "group_audit_pass": not bool(group_audit_failed),
        "not_diagnostic": not bool(data.diagnostic_only),
    }
    if fuzzy.formal_run:
        blocking = [
            key
            for key in ("archive_complete", "eval_states_ok", "not_diagnostic")
            if not formal_conditions[key]
        ]
        if blocking:
            raise RuntimeError(
                "formal_run requested but blocking formal conditions failed: "
                + ", ".join(blocking)
            )
        if not formal_conditions["group_audit_pass"]:
            # The state head may still be trained for the minimal validation, but
            # the run is not formal and cannot support a state-compatibility claim.
            diagnostic_only = True
    formal_ok = (
        bool(fuzzy.formal_run)
        and archive_complete
        and eval_states_ok
        and not group_audit_failed
        and not bool(data.diagnostic_only)
        and not bool(diagnostic_only)
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
    response_head_length = (
        target_length if float(fuzzy.response_auxiliary_weight) > 0.0 else 0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FuzzyResponseModel(
        config.model.name,
        model_params,
        response_head_length,
        group_count=group_map.group_count,
        head_hidden=head_hidden,
        dropout=head_dropout,
        bag_feature_dropout=fuzzy.bag_feature_dropout,
    ).to(device)
    print(
        "stage=first_batch (materializes bag pooling, state head and optional response head)",
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
            mode="max",
            factor=config.train.lr_factor,
            patience=config.train.lr_patience,
            min_lr=config.train.min_learning_rate,
        )
    progress_options = {
        "mode": config.train.progress,
        "log_every": config.train.log_every_batches,
    }
    checkpoint_path = output_root / "models" / f"{config.model.name}_state_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.train.epochs + 1):
        train_loss, train_components, _ = _run_state_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            progress=progress_options,
            desc=f"train epoch={epoch}",
            response_auxiliary_weight=float(fuzzy.response_auxiliary_weight),
            mean_consistency_weight=fuzzy.mean_consistency_weight,
        )
        validation_metrics, validation_per_state, _ = _evaluate_state(
            model,
            validation_loader,
            device,
            normalization,
            group_map,
            progress=progress_options,
            desc=f"validation epoch={epoch}",
            response_auxiliary_weight=float(fuzzy.response_auxiliary_weight),
            mean_consistency_weight=fuzzy.mean_consistency_weight,
        )
        selection = float(validation_metrics["group"]["mrr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_group_ce": train_components.get("group_ce"),
            "train_group_accuracy": train_components.get("group_accuracy"),
            "validation_loss": validation_metrics["loss"],
            "validation_group_top1": validation_metrics["group"]["top1_accuracy"],
            "validation_group_top3": validation_metrics["group"]["top3_accuracy"],
            "validation_group_mrr": validation_metrics["group"]["mrr"],
            "validation_group_nll": validation_metrics["group"]["nll"],
            "validation_group_ece": validation_metrics["group"]["ece"],
            "validation_ad_rel_abs_error": validation_metrics["group"].get(
                "ad_rel_abs_error"
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if "response" in validation_metrics:
            row["validation_response_nll"] = validation_metrics["response"]["nll"]
        history.append(row)
        (output_root / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if selection > best_score:
            best_score = selection
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_name": config.model.name,
                    "model_params": model_params,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "task": "kw51_fuzzy_state",
                    "primary_target": fuzzy.primary_target,
                    "target_length": response_head_length,
                    "group_count": group_map.group_count,
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
            f"epoch={epoch} train_group_ce={train_components.get('group_ce'):.4f} "
            f"val_group_top1={validation_metrics['group']['top1_accuracy']:.4f} "
            f"val_group_mrr={validation_metrics['group']['mrr']:.4f} "
            f"val_group_nll={validation_metrics['group']['nll']:.4f} "
            f"val_group_ece={validation_metrics['group']['ece']:.4f}",
            flush=True,
        )
        if config.train.early_stopping_patience and stale >= config.train.early_stopping_patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, validation_per_state, validation_predictions = _evaluate_state(
        model,
        validation_loader,
        device,
        normalization,
        group_map,
        progress=progress_options,
        desc="validation best-checkpoint",
        response_auxiliary_weight=float(fuzzy.response_auxiliary_weight),
        mean_consistency_weight=fuzzy.mean_consistency_weight,
    )
    test_metrics, test_per_state, test_predictions = _evaluate_state(
        model,
        test_loader,
        device,
        normalization,
        group_map,
        progress=progress_options,
        desc="test best-checkpoint",
        response_auxiliary_weight=float(fuzzy.response_auxiliary_weight),
        mean_consistency_weight=fuzzy.mean_consistency_weight,
    )
    validation_predictions_path = output_root / "validation_predictions.json"
    test_predictions_path = output_root / "test_predictions.json"
    validation_predictions_path.write_text(
        json.dumps(validation_predictions, indent=2), encoding="utf-8"
    )
    test_predictions_path.write_text(
        json.dumps(test_predictions, indent=2), encoding="utf-8"
    )
    position_only = _position_only_group_baseline(
        train_dataset, test_predictions, group_map.group_count
    )
    split_states = {
        role: sorted(
            {record.structural_state_id for record in items},
            key=lambda state: (0, int(state)) if state.isdigit() else (1, state),
        )
        for role, items in records.items()
    }
    report = {
        "task": "kw51_fuzzy_response",
        "primary_target": fuzzy.primary_target,
        "target_semantics": (
            "candidate FEM Ad_rel parameter-family compatibility; not field "
            "stiffness, not component damage, not a calibrated field parameter"
        ),
        "phase_policy": (
            "pre/during/post are weak engineering-phase evidence; not training "
            "labels; post != healthy; during != a unique FEM state"
        ),
        "id_policy": (
            "Id_rel and exact 25-state identification are diagnostics; static "
            "LOS cannot identify Id_rel; the ~0.2 exact top-1 cap is conditional "
            "on indistinguishable Id duplicates and a uniform within-group prior"
        ),
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "group_count": group_map.group_count,
        "group_values": [
            group_map.group_value(index) for index in range(group_map.group_count)
        ],
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
        "response_auxiliary_weight": float(fuzzy.response_auxiliary_weight),
        "normalization": normalization.to_dict(),
        "group_audit": group_audit,
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
        "best_epoch": best_epoch,
        "selection_metric": "validation_group_mrr",
        "best_validation_group_mrr": best_score,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "per_state": {
            "validation": validation_per_state,
            "test": test_per_state,
        },
        "state_level": {
            "validation": _state_level_summary(validation_per_state),
            "test": _state_level_summary(test_per_state),
        },
        "baselines": {
            "test": {
                "group_prior": {
                    "accuracy": 1.0 / float(group_map.group_count),
                    "n": len(test_predictions),
                },
                "bag_mean_nearest_centroid": (
                    None if group_audit is None
                    else {
                        "accuracy": group_audit["top1_accuracy"],
                        "top3": group_audit["top3_accuracy"],
                        "mrr": group_audit["mrr"],
                        "n": group_audit["test_refs"],
                        "note": "Data-level audit sample; same test states.",
                    }
                ),
                "position_only": position_only,
                "model": {
                    "top1_accuracy": test_metrics["group"]["top1_accuracy"],
                    "top3_accuracy": test_metrics["group"]["top3_accuracy"],
                    "mrr": test_metrics["group"]["mrr"],
                    "nll": test_metrics["group"]["nll"],
                    "ece": test_metrics["group"]["ece"],
                    "ad_rel_abs_error": test_metrics["group"].get(
                        "ad_rel_abs_error"
                    ),
                },
            }
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
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"training_report={report_path} run_status={report['run_status']}",
        flush=True,
    )
    return report
