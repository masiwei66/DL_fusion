"""Region-stratified fuzzy 5-group training and formal gating."""
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
    build_deck_geometry,
    fit_fuzzy_normalization,
    make_fuzzy_query_refs,
)
from ..data.fuzzy_region import (
    FuzzyRegionQueryDataset,
    FuzzyRegionSource,
    FuzzyRegionSummaryDataset,
    build_region_layout,
    collate_fuzzy_region_queries,
    collate_fuzzy_region_summaries,
)
from ..models.fuzzy_region import FuzzyRegionModel
from ..models.fuzzy_region_summary import FuzzyRegionSummaryModel
from .fuzzy_losses import fuzzy_state_loss
from .fuzzy_metrics import bootstrap_ci, group_metrics, group_per_state
from .fuzzy_region_audit import (
    DEFAULT_PROTOCOLS,
    configured_protocol,
    run_fuzzy_region_audit,
)
from .fuzzy_runner import _require_archive
from .fuzzy_state_runner import _state_level_summary
from .progress import make_progress


def _cap_records_per_state(records, cap: int):
    if int(cap) <= 0:
        return list(records)
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for record in records:
        state = record.structural_state_id
        if counts[state] < int(cap):
            selected.append(record)
            counts[state] += 1
    return selected


def _run_region_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    train: bool,
    progress: dict[str, Any] | None,
    desc: str | None,
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
        "region_present": [],
        "region_coverage": [],
        "metadata": [],
    }
    with make_progress(desc, total=len(loader), options=progress) as tracker:
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            with torch.set_grad_enabled(train):
                outputs = model(batch)
                loss, components = fuzzy_state_loss(outputs, batch)
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
                collected["group_logits"].append(outputs["group_logits"].detach().cpu())
                collected["group_ids"].append(
                    batch.targets["ad_group_id"].detach().cpu()
                )
                collected["ad_rel"].append(batch.targets["ad_rel"].detach().cpu())
                collected["id_rel"].append(batch.targets["id_rel"].detach().cpu())
                collected["region_present"].append(
                    batch.region_present.detach().cpu()
                )
                collected["region_coverage"].append(
                    batch.region_coverage.detach().cpu()
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


def _evaluate_region(
    model,
    loader,
    device,
    group_map,
    *,
    progress: dict[str, Any] | None,
    desc: str,
):
    loss, components, collected = _run_region_epoch(
        model,
        loader,
        None,
        device,
        train=False,
        progress=progress,
        desc=desc,
        collect=True,
    )
    logits = torch.cat(collected["group_logits"], dim=0)
    group_ids = torch.cat(collected["group_ids"], dim=0)
    ad_rel = torch.cat(collected["ad_rel"], dim=0)
    id_rel = torch.cat(collected["id_rel"], dim=0)
    region_present = torch.cat(collected["region_present"], dim=0)
    region_coverage = torch.cat(collected["region_coverage"], dim=0)
    metadata = collected["metadata"]
    group_values = [
        group_map.group_value(index) for index in range(group_map.group_count)
    ]
    group_result = group_metrics(logits, group_ids, group_values)
    metrics: dict[str, Any] = {
        "group": group_result,
        "loss": float(loss),
        **{f"component_{key}": float(value) for key, value in components.items()},
    }
    probabilities = torch.softmax(logits, dim=-1)
    predicted = probabilities.argmax(dim=-1)
    values = np.asarray(group_values, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
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
        predictions.append(
            {
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
                "region_present": [
                    bool(value) for value in region_present[index].tolist()
                ],
                "region_coverage": region_coverage[index].tolist(),
            }
        )
    per_state = group_per_state(rows)
    return metrics, per_state, predictions


def run_fuzzy_region_training(config) -> dict[str, Any]:
    data = config.data
    fuzzy = config.fuzzy
    if fuzzy is None:
        raise ValueError("kw51 fuzzy region training requires a fuzzy config")
    if fuzzy.observation_mode != "region_bag":
        raise ValueError("run_fuzzy_region_training requires observation_mode=region_bag")
    if fuzzy.primary_target not in ("state", "ad_group"):
        raise ValueError("region training requires primary_target state or ad_group")
    if float(fuzzy.response_auxiliary_weight) != 0.0:
        raise ValueError(
            "region response auxiliary head is not implemented; set "
            "response_auxiliary_weight=0"
        )
    output_root = Path(config.train.output_root)
    if data.split_fold is not None:
        output_root = output_root.parent / f"{output_root.name}_fold{int(data.split_fold)}"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(int(data.seed))
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
    source = FuzzyRegionSource(index, geometry, fuzzy, seed=data.seed)
    normalization = fit_fuzzy_normalization(
        source, records["train"], fuzzy, time_feature_mode=fuzzy.time_feature_mode
    )
    (output_root / "normalization_stats.json").write_text(
        json.dumps(normalization.to_dict(), indent=2), encoding="utf-8"
    )
    layout = build_region_layout(
        geometry,
        mode=fuzzy.region_mode,
        rows=fuzzy.region_rows,
        cols=fuzzy.region_cols,
    )
    (output_root / "region_layout.json").write_text(
        json.dumps(layout.to_dict(), indent=2), encoding="utf-8"
    )
    target_length = int(source._clean_arrays(index.records[0])[0].shape[0])
    if int(data.window_length) != target_length:
        raise ValueError(
            f"data.window_length={data.window_length} does not match archive "
            f"acquisition count {target_length}"
        )
    train_records = _cap_records_per_state(
        records["train"], fuzzy.train_trajectories_per_state
    )
    validation_records = _cap_records_per_state(
        records["validation"], fuzzy.eval_trajectories_per_state
    )
    test_records = _cap_records_per_state(
        records["test"], fuzzy.eval_trajectories_per_state
    )
    train_refs = make_fuzzy_query_refs(train_records, fuzzy.query_realizations)
    validation_refs = make_fuzzy_query_refs(validation_records, 1)
    test_refs = make_fuzzy_query_refs(test_records, 1)
    if fuzzy.region_model == "region_summary":
        train_dataset = FuzzyRegionSummaryDataset(
            train_refs,
            source,
            normalization,
            layout,
            group_map=group_map,
            estimator=fuzzy.region_estimator,
        )
        validation_dataset = FuzzyRegionSummaryDataset(
            validation_refs,
            source,
            normalization,
            layout,
            group_map=group_map,
            estimator=fuzzy.region_estimator,
        )
        test_dataset = FuzzyRegionSummaryDataset(
            test_refs,
            source,
            normalization,
            layout,
            group_map=group_map,
            estimator=fuzzy.region_estimator,
        )
        collate_fn = collate_fuzzy_region_summaries
    else:
        train_dataset = FuzzyRegionQueryDataset(
            train_refs, source, normalization, layout, group_map=group_map
        )
        validation_dataset = FuzzyRegionQueryDataset(
            validation_refs, source, normalization, layout, group_map=group_map
        )
        test_dataset = FuzzyRegionQueryDataset(
            test_refs, source, normalization, layout, group_map=group_map
        )
        collate_fn = collate_fuzzy_region_queries
    loader_kwargs: dict[str, Any] = {
        "collate_fn": collate_fn,
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
    audit = None
    audit_failed = True
    if fuzzy.region_audit:
        audit = run_fuzzy_region_audit(
            index,
            geometry,
            group_map,
            fuzzy,
            seed=data.seed,
            split_seed=data.split_seed,
            train_fraction=data.train_fraction,
            validation_fraction=data.validation_fraction,
            split_mode=data.split_mode,
            protocols=DEFAULT_PROTOCOLS,
            configured=configured_protocol(fuzzy),
        )
        (output_root / "fuzzy_region_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        audit_failed = str(audit.get("status")) != "PASS"
        summary = audit["protocols"][audit["configured_protocol"]["name"]]["summary"]
        print(
            "fuzzy_region_audit status="
            f"{audit['status']} protocol={audit['configured_protocol']['name']} "
            f"top1={summary['state_macro_top1']:.3f} "
            f"mrr={summary['state_macro_mrr']:.3f} "
            f"snr={summary['snr_values']:.3f}",
            flush=True,
        )
        if audit_failed and fuzzy.region_audit_abort_on_fail:
            raise RuntimeError(
                "KW51 fuzzy region audit failed: "
                + "; ".join(
                    audit["protocols"][audit["configured_protocol"]["name"]]["reasons"]
                )
            )
    diagnostic_only = bool(data.diagnostic_only) or (
        bool(fuzzy.region_audit) and audit_failed
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
        "region_audit_pass": not bool(audit_failed),
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
        if not formal_conditions["region_audit_pass"]:
            diagnostic_only = True
    formal_ok = (
        bool(fuzzy.formal_run)
        and archive_complete
        and eval_states_ok
        and not audit_failed
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fuzzy.region_model == "region_summary":
        model_params["input_mode"] = "region"
        model_params["region_pooling"] = fuzzy.region_pooling
        model = FuzzyRegionSummaryModel(
            config.model.name,
            model_params,
            region_count=layout.region_count,
            group_count=group_map.group_count,
            head_hidden=head_hidden,
            dropout=head_dropout,
            bag_feature_dropout=fuzzy.bag_feature_dropout,
        ).to(device)
    else:
        model = FuzzyRegionModel(
            config.model.name,
            model_params,
            region_count=layout.region_count,
            group_count=group_map.group_count,
            region_pooling=fuzzy.region_pooling,
            region_embedding_dim=fuzzy.region_embedding_dim,
            head_hidden=head_hidden,
            dropout=head_dropout,
            bag_feature_dropout=fuzzy.bag_feature_dropout,
            target_length=0,
        ).to(device)
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
    checkpoint_path = (
        output_root / "models" / f"{config.model.name}_region_state_best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.train.epochs + 1):
        train_loss, train_components, _ = _run_region_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            progress=progress_options,
            desc=f"train epoch={epoch}",
        )
        validation_metrics, validation_per_state, _ = _evaluate_region(
            model,
            validation_loader,
            device,
            group_map,
            progress=progress_options,
            desc=f"validation epoch={epoch}",
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
                    "task": "kw51_fuzzy_region_state",
                    "primary_target": "ad_group",
                    "region_layout": layout.to_dict(),
                    "region_pooling": fuzzy.region_pooling,
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
        if (
            config.train.early_stopping_patience
            and stale >= config.train.early_stopping_patience
        ):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, validation_per_state, validation_predictions = _evaluate_region(
        model,
        validation_loader,
        device,
        group_map,
        progress=progress_options,
        desc="validation best-checkpoint",
    )
    test_metrics, test_per_state, test_predictions = _evaluate_region(
        model,
        test_loader,
        device,
        group_map,
        progress=progress_options,
        desc="test best-checkpoint",
    )
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
            key=lambda state: (0, int(state)) if state.isdigit() else (1, state),
        )
        for role, items in records.items()
    }
    audit_summary = None
    audit_configured = None
    if audit is not None:
        audit_configured = audit["configured_protocol"]["name"]
        audit_summary = audit["protocols"][audit_configured]["summary"]
    report = {
        "task": "kw51_fuzzy_region_state",
        "primary_target": "ad_group",
        "observation_mode": fuzzy.observation_mode,
        "target_semantics": (
            "candidate FEM Ad_rel parameter-family compatibility; not field "
            "stiffness, not component damage, not a calibrated field parameter"
        ),
        "claim_scope": (
            "compatibility/ranking among five controlled synthetic Ad_rel "
            "candidate families; not a field structural-health diagnosis"
        ),
        "not_claimed": [
            "field stiffness inversion",
            "component damage classification",
            "healthy/unhealthy state",
            "unique during-retrofit FEM state",
        ],
        "phase_policy": (
            "pre/during/post are weak engineering-phase evidence; not training "
            "labels; post != healthy; during != a unique FEM state"
        ),
        "id_policy": (
            "Id_rel and exact 25-state identification are diagnostics; static "
            "LOS cannot identify Id_rel; the ~0.2 exact top-1 cap is conditional"
        ),
        "model_name": config.model.name,
        "model_params": model_params,
        "region_model": fuzzy.region_model,
        "region_estimator": fuzzy.region_estimator,
        "region_pooling": fuzzy.region_pooling,
        "group_count": group_map.group_count,
        "group_values": [
            group_map.group_value(index) for index in range(group_map.group_count)
        ],
        "data_root": str(data.data_root),
        "fuzzy": asdict(fuzzy),
        "region_layout": layout.to_dict(),
        "split_mode": data.split_mode,
        "split_fold": data.split_fold,
        "split_states": split_states,
        "states": {role: len(items) for role, items in records.items()},
        "trajectories_used": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "trajectory_caps": {
            "train_per_state": int(fuzzy.train_trajectories_per_state),
            "eval_per_state": int(fuzzy.eval_trajectories_per_state),
        },
        "queries": {
            "train": len(train_refs),
            "validation": len(validation_refs),
            "test": len(test_refs),
        },
        "target_length": target_length,
        "normalization": normalization.to_dict(),
        "region_audit": audit,
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
                "region_audit_nearest_centroid": audit_summary,
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
