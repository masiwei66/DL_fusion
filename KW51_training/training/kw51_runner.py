"""KW51 ranking runner using the shared observation and backbone contracts."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dl_fusion.training.lifecycle import (
    BestMetricTracker,
    build_adamw,
    build_scheduler,
    materialize_lazy_parameters,
    restore_model_checkpoint,
    save_model_checkpoint,
    step_scheduler,
)
from dl_fusion.experiment import (
    evaluation_candidate_pools,
    limit_per_state,
    limit_states,
)
from dl_fusion.training.runtime import (
    configure_runtime,
    loader_generator,
    seed_worker,
    write_run_artifacts,
)

from ..data.fem_insar import (
    FEMArchiveIndex,
    FEMRegionalSource,
    FEMRankingDataset,
    FEMQueryBatchSampler,
    RegionConfig,
    build_fem_archive_index,
    collate_fem_pairs,
    derive_ad_groups,
    fit_regional_normalization,
    make_fem_candidate_pairs,
    make_fem_group_pairs,
    make_fem_window_refs,
    split_fem_records,
    split_fem_trajectory_keys,
    trajectory_key,
)
from ..models.kw51_ranking import KW51CandidateScorer
from .kw51_audit import (
    run_exact_state_identifiability_audit,
    run_identifiability_audit,
    summarize_ad_group_cross_validation_audit,
)
from .progress import make_progress


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "query": batch["query"].to(device),
        "candidate": batch["candidate"].to(device),
        "label": batch["label"].to(device),
        "metadata": batch["metadata"],
    }


def _candidate_key(item: dict[str, Any]) -> str:
    if item.get("candidate_group_id") is not None:
        return f"group_{int(item['candidate_group_id'])}"
    return str(item.get("candidate_state_id", ""))


def _resolve_run_status(
    diagnostic_only: bool, audit: dict[str, Any] | None
) -> tuple[bool, bool]:
    """Return (diagnostic_only, audit_failed).

    A failed audit always marks the run diagnostic-only; an explicit
    diagnostic_only flag is preserved even when the audit passes. Formal
    configs should set audit_abort_on_fail so a failed audit stops the run.
    """
    audit_failed = audit is not None and str(audit.get("status")) != "PASS"
    return bool(diagnostic_only) or audit_failed, audit_failed


def _preflight_readiness(
    *,
    smoke_enabled: bool,
    diagnostic_only: bool,
    learnability_audit: bool,
    audit: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    if smoke_enabled:
        return "NOT_FORMAL_CONFIG", ["smoke.enabled=true"]
    if diagnostic_only:
        return "NOT_FORMAL_CONFIG", ["data.diagnostic_only=true"]
    if not learnability_audit:
        return "BLOCKED", ["data.learnability_audit=false"]
    if audit is None:
        return "BLOCKED", ["identifiability audit did not produce a result"]
    if str(audit.get("status")) != "PASS":
        reasons = [str(value) for value in audit.get("reasons", [])]
        return "BLOCKED", reasons or ["identifiability audit did not pass"]
    return "READY_FOR_FORMAL_TRAINING", []


def _archive_provenance(index: FEMArchiveIndex) -> dict[str, Any]:
    manifest = index.run_manifest
    audit = manifest.get("audit_report") or {}
    root = Path(manifest.get("data_root") or index.records[0].path.parent)
    audit_path = next(
        (
            name
            for name in ("fem_archive_audit.json", "fem_archive_audit_live.json")
            if (root / name).exists()
        ),
        None,
    )
    return {
        "schema_version": manifest.get("schema_version"),
        "status": manifest.get("status"),
        "run_manifest_missing": bool(manifest.get("run_manifest_missing", False)),
        "sample_count": manifest.get("sample_count", audit.get("sample_count")),
        "structural_state_count": manifest.get(
            "structural_state_count", audit.get("structural_state_count")
        ),
        "audit_status": audit.get("status"),
        "audit_blockers": audit.get("blockers", []),
        "array_hash_verification": audit.get("array_hash_verification"),
        "audit_report_file": audit_path,
    }


def _state_sort_key(state_id: str) -> tuple[int, Any]:
    try:
        return (0, int(state_id))
    except (TypeError, ValueError):
        return (1, str(state_id))


def _build_group_galleries(
    source: FEMRegionalSource,
    group_map,
    *,
    query_state_ids: list[str],
    exclude_query_state: bool,
) -> dict[str | None, dict[int, dict[str, np.ndarray]]]:
    base = {
        group.group_id: source.aggregate_group(group.group_id, group.state_ids)
        for group in group_map.groups
    }
    galleries: dict[str | None, dict[int, dict[str, np.ndarray]]] = {None: base}
    if exclude_query_state:
        for state_id in query_state_ids:
            galleries[state_id] = {
                group.group_id: source.aggregate_group(
                    group.group_id,
                    group.state_ids,
                    exclude_state_id=state_id,
                )
                for group in group_map.groups
            }
    return galleries


def _assert_same_observation_support(
    query_source: FEMRegionalSource, gallery_source: FEMRegionalSource
) -> None:
    if not np.array_equal(query_source.sample_node_ids, gallery_source.sample_node_ids):
        raise RuntimeError("Query and candidate gallery sampled different FEM nodes")
    if not np.array_equal(
        query_source.observation_coordinates, gallery_source.observation_coordinates
    ):
        raise RuntimeError("Query and candidate gallery use different observation coordinates")
    if not np.array_equal(query_source.point_present_mask, gallery_source.point_present_mask):
        raise RuntimeError("Query and candidate gallery use different observation masks")


def _ranking_metrics(scores: list[float], labels: list[int], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Tie-aware ranking metrics.

    A single positive can share its score with several candidates. The
    expected rank, reciprocal rank and Top-k hit rate average over the
    tie positions uniformly instead of letting Python's stable sort order
    decide the outcome. Pairwise ties count as half a success. With no ties
    these values reduce to the previous hard-rank metrics.
    """
    groups: dict[tuple[str, int, int, int], list[tuple[float, int, str, str]]] = {}
    for score, label, item in zip(scores, labels, metadata):
        key = (
            str(item["query_sample_id"]),
            int(item["query_event"]),
            int(item["query_start"]),
            int(item.get("observation_realization_id", 0)),
        )
        groups.setdefault(key, []).append(
            (
                float(score),
                int(label),
                _candidate_key(item),
                str(item.get("query_state_id", "")),
            )
        )
    ranks: list[float] = []
    reciprocal: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    ndcg: list[float] = []
    pair_correct = 0.0
    pair_total = 0
    tie_groups = 0
    spreads: list[float] = []
    per_state: dict[str, dict[str, float]] = defaultdict(
        lambda: {"groups": 0.0, "top1": 0.0, "mrr": 0.0}
    )
    for rows in groups.values():
        positives = [row for row in rows if row[1] == 1]
        if not positives:
            continue
        positive_score = max(row[0] for row in positives)
        n_better = sum(1 for row in rows if row[0] > positive_score)
        n_tied = sum(1 for row in rows if row[0] == positive_score)
        expected_rank = n_better + (n_tied + 1) / 2.0
        expected_rr = float(
            np.mean([1.0 / (n_better + k) for k in range(1, n_tied + 1)])
        )
        expected_top1 = 1.0 / n_tied if n_better == 0 else 0.0
        expected_top3 = max(0.0, min(n_tied, 3 - n_better)) / n_tied
        ranks.append(expected_rank)
        reciprocal.append(expected_rr)
        top1.append(expected_top1)
        top3.append(expected_top3)
        ordered = sorted(rows, key=lambda row: (-row[0], row[2]))
        gains = np.asarray([row[1] for row in ordered], dtype=np.float64)
        dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
        ideal = np.sort(gains)[::-1]
        idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        pos_scores = [row[0] for row in rows if row[1] == 1]
        neg_scores = [row[0] for row in rows if row[1] == 0]
        for pos in pos_scores:
            for neg in neg_scores:
                pair_correct += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
                pair_total += 1
        tie_groups += int(all(row[0] == rows[0][0] for row in rows))
        spreads.append(max(row[0] for row in rows) - min(row[0] for row in rows))
        state = rows[0][3]
        bucket = per_state[state]
        bucket["groups"] += 1.0
        bucket["top1"] += expected_top1
        bucket["mrr"] += expected_rr
    count = max(len(ranks), 1)
    per_state_payload = {
        state: {
            "query_groups": int(values["groups"]),
            "top1_hit_rate": (
                values["top1"] / values["groups"] if values["groups"] else 0.0
            ),
            "mrr": values["mrr"] / values["groups"] if values["groups"] else 0.0,
        }
        for state, values in sorted(per_state.items(), key=lambda item: _state_sort_key(item[0]))
    }
    return {
        "query_groups": len(ranks),
        "top1_hit_rate": float(np.mean(top1)) if top1 else 0.0,
        "top3_hit_rate": float(np.mean(top3)) if top3 else 0.0,
        "mrr": float(np.mean(reciprocal)) if reciprocal else 0.0,
        "ndcg": float(np.mean(ndcg)) if ndcg else 0.0,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "tie_group_fraction": tie_groups / count,
        "mean_score_spread": float(np.mean(spreads)) if spreads else 0.0,
        "effective_n_states": len(per_state),
        "per_query_state": per_state_payload,
    }


def _listwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metadata: list[dict[str, Any]],
) -> torch.Tensor:
    """Listwise softmax loss over complete query candidate groups."""
    groups: dict[tuple[str, int, int, int], list[int]] = {}
    for index, item in enumerate(metadata):
        key = (
            str(item["query_sample_id"]),
            int(item["query_event"]),
            int(item["query_start"]),
            int(item.get("observation_realization_id", 0)),
        )
        groups.setdefault(key, []).append(index)
    losses: list[torch.Tensor] = []
    for indices in groups.values():
        group_logits = logits[indices]
        group_labels = labels[indices] > 0.5
        if not bool(group_labels.any()) or group_logits.numel() < 2:
            continue
        losses.append(-F.log_softmax(group_logits, dim=0)[group_labels].mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    train: bool,
    progress: dict[str, Any] | None = None,
    desc: str | None = None,
    epoch: int | None = None,
    global_step_offset: int = 0,
    on_train_step: Callable[[dict[str, Any]], None] | None = None,
    loss_weight: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    model.train(train)
    total_loss = 0.0
    total_count = 0
    scores: list[float] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    with make_progress(desc, total=len(loader), options=progress) as tracker:
        for batch_index, batch in enumerate(loader, start=1):
            batch_device = _move_batch(batch, device)
            with torch.set_grad_enabled(train):
                logits = model.forward_pair(batch_device)
                loss = _listwise_ranking_loss(logits, batch_device["label"], batch_device["metadata"])
                loss = loss * float(loss_weight)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            total_loss += float(loss.detach()) * len(batch["label"])
            total_count += len(batch["label"])
            if train and on_train_step is not None:
                on_train_step(
                    {
                        "epoch": int(epoch) if epoch is not None else None,
                        "batch": batch_index,
                        "global_step": global_step_offset + batch_index,
                        "batch_loss": float(loss.detach()),
                        "epoch_running_loss": total_loss / max(total_count, 1),
                    }
                )
            scores.extend(logits.detach().cpu().numpy().reshape(-1).tolist())
            labels.extend(batch["label"].cpu().numpy().astype(int).reshape(-1).tolist())
            metadata.extend(batch["metadata"])
            tracker.update(
                batch_index,
                average_loss=total_loss / max(total_count, 1),
            )
    return total_loss / max(total_count, 1), _ranking_metrics(scores, labels, metadata)


def _as_identifier(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _collect_ranking_predictions(model, loader, device) -> list[dict[str, Any]]:
    """Collect per-candidate scores and ranks for an auditable report."""
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch_device = _move_batch(batch, device)
            logits = model.forward_pair(batch_device).detach().cpu().numpy().reshape(-1)
            labels = batch["label"].detach().cpu().numpy().reshape(-1).tolist()
            for score, label, item in zip(logits.tolist(), labels, batch["metadata"]):
                row = {
                    "query_state_id": _as_identifier(item["query_state_id"]),
                    "query_sample_id": str(item["query_sample_id"]),
                    "candidate_state_id": _as_identifier(item["candidate_state_id"]),
                    "query_event": int(item["query_event"]),
                    "candidate_event": int(item["candidate_event"]),
                    "query_start": int(item["query_start"]),
                    "candidate_start": int(item["candidate_start"]),
                    "observation_realization_id": int(item["observation_realization_id"]),
                    "candidate_state_score": float(score),
                    "label": int(label),
                }
                for key in (
                    "candidate_group_id",
                    "candidate_ad_group_id",
                    "candidate_ad_rel",
                    "candidate_kind",
                    "candidate_trajectory_count",
                    "query_ad_group_id",
                    "query_ad_rel",
                    "query_trajectory_key",
                ):
                    if key in item:
                        row[key] = item[key]
                rows.append(row)
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["query_sample_id"],
            row["query_event"],
            row["query_start"],
            row["observation_realization_id"],
        )
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        group.sort(
            key=lambda row: (
                -row["candidate_state_score"],
                str(row.get("candidate_group_id", row["candidate_state_id"])),
            )
        )
        for rank, row in enumerate(group, start=1):
            row["candidate_state_rank"] = rank
            tie_count = sum(
                1
                for other in group
                if other["candidate_state_score"] == row["candidate_state_score"]
            )
            n_better = sum(
                1
                for other in group
                if other["candidate_state_score"] > row["candidate_state_score"]
            )
            row["tie_count"] = tie_count
            row["expected_rank"] = n_better + (tie_count + 1) / 2.0
    return rows


def _write_plot(
    history: list[dict[str, Any]],
    path: Path,
    step_history: list[dict[str, Any]] | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    step_history = step_history or []
    if step_history:
        steps = np.asarray([row["global_step"] for row in step_history], dtype=int)
        batch_losses = np.asarray([row["batch_loss"] for row in step_history], dtype=float)
        axes[0].plot(
            steps,
            batch_losses,
            color="tab:blue",
            alpha=0.22,
            linewidth=0.7,
            label="train batch loss",
        )
        window = min(25, max(1, len(batch_losses) // 40))
        if window > 1:
            rolling_loss = np.convolve(
                batch_losses,
                np.ones(window, dtype=float) / window,
                mode="valid",
            )
            axes[0].plot(
                steps[window - 1 :],
                rolling_loss,
                color="tab:blue",
                linewidth=1.6,
                label=f"train moving average ({window} batches)",
            )
        validation_steps = [row["global_step_end"] for row in history]
        axes[0].plot(
            validation_steps,
            [row["validation_loss"] for row in history],
            color="tab:orange",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="validation epoch mean",
        )
        axes[0].set_xlabel("Optimizer step")
        axes[0].set_yscale("symlog", linthresh=1e-6)
    else:
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
        axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
        axes[0].set_xlabel("Epoch")
    axes[0].set_title("KW51 ranking loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["validation_mrr"] for row in history], marker="o", label="MRR")
    axes[1].plot(
        epochs,
        [row["validation_top1_hit_rate"] for row in history],
        marker="o",
        label="Top-1",
    )
    axes[1].plot(epochs, [row["validation_ndcg"] for row in history], marker="o", label="NDCG")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation ranking metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_kw51_training(config, *, preflight_only: bool = False) -> dict[str, Any]:
    data = config.data
    if (
        data.ranking_target == "ad_group"
        and not data.diagnostic_only
        and not config.smoke.enabled
    ):
        requirements = []
        if not data.learnability_audit:
            requirements.append("learnability_audit=true")
        if not data.audit_abort_on_fail:
            requirements.append("audit_abort_on_fail=true")
        if data.audit_min_oracle_group_accuracy < 0.9:
            requirements.append("audit_min_oracle_group_accuracy>=0.9")
        if data.audit_state_cv_folds < 2:
            requirements.append("audit_state_cv_folds>=2")
        if data.gallery_exclusion != "query_state":
            requirements.append("gallery_exclusion=query_state")
        if requirements:
            raise ValueError(
                "Formal KW51 ad-group runs require the fail-closed training contract: "
                + ", ".join(requirements)
            )
    device = configure_runtime(config.train)
    output_root = Path(config.train.output_root)
    if data.split_fold is not None:
        output_root = output_root.parent / f"{output_root.name}_fold{int(data.split_fold)}"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    index = build_fem_archive_index(data.data_root)
    if data.require_complete_archive:
        run_status = str(index.run_manifest.get("status", "unknown"))
        audit = index.run_manifest.get("audit_report") or {}
        if audit.get("status") and str(audit.get("status")) != "PASS_FEM_ARCHIVE":
            raise RuntimeError(
                f"FEM archive audit is not passing: {audit.get('status')!r}"
            )
        if audit.get("blockers"):
            raise RuntimeError(
                f"FEM archive audit has blockers: {audit['blockers']}"
            )
        requested = int(
            index.run_manifest.get(
                "requested_samples",
                index.run_manifest.get("sample_count", audit.get("sample_count", data.expected_samples)),
            )
            or 0
        )
        expected_samples = max(int(data.expected_samples), requested)
        if run_status in {"active", "running", "in_progress"}:
            raise RuntimeError(
                f"FEM archive is still {run_status}: found {len(index.records)} files; "
                "wait for archive finalization and Gate A before formal training"
            )
        if expected_samples and len(index.records) < expected_samples:
            raise RuntimeError(
                f"FEM archive is incomplete: found {len(index.records)} files, expected at least {expected_samples}"
            )
        if data.expected_structural_states and len(index.state_ids) < data.expected_structural_states:
            raise RuntimeError(
                f"FEM archive has {len(index.state_ids)} structural states, expected {data.expected_structural_states}"
            )
        audited_states = audit.get("structural_state_count")
        if audited_states is not None and int(audited_states) != len(index.state_ids):
            raise RuntimeError(
                "FEM archive audit state count does not match indexed files: "
                f"audit={audited_states}, indexed={len(index.state_ids)}"
            )
    region_config = RegionConfig(
        los_vector=tuple(data.los_vector),
        reference_mode=data.reference_mode,
        unit_scale=data.unit_scale,
        noise_std_mm=data.noise_std_mm,
        point_bias_std_mm=data.point_bias_std_mm,
        missing_probability=data.missing_probability,
        block_dropout_probability=data.block_dropout_probability,
        block_dropout_fraction=data.block_dropout_fraction,
        min_quality=data.min_quality,
        normalization_max_records=data.normalization_max_records,
        input_mode=data.input_mode,
        region_grid_size_m=data.region_grid_size_m,
        coordinate_scale_to_m=data.coordinate_scale_to_m,
        sparse_point_count=data.sparse_point_count,
        min_points_per_region=data.min_points_per_region,
        seed=data.seed,
    )
    group_map = None
    if data.ranking_target == "ad_group":
        group_map = derive_ad_groups(
            index,
            parameter=data.ad_group_parameter,
            expected_count=data.ad_group_count,
        )
    acquisition_count = max(
        (int(record.metadata.get("acquisition_count", 0)) for record in index.records),
        default=0,
    )
    if acquisition_count and data.window_length > acquisition_count:
        raise RuntimeError(
            f"window_length={data.window_length} exceeds archive acquisition count "
            f"{acquisition_count}"
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
    trajectory_split = None
    if group_map is not None:
        trajectory_split = split_fem_trajectory_keys(
            index,
            gallery_fraction=data.gallery_trajectory_fraction,
            train_fraction=data.train_query_trajectory_fraction,
            validation_fraction=data.validation_query_trajectory_fraction,
            test_fraction=data.test_query_trajectory_fraction,
            seed=data.trajectory_split_seed,
        )
        role_keys = {
            "train": set(trajectory_split.train),
            "validation": set(trajectory_split.validation),
            "test": set(trajectory_split.test),
        }
        records = {
            role: [record for record in rows if trajectory_key(record) in role_keys[role]]
            for role, rows in records.items()
        }
    if config.smoke.enabled:
        if group_map is None:
            records = limit_states(records, config.smoke)
        else:
            selected_train_states = {
                state_id
                for group in group_map.groups
                for state_id in sorted(
                    set(group.state_ids)
                    & {record.structural_state_id for record in records["train"]},
                    key=_state_sort_key,
                )[: config.smoke.states_per_split]
            }
            if any(
                len(selected_train_states & set(group.state_ids)) < 2
                for group in group_map.groups
            ):
                raise ValueError(
                    "KW51 ad-group smoke needs at least two training states per group"
                )
            records["train"] = [
                record for record in records["train"]
                if record.structural_state_id in selected_train_states
            ]
        records = {
            role: limit_per_state(
                items,
                config.smoke.records_per_state,
                distinct_by=trajectory_key,
            )
            for role, items in records.items()
        }
        if any(
            len(rows) < config.smoke.states_per_split * config.smoke.records_per_state
            for rows in records.values()
        ):
            raise ValueError(
                "KW51 smoke could not find the requested number of distinct "
                "trajectories for each selected state"
            )

    eval_stride = data.eval_stride or data.stride
    refs = {
        "train": make_fem_window_refs(records["train"], data.window_length, data.stride),
        "validation": make_fem_window_refs(records["validation"], data.window_length, eval_stride),
        "test": make_fem_window_refs(records["test"], data.window_length, eval_stride),
    }
    if config.smoke.enabled:
        refs = {
            role: limit_per_state(items, config.smoke.windows_per_state)
            for role, items in refs.items()
        }

    train_state_ids = sorted({record.structural_state_id for record in records["train"]}, key=_state_sort_key)
    if group_map is not None:
        source_state_ids = set(index.state_ids)
    elif (
        data.candidate_pool_scope == "all_states" and not config.smoke.enabled
    ):
        source_state_ids = set(index.state_ids)
    else:
        source_state_ids = {record.structural_state_id for rows in records.values() for record in rows}
    source_records = [record for record in index.records if record.structural_state_id in source_state_ids]
    if config.smoke.enabled:
        source_records = limit_per_state(
            source_records,
            config.smoke.records_per_state,
            distinct_by=trajectory_key,
        )
    source_index = FEMArchiveIndex(tuple(source_records), index.run_manifest)
    source = FEMRegionalSource(source_index, region_config)
    candidate_source = None
    candidate_gallery_records: list = []
    candidate_galleries = None
    gallery_state_ids: list[str] = []
    gallery_trajectory_keys: list[str] = []
    if group_map is not None:
        gallery_keys = set(trajectory_split.gallery)
        gallery_state_ids = train_state_ids
        candidate_gallery_records = [
            record for record in index.records
            if record.structural_state_id in set(train_state_ids)
            and trajectory_key(record) in gallery_keys
        ]
        candidate_source = FEMRegionalSource(
            FEMArchiveIndex(tuple(candidate_gallery_records), index.run_manifest),
            region_config,
        )
        _assert_same_observation_support(source, candidate_source)
        gallery_trajectory_keys = sorted(
            {trajectory_key(record) for record in candidate_gallery_records}
        )
        train_states_per_group = {
            group.group_id: sorted(
                set(group.state_ids) & set(train_state_ids), key=_state_sort_key
            )
            for group in group_map.groups
        }
        minimum_states = 2 if data.gallery_exclusion == "query_state" else 1
        insufficient = {
            group_id: states
            for group_id, states in train_states_per_group.items()
            if len(states) < minimum_states
        }
        if insufficient:
            raise ValueError(
                "Candidate gallery has too few training states per Ad group: "
                f"{insufficient}; need at least {minimum_states}"
            )
        normalization_records = [*candidate_gallery_records, *records["train"]]
    else:
        normalization_records = records["train"]
    print(
        f"observation_adapter=input_mode:{data.input_mode} "
        f"sparse_points:{len(source.sample_node_ids)} "
        f"grid_size_m:{data.region_grid_size_m:g} "
        f"observation_tokens:{source.region_count} "
        f"covered_regions:{int(source.point_present_mask.sum())} "
        f"source_records:{len(source_records)} "
        f"gallery_records:{len(candidate_gallery_records)}",
        flush=True,
    )
    normalization = fit_regional_normalization(
        source,
        normalization_records,
        max_records=data.normalization_max_records,
        per_region_offset=(data.normalization_scope == "per_region_offset"),
        records_per_state=data.normalization_records_per_state,
        time_feature_mode=data.time_feature_mode,
    )
    fitted_state_ids = set(normalization.fitted_state_ids)
    if not fitted_state_ids.issubset(set(train_state_ids)):
        raise RuntimeError(
            "Normalization fitted_state_ids contain non-training states: "
            f"{sorted(fitted_state_ids - set(train_state_ids))}"
        )
    normalization_path = output_root / "normalization_stats.json"
    normalization_path.write_text(json.dumps(normalization.to_dict(), indent=2), encoding="utf-8")
    if group_map is not None:
        candidate_galleries = _build_group_galleries(
            candidate_source,
            group_map,
            query_state_ids=train_state_ids,
            exclude_query_state=(data.gallery_exclusion == "query_state"),
        )
    all_source_state_ids = source_index.state_ids
    if group_map is None:
        evaluation_pools = evaluation_candidate_pools(
            records, all_source_state_ids, data.candidate_pool_scope
        )
        train_candidate_state_ids = sorted({record.structural_state_id for record in records["train"]})
        candidate_pool_manifest = {
            "train": train_candidate_state_ids,
            "validation": evaluation_pools["validation"],
            "test": evaluation_pools["test"],
        }
    else:
        group_ids = [str(group.group_id) for group in group_map.groups]
        evaluation_pools = {"validation": group_ids, "test": group_ids}
        candidate_pool_manifest = {
            "train": group_ids,
            "validation": group_ids,
            "test": group_ids,
        }
    write_run_artifacts(
        output_root,
        config,
        refs,
        device=device,
        inventory_records=index.records,
        run_manifest={
            "archive": _archive_provenance(index),
            "ranking_target": data.ranking_target,
            "candidate_pool_scope": (
                "training_state_frozen_gallery"
                if group_map is not None
                else data.candidate_pool_scope
            ),
            "candidate_pool_ids": candidate_pool_manifest,
            "gallery_exclusion": data.gallery_exclusion,
            "trajectory_split": (
                trajectory_split.to_dict() if trajectory_split is not None else None
            ),
            "candidate_gallery_state_ids": gallery_state_ids,
            "candidate_gallery_trajectory_keys": gallery_trajectory_keys,
            "window_length": data.window_length,
            "train_stride": data.stride,
            "eval_stride": eval_stride,
            "observation_seed": data.seed,
            "pair_seed": data.split_seed,
            "normalization_stats": normalization_path.name,
        },
    )
    identifiability_audit = None
    if data.learnability_audit and group_map is not None:
        print(
            "stage=identifiability_audit (validation queries vs frozen train gallery)",
            flush=True,
        )
        identifiability_audit = run_identifiability_audit(
            source,
            group_map,
            records["validation"],
            normalization,
            data,
            candidate_gallery=candidate_galleries[None],
            gallery_state_ids=gallery_state_ids,
            gallery_trajectory_keys=gallery_trajectory_keys,
            seed=data.seed,
        )
        audit_path = output_root / "identifiability_audit.json"
        audit_path.write_text(
            json.dumps(identifiability_audit, indent=2), encoding="utf-8"
        )
        print(
            f"identifiability_audit status={identifiability_audit['status']} "
            f"oracle={identifiability_audit['oracle_group_accuracy_noisy']:.3f} "
            f"snr={identifiability_audit['signal_to_noise']:.3f}",
            flush=True,
        )
    elif data.learnability_audit and data.ranking_target == "exact_state":
        print(
            "stage=identifiability_audit (exact-state nearest-centroid check)",
            flush=True,
        )
        identifiability_audit = run_exact_state_identifiability_audit(
            source,
            list(source.index.records),
            normalization,
            data,
            seed=data.seed,
        )
        audit_path = output_root / "identifiability_audit.json"
        audit_path.write_text(
            json.dumps(identifiability_audit, indent=2), encoding="utf-8"
        )
        print(
            f"identifiability_audit status={identifiability_audit['status']} "
            f"oracle={identifiability_audit['oracle_state_accuracy_noisy']:.3f} "
            f"snr={identifiability_audit['signal_to_noise']:.3f}",
            flush=True,
        )
    cross_validation_audit = None
    if group_map is not None and data.audit_state_cv_folds:
        print(
            "stage=identifiability_cross_validation "
            f"folds={data.audit_state_cv_folds} full_state_coverage=True",
            flush=True,
        )
        fold_audits = []
        gallery_keys = set(trajectory_split.gallery)
        role_keys = {
            "train": set(trajectory_split.train),
            "validation": set(trajectory_split.validation),
            "test": set(trajectory_split.test),
        }
        for fold in range(data.audit_state_cv_folds):
            fold_records = split_fem_records(
                index,
                data.split_seed,
                data.train_fraction,
                data.validation_fraction,
                group_map=group_map,
                split_mode=data.split_mode,
                fold=fold,
            )
            fold_records = {
                role: [
                    record
                    for record in rows
                    if trajectory_key(record) in role_keys[role]
                ]
                for role, rows in fold_records.items()
            }
            fold_train_state_ids = sorted(
                {record.structural_state_id for record in fold_records["train"]},
                key=_state_sort_key,
            )
            fold_gallery_records = [
                record
                for record in index.records
                if record.structural_state_id in set(fold_train_state_ids)
                and trajectory_key(record) in gallery_keys
            ]
            fold_candidate_source = FEMRegionalSource(
                FEMArchiveIndex(tuple(fold_gallery_records), index.run_manifest),
                region_config,
            )
            _assert_same_observation_support(source, fold_candidate_source)
            fold_normalization = fit_regional_normalization(
                source,
                [*fold_gallery_records, *fold_records["train"]],
                max_records=data.normalization_max_records,
                per_region_offset=(data.normalization_scope == "per_region_offset"),
                records_per_state=data.normalization_records_per_state,
                time_feature_mode=data.time_feature_mode,
            )
            if not set(fold_normalization.fitted_state_ids).issubset(
                set(fold_train_state_ids)
            ):
                raise RuntimeError(
                    f"Cross-validation fold {fold} normalization used a held-out state"
                )
            fold_galleries = _build_group_galleries(
                fold_candidate_source,
                group_map,
                query_state_ids=fold_train_state_ids,
                exclude_query_state=(data.gallery_exclusion == "query_state"),
            )
            fold_audit = run_identifiability_audit(
                source,
                group_map,
                fold_records["validation"],
                fold_normalization,
                data,
                candidate_gallery=fold_galleries[None],
                gallery_state_ids=fold_train_state_ids,
                gallery_trajectory_keys=sorted(
                    {trajectory_key(record) for record in fold_gallery_records}
                ),
                seed=data.seed + fold,
            )
            fold_audit["fold"] = fold
            fold_audits.append(fold_audit)
        cross_validation_audit = summarize_ad_group_cross_validation_audit(
            fold_audits,
            expected_state_ids=list(index.state_ids),
            required_fold_count=data.audit_state_cv_folds,
            minimum_oracle_accuracy=data.audit_min_oracle_group_accuracy,
        )
        cv_audit_path = output_root / "identifiability_cv_audit.json"
        cv_audit_path.write_text(
            json.dumps(cross_validation_audit, indent=2), encoding="utf-8"
        )
        print(
            f"identifiability_cv status={cross_validation_audit['status']} "
            f"folds={cross_validation_audit['fold_count']} "
            f"states={cross_validation_audit['validation_state_count']} "
            f"oracle={cross_validation_audit['oracle_group_accuracy_noisy']:.3f}",
            flush=True,
        )
    gate_audit = cross_validation_audit or identifiability_audit
    if (
        data.audit_abort_on_fail
        and gate_audit is not None
        and gate_audit["status"] != "PASS"
        and not preflight_only
    ):
        raise RuntimeError(
            "KW51 identifiability audit failed: "
            + "; ".join(gate_audit.get("reasons", []))
        )
    diagnostic_only, audit_failed = _resolve_run_status(
        bool(data.diagnostic_only), gate_audit
    )
    if diagnostic_only:
        print(
            f"run_status=DIAGNOSTIC_ONLY "
            f"diagnostic_only={bool(data.diagnostic_only)} "
            f"audit_failed={audit_failed}",
            flush=True,
        )
    if preflight_only:
        preflight_status, preflight_reasons = _preflight_readiness(
            smoke_enabled=bool(config.smoke.enabled),
            diagnostic_only=bool(data.diagnostic_only),
            learnability_audit=bool(data.learnability_audit),
            audit=gate_audit,
        )
        preflight_report = {
            "schema": "kw51_training_preflight_v1",
            "status": preflight_status,
            "can_start_formal_training": preflight_status == "READY_FOR_FORMAL_TRAINING",
            "training_started": False,
            "reasons": preflight_reasons,
            "task": "kw51_candidate_ranking",
            "ranking_target": data.ranking_target,
            "input_mode": data.input_mode,
            "data_root": str(data.data_root),
            "archive": _archive_provenance(index),
            "archive_counts": {
                "records": len(index.records),
                "structural_states": len(index.state_ids),
            },
            "state_split": {
                role: sorted(
                    {record.structural_state_id for record in items},
                    key=_state_sort_key,
                )
                for role, items in records.items()
            },
            "trajectory_split": (
                trajectory_split.to_dict() if trajectory_split is not None else None
            ),
            "candidate_gallery": (
                {
                    "state_ids": gallery_state_ids,
                    "trajectory_keys": gallery_trajectory_keys,
                    "validation_test_share_same_gallery": True,
                    "groups": [
                        {
                            "group_id": group.group_id,
                            "state_ids": sorted(
                                set(group.state_ids) & set(gallery_state_ids),
                                key=_state_sort_key,
                            ),
                            "record_count": int(
                                candidate_galleries[None][group.group_id]["trajectory_count"]
                            ),
                        }
                        for group in group_map.groups
                    ]
                    if group_map is not None
                    else None,
                }
            ),
            "normalization": {
                "scope": normalization.scope,
                "fitted_state_ids": list(normalization.fitted_state_ids),
            },
            "identifiability_audit": identifiability_audit,
            "identifiability_cv_audit": cross_validation_audit,
            "gate_audit": gate_audit,
            "run_metadata": str(output_root / "run_metadata.json"),
            "resolved_config": str(output_root / "resolved_config.json"),
            "split_manifest": str(output_root / "split_manifest.json"),
        }
        preflight_path = output_root / "preflight_report.json"
        preflight_path.write_text(
            json.dumps(preflight_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"preflight_status={preflight_status} report={preflight_path} "
            "training_started=False",
            flush=True,
        )
        return preflight_report
    if group_map is not None:
        train_pairs = make_fem_group_pairs(
            refs["train"], group_map, realizations=data.pair_realizations
        )
        validation_pairs = make_fem_group_pairs(
            refs["validation"], group_map, realizations=1
        )
        test_pairs = make_fem_group_pairs(refs["test"], group_map, realizations=1)
        candidate_pool_sizes = {role: group_map.group_count for role in ("train", "validation", "test")}
        num_candidates = group_map.group_count
    else:
        train_pairs = make_fem_candidate_pairs(
            refs["train"], train_state_ids,
            negative_candidates_per_window=data.negative_candidates_per_window,
            realizations=data.pair_realizations,
            seed=data.split_seed,
        )
        validation_pairs = make_fem_candidate_pairs(
            refs["validation"], evaluation_pools["validation"],
            negative_candidates_per_window=max(len(evaluation_pools["validation"]) - 1, 1),
            realizations=1,
            seed=data.split_seed + 1,
        )
        test_pairs = make_fem_candidate_pairs(
            refs["test"], evaluation_pools["test"],
            negative_candidates_per_window=max(len(evaluation_pools["test"]) - 1, 1),
            realizations=1,
            seed=data.split_seed + 2,
        )
        candidate_pool_sizes = {
            "train": len(train_state_ids),
            "validation": len(evaluation_pools["validation"]),
            "test": len(evaluation_pools["test"]),
        }
        num_candidates = 1 + min(
            data.negative_candidates_per_window,
            max(len(train_state_ids) - 1, 1),
        )
    train_dataset = FEMRankingDataset(
        train_pairs,
        source,
        normalization,
        group_map=group_map,
        gallery_exclusion=data.gallery_exclusion,
        candidate_galleries=candidate_galleries,
    )
    validation_dataset = FEMRankingDataset(
        validation_pairs,
        source,
        normalization,
        group_map=group_map,
        gallery_exclusion=data.gallery_exclusion,
        candidate_galleries=candidate_galleries,
    )
    test_dataset = FEMRankingDataset(
        test_pairs,
        source,
        normalization,
        group_map=group_map,
        gallery_exclusion=data.gallery_exclusion,
        candidate_galleries=candidate_galleries,
    )
    print(
        f"stage=datasets train_pairs={len(train_pairs)} "
        f"validation_pairs={len(validation_pairs)} test_pairs={len(test_pairs)} "
        f"num_candidates={num_candidates}",
        flush=True,
    )
    loader_kwargs = {
        "collate_fn": collate_fem_pairs,
        "num_workers": config.train.num_workers,
        "pin_memory": config.train.pin_memory,
        "worker_init_fn": seed_worker,
    }
    if config.train.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.train.prefetch_factor
        loader_kwargs["persistent_workers"] = config.train.persistent_workers
    train_batch_sampler = FEMQueryBatchSampler(train_dataset, config.train.batch_size, seed=config.train.seed, shuffle=True)
    validation_batch_sampler = FEMQueryBatchSampler(validation_dataset, config.train.batch_size, seed=config.train.seed + 1, shuffle=False)
    test_batch_sampler = FEMQueryBatchSampler(test_dataset, config.train.batch_size, seed=config.train.seed + 2, shuffle=False)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        generator=loader_generator(config.train.seed), **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, batch_sampler=validation_batch_sampler,
        generator=loader_generator(config.train.seed + 1), **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_sampler=test_batch_sampler,
        generator=loader_generator(config.train.seed + 2), **loader_kwargs
    )
    model = KW51CandidateScorer(config.model.name, config.model.kwargs).to(device)
    # Materialize LazyLinear and the candidate gallery before creating the optimizer.
    print(
        "stage=first_batch (builds the candidate gallery and materializes LazyLinear)",
        flush=True,
    )
    first_batch = next(iter(train_loader))
    materialize_lazy_parameters(
        model,
        first_batch,
        lambda current: model.forward_pair(_move_batch(current, device)),
    )
    optimizer = build_adamw(
        model,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        name=config.train.lr_scheduler,
        epochs=config.train.epochs,
        factor=config.train.lr_factor,
        patience=config.train.lr_patience,
        min_learning_rate=config.train.min_learning_rate,
    )
    checkpoint_path = output_root / "models" / f"{config.model.name}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    progress_options = {
        "mode": config.train.progress,
        "log_every": config.train.log_every_batches,
    }
    history: list[dict[str, Any]] = []
    step_history: list[dict[str, Any]] = []
    step_history_path = output_root / "step_history.jsonl"
    step_history_path.write_text("", encoding="utf-8")
    global_step = 0

    def record_train_step(row: dict[str, Any]) -> None:
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        step_history.append(row)
        with step_history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    best_metric = BestMetricTracker()
    collapse_epochs = 0
    ranking_loss_weight = (
        config.tasks.candidate_compatibility.loss_weight
        if getattr(config, "tasks", None) is not None
        else 1.0
    )
    uniform_ranking_loss = ranking_loss_weight * math.log(
        max(int(num_candidates), 2)
    )
    for epoch in range(1, config.train.epochs + 1):
        train_batch_sampler.set_epoch(epoch - 1)
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            progress=progress_options,
            desc=f"train epoch={epoch}",
            epoch=epoch,
            global_step_offset=global_step,
            on_train_step=record_train_step,
            loss_weight=ranking_loss_weight,
        )
        epoch_step_start = global_step + 1
        global_step += len(train_loader)
        validation_loss, validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer,
            device,
            train=False,
            progress=progress_options,
            desc=f"validation epoch={epoch}",
            loss_weight=ranking_loss_weight,
        )
        if data.loss_collapse_patience > 0:
            baseline_loss = (
                abs(train_loss - uniform_ranking_loss)
                <= data.loss_collapse_tolerance
            )
            score_collapse = (
                abs(train_metrics.get("pairwise_accuracy", 0.5) - 0.5) <= 0.02
                and train_metrics.get("mean_score_spread", float("inf")) <= 1.0e-4
            )
            collapsed = baseline_loss and (
                train_metrics.get("tie_group_fraction", 0.0) >= 0.95
                or score_collapse
            )
            collapse_epochs = collapse_epochs + 1 if collapsed else 0
            if collapse_epochs >= data.loss_collapse_patience:
                diagnostic = {
                    "reason": "loss_collapse",
                    "epoch": epoch,
                    "num_candidates": int(num_candidates),
                    "target_loss": uniform_ranking_loss,
                    "train_loss": train_loss,
                    "tolerance": data.loss_collapse_tolerance,
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                    "history": history,
                    "ranking_target": data.ranking_target,
                    "normalization_scope": data.normalization_scope,
                    "time_feature_mode": data.time_feature_mode,
                    "note": (
                        "Training loss is indistinguishable from uniform logits and "
                        "candidate scores are tied. Check the identifiability audit, "
                        "input normalization and candidate construction before tuning."
                    ),
                }
                (output_root / "failure_diagnostic.json").write_text(
                    json.dumps(diagnostic, indent=2), encoding="utf-8"
                )
                raise RuntimeError(
                    "KW51 training loss collapsed to ln(num_candidates) with tied "
                    "candidate scores; see failure_diagnostic.json"
                )
        selection = float(validation_metrics["mrr"])
        row = {
            "epoch": epoch,
            "global_step_start": epoch_step_start,
            "global_step_end": global_step,
            "train_batches": len(train_loader),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_mrr": train_metrics["mrr"],
            "train_pairwise_accuracy": train_metrics["pairwise_accuracy"],
            "train_tie_group_fraction": train_metrics["tie_group_fraction"],
            "validation_mrr": validation_metrics["mrr"],
            "validation_top1_hit_rate": validation_metrics["top1_hit_rate"],
            "validation_top1_expected": validation_metrics["top1_hit_rate"],
            "validation_top3_hit_rate": validation_metrics["top3_hit_rate"],
            "validation_ndcg": validation_metrics["ndcg"],
            "validation_pairwise_accuracy": validation_metrics["pairwise_accuracy"],
            "validation_tie_group_fraction": validation_metrics["tie_group_fraction"],
            "validation_mean_score_spread": validation_metrics["mean_score_spread"],
            "validation_effective_n_states": validation_metrics["effective_n_states"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (output_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if best_metric.update(epoch, selection):
            save_model_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch,
                metadata={
                    "model_name": config.model.name,
                    "model_params": config.model.kwargs,
                    "task": "kw51_candidate_ranking",
                    "ranking_target": data.ranking_target,
                    "gallery_exclusion": data.gallery_exclusion,
                    "diagnostic_only": diagnostic_only,
                    "window_length": data.window_length,
                    "stride": data.stride,
                    "pair_realizations": data.pair_realizations,
                },
            )
        step_scheduler(scheduler, selection)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} "
            f"validation_mrr={validation_metrics['mrr']:.6f} "
            f"validation_top1={validation_metrics['top1_hit_rate']:.6f} "
            f"tie_fraction={validation_metrics['tie_group_fraction']:.3f} "
            f"score_spread={validation_metrics['mean_score_spread']:.3e}",
            flush=True,
        )
        if best_metric.should_stop(config.train.early_stopping_patience):
            break
    restore_model_checkpoint(checkpoint_path, model, device=device)
    # Re-evaluate validation after restoring the selected checkpoint so the
    # report and ranking predictions describe the same model.  During the
    # training loop ``validation_metrics`` refers to the final early-stopping
    # epoch, which can differ from the best validation-MRR checkpoint.
    _, validation_metrics = _run_epoch(
        model,
        validation_loader,
        optimizer,
        device,
        train=False,
        progress=progress_options,
        desc="validation best-checkpoint",
        loss_weight=ranking_loss_weight,
    )
    _, test_metrics = _run_epoch(
        model,
        test_loader,
        optimizer,
        device,
        train=False,
        progress=progress_options,
        desc="test best-checkpoint",
        loss_weight=ranking_loss_weight,
    )
    validation_predictions = _collect_ranking_predictions(model, validation_loader, device)
    test_predictions = _collect_ranking_predictions(model, test_loader, device)
    validation_predictions_path = output_root / "validation_ranking_predictions.json"
    test_predictions_path = output_root / "test_ranking_predictions.json"
    validation_predictions_path.write_text(
        json.dumps(validation_predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test_predictions_path.write_text(
        json.dumps(test_predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_plot(history, output_root / "training_history.png", step_history)
    effective_n = {
        "validation_states": len(
            {record.structural_state_id for record in records["validation"]}
        ),
        "test_states": len({record.structural_state_id for record in records["test"]}),
        "validation_query_groups": validation_metrics["query_groups"],
        "test_query_groups": test_metrics["query_groups"],
        "validation_environment_keys": len(
            {trajectory_key(record) for record in records["validation"]}
        ),
        "test_environment_keys": len(
            {trajectory_key(record) for record in records["test"]}
        ),
        "note": (
            "query groups and environment keys are repeated measurements; the "
            "effective sample size for state-level conclusions is the held-out "
            "structural-state count, not the query-group count"
        ),
    }
    ad_group_payload = None
    exact_state_diagnostic = None
    if group_map is not None:
        split_states = {
            role: {record.structural_state_id for record in items}
            for role, items in records.items()
        }
        ad_group_payload = [
            {
                "group_id": group.group_id,
                "value": group.value,
                "state_ids": list(group.state_ids),
                "state_counts": {
                    role: len(set(group.state_ids) & states)
                    for role, states in split_states.items()
                },
            }
            for group in group_map.groups
        ]
        exact_state_diagnostic = {
            "group_count": group_map.group_count,
            "theoretical_ceiling_top1": float(validation_metrics["top1_hit_rate"])
            / float(group_map.group_count),
            "note": (
                "The exact 25-state top-1 cannot exceed group top-1 divided by the "
                "number of near-duplicate states per identifiable group under this "
                "observable. This is a theoretical ceiling, not a measured 25-state "
                "accuracy."
            ),
        }
    if data.ranking_target == "ad_group":
        evaluation_setting = "held_out_states_and_environment_keys_frozen_train_gallery"
        gallery_rule = (
            "Candidate groups are aggregated only from training states and gallery "
            "environment keys. Validation and test use the same frozen gallery; their "
            "query states and environment keys are absent from it. Training queries "
            "exclude their own state from each group reference when configured."
        )
        positive_exclusion = (
            "training_query_state_excluded; evaluation_queries_use_frozen_train_gallery"
        )
    else:
        evaluation_setting = "closed_set_exact_state_gallery"
        gallery_rule = (
            "Every exact-state candidate excludes the query's shared environment "
            "trajectory; the matching state also excludes the query sample."
        )
        positive_exclusion = (
            "query_trajectory_excluded_from_every_state_aggregate"
        )
    report = {
        "task": "kw51_candidate_ranking",
        "primary_outputs": ["candidate_state_score", "candidate_state_rank", "top_k_candidates"],
        "inference_outputs": (
            list(config.inference.outputs)
            if getattr(config, "inference", None) is not None
            else ["candidate_compatibility"]
        ),
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "data_root": str(data.data_root),
        "archive_schema": "kw51_fem_archive_v2",
        "ranking_target": data.ranking_target,
        "ad_group_parameter": data.ad_group_parameter if group_map is not None else None,
        "ad_groups": ad_group_payload,
        "split_mode": data.split_mode,
        "split_fold": data.split_fold,
        "split_states": {
            role: sorted(
                {record.structural_state_id for record in items},
                key=_state_sort_key,
            )
            for role, items in records.items()
        },
        "trajectory_split": (
            trajectory_split.to_dict() if trajectory_split is not None else None
        ),
        "candidate_gallery": (
            {
                "state_ids": gallery_state_ids,
                "trajectory_keys": gallery_trajectory_keys,
                "validation_test_share_same_gallery": True,
                "groups": [
                    {
                        "group_id": group.group_id,
                        "state_ids": sorted(
                            set(group.state_ids) & set(gallery_state_ids),
                            key=_state_sort_key,
                        ),
                        "record_count": int(
                            candidate_galleries[None][group.group_id]["trajectory_count"]
                        ),
                    }
                    for group in group_map.groups
                ],
            }
            if group_map is not None
            else None
        ),
        "gallery_exclusion": data.gallery_exclusion,
        "evaluation_setting": evaluation_setting,
        "diagnostic_only": bool(diagnostic_only),
        "run_status": (
            "SMOKE_ONLY"
            if config.smoke.enabled
            else "DIAGNOSTIC_ONLY"
            if diagnostic_only
            else "FORMAL"
        ),
        "run_kind": "SMOKE_ONLY" if config.smoke.enabled else "EXPERIMENT",
        "audit_failed": bool(audit_failed),
        "selection_metric": "tie_aware_mrr",
        "input_mode": data.input_mode,
        "region_grid_size_m": float(data.region_grid_size_m),
        "coordinate_scale_to_m": float(data.coordinate_scale_to_m),
        "sparse_point_count": int(len(source.sample_node_ids)),
        "min_points_per_region": int(data.min_points_per_region),
        "region_count": int(source.region_count),
        "los_vector": list(data.los_vector),
        "reference_mode": data.reference_mode,
        "unit_scale": data.unit_scale,
        "normalization": normalization.to_dict(),
        "normalization_scope": data.normalization_scope,
        "time_feature_mode": data.time_feature_mode,
        "normalization_stats": str(normalization_path),
        "window_length": data.window_length,
        "stride": data.stride,
        "states": {key: len(value) for key, value in records.items()},
        "windows": {key: len(value) for key, value in refs.items()},
        "pairs": {"train": len(train_pairs), "validation": len(validation_pairs), "test": len(test_pairs)},
        "candidate_pool_scope": (
            "training_state_frozen_gallery"
            if group_map is not None
            else data.candidate_pool_scope
        ),
        "candidate_pool_scope_configured": data.candidate_pool_scope,
        "candidate_pool_kind": "ad_group" if group_map is not None else "structural_state",
        "candidate_pool_state_ids": candidate_pool_manifest,
        "candidate_pool_sizes_by_role": candidate_pool_sizes,
        "candidate_pool_states": (
            group_map.group_count if group_map is not None
            else len(evaluation_pools["validation"])
            if evaluation_pools["validation"] == evaluation_pools["test"]
            else None
        ),
        "smoke_only": bool(config.smoke.enabled),
        "eval_stride": eval_stride,
        "run_metadata": str(output_root / "run_metadata.json"),
        "resolved_config": str(output_root / "resolved_config.json"),
        "split_manifest": str(output_root / "split_manifest.json"),
        "run_requirements": str(output_root / "requirements_frozen.txt"),
        "num_candidates_per_query": int(num_candidates),
        "training_instances": len(train_dataset),
        "pair_realizations": data.pair_realizations,
        "ranking_loss": "listwise_softmax_over_query_candidate_group",
        "ranking_loss_weight": float(ranking_loss_weight),
        "batching": "query_grouped_without_candidate_group_split",
        "query_encoding": "deduplicated_within_batch_by_query_metadata",
        "best_epoch": best_metric.best_epoch,
        "validation_best_mrr": best_metric.best_score,
        "effective_n": effective_n,
        "exact_state_diagnostic": exact_state_diagnostic,
        "identifiability_audit": identifiability_audit,
        "identifiability_cv_audit": cross_validation_audit,
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "weights": {"checkpoint": str(checkpoint_path)},
        "history": str(output_root / "history.json"),
        "step_history": str(step_history_path),
        "plots": {"training_history": str(output_root / "training_history.png")},
        "ranking_predictions": {
            "validation": str(validation_predictions_path),
            "test": str(test_predictions_path),
        },
        "leakage_rule": (
            "query windows are split by structural_state_id and environment trajectory "
            "key; normalization uses training states and training/gallery keys. "
            + gallery_rule
        ),
        "query_view": (
            "FEM XYZ -> deterministic sparse deck nodes -> LOS projection and mm "
            "conversion -> point-level synthetic noise/missingness/quality -> "
            "optional fixed square-grid aggregation (region mode) -> reference removal -> "
            "train-only normalization"
        ),
        "query_augmentation": {
            "noise_std_mm": data.noise_std_mm,
            "point_bias_std_mm": data.point_bias_std_mm,
            "missing_probability": data.missing_probability,
            "block_dropout_probability": data.block_dropout_probability,
            "source_hdf5_degradation": False,
        },
        "candidate_view": (
            "clean LOS response on the same sparse support and observation grid, "
            "aggregated over training states and gallery-only environment keys for "
            "each Ad group (or one structural state in exact_state mode)"
        ),
        "score_head_context": ["quality", "valid_mask", "coverage"],
        "auxiliary_targets": ["Id_rel", "Ad_rel"],
        "candidate_gallery_aggregation": (
            "mean_over_training_states_and_gallery_environment_keys_per_ad_group"
            if group_map is not None
            else "mean_over_all_trajectories_per_structural_state"
        ),
        "positive_candidate_exclusion": positive_exclusion,
        "region_map": {
            "input_mode": data.input_mode,
            "observation_token_count": int(source.region_count),
            "grid_region_count": int(source.region_map.grid.region_count),
            "grid_size_m": float(source.region_map.grid.cell_size_m),
            "grid_origin_xy": list(source.region_map.grid.origin_xy),
            "observation_coordinates_xy": source.observation_coordinates.astype(float).tolist(),
            "centers_xy": source.region_map.point_features.astype(float).tolist(),
            "geometric_deck_node_count": source.region_map.effective_node_count.astype(int).tolist(),
            "sampled_point_count_per_region": source.sampled_points_per_region.astype(int).tolist(),
            "valid_mask": source.region_map.valid_mask.astype(bool).tolist(),
            "weighting": "quality-weighted mean over observed sparse points in each fixed square cell",
        },
        "real_insar_training": False,
        "elapsed_sec": time.perf_counter() - started,
    }
    report_path = output_root / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"training_report={report_path} run_status={report['run_status']}",
        flush=True,
    )
    return report
