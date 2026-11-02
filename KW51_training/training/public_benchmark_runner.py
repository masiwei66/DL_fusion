"""Gate-first Track S benchmark reporting for KW51 public data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from spatial_support_contract import (
    SUPPORT_FEATURE_ORDER,
    SUPPORT_NORMALIZATION_POLICY,
    SUPPORT_TOKEN_SCHEMA,
)

from ..config import FuzzyConfig, load_experiment_config
from ..data.fem_insar import build_fem_archive_index, derive_ad_groups
from ..data.fuzzy_range import (
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
from ..data.support_tokens import (
    FuzzySupportTokenDataset,
    build_support_token_layout,
    collate_fuzzy_support_tokens,
    fit_support_token_value_normalization,
    support_token_inputs_from_window_batch,
    support_token_targets_from_window_batch,
)
from ..models.fuzzy_region import FuzzyRegionModel
from ..models.fuzzy_region_mlp import FuzzyRegionMLPModel
from ..models.fuzzy_region_summary import FuzzyRegionSummaryModel
from ..models.support_response import SupportTokenResponseModel, masked_gaussian_nll
from ..benchmark.validation import (
    canonical_sha256,
    validate_public_split,
    validate_state_manifest,
)
from .fuzzy_losses import dense_region_response_loss
from .fuzzy_metrics import bootstrap_ci, dense_region_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str | Path, *, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def validate_state_cv_splits(
    state_manifest: dict[str, Any], split_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate a complete state-level 5-fold split and mark train-only fit IDs."""
    states = state_manifest.get("states", [])
    state_ids = [str(item["structural_state_id"]) for item in states]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("state manifest contains duplicate structural_state_id values")
    expected = set(state_ids)
    if split_manifest.get("split_unit") != "structural_state_id":
        raise ValueError("split unit must be structural_state_id")
    folds = split_manifest.get("folds", [])
    if len(folds) != 5:
        raise ValueError(f"expected 5 fixed folds, found {len(folds)}")
    family_by_state = {
        str(item["structural_state_id"]): str(item["candidate_family_id"])
        for item in states
    }
    families = sorted(set(family_by_state.values()))
    validated: list[dict[str, Any]] = []
    for expected_fold, item in enumerate(folds):
        if int(item.get("fold", -1)) != expected_fold:
            raise ValueError("fold indices must be the fixed sequence 0..4")
        split_ids: dict[str, list[str]] = {}
        for key in ("train_state_ids", "validation_state_ids", "test_state_ids"):
            ids = [str(value) for value in item.get(key, [])]
            if len(ids) != len(set(ids)):
                raise ValueError(f"fold {expected_fold} has duplicate IDs in {key}")
            unknown = set(ids) - expected
            if unknown:
                raise ValueError(f"fold {expected_fold} has unknown state IDs {sorted(unknown)}")
            split_ids[key] = ids
        train = set(split_ids["train_state_ids"])
        validation = set(split_ids["validation_state_ids"])
        test = set(split_ids["test_state_ids"])
        if train & validation or train & test or validation & test:
            raise ValueError(f"fold {expected_fold} state split overlap detected")
        if train | validation | test != expected:
            raise ValueError(f"fold {expected_fold} does not cover the full state manifest")
        expected_sizes = (3 * len(families), len(families), len(families))
        actual_sizes = (len(train), len(validation), len(test))
        if actual_sizes != expected_sizes:
            raise ValueError(
                f"fold {expected_fold} expected state counts {expected_sizes}, got {actual_sizes}"
            )
        for key, ids, expected_per_family in (
            ("train_state_ids", train, 3),
            ("validation_state_ids", validation, 1),
            ("test_state_ids", test, 1),
        ):
            counts = {family: 0 for family in families}
            for state_id in ids:
                counts[family_by_state[state_id]] += 1
            if any(count != expected_per_family for count in counts.values()):
                raise ValueError(
                    f"fold {expected_fold} is not candidate-family stratified in {key}"
                )
        validated.append(
            {
                "fold": expected_fold,
                **split_ids,
                "fitted_state_ids": sorted(train, key=_state_sort_key),
            }
        )
    return validated


def _state_sort_key(state_id: str) -> tuple[int, Any]:
    try:
        return (0, int(state_id))
    except ValueError:
        return (1, state_id)


def _validate_archive(
    archive_root: Path,
    expected: dict[str, Any],
    state_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not archive_root.exists():
        return {"status": "BLOCKED", "reason": "archive_path_missing", "path": str(archive_root)}
    try:
        index = build_fem_archive_index(archive_root)
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "reason": "archive_index_error",
            "error": f"{type(exc).__name__}: {exc}",
            "path": str(archive_root),
        }
    counts: dict[str, int] = {}
    for record in index.records:
        counts[record.structural_state_id] = counts.get(record.structural_state_id, 0) + 1
    errors: list[str] = []
    if len(index.records) != int(expected.get("sample_count", -1)):
        errors.append("sample_count_mismatch")
    if len(counts) != int(expected.get("structural_state_count", -1)):
        errors.append("structural_state_count_mismatch")
    expected_per_state = int(expected.get("trajectories_per_state", -1))
    if any(count != expected_per_state for count in counts.values()):
        errors.append("trajectories_per_state_mismatch")
    expected_states = {
        str(item["structural_state_id"]) for item in state_manifest["states"]
    }
    if set(counts) != expected_states:
        errors.append("state_ids_do_not_match_manifest")
    family_to_ad = {
        str(item["family_id"]): float(item["Ad_rel"])
        for item in state_manifest["candidate_families"]
    }
    family_by_state = {
        str(item["structural_state_id"]): str(item["candidate_family_id"])
        for item in state_manifest["states"]
    }
    missing_ad = 0
    mismatched_ad = 0
    for record in index.records:
        actual = record.parameters.get("Ad_rel")
        if actual is None:
            missing_ad += 1
        elif record.structural_state_id not in family_by_state:
            mismatched_ad += 1
        elif abs(
            float(actual)
            - family_to_ad[family_by_state[record.structural_state_id]]
        ) > 1.0e-8:
            mismatched_ad += 1
    if missing_ad:
        errors.append("archive_ad_rel_missing")
    if mismatched_ad:
        errors.append("archive_ad_rel_manifest_mismatch")
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "reason": None if not errors else ";".join(errors),
        "sample_count": len(index.records),
        "structural_state_count": len(counts),
        "trajectories_per_state": {
            state: counts[state] for state in sorted(counts, key=_state_sort_key)
        },
        "state_ids": sorted(counts, key=_state_sort_key),
        "run_manifest_schema": index.run_manifest.get("schema_version"),
        "archive_family_parameter_audit": {
            "parameter": "Ad_rel",
            "missing_record_count": missing_ad,
            "mismatched_record_count": mismatched_ad,
            "status": "PASS" if not missing_ad and not mismatched_ad else "BLOCKED",
        },
        "errors": errors,
    }


def _load_training_config(config: dict[str, Any], project_root: Path):
    training_path = _resolve_path(
        config.get("region_observation", {}).get(
            "training_config", "KW51_training/configs/kw51_fuzzy_region_state_lstm.yaml"
        ),
        project_root=project_root,
    )
    return load_experiment_config(training_path)


def _region_fuzzy_config(
    base_config: Any,
    config: dict[str, Any],
    point_budget: int,
    seed: int,
    *,
    summary: bool,
    estimator: str = "mean",
) -> FuzzyConfig:
    if base_config.fuzzy is None:
        raise ValueError("configured region training config has no fuzzy settings")
    region = config.get("region_layout", {})
    observation = config.get("region_observation", {})
    return replace(
        base_config.fuzzy,
        region_point_budget=int(point_budget),
        region_mode=str(region.get("mode", "longitudinal")),
        region_rows=int(region.get("rows", 8)),
        region_cols=int(region.get("cols", 1)),
        region_model="region_summary" if summary else "point_bag",
        region_estimator=str(estimator),
        query_realizations=int(observation.get("query_realizations", 1)),
        normalization_records_per_state=int(
            observation.get("normalization_records_per_state", 1)
        ),
    )


def audit_dense_target_quality(
    config: dict[str, Any],
    *,
    project_root: Path,
    archive_root: Path,
) -> dict[str, Any]:
    """Audit dense target integrity and per-budget observed-region coverage."""
    index = build_fem_archive_index(archive_root)
    if not index.records:
        return {"status": "FAIL", "reason": "archive_has_no_records"}
    base_config = _load_training_config(config, project_root)
    first_by_state: dict[str, Any] = {}
    for record in index.records:
        first_by_state.setdefault(record.structural_state_id, record)
    geometry = build_deck_geometry(index.records[0], base_config.data.los_vector)
    region = config.get("region_layout", {})
    layout = build_region_layout(
        geometry,
        mode=str(region.get("mode", "longitudinal")),
        rows=int(region.get("rows", 8)),
        cols=int(region.get("cols", 1)),
    )
    sample_records = [first_by_state[state] for state in sorted(first_by_state, key=_state_sort_key)]
    target_valid_fractions: list[float] = []
    target_area_coverages: list[float] = []
    target_failures: list[str] = []
    budget_audits: dict[str, Any] = {}
    density_gate = config.get("gates", {}).get("L1", {})
    target_min = float(density_gate.get("target_valid_fraction_min", 0.99))
    region_min = float(
        density_gate.get("observation_region_present_fraction_min", 0.90)
    )
    state_region_min = float(
        density_gate.get("minimum_state_region_present_fraction", 0.75)
    )
    for budget in [int(value) for value in config.get("density", {}).get("point_budgets", [])]:
        fuzzy = _region_fuzzy_config(
            base_config,
            config,
            budget,
            int(config.get("split", {}).get("seed", 0)),
            summary=False,
        )
        source = FuzzyRegionSource(index, geometry, fuzzy, seed=int(config.get("split", {}).get("seed", 0)))
        region_presence: list[float] = []
        per_state: dict[str, Any] = {}
        for record in sample_records:
            query = source.sample_region_query(record, 0, layout)
            valid = np.asarray(query["dense_region_valid"], dtype=bool)
            target = np.asarray(query["dense_region_target"], dtype=np.float64)
            area_coverage = np.asarray(query["dense_region_area_coverage"], dtype=np.float64)
            if not np.isfinite(target[valid]).all():
                target_failures.append(f"state_{record.structural_state_id}:nonfinite_target")
            valid_fraction = float(valid.mean()) if valid.size else 0.0
            area_mean = float(area_coverage[valid].mean()) if valid.any() else 0.0
            target_valid_fractions.append(valid_fraction)
            target_area_coverages.append(area_mean)
            present_fraction = float(np.asarray(query["region_present"], dtype=bool).mean())
            region_presence.append(present_fraction)
            per_state[record.structural_state_id] = {
                "target_valid_fraction": valid_fraction,
                "area_coverage_mean": area_mean,
                "observed_region_present_fraction": present_fraction,
            }
        mean_present = float(np.mean(region_presence)) if region_presence else 0.0
        minimum_present = float(np.min(region_presence)) if region_presence else 0.0
        budget_status = "PASS" if (
            mean_present >= region_min and minimum_present >= state_region_min
        ) else "FAIL"
        budget_audits[str(budget)] = {
            "status": budget_status,
            "sampled_states": len(sample_records),
            "mean_region_present_fraction": mean_present,
            "minimum_state_region_present_fraction": minimum_present,
            "minimum_state_region_present_fraction_required": state_region_min,
            "per_state": per_state,
        }
    target_valid_mean = float(np.mean(target_valid_fractions)) if target_valid_fractions else 0.0
    target_area_mean = float(np.mean(target_area_coverages)) if target_area_coverages else 0.0
    target_status = (
        "PASS"
        if target_valid_mean >= target_min and not target_failures
        else "FAIL"
    )
    overall_status = (
        "PASS"
        if target_status == "PASS"
        and all(item["status"] == "PASS" for item in budget_audits.values())
        else "FAIL"
    )
    return {
        "status": overall_status,
        "target_status": target_status,
        "target_quality": {
            "sampled_states": len(sample_records),
            "valid_fraction_mean": target_valid_mean,
            "valid_fraction_min_required": target_min,
            "area_coverage_mean": target_area_mean,
            "nonfinite_failures": target_failures,
        },
        "density_by_budget": budget_audits,
        "audit_unit": "one_record_per_structural_state",
        "deployment_inputs_exclude_dense_targets": True,
    }


def select_stratified_preflight_records(
    records: list[Any] | tuple[Any, ...],
    *,
    trajectories_per_state: int,
    seed: int,
) -> list[Any]:
    """Select a deterministic, trajectory-distinct sample within each state."""
    if trajectories_per_state < 1:
        raise ValueError("trajectories_per_state must be positive")
    by_state: dict[str, dict[str, Any]] = {}
    for record in records:
        state_id = str(record.structural_state_id)
        trajectory_id = str(record.trajectory_id)
        by_state.setdefault(state_id, {}).setdefault(trajectory_id, record)
    selected: list[Any] = []
    for state_id in sorted(by_state, key=_state_sort_key):
        trajectories = by_state[state_id]
        if len(trajectories) < trajectories_per_state:
            raise ValueError(
                f"state {state_id} has {len(trajectories)} distinct trajectories; "
                f"{trajectories_per_state} required"
            )
        ranked = sorted(
            trajectories.items(),
            key=lambda pair: hashlib.sha256(
                f"{seed}:{state_id}:{pair[0]}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(record for _, record in ranked[:trajectories_per_state])
    return selected


def audit_l1_stratified_target_preflight(
    config: dict[str, Any],
    *,
    project_root: Path,
    archive_root: Path,
) -> dict[str, Any]:
    """Check dense targets and 80-point observation coverage on stratified records."""
    index = build_fem_archive_index(archive_root)
    pilot = config.get("pilot", {})
    preflight = pilot.get("preflight", {})
    per_state_count = int(preflight.get("trajectories_per_state", 5))
    seed = int(pilot.get("seed", config.get("split", {}).get("seed", 0)))
    try:
        records = select_stratified_preflight_records(
            index.records, trajectories_per_state=per_state_count, seed=seed
        )
    except ValueError as exc:
        return {
            "status": "FAIL",
            "reason": "stratified_trajectory_selection_failed",
            "error": str(exc),
            "sampled_record_count": 0,
        }

    base_config = _load_training_config(config, project_root)
    geometry = build_deck_geometry(index.records[0], base_config.data.los_vector)
    region = config.get("region_layout", {})
    layout = build_region_layout(
        geometry,
        mode=str(region.get("mode", "longitudinal")),
        rows=int(region.get("rows", 8)),
        cols=int(region.get("cols", 1)),
    )
    point_budget = int(pilot.get("point_budget", 80))
    fuzzy = _region_fuzzy_config(
        base_config, config, point_budget, seed, summary=False
    )
    source = FuzzyRegionSource(index, geometry, fuzzy, seed=seed)
    target_min = float(preflight.get("minimum_target_valid_fraction", 0.99))
    area_min = float(preflight.get("minimum_area_coverage", 0.99))
    region_min = float(preflight.get("minimum_region_present_fraction", 0.90))
    state_region_min = float(
        preflight.get("minimum_state_region_present_fraction", 0.75)
    )
    expected_time = int(base_config.data.window_length)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    state_presence: dict[str, list[float]] = {}

    for record in records:
        state_id = str(record.structural_state_id)
        row: dict[str, Any] = {
            "sample_id": str(record.sample_id),
            "structural_state_id": state_id,
            "trajectory_id": str(record.trajectory_id),
            "point_budget": point_budget,
            "failures": [],
        }
        try:
            query = source.sample_region_query(record, 0, layout)
            target = np.asarray(query["dense_region_target"], dtype=np.float64)
            valid = np.asarray(query["dense_region_valid"], dtype=bool)
            area_coverage = np.asarray(
                query["dense_region_area_coverage"], dtype=np.float64
            )
            region_present = np.asarray(query["region_present"], dtype=bool).reshape(-1)
            if target.ndim != 2 or target.shape[0] != layout.region_count:
                row["failures"].append("invalid_dense_target_shape")
            if valid.shape != target.shape or area_coverage.shape != target.shape:
                row["failures"].append("dense_target_mask_or_area_shape_mismatch")
            if not np.isfinite(target).all():
                row["failures"].append("nonfinite_dense_target")
            valid_fraction = float(valid.mean()) if valid.size else 0.0
            if valid_fraction < target_min:
                row["failures"].append("target_valid_fraction_below_minimum")
            empty_target_regions = [
                int(region_id)
                for region_id in range(min(valid.shape[0], layout.region_count))
                if not valid[region_id].any()
            ]
            if empty_target_regions:
                row["failures"].append("empty_dense_target_region")
            if not np.isfinite(area_coverage).all():
                row["failures"].append("nonfinite_area_coverage")
            elif np.any((area_coverage < 0.0) | (area_coverage > 1.0 + 1.0e-6)):
                row["failures"].append("area_coverage_out_of_range")
            valid_area = area_coverage[valid] if area_coverage.shape == valid.shape else np.array([])
            area_mean = float(valid_area.mean()) if valid_area.size else 0.0
            area_minimum = float(valid_area.min()) if valid_area.size else 0.0
            if not valid_area.size or area_minimum < area_min:
                row["failures"].append("valid_target_area_coverage_below_minimum")
            time_length = int(target.shape[1]) if target.ndim == 2 else 0
            if time_length != expected_time:
                row["failures"].append("unexpected_time_axis_length")
            if region_present.size != layout.region_count:
                row["failures"].append("invalid_observed_region_mask_shape")
            present_fraction = (
                float(region_present.mean()) if region_present.size else 0.0
            )
            if not region_present.any():
                row["failures"].append("all_observed_regions_empty")
            state_presence.setdefault(state_id, []).append(present_fraction)
            row.update(
                {
                    "dense_target_shape": list(target.shape),
                    "target_valid_fraction": valid_fraction,
                    "area_coverage_mean_on_valid_targets": area_mean,
                    "area_coverage_min_on_valid_targets": area_minimum,
                    "empty_dense_target_regions": empty_target_regions,
                    "observed_region_present_fraction": present_fraction,
                    "observed_region_count": int(region_present.sum()),
                    "empty_observed_regions": np.flatnonzero(~region_present).tolist(),
                    "time_axis_length": time_length,
                    "expected_time_axis_length": expected_time,
                }
            )
        except Exception as exc:
            row["failures"].append(
                f"target_generation_error:{type(exc).__name__}:{exc}"
            )
            state_presence.setdefault(state_id, []).append(0.0)
        if row["failures"]:
            failures.extend(
                {
                    "sample_id": str(record.sample_id),
                    "structural_state_id": state_id,
                    "trajectory_id": str(record.trajectory_id),
                    "reason": reason,
                }
                for reason in row["failures"]
            )
        rows.append(row)

    state_summary = {
        state_id: {
            "sampled_trajectory_count": len(values),
            "mean_observed_region_present_fraction": float(np.mean(values)),
        }
        for state_id, values in sorted(state_presence.items(), key=lambda item: _state_sort_key(item[0]))
    }
    mean_present = float(
        np.mean([row.get("observed_region_present_fraction", 0.0) for row in rows])
    ) if rows else 0.0
    minimum_state_present = min(
        (
            value["mean_observed_region_present_fraction"]
            for value in state_summary.values()
        ),
        default=0.0,
    )
    if mean_present < region_min:
        failures.append({"reason": "mean_observed_region_present_fraction_below_minimum"})
    if minimum_state_present < state_region_min:
        failures.append({"reason": "minimum_state_region_present_fraction_below_minimum"})
    return {
        "schema_version": "kw51_l1_pilot_target_preflight_v1",
        "status": "PASS" if not failures else "FAIL",
        "purpose": "process_validation_not_benchmark_performance",
        "sampled_record_count": len(rows),
        "sampled_state_count": len(state_summary),
        "trajectories_per_state": per_state_count,
        "sampling_method": "deterministic_sha256_rank_within_structural_state",
        "point_budget": point_budget,
        "seed": seed,
        "expected_time_axis_length": expected_time,
        "thresholds": {
            "minimum_target_valid_fraction": target_min,
            "minimum_area_coverage": area_min,
            "minimum_mean_region_present_fraction": region_min,
            "minimum_state_region_present_fraction": state_region_min,
        },
        "mean_observed_region_present_fraction": mean_present,
        "minimum_state_mean_observed_region_present_fraction": minimum_state_present,
        "per_state": state_summary,
        "records": rows,
        "failures": failures,
    }


def _build_fold_data(
    config: dict[str, Any],
    project_root: Path,
    index: Any,
    fold: dict[str, Any],
    *,
    model_name: str,
    point_budget: int,
    seed: int,
) -> dict[str, Any]:
    base_config = _load_training_config(config, project_root)
    is_support_tokens = model_name == "support_token_gru"
    is_summary = model_name.startswith("region_summary_") or model_name == "region_lstm" or model_name == "region_mean_mlp"
    estimator = (
        model_name.removeprefix("region_summary_")
        if model_name.startswith("region_summary_")
        else "mean"
    )
    fuzzy = _region_fuzzy_config(
        base_config,
        config,
        point_budget,
        seed,
        summary=is_summary,
        estimator=estimator,
    )
    if is_support_tokens:
        support_time_mode = str(
            config.get("support_tokens", {}).get("time_feature_mode", "standardized")
        )
        if support_time_mode != "standardized":
            raise ValueError(
                "support-token v1 requires standardized temperature_C and delta_days features"
            )
        fuzzy = replace(fuzzy, time_feature_mode=support_time_mode)
    groups = set(fold["train_state_ids"])
    train_records = [record for record in index.records if record.structural_state_id in groups]
    validation_ids = set(fold["validation_state_ids"])
    validation_records = [record for record in index.records if record.structural_state_id in validation_ids]
    test_ids = set(fold["test_state_ids"])
    test_records = [record for record in index.records if record.structural_state_id in test_ids]
    if not train_records or not validation_records or not test_records:
        raise ValueError(f"fold {fold['fold']} archive split produced an empty record set")
    geometry = build_deck_geometry(index.records[0], base_config.data.los_vector)
    layout_settings = config.get("region_layout", {})
    layout = build_region_layout(
        geometry,
        mode=str(layout_settings.get("mode", "longitudinal")),
        rows=int(layout_settings.get("rows", 8)),
        cols=int(layout_settings.get("cols", 1)),
    )
    source = FuzzyRegionSource(index, geometry, fuzzy, seed=seed)
    normalization = fit_fuzzy_normalization(
        source,
        train_records,
        fuzzy,
        time_feature_mode=fuzzy.time_feature_mode,
    )
    fitted = set(normalization.fitted_state_ids)
    if fitted != groups:
        raise ValueError(
            f"fold {fold['fold']} normalization fitted states do not equal training states"
        )
    group_map = derive_ad_groups(
        index,
        parameter=base_config.data.ad_group_parameter,
        expected_count=base_config.data.ad_group_count,
    )
    train_refs = make_fuzzy_query_refs(train_records, fuzzy.query_realizations)
    validation_refs = make_fuzzy_query_refs(validation_records, 1)
    test_refs = make_fuzzy_query_refs(test_records, 1)
    if is_support_tokens:
        support_config = config.get("support_tokens", {})
        support_layout = build_support_token_layout(
            layout,
            bandwidth_scale=float(support_config.get("bandwidth_scale", 0.75)),
        )
        value_normalization = fit_support_token_value_normalization(
            source,
            train_records,
            layout,
            support_layout,
            minimum_coverage=float(
                support_config.get("minimum_observation_coverage", 0.05)
            ),
        )
        if set(value_normalization.fitted_state_ids) != groups:
            raise ValueError(
                f"fold {fold['fold']} support normalization fitted states do not equal training states"
            )
        collate = collate_fuzzy_support_tokens
        def build_dataset(refs):
            return FuzzySupportTokenDataset(
                refs,
                source,
                normalization,
                layout,
                support_layout,
                value_normalization,
                minimum_observation_coverage=float(
                    support_config.get("minimum_observation_coverage", 0.05)
                ),
                minimum_target_area_coverage=float(
                    support_config.get("minimum_target_area_coverage", 0.99)
                ),
            )
    elif is_summary:
        collate = collate_fuzzy_region_summaries
        def build_dataset(refs):
            return FuzzyRegionSummaryDataset(
                refs,
                source,
                normalization,
                layout,
                group_map=group_map,
                estimator=estimator,
            )
    else:
        collate = collate_fuzzy_region_queries
        def build_dataset(refs):
            return FuzzyRegionQueryDataset(
                refs, source, normalization, layout, group_map=group_map
            )
    datasets = {
        "train": build_dataset(train_refs),
        "validation": build_dataset(validation_refs),
        "test": build_dataset(test_refs),
    }
    batch_size = int(config.get("training", {}).get("batch_size", 16))
    generator = torch.Generator().manual_seed(int(seed) + int(fold["fold"]))
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=0,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        ),
    }
    return {
        "base_config": base_config,
        "fuzzy": fuzzy,
        "source": source,
        "geometry": geometry,
        "layout": layout,
        "support_layout": support_layout if is_support_tokens else None,
        "normalization": normalization,
        "support_value_normalization": value_normalization if is_support_tokens else None,
        "group_map": group_map,
        "loaders": loaders,
        "fitted_state_ids": sorted(fitted, key=_state_sort_key),
    }


def _build_public_model(
    model_name: str, fold_data: dict[str, Any]
) -> torch.nn.Module | None:
    if model_name == "global_mean":
        return None
    base_model = fold_data["base_config"].model
    params = dict(base_model.kwargs)
    head_hidden = int(params.pop("head_hidden", 128))
    dropout = float(params.pop("head_dropout", 0.1))
    layout = fold_data["layout"]
    group_count = fold_data["group_map"].group_count
    if model_name == "support_token_gru":
        support_config = fold_data.get("support_layout")
        if support_config is None:
            raise ValueError("support_token_gru requires support-token fold data")
        return SupportTokenResponseModel(
            time_feature_count=2,
            support_feature_count=int(support_config.features.shape[-1]),
            hidden_size=int(params.pop("hidden_size", head_hidden)),
        )
    if model_name == "region_mean_mlp":
        return FuzzyRegionMLPModel(
            region_count=layout.region_count,
            group_count=group_count,
            hidden_dim=head_hidden,
            dropout=dropout,
        )
    if model_name == "point_lstm":
        params["input_mode"] = "node"
        return FuzzyRegionModel(
            base_model.name,
            params,
            region_count=layout.region_count,
            group_count=group_count,
            region_pooling=fold_data["fuzzy"].region_pooling,
            region_embedding_dim=fold_data["fuzzy"].region_embedding_dim,
            head_hidden=head_hidden,
            dropout=dropout,
            bag_feature_dropout=fold_data["fuzzy"].bag_feature_dropout,
        )
    params["input_mode"] = "region"
    params["region_pooling"] = fold_data["fuzzy"].region_pooling
    return FuzzyRegionSummaryModel(
        base_model.name,
        params,
        region_count=layout.region_count,
        group_count=group_count,
        head_hidden=head_hidden,
        dropout=dropout,
        bag_feature_dropout=fold_data["fuzzy"].bag_feature_dropout,
    )


def _fit_global_mean(loader: Any) -> dict[str, torch.Tensor]:
    weighted_sum: torch.Tensor | None = None
    weighted_square_sum: torch.Tensor | None = None
    total_weight: torch.Tensor | None = None
    for batch in loader:
        target = batch.targets["region_response_target"].double()
        valid = batch.targets["region_response_valid_mask"].double()
        coverage = batch.targets.get("region_area_coverage")
        weight = valid if coverage is None else valid * coverage.double().clamp(0.0, 1.0)
        area_weights = batch.targets.get("region_area_weights")
        if batch.support_features is not None and area_weights is not None:
            weight = weight * area_weights.double().unsqueeze(-1)
        values = target * weight
        if weighted_sum is None:
            weighted_sum = values.sum(dim=0)
            weighted_square_sum = (target.square() * weight).sum(dim=0)
            total_weight = weight.sum(dim=0)
        else:
            weighted_sum += values.sum(dim=0)
            weighted_square_sum += (target.square() * weight).sum(dim=0)
            total_weight += weight.sum(dim=0)
    assert weighted_sum is not None and weighted_square_sum is not None and total_weight is not None
    mean = weighted_sum / total_weight.clamp_min(1.0)
    variance = (weighted_square_sum / total_weight.clamp_min(1.0) - mean.square()).clamp_min(1.0e-6)
    global_mean = (weighted_sum.sum() / total_weight.sum().clamp_min(1.0)).float()
    global_scale = variance.mean().sqrt().float().clamp_min(1.0e-3)
    return {"mean": global_mean, "scale": global_scale}


def _l1_output(
    model: torch.nn.Module | None,
    batch: Any,
    global_baseline: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if model is not None:
        if isinstance(model, SupportTokenResponseModel):
            outputs = model(support_token_inputs_from_window_batch(batch))
            return {
                "region_response_mean": outputs["mean"].transpose(1, 2),
                "region_response_scale": outputs["scale"].transpose(1, 2),
            }
        return model(batch)
    if global_baseline is None:
        raise ValueError("global mean baseline was not fitted")
    shape = batch.targets["region_response_target"].shape
    return {
        "region_response_mean": global_baseline["mean"].expand(shape[0], *shape[1:]),
        "region_response_scale": global_baseline["scale"].expand(shape[0], *shape[1:]),
    }


def _evaluate_l1(
    model: torch.nn.Module | None,
    loader: Any,
    *,
    global_baseline: dict[str, torch.Tensor] | None = None,
    state_seed: int = 0,
) -> dict[str, Any]:
    if model is not None:
        model.eval()
    means: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            outputs = _l1_output(model, batch, global_baseline)
            means.append(outputs["region_response_mean"].detach().cpu())
            scales.append(outputs["region_response_scale"].detach().cpu())
            targets.append(batch.targets["region_response_target"].detach().cpu())
            masks.append(batch.targets["region_response_valid_mask"].detach().cpu())
            metadata.extend(batch.metadata)
    mean = torch.cat(means)
    scale = torch.cat(scales)
    target = torch.cat(targets)
    valid = torch.cat(masks).bool()
    per_state: dict[str, Any] = {}
    states = [str(item["structural_state_id"]) for item in metadata]
    for state in sorted(set(states), key=_state_sort_key):
        indices = torch.as_tensor([i for i, value in enumerate(states) if value == state])
        per_state[state] = dense_region_metrics(
            mean.index_select(0, indices),
            scale.index_select(0, indices),
            target.index_select(0, indices),
            valid.index_select(0, indices),
        )
    per_region = {
        str(region): dense_region_metrics(
            mean[:, region], scale[:, region], target[:, region], valid[:, region]
        )
        for region in range(mean.shape[1])
    }
    density_keys = [
        str(item.get("point_budget", item.get("point_count", "unknown")))
        for item in metadata
    ]
    per_density: dict[str, Any] = {}
    for density in sorted(set(density_keys)):
        indices = torch.as_tensor([i for i, value in enumerate(density_keys) if value == density])
        per_density[density] = dense_region_metrics(
            mean.index_select(0, indices),
            scale.index_select(0, indices),
            target.index_select(0, indices),
            valid.index_select(0, indices),
        )
    scalar_metrics = ("mae", "rmse", "crps", "coverage_50", "coverage_90")
    macro = {
        key: float(np.mean([float(value[key]) for value in per_state.values()]))
        if per_state
        else 0.0
        for key in scalar_metrics
    }
    ci = {
        key: bootstrap_ci(
            [float(value[key]) for value in per_state.values()],
            repeats=1000,
            seed=int(state_seed) + offset,
        )
        for offset, key in enumerate(scalar_metrics)
    }
    return {
        "overall": dense_region_metrics(mean, scale, target, valid),
        "per_state": per_state,
        "per_region": per_region,
        "per_density": per_density,
        "state_macro_mean": macro,
        "state_bootstrap_ci95": ci,
        "state_count": len(per_state),
    }


def _fit_one_l1_run(
    config: dict[str, Any],
    project_root: Path,
    index: Any,
    fold: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    run_started = time.perf_counter()
    seed = int(run["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fold_data = _build_fold_data(
        config,
        project_root,
        index,
        fold,
        model_name=str(run["model"]),
        point_budget=int(run["point_budget"]),
        seed=seed,
    )
    model = _build_public_model(str(run["model"]), fold_data)
    baseline = _fit_global_mean(fold_data["loaders"]["train"]) if model is None else None
    epochs = int(config.get("training", {}).get("epochs", 20))
    patience = int(config.get("training", {}).get("early_stopping_patience", 6))
    validation: dict[str, Any]
    selected_epoch = 0
    history: list[dict[str, float]] = []
    input_leak_check = "not_applicable_global_mean_uses_train_targets_only"
    if model is not None:
        first_batch = next(iter(fold_data["loaders"]["train"]))
        with torch.no_grad():
            _l1_output(model, first_batch, None)
        if config.get("pilot", {}).get("enabled", False):
            input_leak_check = _assert_dense_targets_not_model_inputs(
                model, first_batch
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("training", {}).get("learning_rate", 1.0e-3)),
            weight_decay=float(config.get("training", {}).get("weight_decay", 1.0e-4)),
        )
        best_score = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in fold_data["loaders"]["train"]:
                if isinstance(model, SupportTokenResponseModel):
                    predictions = model(support_token_inputs_from_window_batch(batch))
                    loss = masked_gaussian_nll(
                        predictions, support_token_targets_from_window_batch(batch)
                    )
                else:
                    outputs = _l1_output(model, batch, None)
                    loss, _ = dense_region_response_loss(
                        outputs["region_response_mean"],
                        outputs["region_response_scale"],
                        batch.targets["region_response_target"],
                        batch.targets["region_response_valid_mask"],
                        batch.targets.get("region_area_coverage"),
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(loss.detach()))
            validation = _evaluate_l1(
                model,
                fold_data["loaders"]["validation"],
                state_seed=seed + int(fold["fold"]),
            )
            score = float(validation["state_macro_mean"]["crps"])
            history.append(
                {
                    "epoch": float(epoch),
                    "train_nll": float(np.mean(train_losses)) if train_losses else 0.0,
                    "validation_state_macro_crps": score,
                }
            )
            if score < best_score:
                best_score = score
                selected_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if patience > 0 and stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
    validation = _evaluate_l1(
        model,
        fold_data["loaders"]["validation"],
        global_baseline=baseline,
        state_seed=seed + int(fold["fold"]),
    )
    test = _evaluate_l1(
        model,
        fold_data["loaders"]["test"],
        global_baseline=baseline,
        state_seed=seed + int(fold["fold"]) + 17,
    )
    expected_fitted = sorted(fold["train_state_ids"], key=_state_sort_key)
    if fold_data["fitted_state_ids"] != expected_fitted:
        raise ValueError("fold normalization contains a non-training state")
    return {
        "status": "completed",
        "selected_epoch": selected_epoch,
        "fitted_state_ids": fold_data["fitted_state_ids"],
        "validation": validation,
        "test": test,
        "selection_metric": "validation_state_macro_crps",
        "history": history,
        "l2_status": "not_run_identifiability_fail",
        "input_leak_check": input_leak_check,
        "support_token_contract": (
            {
                "schema": SUPPORT_TOKEN_SCHEMA,
                "input_axes": "values, mask, quality, coverage=[B,T,R]; support_features=[B,R,4]",
                "target_axes": "[B,T,R] at model seam; [B,R,T] in KW51 reporting adapter",
                "support_feature_order": list(SUPPORT_FEATURE_ORDER),
                "normalization_policy": SUPPORT_NORMALIZATION_POLICY,
                "normalization_fitted_state_ids": fold_data[
                    "support_value_normalization"
                ].fitted_state_ids,
                "time_feature_order": ["temperature_C", "delta_days"],
                "time_feature_mode": fold_data["fuzzy"].time_feature_mode,
                "time_normalization_fitted_state_ids": fold_data[
                    "normalization"
                ].fitted_state_ids,
                "dense_targets_in_model_inputs": False,
            }
            if isinstance(model, SupportTokenResponseModel)
            else None
        ),
        "elapsed_seconds": time.perf_counter() - run_started,
    }


def _assert_dense_targets_not_model_inputs(
    model: torch.nn.Module, batch: Any
) -> str:
    keys = (
        "region_response_target",
        "region_response_valid_mask",
        "region_area_coverage",
        "region_area_weights",
    )
    original = {key: batch.targets[key] for key in keys if key in batch.targets}
    if not original:
        raise ValueError("pilot batch is missing dense target fields")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            baseline = _l1_output(model, batch, None)
            for key, value in original.items():
                if key == "region_response_valid_mask":
                    batch.targets[key] = ~value.bool()
                else:
                    batch.targets[key] = torch.full_like(value, 123.456)
            perturbed = _l1_output(model, batch, None)
            if baseline.keys() != perturbed.keys() or any(
                not torch.equal(baseline[key], perturbed[key]) for key in baseline
            ):
                raise ValueError("model outputs changed when dense targets were perturbed")
    finally:
        batch.targets.update(original)
        model.train(was_training)
    return "PASS"


def _assert_finite_tree(value: Any, *, path: str = "metrics") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"pilot produced non-finite metric at {path}: {value}")


def _validate_pilot_run_result(
    run: dict[str, Any], fold: dict[str, Any], result: dict[str, Any], *, region_count: int
) -> None:
    test = result["test"]
    validation = result["validation"]
    _assert_finite_tree(test, path="test")
    _assert_finite_tree(validation, path="validation")
    expected_test_states = set(fold["test_state_ids"])
    actual_test_states = set(test.get("per_state", {}))
    if actual_test_states != expected_test_states:
        raise ValueError(
            f"fold {fold['fold']} has incomplete test states: "
            f"expected={sorted(expected_test_states)}, actual={sorted(actual_test_states)}"
        )
    if set(result["fitted_state_ids"]) != set(fold["train_state_ids"]):
        raise ValueError(f"fold {fold['fold']} normalization is not train-state-only")
    if set(result["fitted_state_ids"]) & (
        set(fold["validation_state_ids"]) | expected_test_states
    ):
        raise ValueError(f"fold {fold['fold']} validation/test state leaked into normalization")
    if len(test.get("per_region", {})) != region_count:
        raise ValueError(f"fold {fold['fold']} test metrics lack region strata")
    if set(test.get("per_density", {})) != {str(run["point_budget"])}:
        raise ValueError(f"fold {fold['fold']} test metrics lack the configured density")
    if result.get("selection_metric") != "validation_state_macro_crps":
        raise ValueError("pilot model selection did not use validation state-macro CRPS")
    if run["model"] in {"region_mean_mlp", "support_token_gru"} and result.get("input_leak_check") != "PASS":
        raise ValueError("pilot dense-target model-input isolation check did not pass")


def _aggregate_completed_l1_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        result = run["tasks"]["L1"]
        if result["status"] == "completed":
            grouped.setdefault((run["model"], int(run["point_budget"])), []).append(run)
    by_model_density: dict[str, Any] = {}
    metrics = ("mae", "rmse", "crps", "coverage_50", "coverage_90")
    for (model, budget), items in sorted(grouped.items()):
        state_values: dict[str, dict[str, list[float]]] = {}
        for run in items:
            for state, values in run["tasks"]["L1"]["per_state"].items():
                target = state_values.setdefault(state, {key: [] for key in metrics})
                for key in metrics:
                    target[key].append(float(values[key]))
        state_macro = {
            key: float(
                np.mean(
                    [np.mean(values[key]) for values in state_values.values()]
                )
            )
            if state_values
            else 0.0
            for key in metrics
        }
        ci = {
            key: bootstrap_ci(
                [float(np.mean(values[key])) for values in state_values.values()],
                repeats=5000,
                seed=20261015 + budget,
            )
            for key in metrics
        }
        by_model_density[f"{model}@{budget}"] = {
            "status": "completed",
            "completed_runs": len(items),
            "state_count": len(state_values),
            "state_macro_mean": state_macro,
            "state_bootstrap_ci95": ci,
        }
    return {
        "status": "completed" if grouped else "not_evaluated",
        "aggregation_unit": "structural_state_id",
        "by_model_density": by_model_density,
    }


def _gate_status(task: str, gate: dict[str, Any]) -> tuple[str, str | None]:
    status = str(gate.get("status", "not_configured")).upper()
    if task == "L2" and status == "FAIL":
        return "not_run_identifiability_fail", str(
            gate.get("not_run_reason", "not_run_identifiability_fail")
        )
    if task == "L1" and status == "FAIL":
        return "not_run_l1_quality_gate_fail", str(
            gate.get("not_run_reason", "dense_target_or_density_gate_fail")
        )
    if task == "L1" and status != "PASS":
        return "not_run_dense_target_quality_audit", str(
            gate.get("status", "dense_target_quality_audit_required")
        )
    return "not_run_execution_not_requested", "l1_gate_passed_but_execution_not_requested"


def _write_summary_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "fold",
        "seed",
        "point_budget",
        "task",
        "status",
        "not_run_reason",
        "overall_metrics_json",
        "per_state_json",
        "per_region_json",
        "per_density_json",
        "mean_ci95_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for task in ("L1", "L2"):
                result = run["tasks"][task]
                writer.writerow(
                    {
                        "model": run["model"],
                        "fold": run["fold"],
                        "seed": run["seed"],
                        "point_budget": run["point_budget"],
                        "task": task,
                        "status": result["status"],
                        "not_run_reason": result["not_run_reason"] or "",
                        "overall_metrics_json": json.dumps(result["overall"], sort_keys=True),
                        "per_state_json": json.dumps(result["per_state"], sort_keys=True),
                        "per_region_json": json.dumps(result["per_region"], sort_keys=True),
                        "per_density_json": json.dumps(result["per_density"], sort_keys=True),
                        "mean_ci95_json": json.dumps(result["mean_ci95"], sort_keys=True),
                    }
                )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pilot_manifest_payload(pilot: dict[str, Any]) -> dict[str, Any]:
    return {
        "purpose": str(pilot["purpose"]),
        "not_for_publication": bool(pilot["not_for_publication"]),
        "model": str(pilot["model"]),
        "seed": int(pilot["seed"]),
        "point_budget": int(pilot["point_budget"]),
        "folds": [int(value) for value in pilot["folds"]],
        "epochs": int(pilot["epochs"]),
        "note": str(pilot["note"]),
    }


def build_public_benchmark_report(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    archive_root: str | Path | None = None,
    run_l1_audit: bool = False,
    execute_l1: bool = False,
) -> tuple[dict[str, Any], Path]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parents[1].parent
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    contract_path = _resolve_path(config["contract"], project_root=project_root)
    state_manifest_path = _resolve_path(config["state_manifest"], project_root=project_root)
    split_path = _resolve_path(config["split_manifest"], project_root=project_root)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    validate_state_manifest(state_manifest)
    validate_public_split(split_manifest, state_manifest)
    folds = validate_state_cv_splits(state_manifest, split_manifest)
    pilot_config = config.get("pilot", {})
    pilot_enabled = bool(pilot_config.get("enabled", False))
    pilot_started = time.perf_counter()
    configured_output = config.get(
        "output_root",
        config.get("real_insar", {}).get("output_root", "runs/public_benchmark_v1"),
    )
    output_dir = (
        Path(output_root).resolve()
        if output_root
        else _resolve_path(configured_output, project_root=project_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_manifest = _pilot_manifest_payload(pilot_config) if pilot_enabled else None
    if pilot_manifest is not None:
        pilot_manifest["run_status"] = "NOT_RUN"
        _write_json(output_dir / "pilot_manifest.json", pilot_manifest)

    configured_archive = archive_root or config.get("source_archive", "")
    archive_path = _resolve_path(configured_archive, project_root=project_root)
    archive_audit = _validate_archive(
        archive_path,
        config.get("expected_archive", {}),
        state_manifest,
    )
    l1_audit: dict[str, Any] | None = None
    target_preflight: dict[str, Any] | None = None
    if run_l1_audit or execute_l1:
        if archive_audit["status"] == "PASS":
            l1_audit = audit_dense_target_quality(
                config,
                project_root=project_root,
                archive_root=archive_path,
            )
        else:
            l1_audit = {
                "status": "BLOCKED",
                "reason": "archive_validation_not_passed",
            }
        if pilot_enabled:
            if archive_audit["status"] == "PASS":
                target_preflight = audit_l1_stratified_target_preflight(
                    config,
                    project_root=project_root,
                    archive_root=archive_path,
                )
            else:
                target_preflight = {
                    "schema_version": "kw51_l1_pilot_target_preflight_v1",
                    "status": "BLOCKED",
                    "reason": "archive_validation_not_passed",
                    "sampled_record_count": 0,
                    "failures": [{"reason": "archive_validation_not_passed"}],
                }
            _write_json(output_dir / "pilot_target_preflight.json", target_preflight)
            if l1_audit is not None:
                l1_audit["stratified_preflight"] = target_preflight
                if target_preflight["status"] != "PASS":
                    l1_audit["status"] = "FAIL"
                    l1_audit["reason"] = "stratified_target_preflight_failed"
    gate_config = config.get("gates", {})
    l1_gate = dict(gate_config.get("L1", {}))
    if l1_audit is not None:
        l1_gate["status"] = l1_audit["status"]
        l1_gate["not_run_reason"] = (
            "l1_quality_gate_fail" if l1_audit["status"] != "PASS" else None
        )
    task_status: dict[str, dict[str, Any]] = {}
    for task in ("L1", "L2"):
        effective_gate = l1_gate if task == "L1" else gate_config.get(task, {})
        status, reason = _gate_status(task, effective_gate)
        if task == "L1" and archive_audit["status"] != "PASS":
            status, reason = "not_run_archive_validation_fail", "archive_validation_not_passed"
        task_status[task] = {
            "status": status,
            "configured_gate": effective_gate,
            "quality_audit": l1_audit if task == "L1" else None,
            "not_run_reason": reason,
            "metrics_status": "not_evaluated",
        }

    models = [str(value) for value in config.get("models", [])]
    seeds = [int(value) for value in config.get("seeds", [])]
    point_budgets = [int(value) for value in config.get("density", {}).get("point_budgets", [])]
    runs: list[dict[str, Any]] = []
    for fold in folds:
        for model in models:
            for seed in seeds:
                for point_budget in point_budgets:
                    runs.append(
                        {
                            "model": model,
                            "seed": seed,
                            "point_budget": point_budget,
                            "fold": fold["fold"],
                            "train_state_ids": fold["train_state_ids"],
                            "validation_state_ids": fold["validation_state_ids"],
                            "test_state_ids": fold["test_state_ids"],
                            "fitted_state_ids": fold["fitted_state_ids"],
                            "metrics": {},
                            "tasks": {
                                task: {
                                    "status": task_status[task]["status"],
                                    "not_run_reason": task_status[task]["not_run_reason"],
                                    "metrics_status": "not_evaluated",
                                    "overall": {},
                                    "per_state": {},
                                    "per_region": {},
                                    "per_density": {},
                                    "mean_ci95": {},
                                }
                                for task in ("L1", "L2")
                            },
                        }
                    )
    l1_aggregate: dict[str, Any] = {
        "status": "not_evaluated",
        "mean": {},
        "state_macro_mean": {},
        "state_bootstrap_ci95": {},
        "per_region": {},
        "per_density": {},
    }
    report: dict[str, Any] = {
        "schema_version": "kw51_public_benchmark_report_v1",
        "claim_scope": {
            "track_s": contract["claim_scope"]["track_s"],
            "track_r": contract["claim_scope"]["track_r"],
            "track_r_state_accuracy": False,
            "excluded_claims": contract["claim_scope"]["excluded_claims"],
        },
        "contract_hash": _sha256(contract_path),
        "split_hash": _sha256(split_path),
        "state_manifest": {
            "hash": _sha256(state_manifest_path),
            "state_ids": sorted(
                (str(item["structural_state_id"]) for item in state_manifest["states"]),
                key=_state_sort_key,
            ),
            "state_count": len(state_manifest["states"]),
            "trajectory_count": state_manifest["trajectory_count"],
            "trajectory_count_per_state": state_manifest["trajectory_count_per_state"],
        },
        "archive_validation": archive_audit,
        "L1_quality_audit": l1_audit,
        "L1_pilot_target_preflight": target_preflight,
        "pilot": pilot_manifest,
        "folds": folds,
        "task_status": task_status,
        "model_registry": {
            "core_baselines": models,
            "adapter_review_only": config.get("adapter_review_only", []),
            "ablations_enabled": False,
        },
        "model_runs": runs,
        "aggregate": {
            "L1": l1_aggregate,
            "L2": {
                "status": "not_evaluated",
                "mean": {},
                "state_macro_mean": {},
                "state_bootstrap_ci95": {},
                "per_region": {},
                "per_density": {},
            },
        },
        "run_status": "NOT_RUN_GATED",
        "not_run_reasons": [],
    }
    if execute_l1 and task_status["L1"]["status"] == "not_run_execution_not_requested":
        training_started = time.perf_counter()
        index = build_fem_archive_index(archive_path)
        folds_by_id = {int(item["fold"]): item for item in folds}
        for run_index, run in enumerate(runs, start=1):
            run_fold = folds_by_id[int(run["fold"])]
            result = _fit_one_l1_run(
                config,
                project_root,
                index,
                run_fold,
                run,
            )
            test_metrics = result["test"]
            if pilot_enabled:
                _validate_pilot_run_result(
                    run,
                    run_fold,
                    result,
                    region_count=int(config.get("region_layout", {}).get("rows", 8))
                    * int(config.get("region_layout", {}).get("cols", 1)),
                )
            run["selected_epoch"] = result["selected_epoch"]
            run["fitted_state_ids"] = result["fitted_state_ids"]
            run["selection_metric"] = result["selection_metric"]
            run["validation_metrics"] = result["validation"]
            run["validation_state_macro_crps"] = result["validation"][
                "state_macro_mean"
            ]["crps"]
            run["test_metric_summary"] = test_metrics["overall"]
            run["input_leak_check"] = result["input_leak_check"]
            run["support_token_contract"] = result["support_token_contract"]
            run["elapsed_seconds"] = result["elapsed_seconds"]
            run["metrics"]["L1"] = test_metrics
            run["tasks"]["L1"].update(
                {
                    "status": "completed",
                    "not_run_reason": None,
                    "metrics_status": "evaluated",
                    "overall": test_metrics["overall"],
                    "per_state": test_metrics["per_state"],
                    "per_region": test_metrics["per_region"],
                    "per_density": test_metrics["per_density"],
                    "mean_ci95": test_metrics["state_bootstrap_ci95"],
                    "state_macro_mean": test_metrics["state_macro_mean"],
                }
            )
            if pilot_enabled:
                print(
                    f"pilot_l1_run={run_index}/{len(runs)} model={run['model']} "
                    f"fold={run['fold']} selected_epoch={run['selected_epoch']} "
                    f"validation_crps={run['validation_state_macro_crps']:.6g} "
                    f"elapsed_seconds={run['elapsed_seconds']:.2f}",
                    flush=True,
                )
            elif run_index % 10 == 0:
                print(f"completed_l1_runs={run_index}/{len(runs)}", flush=True)
        task_status["L1"].update(
            {
                "status": "completed",
                "not_run_reason": None,
                "metrics_status": "evaluated",
            }
        )
        report["aggregate"]["L1"] = _aggregate_completed_l1_runs(runs)
        report["run_status"] = "COMPLETED_L1_L2_BLOCKED"
        if pilot_enabled and pilot_manifest is not None:
            trainable_runs = [run for run in runs if run["model"] != "global_mean"]
            baseline_runs = [run for run in runs if run["model"] == "global_mean"]
            neural_mean_seconds = float(
                np.mean([run["elapsed_seconds"] for run in trainable_runs])
            ) if trainable_runs else 0.0
            baseline_mean_seconds = float(
                np.mean([run["elapsed_seconds"] for run in baseline_runs])
            ) if baseline_runs else 0.0
            configured_full = yaml.safe_load(
                (project_root / "KW51_training/configs/kw51_public_benchmark_v1.yaml")
                .read_text(encoding="utf-8")
            )
            full_epochs = int(configured_full.get("training", {}).get("epochs", 20))
            full_neural_runs = (
                max(len(configured_full.get("models", [])) - 1, 0)
                * len(configured_full.get("seeds", []))
                * len(configured_full.get("density", {}).get("point_budgets", []))
                * len(folds)
            )
            full_baseline_runs = (
                len(configured_full.get("seeds", []))
                * len(configured_full.get("density", {}).get("point_budgets", []))
                * len(folds)
            )
            projected_seconds = (
                neural_mean_seconds
                * max(full_epochs, 1)
                / max(int(pilot_config.get("epochs", 1)), 1)
                * full_neural_runs
                + baseline_mean_seconds * full_baseline_runs
            )
            pilot_manifest.update(
                {
                    "run_status": report["run_status"],
                    "completed_run_count": len(runs),
                    "trainable_run_count": len(trainable_runs),
                    "total_elapsed_seconds": time.perf_counter() - pilot_started,
                    "training_sweep_elapsed_seconds": time.perf_counter() - training_started,
                    "per_run": [
                        {
                            "model": run["model"],
                            "fold": run["fold"],
                            "selected_epoch": run["selected_epoch"],
                            "validation_state_macro_crps": run[
                                "validation_state_macro_crps"
                            ],
                            "test_metrics": run["test_metric_summary"],
                            "elapsed_seconds": run["elapsed_seconds"],
                            "input_leak_check": run["input_leak_check"],
                        }
                        for run in runs
                    ],
                    "full_sweep_projection": {
                        "run_count": full_neural_runs + full_baseline_runs,
                        "neural_run_count": full_neural_runs,
                        "baseline_run_count": full_baseline_runs,
                        "estimated_cpu_seconds": projected_seconds,
                        "estimated_cpu_hours": projected_seconds / 3600.0,
                        "method": "linear scaling from pilot per-run timing and full configured epoch count",
                        "caveat": "single pilot architecture timing is only a rough proxy for the configured full model matrix",
                    },
                }
            )
            report["pilot"] = pilot_manifest
    elif l1_audit is not None and l1_audit["status"] != "PASS":
        report["run_status"] = "NOT_RUN_L1_GATE_FAIL"
    elif l1_audit is not None:
        report["run_status"] = "NOT_RUN_EXECUTION_NOT_REQUESTED"
    report["task_status"] = task_status
    report["model_runs"] = runs
    report["not_run_reasons"] = [
        {"task": task, "reason": task_status[task]["not_run_reason"]}
        for task in ("L1", "L2")
        if task_status[task]["not_run_reason"]
    ]
    report_path = output_dir / "benchmark_report.json"
    csv_path = output_dir / "benchmark_summary.csv"
    if pilot_manifest is not None:
        pilot_manifest["run_status"] = report["run_status"]
        pilot_manifest["total_elapsed_seconds"] = time.perf_counter() - pilot_started
        report["pilot"] = pilot_manifest
        _write_json(output_dir / "pilot_manifest.json", pilot_manifest)
    _write_json(report_path, report)
    _write_summary_csv(csv_path, runs)
    return report, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="KW51_training/configs/kw51_public_benchmark_v1.yaml",
    )
    parser.add_argument("--output-root")
    parser.add_argument("--data-root", help="Override source_archive for archive validation")
    parser.add_argument(
        "--audit-l1",
        action="store_true",
        help="Run the dense-target and per-budget region-coverage readiness audit",
    )
    parser.add_argument(
        "--execute-l1",
        action="store_true",
        help="Run fixed-fold L1 training only when its readiness audit passes",
    )
    args = parser.parse_args(argv)
    report, csv_path = build_public_benchmark_report(
        args.config,
        output_root=args.output_root,
        archive_root=args.data_root,
        run_l1_audit=args.audit_l1,
        execute_l1=args.execute_l1,
    )
    print(
        f"run_status={report['run_status']} "
        f"l1={report['task_status']['L1']['status']} "
        f"l2={report['task_status']['L2']['status']} "
        f"csv={csv_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
