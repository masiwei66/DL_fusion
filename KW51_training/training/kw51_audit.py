"""Pre-training identifiability audit for KW51 Ad-group ranking."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ..data.fem_insar import (
    AdGroupMap,
    FEMArchiveRecord,
    FEMRegionalSource,
    FEMWindowRef,
    RegionalNormalization,
    trajectory_key,
)


def _rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def _noise_scale(normalization: RegionalNormalization) -> float:
    if normalization.scope == "per_region_offset":
        return max(float(normalization.residual_std_mm), 1.0e-12)
    return max(float(normalization.std_mm), 1.0e-12)


def _state_sort_key(state_id: str) -> tuple[int, Any]:
    try:
        return (0, int(state_id))
    except (TypeError, ValueError):
        return (1, str(state_id))


def _sample_queries(
    group_map: AdGroupMap,
    records: list[FEMArchiveRecord],
    data,
    rng: np.random.Generator,
) -> list[tuple[int, FEMArchiveRecord]]:
    by_state: dict[str, list[FEMArchiveRecord]] = defaultdict(list)
    for record in records:
        by_state[record.structural_state_id].append(record)
    queries: list[tuple[int, FEMArchiveRecord]] = []
    for group in group_map.groups:
        states = list(group.state_ids)
        if len(states) > int(data.audit_states_per_group):
            states = rng.choice(
                states, size=int(data.audit_states_per_group), replace=False
            ).tolist()
        for state_id in sorted(states, key=_state_sort_key):
            items = by_state.get(state_id, [])
            if not items:
                continue
            take = min(int(data.audit_trajectories_per_state), len(items))
            indices = rng.choice(len(items), size=take, replace=False)
            for index in indices:
                queries.append((group.group_id, items[int(index)]))
    if not queries:
        raise ValueError("Identifiability audit could not sample any FEM queries")
    return queries


def run_identifiability_audit(
    source: FEMRegionalSource,
    group_map: AdGroupMap,
    records: list[FEMArchiveRecord],
    normalization: RegionalNormalization,
    data,
    *,
    candidate_gallery: dict[int, dict[str, np.ndarray]],
    gallery_state_ids: list[str],
    gallery_trajectory_keys: list[str],
    seed: int = 0,
) -> dict[str, Any]:
    """Sanity-check the configured Ad-group gallery before training.

    Queries come only from validation states and validation environment keys.
    Candidate centers are the same frozen, training-state-only gallery used by
    validation and test. The audit is a data-level gate, not a model score.
    """
    rng = np.random.default_rng(int(seed))
    gallery_exclusion = str(getattr(data, "gallery_exclusion", "query_trajectory"))
    queries = _sample_queries(group_map, records, data, rng)
    raw_centers = candidate_gallery
    centers = {
        group_id: normalization.transform_values(
            center["values"], center["valid_mask"]
        )
        for group_id, center in raw_centers.items()
    }
    group_ids = sorted(centers)
    between_values = []
    for left_index, left in enumerate(group_ids):
        for right in group_ids[left_index + 1:]:
            common = np.asarray(raw_centers[left]["valid_mask"], dtype=bool) & np.asarray(
                raw_centers[right]["valid_mask"], dtype=bool
            )
            if common.any():
                between_values.append(_rms_distance(centers[left][common], centers[right][common]))
    between_group_rms = float(np.mean(between_values)) if between_values else 0.0
    within_values: list[float] = []
    clean_hits = 0
    noisy_hits = 0
    noisy_total = 0
    for group_id, record in queries:
        candidates = centers
        candidate_masks = {
            candidate_id: np.asarray(raw_centers[candidate_id]["valid_mask"], dtype=bool)
            for candidate_id in group_ids
        }
        clean = source.clean_view(record)
        clean_query = normalization.transform_values(
            clean["values"], clean["valid_mask"]
        )
        clean_valid = np.asarray(clean["valid_mask"], dtype=bool)
        target_valid = clean_valid & candidate_masks[group_id]
        if target_valid.any():
            within_values.append(
                _rms_distance(clean_query[target_valid], candidates[group_id][target_valid])
            )
        distances = {
            candidate_id: _rms_distance(
                clean_query[clean_valid & candidate_masks[candidate_id]],
                candidate[clean_valid & candidate_masks[candidate_id]],
            )
            if (clean_valid & candidate_masks[candidate_id]).any()
            else float("inf")
            for candidate_id, candidate in candidates.items()
        }
        clean_hits += int(min(distances, key=distances.get) == group_id)
        length = int(clean["values"].shape[0])
        for realization in range(3):
            degraded = source.query_window(
                FEMWindowRef(record, 0, length), realization
            )
            noisy_query = normalization.transform_values(
                degraded["values"], degraded["valid_mask"]
            )
            noisy_valid = np.asarray(degraded["valid_mask"], dtype=bool)
            distances = {
                candidate_id: _rms_distance(
                    noisy_query[noisy_valid & candidate_masks[candidate_id]],
                    candidate[noisy_valid & candidate_masks[candidate_id]],
                )
                if (noisy_valid & candidate_masks[candidate_id]).any()
                else float("inf")
                for candidate_id, candidate in candidates.items()
            }
            noisy_hits += int(min(distances, key=distances.get) == group_id)
            noisy_total += 1
    within_group_rms = float(np.mean(within_values)) if within_values else 0.0
    scale = _noise_scale(normalization)
    noise_norm = float(data.noise_std_mm) / scale
    bias_norm = float(data.point_bias_std_mm) / scale
    noise_rms = float(np.sqrt(noise_norm * noise_norm + bias_norm * bias_norm))
    signal_to_noise = between_group_rms / max(
        float(np.sqrt(within_group_rms * within_group_rms + noise_rms * noise_rms)),
        1.0e-12,
    )
    clean_accuracy = clean_hits / max(len(queries), 1)
    noisy_accuracy = noisy_hits / max(noisy_total, 1)
    oracle_ok = noisy_accuracy >= float(data.audit_min_oracle_group_accuracy)
    snr_ok = signal_to_noise >= float(data.audit_min_signal_to_noise)
    gate_ok = oracle_ok
    reasons: list[str] = []
    warnings: list[str] = []
    if not oracle_ok:
        reasons.append(
            f"noisy oracle {noisy_accuracy:.3f} < "
            f"{data.audit_min_oracle_group_accuracy:.3f}"
        )
    if not snr_ok:
        warnings.append(
            f"rms signal_to_noise {signal_to_noise:.3f} < "
            f"{data.audit_min_signal_to_noise:.3f}; diagnostic only for Ad-group ranking"
        )
    return {
        "schema": "kw51_identifiability_audit_v2",
        "status": "PASS" if gate_ok else "FAIL",
        "reasons": reasons,
        "warnings": warnings,
        "parameter": group_map.parameter,
        "group_count": group_map.group_count,
        "groups": [
            {
                "group_id": group.group_id,
                "value": group.value,
                "state_ids": list(group.state_ids),
                "trajectory_count": int(raw_centers[group.group_id]["trajectory_count"]),
            }
            for group in group_map.groups
        ],
        "query_count": len(queries),
        "query_state_ids": sorted({record.structural_state_id for _, record in queries}, key=_state_sort_key),
        "query_trajectory_keys": sorted({trajectory_key(record) for _, record in queries}),
        "gallery_state_ids": sorted(gallery_state_ids, key=_state_sort_key),
        "gallery_trajectory_keys": sorted(gallery_trajectory_keys),
        "noisy_trial_count": noisy_total,
        "between_group_rms": between_group_rms,
        "within_group_rms": within_group_rms,
        "noise_rms": noise_rms,
        "noise_scale_mm": scale,
        "signal_to_noise": signal_to_noise,
        "oracle_group_accuracy_clean": clean_accuracy,
        "oracle_group_accuracy_noisy": noisy_accuracy,
        "thresholds": {
            "min_oracle_group_accuracy": float(data.audit_min_oracle_group_accuracy),
            "min_signal_to_noise": float(data.audit_min_signal_to_noise),
            "signal_to_noise_is_gate": False,
            "noisy_oracle_is_gate": True,
        },
        "normalization_scope": normalization.scope,
        "time_feature_mode": normalization.time_feature_mode,
        "gallery_exclusion": gallery_exclusion,
        "evaluation_setting": (
            "open_set_query_state_held_out"
            if gallery_exclusion == "query_state"
            else "closed_set_frozen_gallery"
        ),
        "note": (
            "Validation-only data gate using a frozen gallery built from training "
            "states and disjoint environment keys. The noisy nearest-centroid oracle "
            "is the task-aligned gate; RMS signal-to-noise is reported as a diagnostic "
            "because it does not measure candidate ranking accuracy. This is not a "
            "model performance estimate."
        ),
    }


def summarize_ad_group_cross_validation_audit(
    fold_audits: list[dict[str, Any]],
    *,
    expected_state_ids: list[str],
    required_fold_count: int,
    minimum_oracle_accuracy: float,
) -> dict[str, Any]:
    """Require noisy oracle success and one validation role for every state."""
    expected = {str(state_id) for state_id in expected_state_ids}
    state_counts: Counter[str] = Counter(
        str(state_id)
        for audit in fold_audits
        for state_id in audit.get("query_state_ids", [])
    )
    observed = set(state_counts)
    missing_states = sorted(expected - observed)
    unexpected_states = sorted(observed - expected)
    repeated_states = sorted(state for state, count in state_counts.items() if count != 1)
    reasons: list[str] = []
    if len(fold_audits) != int(required_fold_count):
        reasons.append(
            f"cross-validation produced {len(fold_audits)} folds; "
            f"expected {required_fold_count}"
        )
    if missing_states or unexpected_states or repeated_states:
        reasons.append(
            "cross-validation validation-state coverage is not an exact partition"
        )

    fold_rows = []
    total_trials = 0
    correct_trials = 0.0
    failed_folds: list[int] = []
    for position, audit in enumerate(fold_audits):
        fold = int(audit.get("fold", position))
        accuracy = float(audit.get("oracle_group_accuracy_noisy", 0.0))
        trials = int(audit.get("noisy_trial_count", 0))
        passed = str(audit.get("status")) == "PASS" and accuracy >= float(
            minimum_oracle_accuracy
        )
        if not passed:
            failed_folds.append(fold)
            reasons.append(
                f"fold {fold} noisy oracle {accuracy:.3f} < "
                f"{minimum_oracle_accuracy:.3f} or audit failed"
            )
        total_trials += trials
        correct_trials += accuracy * trials
        fold_rows.append(
            {
                "fold": fold,
                "status": str(audit.get("status", "MISSING")),
                "query_count": int(audit.get("query_count", 0)),
                "noisy_trial_count": trials,
                "oracle_group_accuracy_noisy": accuracy,
                "oracle_group_accuracy_clean": float(
                    audit.get("oracle_group_accuracy_clean", 0.0)
                ),
                "signal_to_noise": float(audit.get("signal_to_noise", 0.0)),
                "query_state_ids": sorted(
                    str(value) for value in audit.get("query_state_ids", [])
                ),
                "gallery_state_ids": sorted(
                    str(value) for value in audit.get("gallery_state_ids", [])
                ),
                "gallery_trajectory_keys": sorted(
                    str(value) for value in audit.get("gallery_trajectory_keys", [])
                ),
            }
        )

    return {
        "schema": "kw51_ad_group_identifiability_cv_v1",
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "fold_count": len(fold_audits),
        "required_fold_count": int(required_fold_count),
        "validation_state_count": len(observed),
        "expected_state_count": len(expected),
        "validation_state_ids": sorted(observed),
        "missing_state_ids": missing_states,
        "unexpected_state_ids": unexpected_states,
        "repeated_state_ids": repeated_states,
        "state_coverage_complete": (
            not missing_states and not unexpected_states and not repeated_states
        ),
        "query_count": sum(int(audit.get("query_count", 0)) for audit in fold_audits),
        "noisy_trial_count": total_trials,
        "oracle_group_accuracy_noisy": (
            correct_trials / total_trials if total_trials else 0.0
        ),
        "failed_folds": failed_folds,
        "thresholds": {
            "minimum_oracle_group_accuracy_per_fold": float(minimum_oracle_accuracy),
            "all_folds_must_pass": True,
            "signal_to_noise_is_gate": False,
        },
        "folds": fold_rows,
        "note": (
            "Each structural state is held out for validation once. Every fold uses "
            "a candidate gallery built only from its training states and disjoint "
            "environment keys. The nearest-centroid oracle is a data-level gate, "
            "not a learned-model performance estimate."
        ),
    }


def _sample_exact_state_queries(
    records: list[FEMArchiveRecord],
    data,
    rng: np.random.Generator,
) -> list[FEMArchiveRecord]:
    """Sample a bounded, state-covering query set for exact-state audits."""
    by_state: dict[str, list[FEMArchiveRecord]] = defaultdict(list)
    for record in records:
        by_state[record.structural_state_id].append(record)
    state_ids = sorted(by_state, key=lambda value: _state_sort_key(value))
    # The default archive has 25 states (five Ad groups x five Id values). Keep
    # the audit bounded for larger archives while retaining deterministic state
    # coverage. ``audit_states_per_group`` is the existing state-budget knob.
    max_states = max(2, int(data.audit_states_per_group) * 5)
    if len(state_ids) > max_states:
        state_ids = sorted(
            rng.choice(state_ids, size=max_states, replace=False).tolist(),
            key=lambda value: _state_sort_key(value),
        )
    queries: list[FEMArchiveRecord] = []
    for state_id in state_ids:
        items = by_state[state_id]
        take = min(int(data.audit_trajectories_per_state), len(items))
        if take <= 0:
            continue
        indices = rng.choice(len(items), size=take, replace=False)
        queries.extend(items[int(index)] for index in indices)
    if not queries:
        raise ValueError("Exact-state identifiability audit could not sample FEM queries")
    return queries


def run_exact_state_identifiability_audit(
    source: FEMRegionalSource,
    records: list[FEMArchiveRecord],
    normalization: RegionalNormalization,
    data,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Audit whether the configured observable can separate exact FEM states.

    This is deliberately a data-level nearest-centroid check. It runs the same
    trajectory exclusion and query degradation as candidate ranking, so an
    exact-state run is marked diagnostic-only before expensive model training
    when the labels are indistinguishable under the configured observation.
    """
    rng = np.random.default_rng(int(seed))
    queries = _sample_exact_state_queries(records, data, rng)
    state_ids = sorted(
        {record.structural_state_id for record in records}, key=_state_sort_key
    )
    clean_hits = 0
    noisy_hits = 0
    noisy_total = 0
    clean_distances: list[float] = []
    noisy_distances: list[float] = []
    pairwise_state_distances: list[float] = []

    def masked_rms(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
        valid = np.asarray(valid, dtype=bool)
        if not valid.any():
            return float("inf")
        return _rms_distance(left[valid], right[valid])

    # Use the full clean gallery to measure the separability of state centers.
    centers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for state_id in state_ids:
        aggregate = source.aggregate_state(state_id)
        centers[state_id] = (
            normalization.transform_values(aggregate["values"], aggregate["valid_mask"]),
            np.asarray(aggregate["valid_mask"], dtype=bool),
        )
    for left_index, left in enumerate(state_ids):
        for right in state_ids[left_index + 1:]:
            left_values, left_valid = centers[left]
            right_values, right_valid = centers[right]
            pairwise_state_distances.append(
                masked_rms(left_values, right_values, left_valid & right_valid)
            )

    for record in queries:
        exclusion_key = trajectory_key(record)
        candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for state_id in state_ids:
            aggregate = source.aggregate_state(
                state_id,
                exclude_sample_id=record.sample_id
                if state_id == record.structural_state_id
                else None,
                exclude_trajectory_key=exclusion_key,
            )
            candidates[state_id] = (
                normalization.transform_values(
                    aggregate["values"], aggregate["valid_mask"]
                ),
                np.asarray(aggregate["valid_mask"], dtype=bool),
            )
        clean = source.clean_view(record)
        clean_values = normalization.transform_values(clean["values"], clean["valid_mask"])
        clean_valid = np.asarray(clean["valid_mask"], dtype=bool)
        clean_scores = {
            state_id: masked_rms(clean_values, values, clean_valid & valid)
            for state_id, (values, valid) in candidates.items()
        }
        target = record.structural_state_id
        clean_hits += int(min(clean_scores, key=clean_scores.get) == target)
        clean_distances.append(clean_scores[target])
        for realization in range(3):
            degraded = source.query_window(
                FEMWindowRef(record, 0, int(clean["values"].shape[0])), realization
            )
            noisy_values = normalization.transform_values(
                degraded["values"], degraded["valid_mask"]
            )
            noisy_valid = np.asarray(degraded["valid_mask"], dtype=bool)
            noisy_scores = {
                state_id: masked_rms(noisy_values, values, noisy_valid & valid)
                for state_id, (values, valid) in candidates.items()
            }
            noisy_hits += int(min(noisy_scores, key=noisy_scores.get) == target)
            noisy_total += 1
            noisy_distances.append(noisy_scores[target])

    between_state_rms = float(np.mean(pairwise_state_distances)) if pairwise_state_distances else 0.0
    within_state_rms = float(np.mean(clean_distances)) if clean_distances else 0.0
    scale = _noise_scale(normalization)
    noise_norm = float(data.noise_std_mm) / scale
    bias_norm = float(data.point_bias_std_mm) / scale
    noise_rms = float(np.sqrt(noise_norm * noise_norm + bias_norm * bias_norm))
    signal_to_noise = between_state_rms / max(
        float(np.sqrt(within_state_rms * within_state_rms + noise_rms * noise_rms)),
        1.0e-12,
    )
    clean_accuracy = clean_hits / max(len(queries), 1)
    noisy_accuracy = noisy_hits / max(noisy_total, 1)
    oracle_ok = noisy_accuracy >= float(data.audit_min_oracle_group_accuracy)
    snr_ok = signal_to_noise >= float(data.audit_min_signal_to_noise)
    gate_ok = oracle_ok and snr_ok
    reasons: list[str] = []
    warnings: list[str] = []
    if not oracle_ok:
        reasons.append(
            f"noisy exact-state oracle {noisy_accuracy:.3f} < "
            f"{data.audit_min_oracle_group_accuracy:.3f}"
        )
    if not snr_ok:
        reasons.append(
            f"exact-state rms signal_to_noise {signal_to_noise:.3f} < "
            f"{data.audit_min_signal_to_noise:.3f}"
        )
    return {
        "schema": "kw51_exact_state_identifiability_audit_v1",
        "status": "PASS" if gate_ok else "FAIL",
        "reasons": reasons,
        "warnings": warnings,
        "state_count": len(state_ids),
        "state_ids": state_ids,
        "query_count": len(queries),
        "noisy_trial_count": noisy_total,
        "between_state_rms": between_state_rms,
        "within_state_rms": within_state_rms,
        "noise_rms": noise_rms,
        "noise_scale_mm": scale,
        "signal_to_noise": signal_to_noise,
        "oracle_state_accuracy_clean": clean_accuracy,
        "oracle_state_accuracy_noisy": noisy_accuracy,
        "target_state_distance_clean_mean": float(np.mean(clean_distances))
        if clean_distances else 0.0,
        "target_state_distance_noisy_mean": float(np.mean(noisy_distances))
        if noisy_distances else 0.0,
        "thresholds": {
            "min_oracle_accuracy": float(data.audit_min_oracle_group_accuracy),
            "min_signal_to_noise": float(data.audit_min_signal_to_noise),
            "both_gates_required": True,
        },
        "normalization_scope": normalization.scope,
        "time_feature_mode": normalization.time_feature_mode,
        "gallery_exclusion": str(getattr(data, "gallery_exclusion", "query_trajectory")),
        "note": (
            "Data-level nearest-centroid check for the exact structural-state target. "
            "A failed result means the configured sparse observation and degradation "
            "cannot support exact-state model claims; it is not a model score."
        ),
    }
