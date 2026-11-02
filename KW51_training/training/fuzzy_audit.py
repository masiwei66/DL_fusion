"""Learnability audit for fuzzy-range KW51 response prediction."""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..data.fem_insar import FEMArchiveRecord
from ..data.fuzzy_range import FuzzyRangeSource


def run_fuzzy_audit(
    source: FuzzyRangeSource,
    records: list[FEMArchiveRecord],
    target_mean_mm: float,
    *,
    audit_records: int,
    noise_std_mm: float,
    point_bias_std_mm: float,
    min_snr: float,
) -> dict[str, Any]:
    by_state: dict[str, list[FEMArchiveRecord]] = {}
    for record in records:
        by_state.setdefault(record.structural_state_id, []).append(record)
    states = sorted(
        by_state,
        key=lambda state: (0, int(state)) if str(state).isdigit() else (1, str(state)),
    )
    selected: list[FEMArchiveRecord] = []
    state_positions = {state: 0 for state in states}
    while len(selected) < max(int(audit_records), 1):
        added = False
        for state in states:
            if len(selected) >= max(int(audit_records), 1):
                break
            items = by_state[state]
            position = state_positions[state]
            if position < len(items):
                selected.append(items[position])
                state_positions[state] = position + 1
                added = True
        if not added:
            break
    if not selected:
        raise ValueError("Fuzzy audit requires at least one training record")
    bag_maes: list[float] = []
    global_maes: list[float] = []
    target_stds: list[float] = []
    skewness_values: list[float] = []
    kurtosis_values: list[float] = []
    k_values = (16, 32, 64)
    k_mean_diffs: dict[int, list[float]] = {k: [] for k in k_values}
    k_std_diffs: dict[int, list[float]] = {k: [] for k in k_values}
    for record in selected:
        result = source.sample_query(record, realization=0)
        values = np.asarray(result["values"], dtype=np.float64)
        valid = np.asarray(result["valid_mask"], dtype=bool)
        target = np.asarray(result["target_mean"], dtype=np.float64)
        target_std = np.asarray(result["target_std"], dtype=np.float64)
        denominator = np.maximum(valid.sum(axis=1), 1)
        bag_mean = np.where(valid, values, 0.0).sum(axis=1) / denominator
        bag_maes.append(float(np.mean(np.abs(bag_mean - target))))
        global_maes.append(float(np.mean(np.abs(target_mean_mm - target))))
        target_stds.append(float(np.mean(target_std)))
        samples = np.asarray(result["target_samples"], dtype=np.float64)
        sample_valid = np.asarray(result["target_sample_valid"], dtype=bool)
        for time_index in range(samples.shape[1]):
            samples_at_time = samples[sample_valid[:, time_index], time_index]
            if samples_at_time.size <= 2:
                continue
            mean = float(samples_at_time.mean())
            std = float(samples_at_time.std())
            if std <= 1.0e-9:
                continue
            centered = samples_at_time - mean
            skewness_values.append(float(np.mean(centered**3) / std**3))
            kurtosis_values.append(float(np.mean(centered**4) / std**4 - 3.0))
        big_k = max(64, int(source.config.target_range_count))
        big_config = dataclasses.replace(source.config, target_range_count=big_k)
        big_source = type(source).__new__(type(source))
        big_source.geometry = source.geometry
        big_source.config = big_config
        big_source.seed = source.seed
        big_source._cache = source._cache
        big_source.cache_limit = source.cache_limit
        big_result = big_source.sample_query(record, realization=0)
        big_samples = np.asarray(big_result["target_samples"], dtype=np.float64)
        big_valid = np.asarray(big_result["target_sample_valid"], dtype=bool)

        def prefix_stats(count: int):
            subset = big_samples[:count]
            subset_valid = big_valid[:count]
            counts = subset_valid.sum(axis=0)
            subset_mean = np.where(
                counts > 0,
                (subset * subset_valid).sum(axis=0) / np.maximum(counts, 1),
                0.0,
            )
            subset_variance = np.where(
                counts > 0,
                (((subset - subset_mean[None, :]) ** 2) * subset_valid).sum(axis=0)
                / np.maximum(counts, 1),
                0.0,
            )
            return subset_mean, np.sqrt(np.maximum(subset_variance, 0.0)), counts

        reference_mean, reference_std, reference_counts = prefix_stats(
            big_samples.shape[0]
        )
        for count in k_values:
            if count > big_samples.shape[0]:
                continue
            subset_mean, subset_std, subset_counts = prefix_stats(count)
            valid = (subset_counts > 0) & (reference_counts > 0)
            if valid.any():
                k_mean_diffs[count].append(
                    float(np.mean(np.abs(subset_mean - reference_mean)[valid]))
                )
                k_std_diffs[count].append(
                    float(np.mean(np.abs(subset_std - reference_std)[valid]))
                )
    bag_mean_mae = float(np.mean(bag_maes))
    global_mean_mae = float(np.mean(global_maes))
    target_std = float(np.mean(target_stds))
    noise = float(np.sqrt(noise_std_mm ** 2 + point_bias_std_mm ** 2))
    snr = target_std / max(noise, 1.0e-9)
    informative = bag_mean_mae < global_mean_mae
    snr_ok = snr >= float(min_snr)
    reasons: list[str] = []
    if not informative:
        reasons.append(
            f"bag-mean MAE {bag_mean_mae:.4f} is not below global-mean MAE "
            f"{global_mean_mae:.4f}"
        )
    if not snr_ok:
        reasons.append(f"target std / noise SNR {snr:.3f} < {min_snr:.3f}")
    skewness = float(np.mean(skewness_values)) if skewness_values else 0.0
    kurtosis = float(np.mean(kurtosis_values)) if kurtosis_values else 0.0
    return {
        "schema": "kw51_fuzzy_audit_v2",
        "status": "PASS" if informative and snr_ok else "FAIL",
        "evidence_scope": "data_learnability_only",
        "reasons": reasons,
        "records": len(selected),
        "states_covered": len({record.structural_state_id for record in selected}),
        "bag_mean_mae_mm": bag_mean_mae,
        "global_mean_mae_mm": global_mean_mae,
        "target_std_mm": target_std,
        "noise_std_mm": noise,
        "snr": snr,
        "informative": bool(informative),
        "snr_ok": bool(snr_ok),
        "target_skewness_mean": skewness,
        "target_excess_kurtosis_mean": kurtosis,
        "target_skewed_fraction": float(
            np.mean([abs(value) > 1.0 for value in skewness_values])
        ) if skewness_values else 0.0,
        "k_sensitivity": {
            "reference_k": max(64, int(source.config.target_range_count)),
            "prefix_comparison": {
                str(count): {
                    "mean_abs_mean_diff_mm": (
                        float(np.mean(k_mean_diffs[count]))
                        if k_mean_diffs[count]
                        else None
                    ),
                    "mean_abs_std_diff_mm": (
                        float(np.mean(k_std_diffs[count]))
                        if k_std_diffs[count]
                        else None
                    ),
                }
                for count in k_values
            },
            "note": (
                "Prefix comparison uses the same seeded range sequence; k=64 "
                "is the reference. It measures Monte Carlo sensitivity of the "
                "target mean/std to the number of latent range hypotheses."
            ),
        },
        "note": (
            "Stratified by structural_state_id across training states. Bag-mean "
            "vs global-mean MAE checks whether the fuzzy bag carries "
            "range-response information; SNR compares the latent-range target "
            "spread with the configured point noise. This is a data-level "
            "learnability check, not model performance evidence."
        ),
    }
