"""Reproducible region-protocol audit for the KW51 5-group task.

This is a data-level audit: it uses masked region means, nearest-centroid and
ridge baselines, location/coverage ablations and multiple realizations/seeds.
The model-level validation is the region state head trained after this gate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from ..config import FuzzyConfig
from ..data.fem_insar import (
    AdGroupMap,
    FEMArchiveIndex,
    split_fem_records,
)
from ..data.fuzzy_range import DeckGeometry
from ..data.fuzzy_region import (
    FuzzyRegionSource,
    RegionLayout,
    build_region_layout,
    estimate_region_values,
)
from .fuzzy_metrics import bootstrap_ci


@dataclass(frozen=True)
class RegionProtocol:
    name: str
    mode: str
    rows: int
    cols: int


DEFAULT_PROTOCOLS: tuple[RegionProtocol, ...] = (
    RegionProtocol("whole", "longitudinal", 1, 1),
    RegionProtocol("2_long", "longitudinal", 2, 1),
    RegionProtocol("4_long", "longitudinal", 4, 1),
    RegionProtocol("8_long", "longitudinal", 8, 1),
    RegionProtocol("4x2", "grid", 4, 2),
)


def configured_protocol(fuzzy: FuzzyConfig) -> RegionProtocol:
    rows = int(fuzzy.region_rows)
    cols = int(fuzzy.region_cols)
    if fuzzy.region_mode == "longitudinal":
        name = f"{rows}_long"
    else:
        name = f"{rows}x{cols}"
    return RegionProtocol(name, fuzzy.region_mode, rows, cols)


def _one_record_per_state(records: Iterable[Any]) -> list[Any]:
    selected: dict[str, Any] = {}
    for record in records:
        selected.setdefault(record.structural_state_id, record)
    return [selected[state] for state in sorted(selected, key=_state_key)]


def _state_key(state: str) -> tuple[int, Any]:
    try:
        return (0, int(state))
    except (TypeError, ValueError):
        return (1, str(state))


def _aggregate_state(
    source: FuzzyRegionSource,
    layout: RegionLayout,
    record: Any,
    realizations: int,
    estimator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    region_count = layout.region_count
    sums: np.ndarray | None = None
    time_counts: np.ndarray | None = None
    present_counts = np.zeros(region_count, dtype=np.float64)
    coverage_sum = np.zeros(region_count, dtype=np.float64)
    coverage_count = np.zeros(region_count, dtype=np.float64)
    per_realization: list[np.ndarray] = []
    for realization in range(int(realizations)):
        query = source.sample_region_query(record, realization, layout)
        values = np.asarray(query["values"], dtype=np.float64)
        valid = np.asarray(query["valid_mask"], dtype=bool)
        region_index = np.asarray(query["region_index"], dtype=np.int64)
        present = np.asarray(query["region_present"], dtype=bool)
        coverage = np.asarray(query["region_coverage"], dtype=np.float64)
        coords = np.asarray(query["point_coords"], dtype=np.float64)
        estimates, estimate_valid = estimate_region_values(
            values,
            valid,
            region_index,
            present,
            coords,
            layout,
            estimator=estimator,
        )
        if sums is None:
            sums = np.zeros_like(estimates)
            time_counts = np.zeros_like(estimates)
        for index in range(region_count):
            if not bool(present[index]):
                continue
            active = estimate_valid[index]
            sums[index, active] += estimates[index, active]
            time_counts[index, active] += 1.0
            present_counts[index] += 1.0
            coverage_sum[index] += coverage[index]
            coverage_count[index] += 1.0
        per_realization.append(
            np.concatenate(
                [
                    estimates.reshape(-1),
                    present.astype(np.float64),
                    coverage,
                ]
            )
        )
    assert sums is not None and time_counts is not None
    mean_series = sums / np.maximum(time_counts, 1.0)
    present_fraction = present_counts / max(int(realizations), 1)
    mean_coverage = coverage_sum / np.maximum(coverage_count, 1.0)
    feature = np.concatenate(
        [mean_series.reshape(-1), present_fraction, mean_coverage]
    )
    return feature, present_fraction, mean_coverage, per_realization


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def _centroid_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    group_count: int,
    *,
    ridge: float | None = None,
) -> np.ndarray:
    """Return [N, G] scores (negative distance or ridge logits)."""
    train_z, test_z = _standardize(train_x, test_x)
    if ridge is not None:
        labels = np.eye(group_count)[train_y]
        gram = train_z.T @ train_z + float(ridge) * np.eye(train_z.shape[1])
        weights = np.linalg.solve(gram, train_z.T @ labels)
        return test_z @ weights
    centroids = np.stack(
        [
            train_z[train_y == group].mean(axis=0)
            if np.any(train_y == group)
            else np.zeros(train_z.shape[1], dtype=train_z.dtype)
            for group in range(group_count)
        ]
    )
    distances = ((test_z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    return -distances


def _rank_metrics(
    scores: np.ndarray, labels: np.ndarray, group_count: int
) -> dict[str, Any]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities = probabilities / np.maximum(
        probabilities.sum(axis=1, keepdims=True), 1.0e-12
    )
    predictions = probabilities.argmax(axis=1)
    order = np.argsort(-probabilities, axis=1)
    ranks = np.asarray(
        [
            int(np.flatnonzero(order[index] == labels[index])[0]) + 1
            for index in range(len(labels))
        ]
    )
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    confusion = np.zeros((group_count, group_count), dtype=np.int64)
    for true, predicted in zip(labels, predictions):
        confusion[int(true), int(predicted)] += 1
    return {
        "top1_accuracy": float((predictions == labels).mean()),
        "top3_accuracy": float((ranks <= 3).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "nll": float((-np.log(np.maximum(true_probabilities, 1.0e-12))).mean()),
        "ranks": ranks.tolist(),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
        "confusion_matrix": confusion.tolist(),
        "num_states": int(len(labels)),
        "probabilities": probabilities.tolist(),
    }


def _variance_snr(features: np.ndarray, labels: np.ndarray, group_count: int) -> float:
    overall = features.mean(axis=0)
    between: list[float] = []
    within: list[float] = []
    for group in range(group_count):
        members = labels == group
        if not np.any(members):
            continue
        group_mean = features[members].mean(axis=0)
        between.append(float(np.mean((group_mean - overall) ** 2)))
        within.append(float(np.mean((features[members] - group_mean) ** 2)))
    if not between or not within:
        return 0.0
    numerator = float(np.mean(between))
    denominator = float(np.mean(within))
    return float(np.sqrt(numerator / max(denominator, 1.0e-12)))


def _permute_regions(
    feature: np.ndarray,
    region_count: int,
    window_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    order = rng.permutation(region_count)
    values = feature[: region_count * window_length].reshape(region_count, window_length)
    rest = feature[region_count * window_length :].reshape(2, region_count)
    permuted = np.concatenate(
        [values[order].reshape(-1), rest[:, order].reshape(-1)]
    )
    return permuted


def _protocol_layout(protocol: RegionProtocol, geometry: DeckGeometry) -> RegionLayout:
    return build_region_layout(
        geometry, mode=protocol.mode, rows=protocol.rows, cols=protocol.cols
    )


def _evaluate_seed(
    source: FuzzyRegionSource,
    layout: RegionLayout,
    train_records: list[Any],
    test_records: list[Any],
    group_map: AdGroupMap,
    realizations: int,
    group_count: int,
    estimator: str,
) -> dict[str, Any]:
    train_features: list[np.ndarray] = []
    train_labels: list[int] = []
    train_realizations: list[list[np.ndarray]] = []
    for record in train_records:
        feature, _, _, per_realization = _aggregate_state(
            source, layout, record, realizations, estimator=estimator
        )
        train_features.append(feature)
        train_labels.append(int(group_map.state_to_group[record.structural_state_id]))
        train_realizations.append(per_realization)
    test_features: list[np.ndarray] = []
    test_labels: list[int] = []
    for record in test_records:
        feature, _, _, _ = _aggregate_state(
            source, layout, record, realizations, estimator=estimator
        )
        test_features.append(feature)
        test_labels.append(int(group_map.state_to_group[record.structural_state_id]))
    train_x = np.stack(train_features)
    test_x = np.stack(test_features)
    train_y = np.asarray(train_labels, dtype=np.int64)
    test_y = np.asarray(test_labels, dtype=np.int64)
    # Infer the temporal length from the feature length.
    rest = 2 * layout.region_count
    temporal_length = (train_x.shape[1] - rest) // layout.region_count
    values_slice = slice(0, layout.region_count * temporal_length)
    coverage_slice = slice(layout.region_count * temporal_length, train_x.shape[1])
    scores = _centroid_scores(
        train_x, train_y, test_x, group_count, ridge=None
    )
    values_metrics = _rank_metrics(scores, test_y, group_count)
    ridge_metrics = _rank_metrics(
        _centroid_scores(train_x, train_y, test_x, group_count, ridge=1.0),
        test_y,
        group_count,
    )
    coverage_metrics = _rank_metrics(
        _centroid_scores(
            train_x[:, coverage_slice],
            train_y,
            test_x[:, coverage_slice],
            group_count,
            ridge=None,
        ),
        test_y,
        group_count,
    )
    rng = np.random.default_rng(20261027)
    train_unordered = np.stack(
        [
            _permute_regions(row, layout.region_count, temporal_length, rng)
            for row in train_x
        ]
    )
    test_unordered = np.stack(
        [
            _permute_regions(row, layout.region_count, temporal_length, rng)
            for row in test_x
        ]
    )
    unordered_metrics = _rank_metrics(
        _centroid_scores(
            train_unordered, train_y, test_unordered, group_count, ridge=None
        ),
        test_y,
        group_count,
    )
    shuffled_rng = np.random.default_rng(20261028)
    shuffled_y = shuffled_rng.permutation(train_y)
    shuffled_metrics = _rank_metrics(
        _centroid_scores(train_x, shuffled_y, test_x, group_count, ridge=None),
        test_y,
        group_count,
    )
    snr = _variance_snr(train_x[:, values_slice], train_y, group_count)
    noise_variances: list[float] = []
    for realizations_rows in train_realizations:
        if len(realizations_rows) < 2:
            continue
        stacked = np.stack(realizations_rows)
        noise_variances.append(float(np.mean(stacked.var(axis=0))))
    bag_noise = float(np.mean(noise_variances)) if noise_variances else 0.0
    group_means = np.stack(
        [
            train_x[:, values_slice][train_y == group].mean(axis=0)
            if np.any(train_y == group)
            else np.zeros(
                train_x[:, values_slice].shape[1], dtype=train_x.dtype
            )
            for group in range(group_count)
        ]
    )
    occupied = np.asarray(
        [np.any(train_y == group) for group in range(group_count)], dtype=bool
    )
    between = (
        float(
            np.mean(
                (
                    group_means[occupied]
                    - train_x[:, values_slice].mean(axis=0)
                )
                ** 2
            )
        )
        if occupied.any()
        else 0.0
    )
    bag_snr = float(np.sqrt(between / max(bag_noise, 1.0e-12)))
    per_group = {}
    for group in range(group_count):
        mask = test_y == group
        per_group[str(group)] = {
            "states": int(mask.sum()),
            "accuracy": float(
                (np.asarray(values_metrics["predictions"])[mask] == group).mean()
            )
            if np.any(mask)
            else 0.0,
        }
    return {
        "values": values_metrics,
        "ridge": ridge_metrics,
        "coverage_only": coverage_metrics,
        "values_unordered": unordered_metrics,
        "shuffled_labels": shuffled_metrics,
        "snr_values": snr,
        "snr_bag": bag_snr,
        "bag_noise_variance": bag_noise,
        "per_group": per_group,
    }


def _run_fuzzy_region_audit_single(
    index: FEMArchiveIndex,
    geometry: DeckGeometry,
    group_map: AdGroupMap,
    fuzzy: FuzzyConfig,
    *,
    seed: int,
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
    split_mode: str,
    fold_indices: Iterable[int] | None = None,
    protocols: Iterable[RegionProtocol] | None = None,
    configured: RegionProtocol | None = None,
    min_accuracy: float | None = None,
    min_mrr: float | None = None,
    min_snr: float | None = None,
    min_margin: float | None = None,
    max_seed_std: float | None = None,
) -> dict[str, Any]:
    configured = configured or configured_protocol(fuzzy)
    protocol_list = list(protocols or DEFAULT_PROTOCOLS)
    if not any(
        item.name == configured.name
        and item.mode == configured.mode
        and item.rows == configured.rows
        and item.cols == configured.cols
        for item in protocol_list
    ):
        protocol_list.append(configured)
    if split_mode != "ad_group_stratified":
        # Random splits have no fold semantics; one pass is enough.
        folds = [None]
    elif fold_indices is None:
        folds = list(range(int(fuzzy.region_audit_folds)))
    else:
        folds = [int(fold) for fold in fold_indices]
    realizations = int(fuzzy.region_audit_realizations)
    seed_count = int(fuzzy.region_audit_seeds)
    min_accuracy = (
        float(fuzzy.region_audit_min_state_accuracy)
        if min_accuracy is None
        else float(min_accuracy)
    )
    min_mrr = (
        float(fuzzy.region_audit_min_mrr) if min_mrr is None else float(min_mrr)
    )
    min_snr = (
        float(fuzzy.region_audit_min_snr) if min_snr is None else float(min_snr)
    )
    min_margin = (
        float(fuzzy.region_audit_min_margin)
        if min_margin is None
        else float(min_margin)
    )
    max_seed_std = (
        float(fuzzy.region_audit_max_seed_std)
        if max_seed_std is None
        else float(max_seed_std)
    )
    sources = [
        FuzzyRegionSource(index, geometry, fuzzy, seed=int(seed) + 100003 * offset)
        for offset in range(seed_count)
    ]
    layouts = {
        protocol.name: _protocol_layout(protocol, geometry)
        for protocol in protocol_list
    }
    results: dict[str, Any] = {}
    for protocol in protocol_list:
        layout = layouts[protocol.name]
        fold_results: list[dict[str, Any]] = []
        all_top1: list[float] = []
        all_mrr: list[float] = []
        all_nll: list[float] = []
        seed_top1_values: list[float] = []
        coverage_top1_values: list[float] = []
        unordered_top1_values: list[float] = []
        snr_values: list[float] = []
        snr_bags: list[float] = []
        confusion = np.zeros(
            (group_map.group_count, group_map.group_count), dtype=np.int64
        )
        for fold in folds:
            records = split_fem_records(
                index,
                split_seed,
                train_fraction,
                validation_fraction,
                group_map=group_map,
                split_mode=split_mode,
                fold=fold,
            )
            train_records = _one_record_per_state(records["train"])
            test_records = _one_record_per_state(records["test"])
            seed_results = []
            for source in sources:
                seed_results.append(
                    _evaluate_seed(
                        source,
                        layout,
                        train_records,
                        test_records,
                        group_map,
                        realizations,
                        group_map.group_count,
                        estimator=fuzzy.region_estimator,
                    )
                )
            seed_metrics = [item["values"] for item in seed_results]
            top1s = [float(item["top1_accuracy"]) for item in seed_metrics]
            mrrs = [float(item["mrr"]) for item in seed_metrics]
            nlls = [float(item["nll"]) for item in seed_metrics]
            seed_mean_top1 = float(np.mean(top1s))
            seed_std_top1 = float(np.std(top1s))
            seed_top1_values.extend(top1s)
            snr_values.extend(float(item["snr_values"]) for item in seed_results)
            snr_bags.extend(float(item["snr_bag"]) for item in seed_results)
            coverage_top1_values.append(
                float(np.mean([item["coverage_only"]["top1_accuracy"] for item in seed_results]))
            )
            unordered_top1_values.append(
                float(np.mean([item["values_unordered"]["top1_accuracy"] for item in seed_results]))
            )
            for item in seed_results:
                confusion += np.asarray(item["values"]["confusion_matrix"], dtype=np.int64)
            # State-level outcomes are computed per seed and then averaged over
            # seeds. Requiring top-1 in every seed would severely understate
            # accuracy when only a handful of test states exist.
            ranks_by_seed = np.stack(
                [np.asarray(item["values"]["ranks"], dtype=np.float64) for item in seed_results]
            )
            per_state_top1 = (ranks_by_seed == 1).mean(axis=0)
            per_state_mrr = (1.0 / ranks_by_seed).mean(axis=0)
            labels = np.asarray(seed_metrics[0]["labels"], dtype=np.int64)
            all_top1.extend(per_state_top1.tolist())
            all_mrr.extend(per_state_mrr.tolist())
            all_nll.append(float(np.mean(nlls)))
            fold_results.append(
                {
                    "fold": fold,
                    "seed_top1": top1s,
                    "seed_mrr": mrrs,
                    "seed_nll": nlls,
                    "mean_top1": seed_mean_top1,
                    "std_top1": seed_std_top1,
                    "coverage_top1": coverage_top1_values[-1],
                    "unordered_top1": unordered_top1_values[-1],
                    "snr_values": snr_values[-1],
                    "snr_bag": snr_bags[-1],
                    "state_ranks_mean_over_seeds": ranks_by_seed.mean(axis=0).tolist(),
                    "state_top1_fraction_over_seeds": per_state_top1.tolist(),
                    "state_mrr_mean_over_seeds": per_state_mrr.tolist(),
                    "state_labels": labels.tolist(),
                }
            )
        mean_top1 = float(np.mean(all_top1)) if all_top1 else 0.0
        mean_mrr = float(np.mean(all_mrr)) if all_mrr else 0.0
        mean_nll = float(np.mean(all_nll)) if all_nll else 0.0
        mean_coverage = (
            float(np.mean(coverage_top1_values)) if coverage_top1_values else 0.0
        )
        mean_unordered = (
            float(np.mean(unordered_top1_values))
            if unordered_top1_values
            else 0.0
        )
        mean_snr = float(np.mean(snr_values)) if snr_values else 0.0
        mean_bag_snr = float(np.mean(snr_bags)) if snr_bags else 0.0
        seed_std = (
            float(np.mean([item["std_top1"] for item in fold_results]))
            if fold_results
            else 0.0
        )
        top1_ci = bootstrap_ci(all_top1)
        mrr_ci = bootstrap_ci(all_mrr)
        chance = 1.0 / float(group_map.group_count)
        margin = mean_top1 - max(mean_coverage, 0.0)
        reasons: list[str] = []
        if mean_top1 < float(min_accuracy):
            reasons.append(
                f"state-macro top1 {mean_top1:.3f} < {float(min_accuracy):.3f}"
            )
        if top1_ci[0] <= chance:
            reasons.append(
                f"top1 lower CI {top1_ci[0]:.3f} <= chance {chance:.3f}"
            )
        if mean_mrr < float(min_mrr):
            reasons.append(
                f"state-macro MRR {mean_mrr:.3f} < {float(min_mrr):.3f}"
            )
        if mean_snr < float(min_snr):
            reasons.append(f"feature SNR {mean_snr:.3f} < {float(min_snr):.3f}")
        if margin < float(min_margin):
            reasons.append(
                f"margin over coverage-only {margin:.3f} < {float(min_margin):.3f}"
            )
        if seed_std > float(max_seed_std):
            reasons.append(
                f"seed std {seed_std:.3f} > {float(max_seed_std):.3f}"
            )
        if mean_top1 - mean_unordered < float(min_margin):
            reasons.append(
                "region identity ablation does not reduce accuracy by the "
                f"required margin ({mean_top1 - mean_unordered:.3f} < "
                f"{float(min_margin):.3f})"
            )
        results[protocol.name] = {
            "protocol": {
                "name": protocol.name,
                "mode": protocol.mode,
                "rows": protocol.rows,
                "cols": protocol.cols,
                "region_count": layout.region_count,
            },
            "status": "PASS" if not reasons else "FAIL",
            "reasons": reasons,
            "folds": fold_results,
            "summary": {
                "state_macro_top1": mean_top1,
                "state_macro_top1_ci95": top1_ci,
                "state_macro_mrr": mean_mrr,
                "state_macro_mrr_ci95": mrr_ci,
                "state_macro_nll": mean_nll,
                "coverage_only_top1": mean_coverage,
                "values_unordered_top1": mean_unordered,
                "margin_over_coverage": margin,
                "snr_values": mean_snr,
                "snr_bag": mean_bag_snr,
                "seed_std_top1": seed_std,
                "effective_state_evaluations": int(len(all_top1)),
                "folds": len(folds),
                "seeds": seed_count,
                "realizations_per_state": realizations,
                "chance_top1": chance,
                "confusion_matrix": confusion.tolist(),
            },
        }
    configured_result = results[configured.name]
    return {
        "schema": "kw51_fuzzy_region_audit_v1",
        "status": configured_result["status"],
        "configured_protocol": configured_result["protocol"],
        "protocols": results,
        "thresholds": {
            "min_state_accuracy": float(min_accuracy),
            "min_mrr": float(min_mrr),
            "min_snr": float(min_snr),
            "min_margin": float(min_margin),
            "max_seed_std": float(max_seed_std),
        },
        "evidence_scope": "data_learnability_only",
        "notes": [
            "State-macro metrics treat each structural state as the unit; "
            "trajectories within a state are near-replicates.",
            "The audit is a data-level baseline, not a model-level CV and not "
            "a field-parameter accuracy.",
            "Coverage-only and unordered-region ablations test whether the "
            "response pattern rather than location/coverage shortcuts drives "
            "the result.",
        ],
    }


def run_fuzzy_region_audit(
    index: FEMArchiveIndex,
    geometry: DeckGeometry,
    group_map: AdGroupMap,
    fuzzy: FuzzyConfig,
    *,
    seed: int,
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
    split_mode: str,
    fold_indices: Iterable[int] | None = None,
    protocols: Iterable[RegionProtocol] | None = None,
    configured: RegionProtocol | None = None,
    min_accuracy: float | None = None,
    min_mrr: float | None = None,
    min_snr: float | None = None,
    min_margin: float | None = None,
    max_seed_std: float | None = None,
    point_budgets: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Run the region audit over one or more deployable point budgets."""
    budget_values = (
        fuzzy.region_audit_point_budgets
        if point_budgets is None
        else point_budgets
    )
    budgets = [int(value) for value in budget_values]
    if not budgets:
        budgets = [int(fuzzy.region_point_budget)]
    configured_budget = int(fuzzy.region_point_budget)
    if configured_budget not in budgets:
        budgets.append(configured_budget)
    singles: dict[int, dict[str, Any]] = {}
    for budget in budgets:
        single_fuzzy = replace(fuzzy, region_point_budget=int(budget))
        singles[budget] = _run_fuzzy_region_audit_single(
            index,
            geometry,
            group_map,
            single_fuzzy,
            seed=seed,
            split_seed=split_seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            split_mode=split_mode,
            fold_indices=fold_indices,
            protocols=protocols,
            configured=configured,
            min_accuracy=min_accuracy,
            min_mrr=min_mrr,
            min_snr=min_snr,
            min_margin=min_margin,
            max_seed_std=max_seed_std,
        )
    configured_single = singles[configured_budget]
    combined_protocols: dict[str, Any] = {}
    for name, entry in configured_single["protocols"].items():
        combined_protocols[name] = {
            "protocol": entry["protocol"],
            "budgets": {
                str(budget): singles[budget]["protocols"][name]
                for budget in budgets
            },
            "status": entry["status"],
            "summary": entry["summary"],
            "reasons": entry["reasons"],
            "folds": entry["folds"],
        }
    configured_name = configured_single["configured_protocol"]["name"]
    result = dict(configured_single)
    result["protocols"] = combined_protocols
    result["point_budgets"] = [int(budget) for budget in budgets]
    result["configured_budget"] = configured_budget
    result["status"] = combined_protocols[configured_name]["status"]
    result["notes"] = list(configured_single["notes"]) + [
        "Point-budget sweep: each budget is a separate deterministic bag protocol.",
        "The top-level status uses the configured point budget.",
    ]
    return result
