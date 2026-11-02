"""Data-level five-group identifiability audit for the fuzzy state task."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..data.fem_insar import AdGroupMap
from ..data.fuzzy_range import FuzzyQueryRef, FuzzyRangeSource


def _state_sort_key(state_id: str) -> tuple[int, Any]:
    try:
        return (0, int(state_id))
    except (TypeError, ValueError):
        return (1, str(state_id))


def _sample_refs(
    refs: list[FuzzyQueryRef], max_per_state: int
) -> list[FuzzyQueryRef]:
    grouped: dict[str, list[FuzzyQueryRef]] = defaultdict(list)
    for ref in refs:
        grouped[ref.record.structural_state_id].append(ref)
    selected: list[FuzzyQueryRef] = []
    for state in sorted(grouped, key=_state_sort_key):
        selected.extend(grouped[state][: max(int(max_per_state), 1)])
    return selected


def _bag_mean_feature(
    source: FuzzyRangeSource, ref: FuzzyQueryRef
) -> tuple[np.ndarray, float]:
    result = source.sample_query(
        ref.record, ref.realization, with_targets=False
    )
    values = np.asarray(result["values"], dtype=np.float64)
    valid = np.asarray(result["valid_mask"], dtype=bool)
    counts = np.maximum(valid.sum(axis=1), 1)
    feature = np.where(valid, values, 0.0).sum(axis=1) / counts
    return feature, float(valid.sum(axis=1).mean())


def _rank_metrics(
    true_group: int, group_ids: list[int], distances: np.ndarray
) -> tuple[int, int]:
    order = np.argsort(distances)
    rank = int(np.flatnonzero(np.asarray(group_ids)[order] == true_group)[0]) + 1
    return rank, int(rank <= 3)


def run_fuzzy_group_audit(
    source: FuzzyRangeSource,
    group_map: AdGroupMap,
    train_refs: list[FuzzyQueryRef],
    test_refs: list[FuzzyQueryRef],
    *,
    max_refs_per_state: int = 3,
    noise_std_mm: float,
    point_bias_std_mm: float,
    min_accuracy: float,
    min_snr: float = 1.0,
) -> dict[str, Any]:
    """Nearest-centroid group audit on the bag-mean response of the current bags.

    This is a data-level baseline under the configured bag rules. The model-level
    validation is the five-group state head and its state-level cross-validation.
    """
    train = _sample_refs(train_refs, max_refs_per_state)
    test = _sample_refs(test_refs, max_refs_per_state)
    if not train or not test:
        raise ValueError("Fuzzy group audit requires non-empty train and test refs")
    train_features: list[np.ndarray] = []
    train_groups: list[int] = []
    valid_counts: list[float] = []
    for ref in train:
        feature, count = _bag_mean_feature(source, ref)
        train_features.append(feature)
        train_groups.append(int(group_map.state_to_group[ref.record.structural_state_id]))
        valid_counts.append(count)
    train_array = np.stack(train_features)
    train_groups_array = np.asarray(train_groups, dtype=np.int64)
    group_ids = list(range(group_map.group_count))
    centroids: dict[int, np.ndarray] = {}
    for group in group_ids:
        members = train_array[train_groups_array == group]
        if members.size:
            centroids[group] = members.mean(axis=0)
    if not centroids:
        raise ValueError("No training groups available for the fuzzy group audit")
    within: list[float] = []
    for feature, group in zip(train_array, train_groups_array):
        if group in centroids:
            within.append(
                float(np.sqrt(np.mean((feature - centroids[group]) ** 2)))
            )
    between: list[float] = []
    available = sorted(centroids)
    for left_index, left in enumerate(available):
        for right in available[left_index + 1:]:
            between.append(
                float(np.sqrt(np.mean((centroids[left] - centroids[right]) ** 2)))
            )
    ranks: list[int] = []
    top3_hits = 0
    predictions: list[int] = []
    true_groups: list[int] = []
    test_counts: list[float] = []
    for ref in test:
        feature, count = _bag_mean_feature(source, ref)
        true_group = int(group_map.state_to_group[ref.record.structural_state_id])
        distances = np.asarray(
            [
                np.sqrt(np.mean((feature - centroids[group]) ** 2))
                for group in available
            ]
        )
        prediction = int(available[int(np.argmin(distances))])
        rank, top3_hit = _rank_metrics(true_group, available, distances)
        ranks.append(rank)
        top3_hits += int(top3_hit)
        predictions.append(prediction)
        true_groups.append(true_group)
        test_counts.append(count)
    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(true_groups)))
    top3 = top3_hits / max(len(test), 1)
    mrr = float(np.mean(1.0 / np.asarray(ranks, dtype=np.float64)))
    confusion = np.zeros((group_map.group_count, group_map.group_count), dtype=np.int64)
    for true_group, prediction in zip(true_groups, predictions):
        confusion[true_group, prediction] += 1
    noise_rms = float(
        np.sqrt(noise_std_mm ** 2 + point_bias_std_mm ** 2)
        / np.sqrt(max(float(np.mean(valid_counts + test_counts)), 1.0))
    )
    within_rms = float(np.mean(within)) if within else 0.0
    between_rms = float(np.mean(between)) if between else 0.0
    snr = between_rms / max(
        float(np.sqrt(within_rms ** 2 + noise_rms ** 2)), 1.0e-9
    )
    reasons: list[str] = []
    if accuracy < float(min_accuracy):
        reasons.append(
            f"nearest-centroid group accuracy {accuracy:.3f} < {min_accuracy:.3f}"
        )
    if snr < float(min_snr):
        reasons.append(f"between/within+noise SNR {snr:.3f} < {float(min_snr):.3f}")
    return {
        "schema": "kw51_fuzzy_group_audit_v1",
        "status": "PASS" if not reasons else "FAIL",
        "evidence_scope": "data_learnability_only",
        "reasons": reasons,
        "train_refs": len(train),
        "test_refs": len(test),
        "group_count": group_map.group_count,
        "chance_accuracy": 1.0 / float(group_map.group_count),
        "top1_accuracy": accuracy,
        "top3_accuracy": float(top3),
        "mrr": mrr,
        "confusion_matrix": confusion.tolist(),
        "groups_train": {
            str(group): int(np.sum(train_groups_array == group))
            for group in group_ids
        },
        "groups_test": {
            str(group): int(np.sum(np.asarray(true_groups) == group))
            for group in group_ids
        },
        "between_group_rms_mm": between_rms,
        "within_group_rms_mm": within_rms,
        "noise_rms_mm": noise_rms,
        "snr": snr,
        "thresholds": {
            "min_accuracy": float(min_accuracy),
            "min_snr": float(min_snr),
        },
        "note": (
            "Data-level nearest-centroid baseline on the noisy bag-mean response "
            "under the current fuzzy bag rules. The model-level validation is the "
            "five-group state head and its state-level CV. It is not a measured "
            "field parameter accuracy."
        ),
    }
