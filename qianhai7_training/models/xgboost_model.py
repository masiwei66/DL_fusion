from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import WindowBatch
from ..task_labels import TASK_LABEL_NAMES


def window_features(batch: WindowBatch) -> np.ndarray:
    """Convert point/time observations into a fixed-width tabular vector."""
    values = batch.values.detach().cpu().numpy()
    valid = (batch.valid_mask & batch.point_present_mask[:, None, :]).detach().cpu().numpy()
    quality = batch.quality.detach().cpu().numpy()
    coords = batch.point_features[..., :2].detach().cpu().numpy()
    present = batch.point_present_mask.detach().cpu().numpy()
    map_valid = (batch.mapping_valid_mask & batch.point_present_mask).detach().cpu().numpy()
    map_distance = batch.mapping_distance.detach().cpu().numpy()
    time = batch.time_features.detach().cpu().numpy()
    time_valid = batch.temperature_valid_mask.squeeze(-1).detach().cpu().numpy()

    mask = valid.astype(np.float32)
    count = mask.sum(axis=2)
    denom = np.maximum(count, 1.0)
    mean = (values * mask).sum(axis=2) / denom
    centered = (values - mean[:, :, None]) * mask
    std = np.sqrt((centered ** 2).sum(axis=2) / denom)
    minimum = np.where(valid, values, np.inf).min(axis=2)
    maximum = np.where(valid, values, -np.inf).max(axis=2)
    minimum = np.where(np.isfinite(minimum), minimum, 0.0)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    valid_ratio = count / np.maximum(present.sum(axis=1, keepdims=True), 1.0)

    point_weight = present.astype(np.float32)
    point_denom = np.maximum(point_weight.sum(axis=1, keepdims=True), 1.0)
    coord_mean = (coords * point_weight[:, :, None]).sum(axis=1) / point_denom
    coord_std = np.sqrt(
        (((coords - coord_mean[:, None, :]) ** 2) * point_weight[:, :, None]).sum(axis=1)
        / point_denom
    )
    mapping_count = np.maximum(map_valid.sum(axis=1), 1)
    mapping_distance_mean = (map_distance * map_valid).sum(axis=1) / mapping_count
    mapping_valid_ratio = map_valid.sum(axis=1) / np.maximum(present.sum(axis=1), 1)
    quality_mean = (quality * mask).sum(axis=2) / denom

    time_mask = time_valid.astype(np.float32)
    time_values = time * time_mask[:, :, None]
    features = np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            valid_ratio,
            quality_mean,
            time_values.reshape(time_values.shape[0], -1),
            time_mask,
            coord_mean,
            coord_std,
            mapping_distance_mean[:, None],
            mapping_valid_ratio[:, None],
        ],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class XGBoostModel:
    """Four independent multiclass XGBoost estimators for tabular window features."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int | None = 10,
        **extra_params: Any,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "Training model 'xgboost' requires xgboost. Install it with `pip install xgboost`."
            ) from exc

        self.params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "min_child_weight": float(min_child_weight),
            "reg_lambda": float(reg_lambda),
            "random_state": int(random_state),
            "n_jobs": int(n_jobs),
            "early_stopping_rounds": early_stopping_rounds,
            **extra_params,
        }
        self.models = {
            task: XGBClassifier(
                objective="multi:softprob",
                num_class=4,
                eval_metric="mlogloss",
                tree_method="hist",
                **self.params,
            )
            for task in TASK_LABEL_NAMES
        }
        self.feature_count_: int | None = None

    def predict_proba(self, features: np.ndarray) -> dict[str, np.ndarray]:
        result = {}
        for task, model in self.models.items():
            raw = np.asarray(model.predict_proba(features), dtype=np.float64)
            probabilities = np.zeros((len(features), 4), dtype=np.float64)
            probabilities[:, np.asarray(model.classes_, dtype=np.int64)] = raw
            result[task] = probabilities
        return result

    def parameter_counts(self) -> dict[str, int]:
        total = 0
        for model in self.models.values():
            if getattr(model, "get_booster", None) and hasattr(model, "classes_"):
                total += int(model.get_booster().num_boosted_rounds())
        return {"trainable_parameters": 0, "total_parameters": total}

