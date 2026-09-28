from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch

from ..contracts import WindowBatch


def fixed_features(batch: WindowBatch) -> np.ndarray:
    """Make a fixed-width tabular representation without using point IDs."""
    mask = (batch.valid_mask & batch.point_present_mask[:, None, :]).float()
    values = batch.values
    denom = mask.sum(dim=2).clamp_min(1.0)
    mean = (values * mask).sum(dim=2) / denom
    centered = (values - mean.unsqueeze(-1)) * mask
    std = centered.square().sum(dim=2).div(denom).sqrt()
    minimum = values.masked_fill(mask == 0, float("inf")).amin(dim=2)
    maximum = values.masked_fill(mask == 0, float("-inf")).amax(dim=2)
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    valid_ratio = mask.mean(dim=2)
    # XGBoost requires [B, F]. Reduce the temporal axis with fixed statistics;
    # unlike the neural temporal models, it cannot consume a variable sequence.
    time_mask = batch.temperature_valid_mask.float()
    time = batch.time_features * time_mask
    time_denom = time_mask.sum(dim=1).clamp_min(1.0)
    time_mean = time.sum(dim=1) / time_denom
    time_std = ((time - time_mean.unsqueeze(1)) * time_mask).square().sum(dim=1).div(time_denom).sqrt()
    time_first = time[:, 0, :]
    time_last = time[:, -1, :]
    time_delta = time_last - time_first
    point_mask = batch.point_present_mask.float().unsqueeze(-1)
    coords = (batch.point_features * point_mask).sum(dim=1) / point_mask.sum(dim=1).clamp_min(1.0)
    # Flatten each [B, W] statistic to a stable width using mean/std/first/last.
    value_features = torch.cat([
        mean.mean(dim=1, keepdim=True), mean.std(dim=1, keepdim=True, unbiased=False),
        mean[:, :1], mean[:, -1:],
        std.mean(dim=1, keepdim=True), std.std(dim=1, keepdim=True, unbiased=False),
        minimum.mean(dim=1, keepdim=True), maximum.mean(dim=1, keepdim=True),
        valid_ratio.mean(dim=1, keepdim=True),
    ], dim=1)
    features = torch.cat([value_features, time_mean, time_std, time_first, time_last, time_delta, coords], dim=1)
    return features.detach().cpu().numpy().astype(np.float32)


@dataclass
class XGBoostModel:
    num_classes: int = 40
    severity_classes: int = 6
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42
    state_model: Any = None
    severity_model: Any = None
    material_model: Any = None
    settlement_model: Any = None
    stiffness_model: Any = None
    # XGBoost's sklearn wrapper requires labels in [0, n_classes).  The
    # dataset labels remain global vocabulary IDs; these arrays record the
    # local-to-global mapping used for each fitted classifier.
    state_classes_: np.ndarray | None = None
    severity_classes_: np.ndarray | None = None
    material_classes_: np.ndarray | None = None
    settlement_classes_: np.ndarray | None = None
    stiffness_classes_: np.ndarray | None = None
    timing: dict[str, float] | None = None

    @staticmethod
    def _local_labels(labels: np.ndarray, classes: np.ndarray, split: str, task: str) -> np.ndarray:
        lookup = {int(label): index for index, label in enumerate(classes)}
        unknown = sorted(set(map(int, np.unique(labels))) - set(lookup))
        if unknown:
            raise ValueError(f"{split} {task} contains labels absent from training: {unknown}")
        return np.fromiter((lookup[int(label)] for label in labels), dtype=np.int64, count=len(labels))

    def fit_loader(self, loader: Any) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost model requires: pip install xgboost") from exc
        features: list[np.ndarray] = []
        state_targets: list[np.ndarray] = []
        severity_targets: list[np.ndarray] = []
        material_targets: list[np.ndarray] = []
        settlement_targets: list[np.ndarray] = []
        stiffness_targets: list[np.ndarray] = []
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        started = time.perf_counter()
        extraction_started = started
        print(f"xgboost_stage=feature_extraction batches={total_batches or '?'}", flush=True)
        for batch_index, batch in enumerate(loader, start=1):
            features.append(fixed_features(batch))
            state_targets.append(batch.targets["fine_state_family"].cpu().numpy())
            severity_targets.append(batch.targets["global_severity"].cpu().numpy())
            material_targets.append(batch.targets["material_active"].cpu().numpy())
            settlement_targets.append(batch.targets["settlement_type"].cpu().numpy())
            stiffness_targets.append(batch.targets["stiffness_reduction"].cpu().numpy())
            if batch_index == 1 or batch_index % 100 == 0 or (total_batches and batch_index == total_batches):
                elapsed = time.perf_counter() - started
                rate = batch_index / max(elapsed, 1e-6)
                print(f"xgboost_stage=feature_extraction progress={batch_index}/{total_batches or '?'} rate={rate:.2f} batches_per_sec", flush=True)
        if not features:
            raise ValueError("Cannot fit XGBoost on an empty loader")
        x = np.concatenate(features)
        y_state = np.concatenate(state_targets)
        y_severity = np.concatenate(severity_targets)
        y_material = np.concatenate(material_targets)
        y_settlement = np.concatenate(settlement_targets)
        y_stiffness = np.concatenate(stiffness_targets)
        self.state_classes_ = np.unique(y_state)
        self.severity_classes_ = np.unique(y_severity)
        state_local = np.searchsorted(self.state_classes_, y_state)
        severity_local = np.searchsorted(self.severity_classes_, y_severity)
        material_classes = np.unique(y_material)
        settlement_classes = np.unique(y_settlement)
        stiffness_classes = np.unique(y_stiffness)
        material_local = np.searchsorted(material_classes, y_material)
        settlement_local = np.searchsorted(settlement_classes, y_settlement)
        stiffness_local = np.searchsorted(stiffness_classes, y_stiffness)
        extraction_elapsed = time.perf_counter() - extraction_started
        common = dict(n_estimators=self.n_estimators, max_depth=self.max_depth,
                      learning_rate=self.learning_rate, random_state=self.random_state,
                      tree_method="hist")
        self.state_model = XGBClassifier(num_class=len(self.state_classes_), objective="multi:softprob", **common)
        self.severity_model = XGBClassifier(num_class=len(self.severity_classes_), objective="multi:softprob", **common)
        self.material_model = XGBClassifier(num_class=len(material_classes), objective="multi:softprob", **common)
        self.settlement_model = XGBClassifier(num_class=len(settlement_classes), objective="multi:softprob", **common)
        self.stiffness_model = XGBClassifier(num_class=len(stiffness_classes), objective="multi:softprob", **common)
        print(f"xgboost_stage=fit_state samples={len(x)} features={x.shape[1]}", flush=True)
        state_started = time.perf_counter()
        self.state_model.fit(x, state_local, verbose=False)
        state_elapsed = time.perf_counter() - state_started
        print("xgboost_stage=fit_state complete", flush=True)
        print("xgboost_stage=fit_severity", flush=True)
        severity_started = time.perf_counter()
        self.severity_model.fit(x, severity_local, verbose=False)
        severity_elapsed = time.perf_counter() - severity_started
        print("xgboost_stage=fit_severity complete", flush=True)
        self.material_model.fit(x, material_local, verbose=False)
        self.settlement_model.fit(x, settlement_local, verbose=False)
        self.stiffness_model.fit(x, stiffness_local, verbose=False)
        self.material_classes_ = material_classes
        self.settlement_classes_ = settlement_classes
        self.stiffness_classes_ = stiffness_classes
        self.timing = {
            "feature_extraction_sec": extraction_elapsed,
            "state_fit_sec": state_elapsed,
            "severity_fit_sec": severity_elapsed,
            "fit_total_sec": time.perf_counter() - started,
            "samples": int(x.shape[0]),
            "features": int(x.shape[1]),
            "state_class_values": self.state_classes_.tolist(),
            "severity_class_values": self.severity_classes_.tolist(),
        }

    def predict_batch(self, batch: WindowBatch) -> dict[str, np.ndarray]:
        if self.state_model is None or self.severity_model is None:
            raise RuntimeError("Call fit_loader before predict_batch")
        x = fixed_features(batch)
        if self.state_classes_ is None or self.severity_classes_ is None:
            raise RuntimeError("Fitted class mappings are unavailable")
        return {
            "state_family": self.state_model.predict_proba(x),
            "state_family_class_values": self.state_classes_.copy(),
            "global_severity": self.severity_model.predict_proba(x),
            "global_severity_class_values": self.severity_classes_.copy(),
            "material_active": self.material_model.predict_proba(x),
            "settlement_type": self.settlement_model.predict_proba(x),
            "stiffness_reduction": self.stiffness_model.predict_proba(x),
        }

    def evaluate_arrays(
        self,
        features: np.ndarray,
        state_labels: np.ndarray,
        severity_labels: np.ndarray,
        split: str,
    ) -> dict[str, Any]:
        if self.state_model is None or self.severity_model is None:
            raise RuntimeError("Call fit_loader before evaluate_arrays")
        if self.state_classes_ is None or self.severity_classes_ is None:
            raise RuntimeError("Fitted class mappings are unavailable")
        state_pred = self.state_model.predict(features)
        severity_pred = self.severity_model.predict(features)
        if state_pred.ndim > 1:
            state_pred = state_pred.reshape(state_pred.shape[0], -1).argmax(axis=1)
        if severity_pred.ndim > 1:
            severity_pred = severity_pred.reshape(severity_pred.shape[0], -1).argmax(axis=1)
        return {
            "state_family": {
                "y_true": state_labels,
                "y_pred": self.state_classes_[state_pred.astype(np.int64)].reshape(-1),
                "y_prob": self.state_model.predict_proba(features),
                "class_values": self.state_classes_,
            },
            "global_severity": {
                "y_true": severity_labels,
                "y_pred": self.severity_classes_[severity_pred.astype(np.int64)].reshape(-1),
                "y_prob": self.severity_model.predict_proba(features),
                "class_values": self.severity_classes_,
            },
        }
