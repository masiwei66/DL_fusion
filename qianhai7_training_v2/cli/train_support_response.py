from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(_PROJECT_ROOT))
    __package__ = "qianhai7_training_v2.cli"

import numpy as np
import torch
from torch.utils.data import DataLoader

from spatial_support_contract import (
    SUPPORT_FEATURE_ORDER,
    SUPPORT_NORMALIZATION_POLICY,
    SUPPORT_TOKEN_SCHEMA,
    masked_gaussian_nll,
)
from ..config import load_experiment_config
from ..data.hdf5_source import HDF5StateSource
from ..data.manifest import build_archive_index
from ..data.normalization import fit_normalization_stats
from ..data.observation import ObservationConfig
from ..data.support_tokens import (
    QianhaiSupportTokenDataset,
    build_spatial_support_layout,
    collate_support_token_samples,
    fit_support_token_value_normalization,
)
from ..data.windows import make_window_refs, split_by_state
from ..models.support_response import SpatialSupportResponseModel


def _metrics(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    nll_sum = 0.0
    crps_sum = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            mean = predictions["mean"]
            scale = predictions["scale"].clamp_min(1.0e-6)
            valid = targets.valid_mask.bool() & torch.isfinite(targets.response)
            weight = valid.to(mean.dtype) * targets.area_coverage.to(mean.dtype)
            if not valid.any():
                continue
            standardized = (targets.response - mean) / scale
            nll = (
                0.5 * standardized.square()
                + torch.log(scale)
                + 0.5 * np.log(2.0 * np.pi)
            )
            phi = torch.exp(-0.5 * standardized.square()) / np.sqrt(2.0 * np.pi)
            cdf = 0.5 * (1.0 + torch.erf(standardized / np.sqrt(2.0)))
            crps = scale * (
                standardized * (2.0 * cdf - 1.0)
                + 2.0 * phi
                - 1.0 / np.sqrt(np.pi)
            )
            nll_sum += float((nll * weight).sum())
            crps_sum += float((crps * weight).sum())
            weight_sum += float(weight.sum())
    if weight_sum <= 0.0:
        raise ValueError("evaluation split contains no valid support targets")
    return {"nll": nll_sum / weight_sum, "crps": crps_sum / weight_sum}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_experiment_config(
        args.config,
        data_root_override=args.data_root,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
    )
    data = config.data
    train_config = config.train
    random.seed(data.seed)
    np.random.seed(data.seed)
    torch.manual_seed(data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(data.seed)
    started = time.perf_counter()

    index = build_archive_index(data.data_root)
    refs = make_window_refs(index, data.window_length, data.stride)
    splits = split_by_state(
        refs,
        data.split_seed,
        data.train_fraction,
        data.validation_fraction,
        stratify=data.split_stratify,
    )
    if any(not splits[name] for name in ("train", "validation", "test")):
        raise ValueError("support-response training requires non-empty state splits")

    mesh = HDF5StateSource(index.records[0].path).read()
    layout = build_spatial_support_layout(
        mesh["node_coords"],
        mesh["deck_node_mask"],
        mesh["node_ids"],
        mesh["deck_element_connectivity"],
        rows=args.rows,
        cols=args.cols,
        bandwidth_scale=args.bandwidth_scale,
    )
    observation = ObservationConfig(
        point_count_min=data.virtual_point_count_min,
        point_count_max=data.virtual_point_count_max,
        noise_std_mm=data.noise_std_mm,
        missing_probability=data.missing_probability,
    )
    time_normalization = fit_normalization_stats(splits["train"])
    value_normalization = fit_support_token_value_normalization(
        splits["train"], observation, data.seed, layout,
        minimum_coverage=args.minimum_observation_coverage,
    )
    expected_states = {
        ref.record.structural_state_id for ref in splits["train"]
    }
    if set(time_normalization.fitted_state_ids) != expected_states:
        raise ValueError("time normalization includes a non-training state")
    if set(value_normalization.fitted_state_ids) != expected_states:
        raise ValueError("support-value normalization includes a non-training state")

    train_data = QianhaiSupportTokenDataset(
        splits["train"], observation, data.seed, time_normalization,
        value_normalization, layout,
        realizations=data.observation_realizations,
        minimum_observation_coverage=args.minimum_observation_coverage,
        minimum_target_area_coverage=args.minimum_target_area_coverage,
    )
    evaluation_realizations = data.validation_realizations or 1
    validation_data = QianhaiSupportTokenDataset(
        splits["validation"], observation, data.seed, time_normalization,
        value_normalization, layout, realizations=evaluation_realizations,
        minimum_observation_coverage=args.minimum_observation_coverage,
        minimum_target_area_coverage=args.minimum_target_area_coverage,
    )
    test_data = QianhaiSupportTokenDataset(
        splits["test"], observation, data.seed, time_normalization,
        value_normalization, layout, realizations=evaluation_realizations,
        minimum_observation_coverage=args.minimum_observation_coverage,
        minimum_target_area_coverage=args.minimum_target_area_coverage,
    )
    generator = torch.Generator().manual_seed(data.seed)
    loaders = {
        "train": DataLoader(
            train_data,
            batch_size=train_config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_support_token_samples,
        ),
        "validation": DataLoader(
            validation_data,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=collate_support_token_samples,
        ),
        "test": DataLoader(
            test_data,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=collate_support_token_samples,
        ),
    }
    device = torch.device(
        "cuda" if train_config.device == "auto" and torch.cuda.is_available()
        else "cpu" if train_config.device == "auto"
        else train_config.device
    )
    model = SpatialSupportResponseModel(
        time_feature_count=2,
        support_feature_count=4,
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    if train_config.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_config.lr_factor,
            patience=train_config.lr_patience,
            min_lr=train_config.min_learning_rate,
        )
    elif train_config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(train_config.epochs, 1),
            eta_min=train_config.min_learning_rate,
        )
    else:
        scheduler = None

    output_root = Path(args.output_root) if args.output_root else train_config.output_root / "support_response"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "support_response_best.pt"
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epoch_seconds: list[float] = []
    history: list[dict[str, float]] = []
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        train_losses: list[float] = []
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = masked_gaussian_nll(predictions, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite support-response loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        validation = _metrics(model, loaders["validation"], device)
        score = float(validation["crps"])
        history.append(
            {
                "epoch": float(epoch),
                "train_nll": float(np.mean(train_losses)),
                "validation_nll": validation["nll"],
                "validation_crps": score,
            }
        )
        epoch_seconds.append(time.perf_counter() - epoch_started)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "support_token_schema": SUPPORT_TOKEN_SCHEMA,
                    "value_normalization": value_normalization.__dict__,
                    "time_normalization": time_normalization.__dict__,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step(score) if train_config.lr_scheduler == "plateau" else scheduler.step()
        print(
            f"epoch={epoch} train_nll={history[-1]['train_nll']:.6f} "
            f"validation_crps={score:.6f} elapsed_sec={epoch_seconds[-1]:.1f}",
            flush=True,
        )
        if train_config.early_stopping_patience and stale_epochs >= train_config.early_stopping_patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _metrics(model, loaders["test"], device)
    report = {
        "schema_version": "qianhai_support_response_report_v1",
        "status": "COMPLETED",
        "task": "support_aggregated_fem_response",
        "support_token_schema": SUPPORT_TOKEN_SCHEMA,
        "support_feature_order": list(SUPPORT_FEATURE_ORDER),
        "normalization_policy": SUPPORT_NORMALIZATION_POLICY,
        "normalization": {
            "support_value_mean": value_normalization.mean,
            "support_value_std": value_normalization.std,
            "support_value_fitted_state_ids": list(value_normalization.fitted_state_ids),
            "time_fitted_state_ids": list(time_normalization.fitted_state_ids),
        },
        "model": {"name": "shared_support_token_gru", "hidden_size": args.hidden_size},
        "selection": {"metric": "validation_support_crps", "best_epoch": best_epoch, "best_value": best_score},
        "history": history,
        "metrics": {"test": test_metrics},
        "splits": {
            name: sorted({ref.record.structural_state_id for ref in values})
            for name, values in splits.items()
        },
        "target_isolation": "dense FEM response is used only for loss and metrics",
        "exact_coordinates": "used by the data adapter for support assignment; excluded from model inputs",
        "time_feature_order": ["temperature_C", "delta_days"],
        "time_feature_normalization": "standardized from unique training states only",
        "timing_sec": {
            "epoch_sec": epoch_seconds,
            "train_total_sec": float(sum(epoch_seconds)),
            "total_process_sec": float(time.perf_counter() - started),
        },
        "checkpoint": str(checkpoint_path),
    }
    report_path = output_root / "training_report.json"
    (report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the separate Qianhai spatial support-token response task."
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "configs" / "qianhai7_v12_baseline.yaml"),
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--bandwidth-scale", type=float, default=0.75)
    parser.add_argument("--minimum-observation-coverage", type=float, default=0.05)
    parser.add_argument("--minimum-target-area-coverage", type=float, default=0.99)
    parser.add_argument("--hidden-size", type=int, default=48)
    args = parser.parse_args()
    report = run(args)
    print(f"training_report={report['report_path']}")


if __name__ == "__main__":
    main()
