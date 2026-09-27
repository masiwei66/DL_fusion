from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    # Allow both `python -m qianhai7_training.cli.train` and direct IDE execution.
    sys.path.insert(0, str(_PROJECT_ROOT))
    __package__ = "qianhai7_training.cli"

import torch
from torch.utils.data import DataLoader
import numpy as np

from ..config import load_experiment_config
from ..data.dataset import QianhaiWindowDataset, collate_window_batches
from ..data.manifest import build_archive_index
from ..data.normalization import fit_normalization_stats
from ..data.observation import ObservationConfig
from ..data.windows import make_window_refs, split_by_state
from ..models.registry import build_model
from ..training.engine import train_one_epoch
from ..training.losses import supervised_loss
from ..training.report import Stopwatch, classification_metrics, count_parameters, multiclass_cross_entropy, write_report


def make_loader(refs, observation, seed, normalization, batch_size, shuffle=False, *, num_workers=0,
                prefetch_factor=2, persistent_workers=False, pin_memory=False):
    options = dict(
        dataset=QianhaiWindowDataset(refs, observation, seed, normalization),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_window_batches,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
        options["persistent_workers"] = persistent_workers
    return DataLoader(**options)


def evaluate_torch(model, loader, device):
    model.eval()
    collected = {key: {"true": [], "prob": []} for key in ("state_family", "global_severity")}
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model.forward_batch(batch)
            loss_sum += float(supervised_loss(outputs, batch)) * batch.batch_size
            sample_count += batch.batch_size
            for key in collected:
                collected[key]["true"].append(batch.targets[key].cpu().numpy())
                collected[key]["prob"].append(torch.softmax(outputs[key], dim=1).cpu().numpy())
    predictions = {}
    for key, items in collected.items():
        y_true = np.concatenate(items["true"])
        y_prob = np.concatenate(items["prob"])
        y_pred = y_prob.argmax(axis=1)
        predictions[key] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "class_values": np.arange(y_prob.shape[1]),
        }
    return loss_sum / max(sample_count, 1), predictions


def pack_metrics(predictions, losses=None):
    results = {}
    for key, values in predictions.items():
        results[key] = classification_metrics(
            values["y_true"], values["y_pred"], values["y_prob"], values["class_values"]
        )
        results[key]["cross_entropy"] = multiclass_cross_entropy(
            values["y_true"], values["y_prob"], values["class_values"]
        )
    if losses is not None:
        results["combined_loss"] = float(losses)
    return results


def evaluate_xgboost(model, loader):
    features, states, severities = [], [], []
    from ..models.xgboost_model import fixed_features
    for batch in loader:
        features.append(fixed_features(batch))
        states.append(batch.targets["state_family"].cpu().numpy())
        severities.append(batch.targets["global_severity"].cpu().numpy())
    if not features:
        raise ValueError("Cannot evaluate on an empty split")
    x = np.concatenate(features)
    state_y, severity_y = np.concatenate(states), np.concatenate(severities)
    task_predictions = model.evaluate_arrays(x, state_y, severity_y, "evaluation")
    metrics = {}
    for key, values in task_predictions.items():
        unsupported = sorted(set(map(int, np.unique(values["y_true"]))) - set(map(int, values["class_values"])))
        task_metrics = classification_metrics(values["y_true"], values["y_pred"], values["y_prob"], values["class_values"])
        task_metrics["cross_entropy"] = multiclass_cross_entropy(values["y_true"], values["y_prob"], values["class_values"])
        task_metrics["unsupported_true_class_values"] = unsupported
        metrics[key] = task_metrics
    losses = [metrics[key]["cross_entropy"] for key in ("state_family", "global_severity")]
    metrics["combined_loss"] = sum(losses) if all(value is not None for value in losses) else None
    return metrics


def save_torch_checkpoint(path, model, optimizer, config, normalization, epoch):
    torch.save({
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "window_length": config.data.window_length,
        "normalization": normalization.__dict__,
        "state_class_values": list(range(getattr(model, "num_classes", 40))),
        "severity_class_values": list(range(6)),
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "qianhai7_v12_baseline.yaml"))
    parser.add_argument("--data-root", default=None, help="Optional override for data.data_root")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = load_experiment_config(
        args.config,
        data_root_override=args.data_root,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
    )
    data = config.data
    train = config.train
    run_clock = Stopwatch()
    index = build_archive_index(data.data_root)
    refs = make_window_refs(index, data.window_length, data.stride)
    splits = split_by_state(refs, data.split_seed, data.train_fraction, data.validation_fraction)
    normalization = fit_normalization_stats(splits["train"])
    normalization_path = Path("runs") / "normalization_stats.json"
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    normalization.save(normalization_path)
    observation = ObservationConfig(data.virtual_point_count_min, data.virtual_point_count_max, data.noise_std_mm, data.missing_probability)
    loader_kwargs = {
        "num_workers": train.num_workers,
        "prefetch_factor": train.prefetch_factor,
        "persistent_workers": train.persistent_workers,
        "pin_memory": train.pin_memory,
    }
    loader = make_loader(splits["train"], observation, data.seed, normalization, train.batch_size, shuffle=True, **loader_kwargs)
    validation_loader = make_loader(splits["validation"], observation, data.seed, normalization, train.batch_size, **loader_kwargs)
    test_loader = make_loader(splits["test"], observation, data.seed, normalization, train.batch_size, **loader_kwargs)
    print(f"states={len(index.records)} windows={len(refs)} train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}")
    print(f"normalization_stats={normalization_path}")
    print(f"model={config.model.name} params={config.model.kwargs}")
    print(f"data_loader=num_workers:{train.num_workers} prefetch_factor:{train.prefetch_factor} persistent_workers:{train.persistent_workers} pin_memory:{train.pin_memory}")
    print("stage=model_initialization", flush=True)
    model = build_model(config.model.name, **config.model.kwargs)
    if config.model.name == "xgboost":
        model.fit_loader(loader)
        weights_dir = train.output_root / "models"
        weights_dir.mkdir(parents=True, exist_ok=True)
        state_path = weights_dir / "xgboost_state.json"
        severity_path = weights_dir / "xgboost_severity.json"
        model.state_model.save_model(state_path)
        model.severity_model.save_model(severity_path)
        validation_metrics = evaluate_xgboost(model, validation_loader)
        test_metrics = evaluate_xgboost(model, test_loader)
        total_elapsed = run_clock.elapsed()
        report = {
            "model_name": config.model.name,
            "model_params": config.model.kwargs,
            "data_root": str(data.data_root),
            "window_length": data.window_length,
            "batch_size": train.batch_size,
            "states": len(index.records),
            "windows": len(refs),
            "train_windows": len(splits["train"]),
            "timing_sec": {**(model.timing or {}), "total_process_sec": total_elapsed},
            "class_mappings": {
                "state_family": (model.state_classes_.tolist() if model.state_classes_ is not None else []),
                "global_severity": (model.severity_classes_.tolist() if model.severity_classes_ is not None else []),
            },
            "weights": {"state_model": str(state_path), "severity_model": str(severity_path)},
            "metrics": {"validation": validation_metrics, "test": test_metrics},
        }
        report_path = train.output_root / "training_report.json"
        write_report(report_path, report)
        print(f"xgboost_fit=complete elapsed_sec={total_elapsed:.1f} report={report_path}", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate, weight_decay=train.weight_decay)
    weights_dir = train.output_root / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)
    epoch_times: list[float] = []
    for epoch in range(train.epochs):
        epoch_started = time.perf_counter()
        loss = train_one_epoch(model, loader, optimizer, device)
        epoch_elapsed = time.perf_counter() - epoch_started
        epoch_times.append(epoch_elapsed)
        print(f"epoch={epoch + 1} train_loss={loss:.6f} elapsed_sec={epoch_elapsed:.1f}", flush=True)
    checkpoint_path = weights_dir / f"{config.model.name}_best.pt"
    save_torch_checkpoint(checkpoint_path, model, optimizer, config, normalization, train.epochs)
    validation_loss, validation_predictions = evaluate_torch(model, validation_loader, device)
    test_loss, test_predictions = evaluate_torch(model, test_loader, device)
    report = {
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "data_root": str(data.data_root),
        "window_length": data.window_length,
        "batch_size": train.batch_size,
        "epochs": train.epochs,
        "states": len(index.records),
        "windows": len(refs),
        "train_windows": len(splits["train"]),
        "parameters": count_parameters(model),
        "timing_sec": {
            "epoch_sec": epoch_times,
            "train_total_sec": sum(epoch_times),
            "total_process_sec": run_clock.elapsed(),
        },
        "weights": {"checkpoint": str(checkpoint_path)},
        "metrics": {
            "validation": pack_metrics(validation_predictions, validation_loss),
            "test": pack_metrics(test_predictions, test_loss),
        },
    }
    report_path = train.output_root / "training_report.json"
    write_report(report_path, report)
    print(f"training_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
