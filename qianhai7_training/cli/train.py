from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


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
from ..data.sampler import RotatingRealizationSampler
from ..data.windows import make_window_refs, missing_train_families, split_by_state, split_distribution
from ..models.registry import build_model
from ..task_labels import (
    TASK_LABEL_LEVEL_NAMES,
    TASK_LABEL_NAMES,
    TASK_LABEL_SCHEMA,
)
from ..training.engine import train_one_epoch
from ..training.losses import supervised_loss
from ..training.plots import plot_final_evaluation, plot_training_history, write_history
from ..training.progress import make_progress
from ..training.report import Stopwatch, classification_metrics, count_parameters, write_report


def describe_run(data, splits, eval_splits, eval_stride, realizations, distribution) -> dict[str, Any]:
    """Data settings and split sizes, so a report can be interpreted later."""
    return {
        "window_length": data.window_length,
        "stride": data.stride,
        "eval_stride": eval_stride,
        "split_stratify": data.split_stratify,
        "split_seed": data.split_seed,
        "data_seed": data.seed,
        "train_fraction": data.train_fraction,
        "validation_fraction": data.validation_fraction,
        "observation_realizations": realizations,
        "noise_std_mm": data.noise_std_mm,
        "missing_probability": data.missing_probability,
        "virtual_point_count_min": data.virtual_point_count_min,
        "virtual_point_count_max": data.virtual_point_count_max,
        "train_windows": len(splits["train"]),
        "validation_windows": len(eval_splits["validation"]),
        "test_windows": len(eval_splits["test"]),
        "train_states": distribution["train"]["unique_states"],
        "validation_states": distribution["validation"]["unique_states"],
        "test_states": distribution["test"]["unique_states"],
        "families_missing_from_train": distribution["missing_from_train"],
    }


def make_loader(refs, observation, seed, normalization, batch_size, shuffle=False, *, num_workers=0,
                prefetch_factor=2, persistent_workers=False, pin_memory=False, realizations=1, sampler=None,
                regional_target_config=None):
    if realizations > 1 and sampler is None:
        raise ValueError(
            "realizations > 1 requires a RotatingRealizationSampler; sequential iteration "
            "would put the same window in a batch once per realization"
        )
    options = dict(
        dataset=QianhaiWindowDataset(refs, observation, seed, normalization, realizations=realizations,
                                     regional_target_config=regional_target_config),
        batch_size=batch_size,
        collate_fn=collate_window_batches,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # DataLoader rejects shuffle and sampler together.
    if sampler is not None:
        options["sampler"] = sampler
    else:
        options["shuffle"] = shuffle
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
        options["persistent_workers"] = persistent_workers
    return DataLoader(**options)


def evaluate_torch(model, loader, device, *, desc=None, progress=None):
    model.eval()
    collected = {key: {"true": [], "prob": []} for key in TASK_LABEL_NAMES}
    loss_sum = 0.0
    sample_count = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    with torch.no_grad(), make_progress(desc, total=total_batches, options=progress) as tracker:
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            outputs = model.forward_batch(batch)
            loss_sum += float(supervised_loss(outputs, batch)) * batch.batch_size
            sample_count += batch.batch_size
            for key in TASK_LABEL_NAMES:
                collected[key]["true"].append(batch.targets[f"target_{key}"].cpu().numpy())
                collected[key]["prob"].append(torch.softmax(outputs[key], dim=-1).cpu().numpy())
            tracker.update(batch_index, average_loss=loss_sum / max(sample_count, 1))
    predictions = {}
    for key, items in collected.items():
        y_true = np.concatenate(items["true"])
        y_prob = np.concatenate(items["prob"])
        y_pred = y_prob.argmax(axis=-1).astype(np.int64)
        predictions[key] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "class_values": np.arange(y_prob.shape[1]),
        }
    return loss_sum / max(sample_count, 1), predictions


def pack_metrics(predictions, losses=None):
    results = {}
    from sklearn.metrics import f1_score

    for key, values in predictions.items():
        metric = classification_metrics(
            values["y_true"],
            values["y_pred"],
            values["y_prob"],
            class_values=np.arange(len(TASK_LABEL_LEVEL_NAMES)),
        )
        metric["macro_f1"] = metric["f1_macro"]
        metric["class_names"] = list(TASK_LABEL_LEVEL_NAMES)
        metric["f1_per_class"] = f1_score(
            values["y_true"],
            values["y_pred"],
            labels=np.arange(len(TASK_LABEL_LEVEL_NAMES)),
            average=None,
            zero_division=0,
        ).tolist()
        results[key] = metric
    if losses is not None:
        results["combined_loss"] = float(losses)
    return results


def save_torch_checkpoint(path, model, optimizer, config, normalization, epoch):
    torch.save({
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "window_length": config.data.window_length,
        "normalization": normalization.__dict__,
        "target_schema": TASK_LABEL_SCHEMA,
        "target_version": "v1",
        "task_names": list(TASK_LABEL_NAMES),
        "class_names": list(TASK_LABEL_LEVEL_NAMES),
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "qianhai7_v12_baseline.yaml"))
    parser.add_argument("--data-root", default=None, help="Optional override for data.data_root")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "log", "off"),
        default=None,
        help="Override train.progress. Use 'bar' to force a tqdm progress bar "
             "when the output is not a terminal (IDE run windows, pipes).",
    )
    args = parser.parse_args()
    config = load_experiment_config(
        args.config,
        data_root_override=args.data_root,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        progress_override=args.progress,
    )
    data = config.data
    train = config.train
    progress_options = {"mode": train.progress, "log_every": train.log_every_batches}
    run_clock = Stopwatch()
    stage_started = time.perf_counter()
    index = build_archive_index(data.data_root)
    print(f"stage=archive_index files={len(index.records)} elapsed_sec={time.perf_counter() - stage_started:.1f}", flush=True)
    stage_started = time.perf_counter()
    refs = make_window_refs(index, data.window_length, data.stride)
    print(f"stage=window_refs windows={len(refs)} window_length={data.window_length} stride={data.stride} elapsed_sec={time.perf_counter() - stage_started:.1f}", flush=True)
    stage_started = time.perf_counter()
    splits = split_by_state(refs, data.split_seed, data.train_fraction, data.validation_fraction, stratify=data.split_stratify)
    eval_stride = data.eval_stride or data.stride
    eval_refs = make_window_refs(index, data.window_length, eval_stride)
    eval_splits = split_by_state(eval_refs, data.split_seed, data.train_fraction, data.validation_fraction, stratify=data.split_stratify)
    distribution = split_distribution(splits)
    missing_families = distribution["missing_from_train"]
    print(
        f"split_audit stratify={data.split_stratify} "
        f"states=train:{distribution['train']['unique_states']}"
        f"/validation:{distribution['validation']['unique_states']}"
        f"/test:{distribution['test']['unique_states']} "
        f"families_missing_from_train={len(missing_families)}"
        + (f" {missing_families}" if missing_families else ""),
        flush=True,
    )
    print(f"stage=split train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])} elapsed_sec={time.perf_counter() - stage_started:.1f}", flush=True)
    stage_started = time.perf_counter()
    normalization = fit_normalization_stats(splits["train"])
    print(f"stage=normalization states={len(normalization.fitted_state_ids)} elapsed_sec={time.perf_counter() - stage_started:.1f}", flush=True)
    normalization_path = train.output_root / "normalization_stats.json"
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    normalization.save(normalization_path)
    observation = ObservationConfig(data.virtual_point_count_min, data.virtual_point_count_max, data.noise_std_mm, data.missing_probability)
    loader_kwargs = {
        "num_workers": train.num_workers,
        "prefetch_factor": train.prefetch_factor,
        "persistent_workers": train.persistent_workers,
        "pin_memory": train.pin_memory,
    }
    realizations = max(int(data.observation_realizations), 1)
    # Always routed through the sampler: shuffling then comes from an explicit
    # seed instead of the global torch RNG, whose state depends on how much
    # randomness model construction consumed first.
    train_sampler = RotatingRealizationSampler(len(splits["train"]), realizations, seed=data.split_seed)
    loader = make_loader(splits["train"], observation, data.seed, normalization, train.batch_size,
                         realizations=realizations, sampler=train_sampler,
                         regional_target_config=config.regional_targets, **loader_kwargs)
    # Evaluation stays on a single deterministic realization so metrics are
    # comparable across epochs and runs.
    validation_loader = make_loader(eval_splits["validation"], observation, data.seed, normalization, train.batch_size,
                                    regional_target_config=config.regional_targets, **loader_kwargs)
    test_loader = make_loader(eval_splits["test"], observation, data.seed, normalization, train.batch_size,
                              regional_target_config=config.regional_targets, **loader_kwargs)
    if realizations > 1:
        per_worker_mib = distribution["train"]["unique_states"] * realizations * 25 / 1024
        print(
            f"observation_realizations={realizations} epoch_rotation={realizations} "
            f"estimated_cache_mib={per_worker_mib * max(train.num_workers, 1):.0f}",
            flush=True,
        )
    print(f"states={len(index.records)} windows={len(refs)} train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}", flush=True)
    print(f"normalization_stats={normalization_path}", flush=True)
    print(f"model={config.model.name} params={config.model.kwargs}", flush=True)
    print(f"data_loader=num_workers:{train.num_workers} prefetch_factor:{train.prefetch_factor} persistent_workers:{train.persistent_workers} pin_memory:{train.pin_memory}", flush=True)
    print("stage=model_initialization", flush=True)
    model = build_model(config.model.name, **config.model.kwargs)
    if config.model.name == "xgboost":
        raise ValueError("xgboost has no adapter for the v3 four-task ordinal target contract yet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    parameter_counts = count_parameters(model)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "unavailable"
    print(
        f"device={device} gpu={gpu_name} total_parameters={parameter_counts['total_parameters']} "
        f"trainable_parameters={parameter_counts['trainable_parameters']}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate, weight_decay=train.weight_decay)
    weights_dir = train.output_root / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)
    epoch_times: list[float] = []
    history: list[dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = 0
    stopped_epoch = 0
    epochs_without_improvement = 0
    history_path = train.output_root / "history.json"
    checkpoint_path = weights_dir / f"{config.model.name}_best.pt"
    selection_tasks = tuple(train.selection_tasks)
    for epoch in range(train.epochs):
        train_sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        loss = train_one_epoch(model, loader, optimizer, device, desc=f"train_batches epoch={epoch + 1}", progress=progress_options)
        epoch_elapsed = time.perf_counter() - epoch_started
        epoch_times.append(epoch_elapsed)
        validation_loss, validation_predictions = evaluate_torch(model, validation_loader, device, desc="validation_batches", progress=progress_options)
        validation_metrics = pack_metrics(validation_predictions)
        selection_score = float(np.mean([
            validation_metrics[key]["macro_f1"] for key in selection_tasks
        ]))
        history_row: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": float(loss),
            "validation_loss": float(validation_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_selection_macro_f1": selection_score,
        }
        for task, task_metrics in validation_metrics.items():
            if task == "combined_loss":
                continue
            history_row[f"validation_{task}_macro_f1"] = float(task_metrics["macro_f1"])
        history.append(history_row)
        write_history(history_path, history)
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_torch_checkpoint(checkpoint_path, model, optimizer, config, normalization, epoch + 1)
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch + 1} train_loss={loss:.6f} validation_loss={validation_loss:.6f} "
            f"validation_macro_f1={selection_score:.6f} "
            f"selection_score={selection_score:.6f} elapsed_sec={epoch_elapsed:.1f}",
            flush=True,
        )
        if train.early_stopping_patience and epochs_without_improvement >= train.early_stopping_patience:
            stopped_epoch = epoch + 1
            print(
                f"early_stopping epoch={stopped_epoch} best_epoch={best_epoch} "
                f"best_value={best_score:.6f} patience={train.early_stopping_patience}",
                flush=True,
            )
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_loss, validation_predictions = evaluate_torch(model, validation_loader, device, desc="validation_batches", progress=progress_options)
    test_loss, test_predictions = evaluate_torch(model, test_loader, device, desc="test_batches", progress=progress_options)
    history_plot_path = train.output_root / "training_history.png"
    plot_training_history(history, history_plot_path)
    test_metrics = pack_metrics(test_predictions, test_loss)
    evaluation_plot_path = train.output_root / "final_test_evaluation.png"
    plot_final_evaluation(test_metrics, evaluation_plot_path)
    report = {
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "target_schema": TASK_LABEL_SCHEMA,
        "task_names": list(TASK_LABEL_NAMES),
        "class_names": list(TASK_LABEL_LEVEL_NAMES),
        "data_root": str(data.data_root),
        "window_length": data.window_length,
        "batch_size": train.batch_size,
        "epochs": train.epochs,
        "states": len(index.records),
        "windows": len(refs),
        "train_windows": len(splits["train"]),
        "data": describe_run(data, splits, eval_splits, eval_stride, realizations, distribution),
        "parameters": count_parameters(model),
        "timing_sec": {
            "epoch_sec": epoch_times,
            "train_total_sec": sum(epoch_times),
            "total_process_sec": run_clock.elapsed(),
        },
        "history": str(history_path),
        "plots": {
            "training_history": str(history_plot_path),
            "final_test_evaluation": str(evaluation_plot_path),
        },
        "weights": {"checkpoint": str(checkpoint_path)},
        "metrics": {
            "validation": pack_metrics(validation_predictions, validation_loss),
            "test": test_metrics,
        },
        "selection": {
            "tasks": list(selection_tasks),
            "metric": "mean_macro_f1",
            "best_epoch": best_epoch,
            "best_value": best_score,
            "early_stopping_patience": train.early_stopping_patience,
            "epochs_run": len(epoch_times),
            "stopped_epoch": stopped_epoch,
        },
    }
    report_path = train.output_root / "training_report.json"
    write_report(report_path, report)
    print(f"training_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
