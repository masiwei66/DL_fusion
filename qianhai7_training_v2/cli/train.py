from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    # Allow both `python -m qianhai7_training_v2.cli.train` and direct IDE execution.
    sys.path.insert(0, str(_PROJECT_ROOT))
    __package__ = "qianhai7_training_v2.cli"

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
from ..models.xgboost_model import XGBoostModel, window_features
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
from dl_fusion.training.lifecycle import (
    BestMetricTracker,
    build_adamw,
    build_scheduler,
    materialize_lazy_parameters,
    restore_model_checkpoint,
    save_model_checkpoint,
    step_scheduler,
)
from dl_fusion.experiment import limit_per_state, limit_states
from dl_fusion.training.runtime import (
    configure_runtime,
    loader_generator,
    seed_worker,
    write_run_artifacts,
)


def describe_run(
    data,
    input_mode,
    splits,
    eval_splits,
    eval_stride,
    realizations,
    validation_realizations,
    distribution,
    region_grid=None,
) -> dict[str, Any]:
    """Data settings and split sizes, so a report can be interpreted later."""
    payload = {
        "input_mode": input_mode,
        "observable_kind": "vertical_displacement_uz_mm",
        "region_grid_size_m": float(data.region_grid_size_m),
        "coordinate_scale_to_m": float(data.coordinate_scale_to_m),
        "min_points_per_region": int(data.min_points_per_region),
        "window_length": data.window_length,
        "stride": data.stride,
        "eval_stride": eval_stride,
        "split_stratify": data.split_stratify,
        "split_seed": data.split_seed,
        "data_seed": data.seed,
        "realization_sampling": data.realization_sampling,
        "train_fraction": data.train_fraction,
        "validation_fraction": data.validation_fraction,
        "observation_realizations": realizations,
        "validation_realizations": validation_realizations,
        "noise_std_mm": data.noise_std_mm,
        "missing_probability": data.missing_probability,
        "virtual_point_count_min": data.virtual_point_count_min,
        "virtual_point_count_max": data.virtual_point_count_max,
        "train_windows": len(splits["train"]),
        "validation_windows": len(eval_splits["validation"]),
        "test_windows": len(eval_splits["test"]),
        "validation_evaluated_samples": len(eval_splits["validation"]) * validation_realizations,
        "test_evaluated_samples": len(eval_splits["test"]) * validation_realizations,
        "train_states": distribution["train"]["unique_states"],
        "validation_states": distribution["validation"]["unique_states"],
        "test_states": distribution["test"]["unique_states"],
        "families_missing_from_train": distribution["missing_from_train"],
    }
    if region_grid is not None:
        payload["region_layout"] = {
            "grid_origin_xy_m": list(region_grid.origin_xy),
            "cell_size_m": float(region_grid.cell_size_m),
            "rows": int(region_grid.rows),
            "columns": int(region_grid.columns),
            "region_count": int(region_grid.region_count),
            "present_mask": region_grid.present_mask.astype(bool).tolist(),
        }
    return payload


def make_loader(refs, observation, seed, normalization, batch_size, shuffle=False, *, num_workers=0,
                prefetch_factor=2, persistent_workers=False, pin_memory=False, realizations=1, sampler=None,
                regional_target_config=None, loader_seed=0):
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
        worker_init_fn=seed_worker,
        generator=loader_generator(loader_seed),
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


def evaluate_torch(model, loader, device, *, desc=None, progress=None, task_weights=None):
    model.eval()
    collected = {key: {"true": [], "prob": []} for key in TASK_LABEL_NAMES}
    loss_sum = 0.0
    sample_count = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    with torch.no_grad(), make_progress(desc, total=total_batches, options=progress) as tracker:
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            outputs = model.forward_batch(batch)
            loss_sum += float(supervised_loss(outputs, batch, task_weights=task_weights)) * batch.batch_size
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
    metadata = {
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "input_mode": config.input_mode,
        "window_length": config.data.window_length,
        "observation_realizations": config.data.observation_realizations,
        "validation_realizations": config.data.validation_realizations or config.data.observation_realizations,
        "realization_sampling": config.data.realization_sampling,
        "normalization": normalization.__dict__,
        "target_schema": TASK_LABEL_SCHEMA,
        "target_version": "v1",
        "task_names": list(TASK_LABEL_NAMES),
        "class_names": list(TASK_LABEL_LEVEL_NAMES),
        "task_loss_weights": config.train.task_loss_weights,
        "lr_scheduler": config.train.lr_scheduler,
    }
    save_model_checkpoint(
        path,
        model,
        optimizer=optimizer,
        epoch=epoch,
        metadata=metadata,
    )


def materialize_lazy_model(model, loader, device: torch.device) -> bool:
    """Initialize lazy task heads from a real batch before optimizer setup."""
    def first_batch():
        try:
            return next(iter(loader))
        except StopIteration as exc:
            raise ValueError(
                "cannot initialize a lazy model from an empty training loader"
            ) from exc

    return materialize_lazy_parameters(
        model,
        first_batch,
        lambda current: model.forward_batch(current.to(device)),
    )


def _xgb_features_and_targets(loader, device="cpu"):
    del device
    feature_rows = []
    targets = {task: [] for task in TASK_LABEL_NAMES}
    for batch in loader:
        feature_rows.append(window_features(batch))
        for task in TASK_LABEL_NAMES:
            targets[task].append(batch.targets[f"target_{task}"].numpy())
    features = np.concatenate(feature_rows, axis=0)
    labels = {task: np.concatenate(rows, axis=0).astype(np.int64) for task, rows in targets.items()}
    return features, labels


def _evaluate_xgb(model, features, labels):
    probabilities = model.predict_proba(features)
    predictions = {}
    for task in TASK_LABEL_NAMES:
        probability = probabilities[task]
        predictions[task] = {
            "y_true": labels[task],
            "y_pred": probability.argmax(axis=1).astype(np.int64),
            "y_prob": probability,
            "class_values": np.arange(len(TASK_LABEL_LEVEL_NAMES)),
        }
    return predictions


def train_xgboost(config, train_loader, validation_loader, test_loader, normalization, splits, eval_splits,
                  distribution, index, refs, eval_stride, run_clock):
    """Train four XGBoost heads while preserving the standard report contract."""
    from sklearn.metrics import log_loss

    stage_started = time.perf_counter()
    train_features, train_labels = _xgb_features_and_targets(train_loader)
    validation_features, validation_labels = _xgb_features_and_targets(validation_loader)
    test_features, test_labels = _xgb_features_and_targets(test_loader)
    print(
        f"xgboost_features train_shape={train_features.shape} validation_shape={validation_features.shape} "
        f"test_shape={test_features.shape} elapsed_sec={time.perf_counter() - stage_started:.1f}",
        flush=True,
    )

    model = XGBoostModel(**config.model.kwargs)
    models_dir = config.train.output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_round = 0
    history = []
    for task, estimator in model.models.items():
        fit_kwargs = {"eval_set": [(validation_features, validation_labels[task])], "verbose": False}
        estimator.fit(train_features, train_labels[task], **fit_kwargs)

    validation_predictions = _evaluate_xgb(model, validation_features, validation_labels)
    validation_metrics = pack_metrics(validation_predictions)
    selection_tasks = tuple(config.train.selection_tasks)
    best_score = float(np.mean([validation_metrics[key]["macro_f1"] for key in selection_tasks]))
    best_round = max(
        int(getattr(estimator, "best_iteration", -1)) + 1
        for estimator in model.models.values()
    )
    test_predictions = _evaluate_xgb(model, test_features, test_labels)
    test_metrics = pack_metrics(test_predictions)
    history.append({
        "epoch": best_round,
        "train_loss": float(np.mean([
            log_loss(train_labels[task], model.models[task].predict_proba(train_features), labels=np.arange(4))
            for task in TASK_LABEL_NAMES
        ])),
        "validation_loss": float(np.mean([
            log_loss(validation_labels[task], model.models[task].predict_proba(validation_features), labels=np.arange(4))
            for task in TASK_LABEL_NAMES
        ])),
        "validation_selection_macro_f1": best_score,
        **{f"validation_{task}_macro_f1": float(validation_metrics[task]["macro_f1"]) for task in TASK_LABEL_NAMES},
    })

    import joblib
    checkpoint_path = models_dir / "xgboost_best.joblib"
    joblib.dump(model, checkpoint_path)
    history_path = config.train.output_root / "history.json"
    write_history(history_path, history)
    history_plot_path = config.train.output_root / "training_history.png"
    plot_training_history(history, history_plot_path)
    evaluation_plot_path = config.train.output_root / "final_test_evaluation.png"
    plot_final_evaluation(test_metrics, evaluation_plot_path)
    report = {
        "task": config.task,
        "run_status": "SMOKE_ONLY" if config.smoke.enabled else "FORMAL",
        "run_kind": "SMOKE_ONLY" if config.smoke.enabled else "EXPERIMENT",
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "target_schema": TASK_LABEL_SCHEMA,
        "task_names": list(TASK_LABEL_NAMES),
        "class_names": list(TASK_LABEL_LEVEL_NAMES),
        "data_root": str(config.data.data_root),
        "window_length": config.data.window_length,
        "batch_size": config.train.batch_size,
        "epochs": config.train.epochs,
        "optimization": {"xgboost_params": config.model.kwargs},
        "states": len(index.records),
        "windows": len(refs),
        "train_windows": len(splits["train"]),
        "data": describe_run(config.data, config.input_mode, splits, eval_splits, eval_stride,
                              config.data.observation_realizations,
                              config.data.validation_realizations or config.data.observation_realizations,
                              distribution, getattr(loader.dataset, "region_grid", None)),
        "parameters": model.parameter_counts(),
        "timing_sec": {"total_process_sec": run_clock.elapsed()},
        "history": str(history_path),
        "plots": {"training_history": str(history_plot_path), "final_test_evaluation": str(evaluation_plot_path)},
        "weights": {"checkpoint": str(checkpoint_path)},
        "run_metadata": str(config.train.output_root / "run_metadata.json"),
        "resolved_config": str(config.train.output_root / "resolved_config.json"),
        "split_manifest": str(config.train.output_root / "split_manifest.json"),
        "run_requirements": str(config.train.output_root / "requirements_frozen.txt"),
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "selection": {
            "tasks": list(selection_tasks), "metric": "mean_macro_f1", "best_epoch": best_round,
            "best_value": best_score, "early_stopping_patience": model.params.get("early_stopping_rounds"),
            "epochs_run": best_round, "stopped_epoch": best_round,
        },
    }
    report_path = config.train.output_root / "training_report.json"
    write_report(report_path, report)
    print(f"training_report={report_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
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
    args = parser.parse_args(argv)
    config = load_experiment_config(
        args.config,
        data_root_override=args.data_root,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        progress_override=args.progress,
    )
    data = config.data
    train = config.train
    device = configure_runtime(train)
    progress_options = {"mode": train.progress, "log_every": train.log_every_batches}
    run_clock = Stopwatch()
    stage_started = time.perf_counter()
    index = build_archive_index(data.data_root, progress=train.progress != "off")
    print(f"stage=archive_index files={len(index.records)} elapsed_sec={time.perf_counter() - stage_started:.1f}", flush=True)
    stage_started = time.perf_counter()
    refs = make_window_refs(index, data.window_length, data.stride)
    print(f"stage=window_refs windows={len(refs)} window_length={data.window_length} stride={data.stride} elapsed_sec={time.perf_counter() - stage_started:.1f}", flush=True)
    stage_started = time.perf_counter()
    splits = split_by_state(refs, data.split_seed, data.train_fraction, data.validation_fraction, stratify=data.split_stratify)
    eval_stride = data.eval_stride or data.stride
    eval_refs = make_window_refs(index, data.window_length, eval_stride)
    eval_splits = split_by_state(eval_refs, data.split_seed, data.train_fraction, data.validation_fraction, stratify=data.split_stratify)
    if config.smoke.enabled:
        splits = limit_states(splits, config.smoke)
        eval_splits = limit_states(eval_splits, config.smoke)
        splits = {role: limit_per_state(items, config.smoke.windows_per_state) for role, items in splits.items()}
        eval_splits = {
            role: limit_per_state(items, config.smoke.windows_per_state)
            for role, items in eval_splits.items()
        }
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
    normalization = fit_normalization_stats(
        splits["train"],
        progress=train.progress != "off",
        coordinate_scale_to_m=data.coordinate_scale_to_m,
    )
    print(f"stage=normalization states={len(normalization.fitted_state_ids)} elapsed_sec={time.perf_counter() - stage_started:.1f}", flush=True)
    normalization_path = train.output_root / "normalization_stats.json"
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    normalization.save(normalization_path)
    artifact_splits = {
        "train": splits["train"],
        "validation": eval_splits["validation"],
        "test": eval_splits["test"],
    }
    write_run_artifacts(
        train.output_root,
        config,
        artifact_splits,
        device=device,
        inventory_records=index.records,
        run_manifest={
            "archive": index.run_manifest,
            "input_mode": config.input_mode,
            "window_length": data.window_length,
            "train_stride": data.stride,
            "eval_stride": eval_stride,
            "observation_seed": data.seed,
            "normalization_stats": normalization_path.name,
        },
    )
    observation = ObservationConfig(
        point_count_min=data.virtual_point_count_min,
        point_count_max=data.virtual_point_count_max,
        noise_std_mm=data.noise_std_mm,
        missing_probability=data.missing_probability,
        input_mode=config.input_mode,
        region_grid_size_m=data.region_grid_size_m,
        min_points_per_region=data.min_points_per_region,
        coordinate_scale_to_m=data.coordinate_scale_to_m,
    )
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
    train_sampler = RotatingRealizationSampler(
        len(splits["train"]),
        realizations,
        seed=train.seed,
        strategy=data.realization_sampling,
    )
    loader = make_loader(splits["train"], observation, data.seed, normalization, train.batch_size,
                         realizations=realizations, sampler=train_sampler,
                         regional_target_config=config.regional_targets, loader_seed=train.seed,
                         **loader_kwargs)
    # Evaluate every configured realization with one deterministic pass. A
    # single fixed realization creates a periodic validation signal when the
    # training sampler rotates or mixes observation layouts.
    validation_realizations = data.validation_realizations or realizations
    validation_sampler = RotatingRealizationSampler(
        len(eval_splits["validation"]),
        validation_realizations,
        seed=train.seed + 1,
        shuffle=False,
        strategy="all",
    )
    validation_loader = make_loader(eval_splits["validation"], observation, data.seed, normalization, train.batch_size,
                                    realizations=validation_realizations, sampler=validation_sampler,
                                    regional_target_config=config.regional_targets, loader_seed=train.seed + 1,
                                    **loader_kwargs)
    test_sampler = RotatingRealizationSampler(
        len(eval_splits["test"]),
        validation_realizations,
        seed=train.seed + 2,
        shuffle=False,
        strategy="all",
    )
    test_loader = make_loader(eval_splits["test"], observation, data.seed, normalization, train.batch_size,
                              realizations=validation_realizations, sampler=test_sampler,
                              regional_target_config=config.regional_targets, loader_seed=train.seed + 2,
                              **loader_kwargs)
    if realizations > 1:
        per_worker_mib = distribution["train"]["unique_states"] * realizations * 25 / 1024
        print(
            f"observation_realizations={realizations} training_sampling={data.realization_sampling} "
            f"validation_realizations={validation_realizations} evaluation_sampling=all "
            f"estimated_cache_mib={per_worker_mib * max(train.num_workers, 1):.0f}",
            flush=True,
        )
    print(f"states={len(index.records)} windows={len(refs)} train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}", flush=True)
    print(f"normalization_stats={normalization_path}", flush=True)
    print(f"model={config.model.name} input_mode={config.input_mode} params={config.model.kwargs}", flush=True)
    print(f"data_loader=num_workers:{train.num_workers} prefetch_factor:{train.prefetch_factor} persistent_workers:{train.persistent_workers} pin_memory:{train.pin_memory}", flush=True)
    print("stage=model_initialization", flush=True)
    model = build_model(config.model.name, **config.model.kwargs)
    if config.model.name == "xgboost":
        train_xgboost(
            config,
            loader,
            validation_loader,
            test_loader,
            normalization,
            splits,
            eval_splits,
            distribution,
            index,
            refs,
            eval_stride,
            run_clock,
        )
        return
    model = model.to(device)
    if materialize_lazy_model(model, loader, device):
        print("stage=lazy_parameters_materialized", flush=True)
    parameter_counts = count_parameters(model)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "unavailable"
    print(
        f"device={device} gpu={gpu_name} total_parameters={parameter_counts['total_parameters']} "
        f"trainable_parameters={parameter_counts['trainable_parameters']}",
        flush=True,
    )
    optimizer = build_adamw(
        model,
        learning_rate=train.learning_rate,
        weight_decay=train.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        name=train.lr_scheduler,
        epochs=train.epochs,
        factor=train.lr_factor,
        patience=train.lr_patience,
        min_learning_rate=train.min_learning_rate,
        threshold=1e-3,
    )
    weights_dir = train.output_root / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)
    epoch_times: list[float] = []
    history: list[dict[str, object]] = []
    best_metric = BestMetricTracker()
    stopped_epoch = 0
    history_path = train.output_root / "history.json"
    checkpoint_path = weights_dir / f"{config.model.name}_best.pt"
    selection_tasks = tuple(train.selection_tasks)
    for epoch in range(train.epochs):
        train_sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        loss = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            desc=f"train_batches epoch={epoch + 1}",
            progress=progress_options,
            task_weights=train.task_loss_weights,
        )
        epoch_elapsed = time.perf_counter() - epoch_started
        epoch_times.append(epoch_elapsed)
        validation_loss, validation_predictions = evaluate_torch(
            model,
            validation_loader,
            device,
            desc="validation_batches",
            progress=progress_options,
            task_weights=train.task_loss_weights,
        )
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
        improved = best_metric.update(epoch + 1, selection_score)
        if improved:
            save_torch_checkpoint(checkpoint_path, model, optimizer, config, normalization, epoch + 1)
        step_scheduler(scheduler, selection_score)
        print(
            f"epoch={epoch + 1} train_loss={loss:.6f} validation_loss={validation_loss:.6f} "
            f"validation_macro_f1={selection_score:.6f} "
            f"selection_score={selection_score:.6f} elapsed_sec={epoch_elapsed:.1f}",
            flush=True,
        )
        if best_metric.should_stop(train.early_stopping_patience):
            stopped_epoch = epoch + 1
            print(
                f"early_stopping epoch={stopped_epoch} best_epoch={best_metric.best_epoch} "
                f"best_value={best_metric.best_score:.6f} patience={train.early_stopping_patience}",
                flush=True,
            )
            break
    restore_model_checkpoint(checkpoint_path, model, device=device)
    validation_loss, validation_predictions = evaluate_torch(
        model,
        validation_loader,
        device,
        desc="validation_batches",
        progress=progress_options,
        task_weights=train.task_loss_weights,
    )
    test_loss, test_predictions = evaluate_torch(
        model,
        test_loader,
        device,
        desc="test_batches",
        progress=progress_options,
        task_weights=train.task_loss_weights,
    )
    history_plot_path = train.output_root / "training_history.png"
    plot_training_history(history, history_plot_path)
    test_metrics = pack_metrics(test_predictions, test_loss)
    evaluation_plot_path = train.output_root / "final_test_evaluation.png"
    plot_final_evaluation(test_metrics, evaluation_plot_path)
    report = {
        "task": config.task,
        "run_status": "SMOKE_ONLY" if config.smoke.enabled else "FORMAL",
        "run_kind": "SMOKE_ONLY" if config.smoke.enabled else "EXPERIMENT",
        "model_name": config.model.name,
        "model_params": config.model.kwargs,
        "target_schema": TASK_LABEL_SCHEMA,
        "task_names": list(TASK_LABEL_NAMES),
        "class_names": list(TASK_LABEL_LEVEL_NAMES),
        "data_root": str(data.data_root),
        "window_length": data.window_length,
        "batch_size": train.batch_size,
        "epochs": train.epochs,
        "optimization": {
            "learning_rate": train.learning_rate,
            "weight_decay": train.weight_decay,
            "lr_scheduler": train.lr_scheduler,
            "lr_factor": train.lr_factor,
            "lr_patience": train.lr_patience,
            "min_learning_rate": train.min_learning_rate,
            "task_loss_weights": train.task_loss_weights,
        },
        "states": len(index.records),
        "windows": len(refs),
        "train_windows": len(splits["train"]),
        "data": describe_run(
            data,
            config.input_mode,
            splits,
            eval_splits,
            eval_stride,
            realizations,
            validation_realizations,
            distribution,
            getattr(loader.dataset, "region_grid", None),
        ),
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
        "run_metadata": str(config.train.output_root / "run_metadata.json"),
        "resolved_config": str(config.train.output_root / "resolved_config.json"),
        "split_manifest": str(config.train.output_root / "split_manifest.json"),
        "run_requirements": str(config.train.output_root / "requirements_frozen.txt"),
        "metrics": {
            "validation": pack_metrics(validation_predictions, validation_loss),
            "test": test_metrics,
        },
        "selection": {
            "tasks": list(selection_tasks),
            "metric": "mean_macro_f1",
            "best_epoch": best_metric.best_epoch,
            "best_value": best_metric.best_score,
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
