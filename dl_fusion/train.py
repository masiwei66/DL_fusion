"""Single config-driven entry point for Qianhai and KW51 experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_task(config_path: Path) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Loading YAML config requires PyYAML") from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    task = str(payload.get("task", "qianhai_four_task"))
    if task not in (
        "qianhai_four_task",
        "kw51_candidate_ranking",
        "kw51_fuzzy_response",
        "kw51_ad_rel_classification",
    ):
        raise ValueError(f"Unsupported task: {task!r}")
    return task


def _forwarded_args(args: argparse.Namespace) -> list[str]:
    forwarded = ["--config", str(args.config)]
    for flag, value in (
        ("--data-root", args.data_root),
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--progress", args.progress),
        ("--fold", args.fold),
    ):
        if value is not None:
            forwarded.extend((flag, str(value)))
    return forwarded


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a configured DL_fusion task")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qianhai7_v12_baseline.yaml",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--progress", choices=("auto", "bar", "log", "off"), default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run KW51 data and identifiability gates without starting training",
    )
    args = parser.parse_args(argv)
    task = _read_task(args.config)

    if task == "qianhai_four_task":
        if args.preflight_only:
            parser.error("--preflight-only is only supported by KW51 candidate ranking")
        if args.fold is not None:
            parser.error("--fold is only supported by KW51 ranking experiments")
        from qianhai7_training_v2.cli.train import main as run_qianhai

        run_qianhai(_forwarded_args(args))
        return

    from KW51_training.config import load_experiment_config

    config = load_experiment_config(
        args.config,
        data_root_override=args.data_root,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        progress_override=args.progress,
        fold_override=args.fold,
    )
    if task == "kw51_candidate_ranking":
        from KW51_training.training.kw51_runner import run_kw51_training

        result = run_kw51_training(config, preflight_only=args.preflight_only)
        if args.preflight_only and not result["can_start_formal_training"]:
            raise SystemExit(2)
        return

    if args.preflight_only:
        parser.error("--preflight-only is only supported by KW51 candidate ranking")

    fuzzy = config.fuzzy
    if fuzzy is None:
        raise ValueError(f"{task} requires a fuzzy config section")
    if task == "kw51_ad_rel_classification":
        if config.tasks is None or config.tasks.selected_task != "ad_rel_classification":
            raise ValueError("kw51_ad_rel_classification requires its task head to be enabled")
        if config.inference is None or config.inference.outputs != ("ad_rel_classification",):
            raise ValueError("classification config must select ad_rel_classification output")
        if config.checkpoint_selection is None or config.checkpoint_selection.metric not in (
            "f1_macro",
            "nll",
        ):
            raise ValueError(
                "classification checkpoint selection must use f1_macro or nll"
            )
    if fuzzy.observation_mode == "region_bag":
        if fuzzy.primary_target not in ("state", "ad_group"):
            raise ValueError("region_bag requires primary_target state or ad_group")
        from KW51_training.training.fuzzy_region_runner import run_fuzzy_region_training

        run_fuzzy_region_training(config)
    elif fuzzy.primary_target in ("state", "ad_group"):
        from KW51_training.training.fuzzy_state_runner import run_fuzzy_state_training

        run_fuzzy_state_training(config)
    else:
        from KW51_training.training.fuzzy_runner import run_fuzzy_training

        run_fuzzy_training(config)


if __name__ == "__main__":
    main()
