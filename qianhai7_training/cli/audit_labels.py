"""Audit four-task ordinal label counts by state split.

The audit uses the same state-level split and target-generation function as
training, while counting each structural state once at its final acquisition.
Window multiplicity and observation realizations are intentionally excluded:
labels describe the structural state, not an individual noisy observation.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(_PROJECT_ROOT))
    __package__ = "qianhai7_training.cli"

from ..config import load_experiment_config
from ..data.hdf5_source import HDF5StateSource
from ..data.manifest import build_archive_index
from ..data.windows import make_window_refs, split_by_state
from ..task_labels import TASK_LABEL_LEVEL_NAMES, TASK_LABEL_NAMES


# Edit these values directly before running this file.
CONFIG_PATH = _PROJECT_ROOT / "configs" / "qianhai7_v12_baseline.yaml"
DATA_ROOT_OVERRIDE: Path | None = None
AUDIT_STRIDE = 6


def _state_refs(refs):
    """Keep one deterministic reference per structural state."""
    selected = {}
    for ref in refs:
        state_id = ref.record.structural_state_id
        current = selected.get(state_id)
        if current is None or ref.target_index > current.target_index:
            selected[state_id] = ref
    return list(selected.values())


def _audit_split(refs):
    counts = {
        task: np.zeros(len(TASK_LABEL_LEVEL_NAMES), dtype=np.int64)
        for task in TASK_LABEL_NAMES
    }
    states = _state_refs(refs)
    for ref in states:
        arrays = HDF5StateSource(ref.record.path).read()
        labels = np.asarray(arrays["task_level_by_acquisition"], dtype=np.int64)
        if labels.ndim != 2 or labels.shape[1] != len(TASK_LABEL_NAMES):
            raise ValueError(f"invalid task label matrix in {ref.record.path}: {labels.shape}")
        if not len(labels):
            raise ValueError(f"empty task label matrix in {ref.record.path}")
        for column, task in enumerate(TASK_LABEL_NAMES):
            counts[task] += np.bincount(
                [int(labels[-1, column])], minlength=len(TASK_LABEL_LEVEL_NAMES)
            )
    return len(states), counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final regional labels by state split.")
    parser.add_argument(
        "--data-root", type=Path, default=DATA_ROOT_OVERRIDE,
        help="dataset directory; overrides the YAML data_root",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help="experiment YAML configuration",
    )
    parser.add_argument(
        "--stride", type=int, default=AUDIT_STRIDE,
        help="window stride used to choose final-state references",
    )
    args = parser.parse_args()
    data_root_override = args.data_root
    config = load_experiment_config(args.config, data_root_override=data_root_override)
    index = build_archive_index(config.data.data_root)
    stride = args.stride or config.data.eval_stride or config.data.stride
    refs = make_window_refs(index, config.data.window_length, stride)
    splits = split_by_state(
        refs,
        config.data.split_seed,
        config.data.train_fraction,
        config.data.validation_fraction,
        stratify=config.data.split_stratify,
    )

    print(
        f"records={len(index.records)} states={len(index.structural_state_ids)} "
        f"window_length={config.data.window_length} stride={stride} "
        f"task_label_schema=qianhai_four_task_ordinal_v1"
    )
    print("counting=one_final_acquisition_per_structural_state")
    for split_name in ("train", "validation", "test"):
        state_count, counts = _audit_split(splits[split_name])
        print(f"[{split_name}] states={state_count}")
        for task in TASK_LABEL_NAMES:
            values = counts[task]
            ratio = values / state_count if state_count else np.zeros(len(TASK_LABEL_LEVEL_NAMES))
            print(f"  {task}:")
            for level, count, fraction in zip(TASK_LABEL_LEVEL_NAMES, values, ratio):
                print(f"    {level}: count={int(count)} ratio={fraction:.6f}")


if __name__ == "__main__":
    main()
