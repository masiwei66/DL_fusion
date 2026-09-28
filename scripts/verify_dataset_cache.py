"""Fingerprint the dataset output so a caching change can be proven transparent.

Run with `--save` before editing the dataset, then without it afterwards: the
digest must match byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qianhai7_training.config import load_experiment_config
from qianhai7_training.data.dataset import QianhaiWindowDataset
from qianhai7_training.data.manifest import build_archive_index
from qianhai7_training.data.normalization import fit_normalization_stats
from qianhai7_training.data.observation import ObservationConfig
from qianhai7_training.data.windows import make_window_refs, split_by_state

FINGERPRINT_KEYS = (
    "values", "valid_mask", "quality", "point_coords", "temperature_C", "delta_days",
    "target_state_family", "target_fine_state_family", "target_global_severity",
    "target_material_active", "target_settlement_type", "target_stiffness_reduction",
    "target_global_severity_value", "target_material_severity",
)


def digest(item: dict) -> str:
    hasher = hashlib.sha256()
    for key in FINGERPRINT_KEYS:
        array = np.ascontiguousarray(item[key])
        hasher.update(key.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    hasher.update(item["metadata"]["structural_state_id"].encode())
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args()

    config = load_experiment_config(str(Path(__file__).resolve().parents[1] / "configs" / "qianhai7_v12_baseline.yaml"))
    data = config.data
    index = build_archive_index(data.data_root)
    refs = make_window_refs(index, data.window_length, data.stride)
    splits = split_by_state(refs, data.split_seed, data.train_fraction, data.validation_fraction)
    normalization = fit_normalization_stats(splits["train"])
    observation = ObservationConfig(
        data.virtual_point_count_min, data.virtual_point_count_max, data.noise_std_mm, data.missing_probability
    )

    # One window per state, spread deterministically across the training split,
    # so every sample pays the cold HDF5 read a shuffled loader would pay.
    by_state: dict[str, list] = {}
    for ref in splits["train"]:
        by_state.setdefault(ref.record.structural_state_id, []).append(ref)
    states = sorted(by_state)
    step = max(1, len(states) // args.samples)
    picked = [by_state[states[i]][0] for i in range(0, len(states), step)][: args.samples]

    dataset = QianhaiWindowDataset(picked, observation, data.seed, normalization)
    started = time.perf_counter()
    digests = [digest(dataset[i]) for i in range(len(picked))]
    elapsed = time.perf_counter() - started

    print(f"states_total={len(states)} sampled={len(picked)}")
    print(f"unique_digests={len(set(digests))}")
    print(f"elapsed_sec={elapsed:.2f} per_sample_ms={elapsed / len(picked) * 1000:.1f}")
    print(f"combined={hashlib.sha256(''.join(digests).encode()).hexdigest()}")

    if args.save is not None:
        args.save.write_text(json.dumps(digests), encoding="utf-8")
        print(f"saved={args.save}")
    if args.compare is not None:
        expected = json.loads(args.compare.read_text(encoding="utf-8"))
        print("MATCH" if expected == digests else "MISMATCH")
        if expected != digests:
            for position, (a, b) in enumerate(zip(expected, digests)):
                if a != b:
                    print(f"  first difference at sample {position}: {picked[position].record.structural_state_id}")
                    break


if __name__ == "__main__":
    main()