"""Verify that realization rotation is deterministic and actually rotates.

Checks:
  (b) the sampler rotates the realization per epoch, keeps the base window order
      reproducible per epoch, and one (state, realization) is stable
  (c) two independent runs with the same seed produce identical batches, both
      with num_workers=0 and across worker processes
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from qianhai7_training.cli.train import make_loader
from qianhai7_training.config import load_experiment_config
from qianhai7_training.data.dataset import QianhaiWindowDataset
from qianhai7_training.data.manifest import build_archive_index
from qianhai7_training.data.normalization import fit_normalization_stats
from qianhai7_training.data.observation import ObservationConfig
from qianhai7_training.data.sampler import RotatingRealizationSampler
from qianhai7_training.data.windows import make_window_refs, split_by_state


def batch_digest(batch) -> str:
    hasher = hashlib.sha256()
    for name, tensor in (("values", batch.values), ("valid_mask", batch.valid_mask),
                         ("point_features", batch.point_features)):
        array = np.ascontiguousarray(tensor.numpy())
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    for key in sorted(batch.targets):
        array = np.ascontiguousarray(batch.targets[key].numpy())
        hasher.update(key.encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def epoch_hashes(loader, sampler, epochs: int, batches: int) -> list[str]:
    digests: list[str] = []
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        per_epoch = []
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            per_epoch.append(batch_digest(batch))
        digests.append(hashlib.sha256("".join(per_epoch).encode()).hexdigest())
    return digests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    config = load_experiment_config(str(_PROJECT_ROOT / "configs" / "qianhai7_v12_baseline.yaml"))
    data, train = config.data, config.train
    index = build_archive_index(data.data_root)
    refs = make_window_refs(index, data.window_length, data.stride)
    splits = split_by_state(refs, data.split_seed, data.train_fraction, data.validation_fraction,
                            stratify=data.split_stratify)
    normalization = fit_normalization_stats(splits["train"])
    observation = ObservationConfig(data.virtual_point_count_min, data.virtual_point_count_max,
                                    data.noise_std_mm, data.missing_probability)
    windows = len(splits["train"])
    realizations = 4

    # (b) rotation mechanics
    sampler = RotatingRealizationSampler(windows, realizations, seed=data.split_seed)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(0)
    repeated = list(sampler)
    print(f"[b] epoch0 realization blocks={sorted({i // windows for i in first})} "
          f"epoch1 realization blocks={sorted({i // windows for i in second})}")
    print(f"[b] epoch0 order reproducible after set_epoch round-trip: {first == repeated}")
    print(f"[b] same window position in both epochs: "
          f"{sorted(i % windows for i in first) == sorted(i % windows for i in second)}")
    print(f"[b] permutation differs between epochs: "
          f"{[i % windows for i in first] != [i % windows for i in second]}")

    dataset = QianhaiWindowDataset(splits["train"], observation, data.seed, normalization, realizations=realizations)
    flat = first[0]
    print(f"[b] same flat index twice identical: {np.array_equal(dataset[flat]['values'], dataset[flat]['values'])}")
    a = dataset[flat]
    b = dataset[flat + windows]
    print(f"[b] same window, different realization differs: {not np.array_equal(a['values'], b['values'])}")
    print(f"[b] realizations present: {sorted({dataset[i]['metadata']['observation_realization_id'] for i in range(0, len(dataset), max(windows // 20, 1))})}")

    # (c) reproducibility
    def build(seed_offset: int = 0):
        local_sampler = RotatingRealizationSampler(windows, realizations, seed=data.split_seed)
        loader = make_loader(splits["train"], observation, data.seed, normalization, train.batch_size,
                             realizations=realizations, sampler=local_sampler,
                             num_workers=args.workers, persistent_workers=args.workers > 0,
                             pin_memory=False)
        return loader, local_sampler

    loader_a, sampler_a = build()
    hashes_a = epoch_hashes(loader_a, sampler_a, args.epochs, args.batches)
    loader_b, sampler_b = build()
    hashes_b = epoch_hashes(loader_b, sampler_b, args.epochs, args.batches)
    print(f"[c] workers={args.workers} epoch hashes={[h[:12] for h in hashes_a]}")
    print(f"[c] two independent runs identical: {hashes_a == hashes_b}")
    print(f"[c] epochs differ from each other (rotation is real): {len(set(hashes_a)) == len(hashes_a)}")

    # realizations=1 must still reshuffle each epoch, but reproducibly across runs.
    def build_single():
        local_sampler = RotatingRealizationSampler(windows, 1, seed=data.split_seed)
        loader = make_loader(splits["train"], observation, data.seed, normalization, train.batch_size,
                             realizations=1, sampler=local_sampler,
                             num_workers=args.workers, persistent_workers=args.workers > 0,
                             pin_memory=False)
        return loader, local_sampler

    loader_s, sampler_s = build_single()
    single_a = epoch_hashes(loader_s, sampler_s, args.epochs, args.batches)
    loader_t, sampler_t = build_single()
    single_b = epoch_hashes(loader_t, sampler_t, args.epochs, args.batches)
    print(f"[c] realizations=1 reshuffles each epoch: {single_a[0] != single_a[1]}")
    print(f"[c] realizations=1 reproducible across runs: {single_a == single_b}")


if __name__ == "__main__":
    main()
