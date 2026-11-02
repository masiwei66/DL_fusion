from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    # Allow both `python -m qianhai7_training_v2.cli.audit_splits` and direct IDE execution.
    sys.path.insert(0, str(_PROJECT_ROOT))
    __package__ = "qianhai7_training_v2.cli"

from ..config import load_experiment_config
from ..data.manifest import build_archive_index
from ..data.windows import make_window_refs, missing_train_families, split_by_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit state-group splits and label distributions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    config = load_experiment_config(args.config, data_root_override=args.data_root)
    index = build_archive_index(config.data.data_root)
    stride = config.data.eval_stride or config.data.stride
    refs = make_window_refs(index, config.data.window_length, stride)
    splits = split_by_state(refs, config.data.split_seed, config.data.train_fraction, config.data.validation_fraction,
                            stratify=config.data.split_stratify)
    print(f"records={len(index.records)} states={len(index.structural_state_ids)} windows={len(refs)} stride={stride} "
          f"stratify={config.data.split_stratify}")
    for name, items in splits.items():
        states = {item.record.structural_state_id for item in items}
        families = collections.Counter(item.record.state_family_id for item in items)
        print(f"[{name}] windows={len(items)} states={len(states)} state_families={len(families)}")
        print("  state_family_windows=" + ", ".join(f"{key}:{value}" for key, value in sorted(families.items())))
    missing = missing_train_families(splits)
    print(f"families_missing_from_train={len(missing)}" + (f" {missing}" if missing else ""))


if __name__ == "__main__":
    main()
