"""Run the region-protocol audit without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_experiment_config
from ..data.fem_insar import build_fem_archive_index, derive_ad_groups
from ..data.fuzzy_range import build_deck_geometry
from ..training.fuzzy_region_audit import (
    DEFAULT_PROTOCOLS,
    run_fuzzy_region_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--protocols",
        default="",
        help="Comma-separated protocol names, e.g. 4_long,8_long.",
    )
    parser.add_argument(
        "--budgets",
        default="",
        help="Comma-separated point budgets, e.g. 40,160,640.",
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if config.fuzzy is None:
        raise ValueError("region audit requires a fuzzy config")
    data = config.data
    index = build_fem_archive_index(data.data_root)
    geometry = build_deck_geometry(index.records[0], data.los_vector)
    group_map = derive_ad_groups(
        index,
        parameter=data.ad_group_parameter,
        expected_count=data.ad_group_count,
    )
    selected_names = [
        name.strip() for name in str(args.protocols).split(",") if name.strip()
    ]
    if selected_names:
        by_name = {protocol.name: protocol for protocol in DEFAULT_PROTOCOLS}
        missing = [name for name in selected_names if name not in by_name]
        if missing:
            raise ValueError(f"Unknown protocols: {missing}")
        protocols = [by_name[name] for name in selected_names]
    else:
        protocols = None
    budgets = [
        int(value.strip())
        for value in str(args.budgets).split(",")
        if value.strip()
    ]
    audit = run_fuzzy_region_audit(
        index,
        geometry,
        group_map,
        config.fuzzy,
        seed=data.seed,
        split_seed=data.split_seed,
        train_fraction=data.train_fraction,
        validation_fraction=data.validation_fraction,
        split_mode=data.split_mode,
        fold_indices=range(int(config.fuzzy.region_audit_folds)),
        protocols=protocols,
        point_budgets=budgets or None,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(config.train.output_root.parent)
        / f"{Path(config.train.output_root).name}_region_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"status={audit['status']} configured_budget={audit['configured_budget']} "
        f"budgets={audit['point_budgets']}"
    )
    for name, entry in audit["protocols"].items():
        summary = entry["summary"]
        print(
            f"{name} status={entry['status']} "
            f"top1={summary['state_macro_top1']:.3f} "
            f"mrr={summary['state_macro_mrr']:.3f} "
            f"snr={summary['snr_values']:.3f}"
        )
    print(f"region_audit={output}")


if __name__ == "__main__":
    main()
