"""CLI for the read-only real InSAR density audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.real_insar_density import audit_real_insar_density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="E:/working/KW51_data/InSAR_DATA/KW51data",
        help="Directory containing KW51 InSAR CSV files.",
    )
    parser.add_argument(
        "--output",
        default="KW51_training/runs/real_insar_density_audit.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()
    paths = sorted(Path(args.data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files under {args.data_dir}")
    audit = audit_real_insar_density(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"tracks={audit['track_count']} total_points={audit['total_points_across_tracks']} "
        f"per_track={audit['points_per_track']} "
        f"upper_bound_per_region={audit['upper_bound_points_per_region_8']:.2f}"
    )
    print(f"density_audit={output}")


if __name__ == "__main__":
    main()
