from __future__ import annotations

import argparse
import collections

from ..data.manifest import build_archive_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    index = build_archive_index(args.data_root)
    counts = collections.Counter(record.state_family_id for record in index.records)
    print(f"files={len(index.records)}")
    print(f"unique_structural_states={len(index.structural_state_ids)}")
    print(f"schema_version={index.run_manifest.get('schema_version')}")
    print("state_family_counts=")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
