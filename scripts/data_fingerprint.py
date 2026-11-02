from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(root: Path, pattern: str) -> dict[str, object]:
    root = root.resolve()
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {root}")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    inventory = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "root": str(root),
        "pattern": pattern,
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "inventory_sha256": hashlib.sha256(inventory.encode("utf-8")).hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a content-level SHA-256 manifest for HDF5 data")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--output", type=Path, default=Path("data_fingerprint.json"))
    args = parser.parse_args()
    report = fingerprint(args.data_root, args.pattern)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"files={report['file_count']} bytes={report['total_size_bytes']} "
        f"inventory_sha256={report['inventory_sha256']} output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
