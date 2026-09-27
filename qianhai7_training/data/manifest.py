from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchiveRecord:
    path: Path
    sample_id: str
    structural_state_id: str
    state_family_id: str
    split_group: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ArchiveIndex:
    records: tuple[ArchiveRecord, ...]
    run_manifest: dict[str, Any]

    @property
    def structural_state_ids(self) -> tuple[str, ...]:
        return tuple(record.structural_state_id for record in self.records)


def build_archive_index(data_root: str | Path) -> ArchiveIndex:
    """Read manifests while keeping large HDF5 arrays lazy."""
    import h5py

    root = Path(data_root)
    run_manifest_path = root / "_run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    else:
        # Each HDF5 is self-describing through its embedded manifest_json. The
        # run-level manifest is useful for provenance, but is not required to
        # build training samples from a valid merged archive.
        run_manifest = {
            "schema_version": "inferred_from_embedded_manifests",
            "data_root": str(root),
            "run_manifest_missing": True,
        }
        warnings.warn(
            f"{run_manifest_path} is missing; using embedded HDF5 manifests. "
            "Dataset-level provenance fields will be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
    records: list[ArchiveRecord] = []
    for path in sorted(root.glob("qh7_state_*.h5")):
        with h5py.File(path, "r") as handle:
            raw = handle["manifest_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            manifest = json.loads(raw)
        sample = manifest["sample_index"]
        records.append(
            ArchiveRecord(
                path=path,
                sample_id=sample["sample_id"],
                structural_state_id=sample["structural_state_id"],
                state_family_id=sample["state_family_id"],
                split_group=sample.get("split_group", sample["structural_state_id"]),
                metadata=manifest,
            )
        )
    if not records:
        raise FileNotFoundError(f"No qh7_state_*.h5 files found under {root}")
    return ArchiveIndex(tuple(records), run_manifest)
