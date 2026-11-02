"""Reproducible runtime setup and auditable run artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def select_device(requested: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"train.device={requested!r} requested CUDA, but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device.index}")
    elif device.type != "cpu":
        raise ValueError("train.device must be auto, cpu, cuda, or cuda:N")
    return device


def configure_runtime(train: Any) -> torch.device:
    if train.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(train.seed)
    np.random.seed(train.seed)
    torch.manual_seed(train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train.seed)
    torch.use_deterministic_algorithms(train.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = train.deterministic
    torch.backends.cuda.matmul.allow_tf32 = not train.deterministic
    torch.backends.cudnn.allow_tf32 = not train.deterministic
    if train.cpu_threads is not None:
        torch.set_num_threads(train.cpu_threads)
    return select_device(train.device)


def loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(plain(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def source_digest() -> str:
    digest = hashlib.sha256()
    for folder in ("dl_fusion", "qianhai7_training_v2", "KW51_training"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_run_artifacts(
    output_root: Path,
    config: Any,
    splits: dict[str, list[Any]],
    *,
    device: torch.device,
    run_manifest: dict[str, Any],
    inventory_records: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Persist resolved config and exact sample/window assignments for migration.

    The data fingerprint covers indexed manifests and file sizes, not HDF5 array
    bytes. The explicit scope prevents confusing it with a full-content hash.
    """
    resolved = plain(config)
    resolved["model"] = {"name": config.model.name, "kwargs": config.model.kwargs}
    resolved.setdefault("train", {})["output_root"] = str(output_root.resolve())
    write_json(output_root / "resolved_config.json", resolved)
    config_sha256 = hashlib.sha256((output_root / "resolved_config.json").read_bytes()).hexdigest()
    membership = {}
    inventory = {}
    data_root = Path(config.data.data_root).resolve()

    def add_inventory(record: Any) -> None:
        path = Path(record.path).resolve()
        relative = path.relative_to(data_root).as_posix()
        inventory[relative] = {
            "size_bytes": path.stat().st_size,
            "manifest": record.metadata,
        }

    for role, items in splits.items():
        rows = []
        for item in items:
            record = getattr(item, "record", item)
            relative = Path(record.path).resolve().relative_to(data_root).as_posix()
            row = {"state_id": str(record.structural_state_id), "sample_id": record.sample_id, "file": relative}
            if hasattr(item, "start"):
                row.update(start=int(item.start), length=int(item.length))
            rows.append(row)
            add_inventory(record)
        membership[role] = rows
    if inventory_records is not None:
        for record in inventory_records:
            add_inventory(record)
    states = {role: {row["state_id"] for row in rows} for role, rows in membership.items()}
    roles = list(states)
    if any(states[left] & states[right] for i, left in enumerate(roles) for right in roles[i + 1:]):
        raise ValueError("structural states overlap across splits")
    write_json(output_root / "split_manifest.json", membership)
    try:
        frozen_packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        frozen_packages = f"pip freeze unavailable: {exc}\n"
    (output_root / "requirements_frozen.txt").write_text(frozen_packages, encoding="utf-8")
    accelerator_packages = {"torch", "torchvision", "torchaudio"}
    portable_lines = [
        line
        for line in frozen_packages.splitlines()
        if line.partition("==")[0].strip().lower() not in accelerator_packages
    ]
    portable_requirements = "\n".join(portable_lines) + "\n"
    (output_root / "requirements_non_torch.txt").write_text(
        portable_requirements, encoding="utf-8"
    )
    versions = {}
    for name in (
        "torch", "numpy", "h5py", "PyYAML", "scikit-learn", "matplotlib",
        "tqdm", "xgboost", "joblib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = {
        "run_kind": "SMOKE_ONLY" if config.smoke.enabled else "EXPERIMENT",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root.resolve()),
        "task": resolved.get("task", "qianhai_four_task"),
        "input_mode": resolved.get("input_mode"),
        "dataset_kind": getattr(config.data, "dataset_kind", "qianhai7"),
        "resolved_config_sha256": config_sha256,
        "frozen_requirements_sha256": hashlib.sha256(frozen_packages.encode("utf-8")).hexdigest(),
        "non_torch_requirements_sha256": hashlib.sha256(portable_requirements.encode("utf-8")).hexdigest(),
        "training_seed": config.train.seed,
        "data_seed": config.data.seed,
        "split_seed": config.data.split_seed,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "host": platform.node(),
        "packages": versions,
        "cuda_version": torch.version.cuda,
        "git_revision": revision,
        "source_sha256": source_digest(),
        "split_sha256": _digest(membership),
        "data_index_sha256": _digest(inventory),
        "data_fingerprint_scope": "indexed embedded manifests and file sizes; NOT full HDF5 contents",
        "indexed_file_count": len(inventory),
        "indexed_data_size_bytes": sum(item["size_bytes"] for item in inventory.values()),
        "run_manifest": run_manifest,
    }
    write_json(output_root / "run_metadata.json", metadata)
    return metadata
