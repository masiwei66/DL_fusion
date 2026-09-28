from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_NAMES = (
    "xgboost",
    "resnet1d",
    "resnet50_1d",
    "lstm",
    "gru",
    "transformer",
    "gnn_lstm",
)


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    window_length: int = 6
    stride: int = 1
    virtual_point_count_min: int = 50
    virtual_point_count_max: int = 100
    noise_std_mm: float = 0.2
    missing_probability: float = 0.05
    seed: int = 42
    split_seed: int = 20260924
    train_fraction: float = 0.7
    validation_fraction: float = 0.15


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "mlp"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    output_root: Path = Path("runs")
    # DataLoader controls. num_workers is the number of CPU worker processes.
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("train.num_workers must be >= 0")
        if self.prefetch_factor < 1:
            raise ValueError("train.prefetch_factor must be >= 1")
        if self.num_workers == 0 and self.persistent_workers:
            object.__setattr__(self, "persistent_workers", False)


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and architecture parameters owned by the config file."""

    name: str = "mlp"
    params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.name not in ("mlp", *SUPPORTED_MODEL_NAMES):
            choices = ", ".join(("mlp", *SUPPORTED_MODEL_NAMES))
            raise ValueError(f"Unknown model {self.name!r}; choose one of: {choices}")

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params or {})


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig


def load_experiment_config(
    path: str | Path,
    *,
    data_root_override: str | Path | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
) -> ExperimentConfig:
    """Load the YAML experiment contract and apply explicit CLI overrides."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Loading YAML config requires PyYAML; install pyyaml") from exc

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw_data = dict(payload.get("data") or {})
    raw_train = dict(payload.get("train") or {})
    raw_model = dict(payload.get("model") or {})

    # Keep the existing train.model_name format valid while allowing the clearer
    # model.name/model.params format for new experiments.
    model_name = str(raw_model.pop("name", raw_train.pop("model_name", "mlp")))
    model_params = raw_model.pop("params", raw_model)
    if model_params is None:
        model_params = {}

    if data_root_override is not None:
        raw_data["data_root"] = str(data_root_override)
    if epochs_override is not None:
        raw_train["epochs"] = epochs_override
    if batch_size_override is not None:
        raw_train["batch_size"] = batch_size_override

    if "data_root" not in raw_data:
        raise ValueError(f"Config {path} is missing data.data_root")
    data = DataConfig(data_root=Path(raw_data.pop("data_root")), **raw_data)
    train = TrainConfig(**raw_train)
    model = ModelConfig(name=model_name, params=model_params)
    return ExperimentConfig(data=data, train=train, model=model)
