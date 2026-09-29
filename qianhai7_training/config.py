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
    eval_stride: int | None = None
    virtual_point_count_min: int = 50
    virtual_point_count_max: int = 100
    noise_std_mm: float = 0.2
    missing_probability: float = 0.05
    seed: int = 42
    split_seed: int = 20260924
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    # Stratify the state-level split by state_family_id to preserve mechanism coverage.
    split_stratify: bool = True
    # Number of cached virtual-observation realizations rotated one per epoch.
    # 1 means a single deterministic realization (no augmentation).
    observation_realizations: int = 1
    noise_scan_std_mm: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)
    missing_scan_probability: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)


@dataclass(frozen=True)
class RegionalTargetConfig:
    """Engineering targets derived from the immutable FEM state parameters."""

    target_schema: str = "regional_anomaly_v1"
    target_version: str = "v1"
    label_mode: str = "engineering_anomaly"
    material_scale_threshold: float = 0.90
    bearing_scale_threshold: float = 0.90
    settlement_ratio_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.label_mode not in ("physical_active", "engineering_anomaly"):
            raise ValueError("regional_targets.label_mode must be physical_active or engineering_anomaly")
        for name in ("material_scale_threshold", "bearing_scale_threshold"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"regional_targets.{name} must be in (0, 1]")
        if float(self.settlement_ratio_threshold) < 0.0:
            raise ValueError("regional_targets.settlement_ratio_threshold must be >= 0")


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
    # Progress output for every training/evaluation loop. "auto" shows a tqdm bar
    # on a terminal and falls back to one log line per log_every_batches when the
    # output is redirected. Other modes: "bar", "log", "off".
    progress: str = "auto"
    # Log-mode frequency. Capped so short loops still emit about ten lines.
    log_every_batches: int = 100
    # Tasks averaged into the best-checkpoint score. Tasks differ by orders of
    # magnitude in difficulty, so the default is the primary task alone.
    selection_tasks: tuple[str, ...] = ("deck_material", "pier_settlement", "pier_material", "bearing_stiffness")
    # Stop once the selection score has not improved for this many epochs; 0 disables.
    early_stopping_patience: int = 10

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("train.num_workers must be >= 0")
        if self.prefetch_factor < 1:
            raise ValueError("train.prefetch_factor must be >= 1")
        if self.log_every_batches < 0:
            raise ValueError("train.log_every_batches must be >= 0")
        if self.progress not in ("auto", "bar", "log", "off"):
            raise ValueError("train.progress must be one of: auto, bar, log, off")
        if self.early_stopping_patience < 0:
            raise ValueError("train.early_stopping_patience must be >= 0")
        if not self.selection_tasks:
            raise ValueError("train.selection_tasks must name at least one task")
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
        if self.name == "xgboost":
            raise ValueError("xgboost is not supported by regional_anomaly_v1")

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params or {})


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig
    regional_targets: RegionalTargetConfig = RegionalTargetConfig()


def load_experiment_config(
    path: str | Path,
    *,
    data_root_override: str | Path | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    progress_override: str | None = None,
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
    raw_targets = dict(payload.get("regional_targets") or {})

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
    if progress_override is not None:
        raw_train["progress"] = progress_override

    if "data_root" not in raw_data:
        raise ValueError(f"Config {path} is missing data.data_root")
    if "output_root" in raw_train:
        raw_train["output_root"] = Path(raw_train["output_root"])
    data = DataConfig(data_root=Path(raw_data.pop("data_root")), **raw_data)
    train = TrainConfig(**raw_train)
    model = ModelConfig(name=model_name, params=model_params)
    regional_targets = RegionalTargetConfig(**raw_targets)
    return ExperimentConfig(data=data, train=train, model=model, regional_targets=regional_targets)
