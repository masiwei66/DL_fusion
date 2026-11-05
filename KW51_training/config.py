from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dl_fusion.experiment import SmokeConfig, model_parameters, validate_training_settings

SUPPORTED_MODEL_NAMES = (
    "resnet1d",
    "resnet50_1d",
    "lstm",
    "gru",
    "transformer",
    "gnn_lstm",
)

KW51_TASK_NAMES = ("candidate_compatibility", "ad_rel_classification")


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    # Dataset adapter selected by the experiment config. The original
    # qianhai7 path remains the default; KW51 uses its FEM-to-InSAR
    # source-domain candidate-ranking adapter inside this checkout.
    dataset_kind: str = "qianhai7"
    canonical_root: Path | None = None
    source_root: Path | None = None
    split_manifest: Path | None = None
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
    # Number of virtual-observation realizations available to the dataset.
    # 1 means a single deterministic realization (no augmentation).
    observation_realizations: int = 1
    # Training realization assignment. balanced_random mixes all realizations
    # within every epoch; rotate is retained only for historical comparisons.
    realization_sampling: str = "balanced_random"
    # Validation/test realization count. None means all training realizations.
    validation_realizations: int | None = None
    # KW51 creates independent query/candidate windows from the two official
    # train events. These fields are ignored by the original qianhai7 path.
    pair_realizations: int = 1
    query_noise_std_fraction: float = 0.0
    query_missing_probability: float = 0.0
    negative_candidates_per_window: int = 1
    # Legacy exact-state candidate-pool setting. Ad-group ranking always uses
    # the training-state gallery constructed by the KW51 runner.
    candidate_pool_scope: str = "all_states"
    noise_scan_std_mm: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)
    missing_scan_probability: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    # Shared observation adapter: sparse FEM observations can be consumed as
    # points or pooled into fixed metric square cells. Legacy footprint fields
    # remain accepted by the parser but no longer affect this adapter.
    input_mode: str = "region"
    region_grid_size_m: float = 10.0
    coordinate_scale_to_m: float = 1.0
    sparse_point_count: int = 80
    min_points_per_region: int = 1
    los_vector: tuple[float, float, float] = (-0.534, -0.105, 0.839)
    reference_mode: str = "region_mean"
    unit_scale: float = 1000.0
    point_bias_std_mm: float = 0.5
    block_dropout_probability: float = 0.10
    block_dropout_fraction: float = 0.15
    min_quality: float = 0.05
    normalization_max_records: int = 512
    # KW51 candidate resolution. "exact_state" preserves the original 25-state
    # ranking; "ad_group" ranks the five identifiable Ad_rel groups instead.
    ranking_target: str = "exact_state"
    ad_group_parameter: str = "Ad_rel"
    ad_group_count: int = 0
    # Training-query exclusion policy for ad-group references. Evaluation
    # queries use the same frozen gallery built from training states.
    gallery_exclusion: str = "query_trajectory"
    # Candidate galleries must be split deterministically by state. The
    # ad-group stratified mode guarantees every group appears in validation and
    # test (with five states per group this yields a 15/5/5 state split).
    split_mode: str = "random"
    # Deterministic state-level fold for ad_group_stratified CV. Fold k uses
    # one state per Ad group as test and the next as validation; across
    # folds 0..G-1 every state is test exactly once.
    split_fold: int | None = None
    # KW51 archives repeat the same environment trajectory keys across states.
    # Keep candidate-gallery and query environments disjoint, and hold separate
    # query keys for training, validation and final testing.
    gallery_trajectory_fraction: float = 0.60
    train_query_trajectory_fraction: float = 0.20
    validation_query_trajectory_fraction: float = 0.10
    test_query_trajectory_fraction: float = 0.10
    trajectory_split_seed: int = 20261015
    normalization_scope: str = "global"
    normalization_records_per_state: int = 0
    time_feature_mode: str = "raw"
    learnability_audit: bool = False
    audit_abort_on_fail: bool = False
    audit_min_oracle_group_accuracy: float = 0.9
    audit_min_signal_to_noise: float = 0.5
    audit_states_per_group: int = 5
    audit_trajectories_per_state: int = 2
    audit_state_cv_folds: int = 0
    loss_collapse_patience: int = 0
    loss_collapse_tolerance: float = 1e-3
    # Explicitly mark an experiment as diagnostic-only. A failed
    # identifiability audit also forces this flag, so runs outside the
    # identifiable regime cannot be mistaken for formal results.
    diagnostic_only: bool = False
    require_complete_archive: bool = True
    expected_samples: int = 2500
    expected_structural_states: int = 25

    def __post_init__(self) -> None:
        if self.dataset_kind not in ("qianhai7", "kw51"):
            raise ValueError("data.dataset_kind must be qianhai7 or kw51")
        if self.window_length < 1 or self.stride < 1:
            raise ValueError("data.window_length and data.stride must be >= 1")
        if self.eval_stride is not None and self.eval_stride < 1:
            raise ValueError("data.eval_stride must be >= 1 when set")
        if self.input_mode not in ("node", "region"):
            raise ValueError("data.input_mode must be node or region")
        if self.region_grid_size_m <= 0:
            raise ValueError("data.region_grid_size_m must be positive")
        if self.coordinate_scale_to_m <= 0:
            raise ValueError("data.coordinate_scale_to_m must be positive")
        if self.sparse_point_count < 1:
            raise ValueError("data.sparse_point_count must be >= 1")
        if self.min_points_per_region < 1:
            raise ValueError("data.min_points_per_region must be >= 1")
        if self.pair_realizations < 1:
            raise ValueError("data.pair_realizations must be >= 1")
        if self.negative_candidates_per_window < 1:
            raise ValueError("data.negative_candidates_per_window must be >= 1")
        if self.candidate_pool_scope not in ("split", "all_states"):
            raise ValueError("data.candidate_pool_scope must be split or all_states")
        if not 0.0 <= float(self.query_noise_std_fraction):
            raise ValueError("data.query_noise_std_fraction must be >= 0")
        if not 0.0 <= float(self.query_missing_probability) < 1.0:
            raise ValueError("data.query_missing_probability must be in [0, 1)")
        if self.observation_realizations < 1:
            raise ValueError("data.observation_realizations must be >= 1")
        if self.realization_sampling not in ("balanced_random", "rotate", "all"):
            raise ValueError(
                "data.realization_sampling must be one of: balanced_random, rotate, all"
            )
        if self.validation_realizations is not None and self.validation_realizations < 1:
            raise ValueError("data.validation_realizations must be >= 1 when provided")
        if self.reference_mode not in ("none", "region_mean", "first_region"):
            raise ValueError("data.reference_mode must be none, region_mean or first_region")
        if self.normalization_max_records < 0:
            raise ValueError("data.normalization_max_records must be >= 0")
        if self.expected_samples < 0 or self.expected_structural_states < 0:
            raise ValueError("expected archive sizes must be >= 0")
        if self.ranking_target not in ("exact_state", "ad_group"):
            raise ValueError("data.ranking_target must be exact_state or ad_group")
        if self.ranking_target == "ad_group" and self.dataset_kind != "kw51":
            raise ValueError("data.ranking_target=ad_group requires dataset_kind=kw51")
        if not str(self.ad_group_parameter).strip():
            raise ValueError("data.ad_group_parameter must not be empty")
        if self.ad_group_count < 0:
            raise ValueError("data.ad_group_count must be >= 0")
        if self.ad_group_count == 1:
            raise ValueError("data.ad_group_count must be 0 (infer) or at least 2")
        if self.gallery_exclusion not in ("query_trajectory", "query_state"):
            raise ValueError(
                "data.gallery_exclusion must be query_trajectory or query_state"
            )
        if self.gallery_exclusion == "query_state" and self.ranking_target != "ad_group":
            raise ValueError(
                "data.gallery_exclusion=query_state requires ranking_target=ad_group"
            )
        if self.split_mode not in ("random", "ad_group_stratified"):
            raise ValueError("data.split_mode must be random or ad_group_stratified")
        if self.split_mode == "ad_group_stratified" and self.ranking_target != "ad_group":
            raise ValueError(
                "data.split_mode=ad_group_stratified requires ranking_target=ad_group"
            )
        if self.split_fold is not None:
            if self.split_fold < 0:
                raise ValueError("data.split_fold must be >= 0")
            if self.split_mode != "ad_group_stratified":
                raise ValueError(
                    "data.split_fold requires split_mode=ad_group_stratified"
                )
        trajectory_fractions = (
            self.gallery_trajectory_fraction,
            self.train_query_trajectory_fraction,
            self.validation_query_trajectory_fraction,
            self.test_query_trajectory_fraction,
        )
        if any(not 0.0 < float(value) < 1.0 for value in trajectory_fractions):
            raise ValueError("KW51 trajectory split fractions must be in (0, 1)")
        if abs(sum(float(value) for value in trajectory_fractions) - 1.0) > 1.0e-8:
            raise ValueError("KW51 trajectory split fractions must sum to 1")
        if self.trajectory_split_seed < 0:
            raise ValueError("data.trajectory_split_seed must be >= 0")
        if self.normalization_scope not in ("global", "per_region_offset"):
            raise ValueError(
                "data.normalization_scope must be global or per_region_offset"
            )
        if self.normalization_records_per_state < 0:
            raise ValueError("data.normalization_records_per_state must be >= 0")
        if self.time_feature_mode not in ("raw", "standardized", "disabled"):
            raise ValueError(
                "data.time_feature_mode must be one of: raw, standardized, disabled"
            )
        if not 0.0 <= float(self.audit_min_oracle_group_accuracy) <= 1.0:
            raise ValueError("data.audit_min_oracle_group_accuracy must be in [0, 1]")
        if self.audit_min_signal_to_noise < 0.0:
            raise ValueError("data.audit_min_signal_to_noise must be >= 0")
        if self.audit_states_per_group < 1 or self.audit_trajectories_per_state < 1:
            raise ValueError(
                "data.audit_states_per_group and audit_trajectories_per_state must be >= 1"
            )
        if self.audit_state_cv_folds < 0:
            raise ValueError("data.audit_state_cv_folds must be >= 0")
        if self.audit_state_cv_folds and (
            self.ranking_target != "ad_group" or self.split_mode != "ad_group_stratified"
        ):
            raise ValueError(
                "data.audit_state_cv_folds requires ad_group ranking and "
                "ad_group_stratified splits"
            )
        if self.loss_collapse_patience < 0:
            raise ValueError("data.loss_collapse_patience must be >= 0")
        if self.loss_collapse_tolerance < 0.0:
            raise ValueError("data.loss_collapse_tolerance must be >= 0")


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
    # Validation-score driven decay is less sensitive than a fixed epoch
    # schedule when observations are stochastic. Set to "none" for a legacy
    # fixed-learning-rate comparison.
    lr_scheduler: str = "plateau"
    lr_factor: float = 0.5
    lr_patience: int = 3
    min_learning_rate: float = 1e-5
    device: str = "auto"
    seed: int = 42
    deterministic: bool = False
    cpu_threads: int | None = None
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
    # Optional relative task weights. Missing tasks default to one. Weights are
    # normalized to keep the all-ones loss numerically identical to four CE sum.
    task_loss_weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        validate_training_settings(self)
        if self.lr_scheduler not in ("none", "plateau", "cosine"):
            raise ValueError("train.lr_scheduler must be one of: none, plateau, cosine")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError("train.lr_factor must be in (0, 1)")
        if self.lr_patience < 0:
            raise ValueError("train.lr_patience must be >= 0")
        if self.min_learning_rate < 0.0:
            raise ValueError("train.min_learning_rate must be >= 0")
        if self.task_loss_weights is not None:
            allowed_tasks = {
                "deck_material",
                "pier_material",
                "pier_settlement",
                "bearing_stiffness",
            }
            unknown = set(self.task_loss_weights) - allowed_tasks
            if unknown:
                raise ValueError(f"train.task_loss_weights contains unknown tasks: {sorted(unknown)}")
            if any(float(value) <= 0.0 for value in self.task_loss_weights.values()):
                raise ValueError("train.task_loss_weights values must be > 0")
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

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params or {})


@dataclass(frozen=True)
class TaskHeadConfig:
    enabled: bool = False
    loss: str = "listwise"
    loss_weight: float = 1.0
    target: str | None = None
    class_values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.loss_weight <= 0.0:
            raise ValueError("task loss_weight must be > 0")
        if self.class_values and any(
            right <= left for left, right in zip(self.class_values, self.class_values[1:])
        ):
            raise ValueError("task class_values must be strictly increasing")


@dataclass(frozen=True)
class TasksConfig:
    candidate_compatibility: TaskHeadConfig = TaskHeadConfig(
        enabled=True, loss="listwise"
    )
    ad_rel_classification: TaskHeadConfig = TaskHeadConfig(
        enabled=False,
        loss="cross_entropy",
        target="Ad_rel",
        class_values=(0.780, 0.805, 0.830, 0.855, 0.880),
    )

    def __post_init__(self) -> None:
        enabled = [
            name
            for name in KW51_TASK_NAMES
            if getattr(self, name).enabled
        ]
        if len(enabled) != 1:
            raise ValueError("exactly one KW51 task must be enabled per run")
        if self.candidate_compatibility.enabled and self.candidate_compatibility.loss != "listwise":
            raise ValueError("candidate_compatibility.loss must be listwise")
        if self.ad_rel_classification.enabled:
            classifier = self.ad_rel_classification
            if classifier.loss != "cross_entropy":
                raise ValueError("ad_rel_classification.loss must be cross_entropy")
            if classifier.target != "Ad_rel":
                raise ValueError("ad_rel_classification.target must be Ad_rel")
            if len(classifier.class_values) != 5:
                raise ValueError("ad_rel_classification.class_values must define five classes")

    @property
    def selected_task(self) -> str:
        return next(name for name in KW51_TASK_NAMES if getattr(self, name).enabled)


@dataclass(frozen=True)
class InferenceConfig:
    outputs: tuple[str, ...] = ("candidate_compatibility",)

    def __post_init__(self) -> None:
        if not self.outputs or any(value not in KW51_TASK_NAMES for value in self.outputs):
            raise ValueError(f"inference.outputs must contain values from {KW51_TASK_NAMES}")


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    task: str = "candidate_compatibility"
    metric: str = "mrr"

    def __post_init__(self) -> None:
        allowed = {
            "candidate_compatibility": {"mrr"},
            "ad_rel_classification": {"f1_macro", "nll"},
        }
        if self.task not in allowed or self.metric not in allowed[self.task]:
            raise ValueError(
                "checkpoint_selection must use candidate_compatibility/mrr or "
                "ad_rel_classification/f1_macro or ad_rel_classification/nll"
            )


@dataclass(frozen=True)
class FuzzyConfig:
    """Fuzzy-range multi-instance response-distribution experiment settings."""

    point_count_min: int = 20
    point_count_max: int = 80
    bag_halo_fraction: float = 0.2
    point_interpolation_k: int = 4
    range_length_m_min: float = 10.0
    range_length_m_max: float = 40.0
    range_width_m_min: float = 4.0
    range_width_m_max: float = 11.0
    range_center_margin_fraction: float = 0.05
    coordinate_jitter_m: float = 1.0
    noise_std_mm: float = 0.2
    point_bias_std_mm: float = 0.1
    missing_probability: float = 0.03
    block_dropout_probability: float = 0.05
    block_dropout_fraction: float = 0.15
    min_quality: float = 0.05
    # The latent range family A(B) sampled for the target distribution.
    target_range_count: int = 32
    target_center_sigma_fraction: float = 0.25
    target_extent_log_sigma: float = 0.25
    target_min_nodes: int = 20
    # Query realizations / train-only normalization controls.
    query_realizations: int = 2
    normalization_records_per_state: int = 2
    time_feature_mode: str = "disabled"
    bag_feature_dropout: float = 0.1
    mean_consistency_weight: float = 0.0
    audit_min_snr: float = 0.5
    audit_abort_on_fail: bool = False
    audit_records: int = 30
    # Primary output of the fuzzy task. "response" is the historical
    # response-distribution task; "state" makes the five Ad_rel candidate
    # groups the primary supervised target and keeps the response head as an
    # optional auxiliary (response_auxiliary_weight > 0).
    primary_target: str = "response"
    response_auxiliary_weight: float = 0.0
    # Data-level 5-group nearest-centroid audit for the state task.
    group_audit: bool = True
    group_audit_min_accuracy: float = 0.4
    group_audit_min_snr: float = 1.0
    group_audit_abort_on_fail: bool = False
    # Region-stratified fuzzy bags: each query covers a fixed set of deck
    # regions, each with its own point bag and coverage mask. This preserves
    # region identity instead of pooling all points into one permutation-
    # invariant bag.
    observation_mode: str = "single_bag"
    region_mode: str = "longitudinal"
    region_rows: int = 8
    region_cols: int = 1
    region_point_budget: int = 160
    region_missing_probability: float = 0.05
    region_min_valid_points: int = 3
    # Optional state-covering caps to avoid treating near-duplicate trajectories
    # as independent data. 0 keeps all records.
    train_trajectories_per_state: int = 0
    eval_trajectories_per_state: int = 0
    # Point-budget sweep used by the region audit. Empty means the configured
    # region_point_budget only.
    region_audit_point_budgets: tuple[int, ...] = ()
    # Region-level audit and formal gating.
    region_pooling: str = "flatten"
    region_embedding_dim: int = 16
    region_estimator: str = "mean"
    region_model: str = "point_bag"
    region_audit: bool = True
    region_audit_realizations: int = 5
    region_audit_seeds: int = 2
    region_audit_folds: int = 5
    region_audit_abort_on_fail: bool = False
    region_audit_min_state_accuracy: float = 0.4
    region_audit_min_mrr: float = 0.6
    region_audit_min_snr: float = 1.0
    region_audit_min_margin: float = 0.15
    region_audit_max_seed_std: float = 0.1
    # A run is FORMAL only when the experimental contract is complete, not just
    # when the data audit passes. Smoke/dev configs set formal_run=false.
    formal_run: bool = False
    formal_min_eval_states: int = 5

    def __post_init__(self) -> None:
        if self.point_count_min < 1 or self.point_count_max < self.point_count_min:
            raise ValueError("fuzzy point counts must satisfy 1 <= min <= max")
        if not 0.0 <= self.bag_halo_fraction < 1.0:
            raise ValueError("fuzzy.bag_halo_fraction must be in [0, 1)")
        if self.point_interpolation_k < 1:
            raise ValueError("fuzzy.point_interpolation_k must be >= 1")
        for name in (
            "range_length_m_min",
            "range_length_m_max",
            "range_width_m_min",
            "range_width_m_max",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"fuzzy.{name} must be > 0")
        if self.range_length_m_max < self.range_length_m_min:
            raise ValueError("fuzzy.range_length_m_max must be >= min")
        if self.range_width_m_max < self.range_width_m_min:
            raise ValueError("fuzzy.range_width_m_max must be >= min")
        if not 0.0 <= self.range_center_margin_fraction < 0.5:
            raise ValueError("fuzzy.range_center_margin_fraction must be in [0, 0.5)")
        for name in ("coordinate_jitter_m", "noise_std_mm", "point_bias_std_mm"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"fuzzy.{name} must be >= 0")
        for name in ("missing_probability", "block_dropout_probability"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"fuzzy.{name} must be in [0, 1)")
        if not 0.0 <= self.block_dropout_fraction <= 1.0:
            raise ValueError("fuzzy.block_dropout_fraction must be in [0, 1]")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("fuzzy.min_quality must be in [0, 1]")
        if self.target_range_count < 2:
            raise ValueError("fuzzy.target_range_count must be >= 2")
        if self.target_center_sigma_fraction < 0.0:
            raise ValueError("fuzzy.target_center_sigma_fraction must be >= 0")
        if self.target_extent_log_sigma < 0.0:
            raise ValueError("fuzzy.target_extent_log_sigma must be >= 0")
        if self.target_min_nodes < 1:
            raise ValueError("fuzzy.target_min_nodes must be >= 1")
        if self.query_realizations < 1:
            raise ValueError("fuzzy.query_realizations must be >= 1")
        if self.normalization_records_per_state < 0:
            raise ValueError("fuzzy.normalization_records_per_state must be >= 0")
        if self.time_feature_mode not in ("raw", "standardized", "disabled"):
            raise ValueError(
                "fuzzy.time_feature_mode must be raw, standardized or disabled"
            )
        if not 0.0 <= self.bag_feature_dropout < 1.0:
            raise ValueError("fuzzy.bag_feature_dropout must be in [0, 1)")
        if self.mean_consistency_weight < 0.0:
            raise ValueError("fuzzy.mean_consistency_weight must be >= 0")
        if self.audit_min_snr < 0.0:
            raise ValueError("fuzzy.audit_min_snr must be >= 0")
        if self.audit_records < 1:
            raise ValueError("fuzzy.audit_records must be >= 1")
        if self.formal_min_eval_states < 1:
            raise ValueError("fuzzy.formal_min_eval_states must be >= 1")
        if self.primary_target not in ("response", "state", "ad_group"):
            raise ValueError(
                "fuzzy.primary_target must be response, state or ad_group"
            )
        if self.response_auxiliary_weight < 0.0:
            raise ValueError("fuzzy.response_auxiliary_weight must be >= 0")
        if not 0.0 <= self.group_audit_min_accuracy <= 1.0:
            raise ValueError("fuzzy.group_audit_min_accuracy must be in [0, 1]")
        if self.group_audit_min_snr < 0.0:
            raise ValueError("fuzzy.group_audit_min_snr must be >= 0")
        if self.observation_mode not in ("single_bag", "region_bag"):
            raise ValueError(
                "fuzzy.observation_mode must be single_bag or region_bag"
            )
        if self.region_mode not in ("longitudinal", "grid"):
            raise ValueError("fuzzy.region_mode must be longitudinal or grid")
        if self.region_pooling not in ("flatten", "mean"):
            raise ValueError("fuzzy.region_pooling must be flatten or mean")
        if self.region_embedding_dim < 1:
            raise ValueError("fuzzy.region_embedding_dim must be >= 1")
        if self.region_estimator not in ("mean", "plane", "quadratic"):
            raise ValueError(
                "fuzzy.region_estimator must be mean, plane or quadratic"
            )
        if self.region_model not in ("point_bag", "region_summary"):
            raise ValueError(
                "fuzzy.region_model must be point_bag or region_summary"
            )
        if self.region_rows < 1 or self.region_cols < 1:
            raise ValueError("fuzzy.region_rows and region_cols must be >= 1")
        if self.region_mode == "longitudinal" and self.region_cols != 1:
            raise ValueError("longitudinal region_mode requires region_cols=1")
        if self.region_point_budget < 1:
            raise ValueError("fuzzy.region_point_budget must be >= 1")
        if self.train_trajectories_per_state < 0:
            raise ValueError(
                "fuzzy.train_trajectories_per_state must be >= 0"
            )
        if self.eval_trajectories_per_state < 0:
            raise ValueError(
                "fuzzy.eval_trajectories_per_state must be >= 0"
            )
        budgets = tuple(int(value) for value in self.region_audit_point_budgets)
        if any(value < 1 for value in budgets):
            raise ValueError(
                "fuzzy.region_audit_point_budgets must all be >= 1"
            )
        if not 0.0 <= self.region_missing_probability <= 1.0:
            raise ValueError("fuzzy.region_missing_probability must be in [0, 1]")
        configured_regions = int(self.region_rows) * int(self.region_cols)
        max_per_region = max(1, int(self.region_point_budget) // configured_regions)
        if not 1 <= self.region_min_valid_points <= max_per_region:
            raise ValueError(
                "fuzzy.region_min_valid_points must be in "
                f"[1, {max_per_region}] for the configured region layout"
            )
        for name in ("region_audit_realizations", "region_audit_seeds", "region_audit_folds"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"fuzzy.{name} must be >= 1")
        for name in (
            "region_audit_min_state_accuracy",
            "region_audit_min_mrr",
            "region_audit_min_margin",
            "region_audit_max_seed_std",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"fuzzy.{name} must be in [0, 1]")
        if self.region_audit_min_snr < 0.0:
            raise ValueError("fuzzy.region_audit_min_snr must be >= 0")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig
    regional_targets: RegionalTargetConfig = RegionalTargetConfig()
    task: str = "qianhai_four_task"
    input_mode: str = "node"
    fuzzy: FuzzyConfig | None = None
    smoke: SmokeConfig = SmokeConfig()
    tasks: TasksConfig | None = None
    inference: InferenceConfig | None = None
    checkpoint_selection: CheckpointSelectionConfig | None = None


def load_experiment_config(
    path: str | Path,
    *,
    data_root_override: str | Path | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    progress_override: str | None = None,
    fold_override: int | None = None,
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
    raw_fuzzy = dict(payload.get("fuzzy") or {})
    # Legacy Gaussian-footprint configs remain readable; these fields are no
    # longer part of the sparse-point -> fixed-square-grid adapter.
    raw_data.pop("region_count", None)
    raw_data.pop("footprint_sigma_m", None)
    task = str(payload.get("task", "qianhai_four_task"))
    if task not in (
        "qianhai_four_task",
        "kw51_candidate_ranking",
        "kw51_fuzzy_response",
        "kw51_ad_rel_classification",
    ):
        raise ValueError(
            "task must be qianhai_four_task, kw51_candidate_ranking, "
            "kw51_fuzzy_response or kw51_ad_rel_classification"
        )
    if task in ("kw51_fuzzy_response", "kw51_ad_rel_classification") and not payload.get("fuzzy"):
        raise ValueError(f"task {task} requires a fuzzy: config section")

    # Keep the existing train.model_name format valid while allowing the clearer
    # model.name/model.params format for new experiments.
    model_name = str(raw_model.pop("name", raw_train.pop("model_name", "mlp")))
    model_params = model_parameters(raw_model)

    # One authoritative observation mode drives both the dataset adapter and
    # the shared backbone. Accept the old model.params location during migration.
    configured_modes = [
        mode for mode in (
            payload.get("input_mode"),
            raw_data.pop("input_mode", None),
            model_params.get("input_mode"),
        ) if mode is not None
    ]
    if len({str(mode) for mode in configured_modes}) > 1:
        raise ValueError("input_mode must agree across top-level, data, and model.params")
    default_mode = "region" if task.startswith("kw51_") else "node"
    input_mode = str(configured_modes[0]) if configured_modes else default_mode
    if input_mode not in ("node", "region"):
        raise ValueError("input_mode must be node or region")
    model_params["input_mode"] = input_mode
    if model_name in ("lstm", "gru", "transformer", "resnet1d", "resnet50_1d"):
        model_params.setdefault(
            "region_pooling", "flatten" if input_mode == "region" else "mean"
        )
    else:
        model_params.pop("region_pooling", None)
    raw_data["input_mode"] = input_mode

    if data_root_override is not None:
        raw_data["data_root"] = str(data_root_override)
    if epochs_override is not None:
        raw_train["epochs"] = epochs_override
    if batch_size_override is not None:
        raw_train["batch_size"] = batch_size_override
    if progress_override is not None:
        raw_train["progress"] = progress_override
    if fold_override is not None:
        raw_data["split_fold"] = int(fold_override)

    if "data_root" not in raw_data:
        raise ValueError(f"Config {path} is missing data.data_root")
    if "output_root" in raw_train:
        raw_train["output_root"] = Path(raw_train["output_root"])
    for path_key in ("canonical_root", "source_root", "split_manifest"):
        if path_key in raw_data and raw_data[path_key] is not None:
            raw_data[path_key] = Path(raw_data[path_key])
    data = DataConfig(data_root=Path(raw_data.pop("data_root")), **raw_data)
    train = TrainConfig(**raw_train)
    model = ModelConfig(name=model_name, params=model_params)
    regional_targets = RegionalTargetConfig(**raw_targets)
    fuzzy = FuzzyConfig(**raw_fuzzy) if raw_fuzzy else None
    raw_tasks = payload.get("tasks")
    tasks = None
    inference = None
    checkpoint_selection = None
    if raw_tasks is not None:
        task_payload = dict(raw_tasks)
        default_candidate_enabled = task == "kw51_candidate_ranking"
        default_classifier_enabled = task == "kw51_ad_rel_classification"
        candidate_payload = dict(task_payload.get("candidate_compatibility") or {})
        classifier_payload = dict(task_payload.get("ad_rel_classification") or {})
        candidate_payload.setdefault("enabled", default_candidate_enabled)
        candidate_payload.setdefault("loss", "listwise")
        candidate_payload.setdefault("loss_weight", 1.0)
        classifier_payload.setdefault("enabled", default_classifier_enabled)
        classifier_payload.setdefault("loss", "cross_entropy")
        classifier_payload.setdefault("loss_weight", 0.2)
        classifier_payload.setdefault("target", "Ad_rel")
        classifier_payload["class_values"] = tuple(
            float(value)
            for value in classifier_payload.get(
                "class_values", (0.780, 0.805, 0.830, 0.855, 0.880)
            )
        )
        tasks = TasksConfig(
            candidate_compatibility=TaskHeadConfig(**candidate_payload),
            ad_rel_classification=TaskHeadConfig(**classifier_payload),
        )
        selected_task = tasks.selected_task
        expected_task = {
            "kw51_candidate_ranking": "candidate_compatibility",
            "kw51_ad_rel_classification": "ad_rel_classification",
        }.get(task)
        if expected_task is not None and selected_task != expected_task:
            raise ValueError(f"task={task} conflicts with tasks.{selected_task}.enabled")
        raw_inference = dict(payload.get("inference") or {})
        default_output = "ad_rel_classification" if selected_task == "ad_rel_classification" else "candidate_compatibility"
        inference = InferenceConfig(
            outputs=tuple(raw_inference.get("outputs", (default_output,)))
        )
        if inference.outputs != (selected_task,):
            raise ValueError("inference.outputs must select the single enabled task")
        raw_selection = dict((payload.get("training") or {}).get("checkpoint_selection") or {})
        default_metric = "f1_macro" if selected_task == "ad_rel_classification" else "mrr"
        checkpoint_selection = CheckpointSelectionConfig(
            task=str(raw_selection.get("task", selected_task)),
            metric=str(raw_selection.get("metric", default_metric)),
        )
        if checkpoint_selection.task != selected_task:
            raise ValueError("checkpoint_selection.task must match the enabled task")
    elif task == "kw51_ad_rel_classification":
        raise ValueError("kw51_ad_rel_classification requires a tasks: config section")
    if task == "kw51_ad_rel_classification":
        if data.dataset_kind != "kw51":
            raise ValueError("kw51_ad_rel_classification requires data.dataset_kind=kw51")
        if data.ranking_target != "ad_group" or data.ad_group_parameter != "Ad_rel":
            raise ValueError(
                "kw51_ad_rel_classification requires ranking_target=ad_group "
                "and ad_group_parameter=Ad_rel"
            )
        if data.split_mode != "ad_group_stratified":
            raise ValueError(
                "kw51_ad_rel_classification requires split_mode=ad_group_stratified"
            )
        if fuzzy is None or fuzzy.primary_target != "ad_group":
            raise ValueError(
                "kw51_ad_rel_classification requires fuzzy.primary_target=ad_group"
            )
    return ExperimentConfig(
        data=data,
        train=train,
        model=model,
        regional_targets=regional_targets,
        task=task,
        input_mode=input_mode,
        fuzzy=fuzzy,
        smoke=SmokeConfig(**(payload.get("smoke") or {})),
        tasks=tasks,
        inference=inference,
        checkpoint_selection=checkpoint_selection,
    )
