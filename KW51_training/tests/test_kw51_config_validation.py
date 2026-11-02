from pathlib import Path

import pytest

from KW51_training.config import DataConfig, load_experiment_config


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_new_ad_group_configs_load() -> None:
    names = (
        "kw51_lstm_5class_clean.yaml",
        "kw51_lstm_5class_clean_flatten.yaml",
        "kw51_lstm_5class_degraded.yaml",
        "kw51_lstm_5class_degraded_flatten.yaml",
        "kw51_lstm_5class_clean_flatten_openset.yaml",
        "kw51_lstm_5class_clean_flatten_openset_cv20.yaml",
    )
    for name in names:
        config = load_experiment_config(CONFIG_ROOT / name)
        assert config.data.ranking_target == "ad_group"
        assert config.data.ad_group_count == 5
        assert config.data.split_mode == "ad_group_stratified"
        assert config.data.window_length == 26
        assert config.data.normalization_scope == "per_region_offset"
        assert config.data.time_feature_mode == "disabled"
        if "_cv" not in name:
            assert config.data.learnability_audit is True
    assert load_experiment_config(CONFIG_ROOT / "kw51_lstm_5class_clean.yaml").data.diagnostic_only is False
    assert load_experiment_config(CONFIG_ROOT / "kw51_lstm_5class_degraded_flatten.yaml").data.diagnostic_only is True
    assert load_experiment_config(CONFIG_ROOT / "kw51_lstm_5class_clean_flatten_openset.yaml").data.gallery_exclusion == "query_state"
    assert load_experiment_config(CONFIG_ROOT / "kw51_lstm_5class_clean_flatten_openset_cv20.yaml").train.epochs == 20


def test_old_config_keeps_legacy_defaults() -> None:
    config = load_experiment_config(CONFIG_ROOT / "kw51_lstm.yaml")

    assert config.data.ranking_target == "exact_state"
    assert config.data.split_mode == "random"
    assert config.data.normalization_scope == "global"
    assert config.data.time_feature_mode == "raw"
    assert config.data.learnability_audit is False
    assert config.data.loss_collapse_patience == 0


def test_invalid_ranking_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="ranking_target"):
        DataConfig(data_root=Path("."), dataset_kind="kw51", ranking_target="bad")


def test_ad_group_stratified_requires_ad_group_target() -> None:
    with pytest.raises(ValueError, match="requires ranking_target=ad_group"):
        DataConfig(
            data_root=Path("."),
            dataset_kind="kw51",
            split_mode="ad_group_stratified",
        )


def test_single_group_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        DataConfig(
            data_root=Path("."),
            dataset_kind="kw51",
            ranking_target="ad_group",
            ad_group_count=1,
        )


def test_invalid_time_feature_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="time_feature_mode"):
        DataConfig(
            data_root=Path("."),
            dataset_kind="kw51",
            time_feature_mode="bad",
        )


def test_invalid_audit_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="audit_min_oracle_group_accuracy"):
        DataConfig(
            data_root=Path("."),
            dataset_kind="kw51",
            audit_min_oracle_group_accuracy=1.5,
        )


def test_invalid_gallery_exclusion_is_rejected() -> None:
    with pytest.raises(ValueError, match="gallery_exclusion"):
        DataConfig(
            data_root=Path("."),
            dataset_kind="kw51",
            gallery_exclusion="bad",
        )


def test_query_state_gallery_exclusion_requires_ad_group() -> None:
    with pytest.raises(ValueError, match="requires ranking_target=ad_group"):
        DataConfig(
            data_root=Path("."),
            dataset_kind="kw51",
            gallery_exclusion="query_state",
        )


def test_cv_config_loads_and_fold_override_works() -> None:
    config = load_experiment_config(
        CONFIG_ROOT / "kw51_lstm_5class_clean_flatten_openset_cv.yaml"
    )
    assert config.data.split_fold == 0
    assert config.data.gallery_exclusion == "query_state"
    assert config.train.epochs == 5
    overridden = load_experiment_config(
        CONFIG_ROOT / "kw51_lstm_5class_clean_flatten_openset_cv.yaml",
        fold_override=4,
    )
    assert overridden.data.split_fold == 4


def test_split_fold_requires_stratified_split() -> None:
    with pytest.raises(ValueError, match="requires split_mode=ad_group_stratified"):
        DataConfig(
            data_root=Path("."),
            dataset_kind="kw51",
            ranking_target="ad_group",
            split_fold=0,
        )


def test_trajectory_split_fractions_must_cover_all_roles() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        DataConfig(
            data_root=Path("."),
            dataset_kind="kw51",
            ranking_target="ad_group",
            gallery_trajectory_fraction=0.6,
            train_query_trajectory_fraction=0.2,
            validation_query_trajectory_fraction=0.1,
            test_query_trajectory_fraction=0.2,
        )
