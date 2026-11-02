from __future__ import annotations

from KW51_training.training.kw51_runner import _preflight_readiness
from KW51_training.training.kw51_audit import summarize_ad_group_cross_validation_audit


def test_preflight_requires_a_passing_identifiability_audit() -> None:
    status, reasons = _preflight_readiness(
        smoke_enabled=False,
        diagnostic_only=False,
        learnability_audit=True,
        audit={"status": "FAIL", "reasons": ["oracle below threshold"]},
    )

    assert status == "BLOCKED"
    assert reasons == ["oracle below threshold"]


def test_preflight_passes_only_for_a_formal_config_with_a_passing_audit() -> None:
    status, reasons = _preflight_readiness(
        smoke_enabled=False,
        diagnostic_only=False,
        learnability_audit=True,
        audit={"status": "PASS", "reasons": []},
    )

    assert status == "READY_FOR_FORMAL_TRAINING"
    assert reasons == []


def test_smoke_or_diagnostic_config_is_never_formal_ready() -> None:
    smoke_status, _ = _preflight_readiness(
        smoke_enabled=True,
        diagnostic_only=False,
        learnability_audit=True,
        audit={"status": "PASS", "reasons": []},
    )
    diagnostic_status, _ = _preflight_readiness(
        smoke_enabled=False,
        diagnostic_only=True,
        learnability_audit=True,
        audit={"status": "PASS", "reasons": []},
    )

    assert smoke_status == "NOT_FORMAL_CONFIG"
    assert diagnostic_status == "NOT_FORMAL_CONFIG"


def test_preflight_blocks_when_the_audit_is_disabled_or_missing() -> None:
    disabled_status, disabled_reasons = _preflight_readiness(
        smoke_enabled=False,
        diagnostic_only=False,
        learnability_audit=False,
        audit=None,
    )
    missing_status, missing_reasons = _preflight_readiness(
        smoke_enabled=False,
        diagnostic_only=False,
        learnability_audit=True,
        audit=None,
    )

    assert disabled_status == "BLOCKED"
    assert disabled_reasons == ["data.learnability_audit=false"]
    assert missing_status == "BLOCKED"
    assert missing_reasons == ["identifiability audit did not produce a result"]


def test_cross_validation_audit_requires_all_states_and_all_folds() -> None:
    folds = [
        {
            "fold": 0,
            "status": "PASS",
            "query_count": 2,
            "noisy_trial_count": 6,
            "oracle_group_accuracy_noisy": 1.0,
            "oracle_group_accuracy_clean": 1.0,
            "signal_to_noise": 0.17,
            "query_state_ids": ["1", "2"],
            "gallery_state_ids": ["3", "4"],
            "gallery_trajectory_keys": ["0001"],
        },
        {
            "fold": 1,
            "status": "PASS",
            "query_count": 2,
            "noisy_trial_count": 6,
            "oracle_group_accuracy_noisy": 1.0,
            "oracle_group_accuracy_clean": 1.0,
            "signal_to_noise": 0.17,
            "query_state_ids": ["3", "4"],
            "gallery_state_ids": ["1", "2"],
            "gallery_trajectory_keys": ["0001"],
        },
    ]

    result = summarize_ad_group_cross_validation_audit(
        folds,
        expected_state_ids=["1", "2", "3", "4"],
        required_fold_count=2,
        minimum_oracle_accuracy=0.9,
    )

    assert result["status"] == "PASS"
    assert result["state_coverage_complete"] is True
    assert result["oracle_group_accuracy_noisy"] == 1.0
    assert result["thresholds"]["signal_to_noise_is_gate"] is False


def test_cross_validation_audit_fails_on_missed_states_or_failed_fold() -> None:
    result = summarize_ad_group_cross_validation_audit(
        [
            {
                "fold": 0,
                "status": "FAIL",
                "query_count": 1,
                "noisy_trial_count": 3,
                "oracle_group_accuracy_noisy": 2 / 3,
                "query_state_ids": ["1"],
            }
        ],
        expected_state_ids=["1", "2"],
        required_fold_count=2,
        minimum_oracle_accuracy=0.9,
    )

    assert result["status"] == "FAIL"
    assert result["missing_state_ids"] == ["2"]
    assert result["failed_folds"] == [0]
