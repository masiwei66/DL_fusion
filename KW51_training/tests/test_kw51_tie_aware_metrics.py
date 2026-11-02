import pytest

from KW51_training.training.kw51_runner import _ranking_metrics, _resolve_run_status


def _metadata(
    sample_id: str = "q1", state_id: str = "1", candidate_id: str = "1"
) -> dict[str, object]:
    return {
        "query_state_id": state_id,
        "query_sample_id": sample_id,
        "candidate_state_id": candidate_id,
        "query_event": 0,
        "query_start": 0,
        "observation_realization_id": 0,
    }


def test_all_tied_candidates_are_scored_by_expected_rank() -> None:
    metadata = [_metadata(candidate_id=str(index)) for index in range(5)]

    metrics = _ranking_metrics([0.0] * 5, [1, 0, 0, 0, 0], metadata)

    assert metrics["query_groups"] == 1
    assert metrics["top1_hit_rate"] == pytest.approx(1.0 / 5.0)
    assert metrics["top3_hit_rate"] == pytest.approx(3.0 / 5.0)
    assert metrics["mrr"] == pytest.approx(
        sum(1.0 / rank for rank in range(1, 6)) / 5.0
    )
    assert metrics["pairwise_accuracy"] == pytest.approx(0.5)
    assert metrics["tie_group_fraction"] == pytest.approx(1.0)
    assert metrics["mean_score_spread"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["effective_n_states"] == 1
    assert metrics["per_query_state"]["1"]["query_groups"] == 1


def test_no_tie_metrics_match_hard_rank_values() -> None:
    metadata = [
        _metadata("traj-a"),
        _metadata("traj-a"),
        _metadata("traj-b"),
        _metadata("traj-b"),
    ]

    metrics = _ranking_metrics([2.0, 1.0, 1.0, 2.0], [1, 0, 1, 0], metadata)

    assert metrics["query_groups"] == 2
    assert metrics["top1_hit_rate"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["pairwise_accuracy"] == pytest.approx(0.5)
    assert metrics["tie_group_fraction"] == pytest.approx(0.0)
    assert metrics["effective_n_states"] == 1


def test_partial_tie_uses_expected_rank_within_the_tie_block() -> None:
    metadata = [_metadata(candidate_id=str(index)) for index in range(3)]

    metrics = _ranking_metrics([1.0, 1.0, 0.0], [1, 0, 0], metadata)

    assert metrics["top1_hit_rate"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5) / 2.0)
    assert metrics["pairwise_accuracy"] == pytest.approx(0.75)


def test_resolve_run_status_marks_failed_audits_as_diagnostic() -> None:
    assert _resolve_run_status(False, None) == (False, False)
    assert _resolve_run_status(False, {"status": "PASS"}) == (False, False)
    assert _resolve_run_status(False, {"status": "FAIL"}) == (True, True)
    assert _resolve_run_status(True, {"status": "PASS"}) == (True, False)
    assert _resolve_run_status(True, {"status": "FAIL"}) == (True, True)
