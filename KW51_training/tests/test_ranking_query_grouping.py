import torch

from KW51_training.training.kw51_runner import _listwise_ranking_loss, _ranking_metrics


def _metadata(sample_id: str) -> dict[str, object]:
    return {
        "query_state_id": "1",
        "query_sample_id": sample_id,
        "query_event": 0,
        "query_start": 0,
        "observation_realization_id": 0,
    }


def test_metrics_keep_same_state_trajectories_as_separate_queries() -> None:
    metadata = [_metadata("traj-a"), _metadata("traj-a"), _metadata("traj-b"), _metadata("traj-b")]

    metrics = _ranking_metrics([2.0, 1.0, 1.0, 2.0], [1, 0, 1, 0], metadata)

    assert metrics["query_groups"] == 2
    assert metrics["top1_hit_rate"] == 0.5
    assert metrics["mrr"] == 0.75


def test_listwise_loss_is_normalized_per_query_sample() -> None:
    metadata = [_metadata("traj-a"), _metadata("traj-a"), _metadata("traj-b"), _metadata("traj-b")]

    loss = _listwise_ranking_loss(
        torch.tensor([5.0, 0.0, 0.0, 5.0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        metadata,
    )

    assert 2.5 < float(loss) < 2.6
