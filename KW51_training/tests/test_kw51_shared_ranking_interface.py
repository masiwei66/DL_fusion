from __future__ import annotations

import numpy as np
import torch

from KW51_training.data.fem_insar import collate_fem_pairs
from KW51_training.models.kw51_ranking import KW51CandidateScorer
from KW51_training.training.kw51_runner import _listwise_ranking_loss


def _side(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(3, 4)).astype(np.float32)
    valid = np.ones((3, 4), dtype=bool)
    return {
        "values": values,
        "valid_mask": valid,
        "quality": valid.astype(np.float32),
        "region_coords": rng.normal(size=(4, 2)).astype(np.float32),
        "point_present_mask": np.asarray([1, 1, 1, 0], dtype=bool),
        "coverage": valid.astype(np.float32),
        "time_features": rng.normal(size=(3, 2)).astype(np.float32),
        "temperature_valid_mask": np.asarray([1, 1, 1], dtype=bool),
        "input_mode": "region",
    }


def test_kw51_collator_builds_shared_window_batches_and_ranking_backpropagates() -> None:
    metadata = [
        {
            "query_sample_id": "query-1",
            "query_event": 0,
            "query_start": 0,
            "observation_realization_id": 0,
            "candidate_state_id": "state-a",
            "candidate_start": 0,
            "candidate_exclusion_key": "",
        },
        {
            "query_sample_id": "query-1",
            "query_event": 0,
            "query_start": 0,
            "observation_realization_id": 0,
            "candidate_state_id": "state-b",
            "candidate_start": 0,
            "candidate_exclusion_key": "",
        },
    ]
    batch = collate_fem_pairs(
        [
            {"query": _side(1), "candidate": _side(2), "label": 1, "metadata": metadata[0]},
            {"query": _side(1), "candidate": _side(3), "label": 0, "metadata": metadata[1]},
        ]
    )

    assert batch["query"].input_mode == "region"
    assert batch["candidate"].input_mode == "region"
    assert batch["query"].region_coverage.shape == (2, 3, 4)
    assert batch["query"].temperature_valid_mask.shape == (2, 3, 1)

    model = KW51CandidateScorer(
        "lstm",
        {
            "input_dim": 6,
            "input_mode": "region",
            "region_pooling": "flatten",
            "hidden_size": 8,
            "num_layers": 1,
        },
    )
    scores = model.forward_pair({**batch, "metadata": metadata})
    loss = _listwise_ranking_loss(scores, batch["label"], metadata)
    loss.backward()

    assert scores.shape == (2,)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())

