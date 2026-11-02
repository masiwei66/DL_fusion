from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from KW51_training.data.fem_insar import (
    FEMArchiveIndex,
    FEMArchiveRecord,
    FEMRankingDataset,
    FEMQueryBatchSampler,
    FEMRegionalSource,
    FEMWindowRef,
    RegionConfig,
    make_fem_group_pairs,
    derive_ad_groups,
)


def _record(state: int, trajectory: int) -> FEMArchiveRecord:
    return FEMArchiveRecord(
        path=Path(f"state_{state}_traj_{trajectory}.h5"),
        sample_id=f"sample_{state}_{trajectory}",
        structural_state_id=str(state),
        trajectory_id=f"kw51_traj_{state:04d}_{trajectory:04d}",
        split_group=f"state_{state:04d}",
        metadata={},
        parameters={"Ad_rel": 0.780},
    )


def _view(state: int, trajectory: int) -> dict[str, np.ndarray]:
    value = float(state) * 10.0 + float(trajectory) / 10.0
    return {
        "values": np.full((4, 2), value, dtype=np.float32),
        "valid_mask": np.ones((4, 2), dtype=bool),
        "quality": np.ones((4, 2), dtype=np.float32),
        "coverage": np.ones((4, 2), dtype=np.float32),
        "temperature": np.zeros(4, dtype=np.float32),
        "delta_days": np.zeros(4, dtype=np.float32),
    }


def _source() -> FEMRegionalSource:
    records = []
    views = {}
    for state in (1, 2, 3, 4):
        for trajectory in (1, 2):
            record = _record(state, trajectory)
            records.append(record)
            views[record.sample_id] = _view(state, trajectory)
    source = object.__new__(FEMRegionalSource)
    source.index = FEMArchiveIndex(tuple(records), {})
    source.config = RegionConfig()
    source._clean_cache = {}
    source._state_cache = {}
    source._group_totals = {}
    source._group_traj_totals = {}
    source._group_state_totals = {}
    source._group_state_traj_totals = {}
    source.observation_coordinates = np.zeros((2, 2), dtype=np.float32)
    source.point_present_mask = np.ones(2, dtype=bool)
    source.clean_view = lambda record: views[record.sample_id]
    return source


def test_group_aggregation_excludes_the_same_environment_key_everywhere() -> None:
    source = _source()

    positive = source.aggregate_group(0, ("1", "2"), exclude_trajectory_key="0001")
    negative = source.aggregate_group(1, ("3", "4"), exclude_trajectory_key="0001")

    # Remaining group-0 records are (state 1, traj 2) and (state 2, traj 2).
    assert np.allclose(positive["values"], 15.2)
    assert np.allclose(negative["values"], 35.2)
    assert int(positive["trajectory_count"]) == 2
    assert int(negative["trajectory_count"]) == 2
    assert positive["valid_mask"].all()


def test_group_aggregation_without_a_matching_key_uses_all_records() -> None:
    source = _source()

    result = source.aggregate_group(0, ("1", "2"), exclude_trajectory_key="9999")

    assert np.allclose(result["values"], 15.15)
    assert int(result["trajectory_count"]) == 4


def test_group_aggregation_rejects_changed_state_membership() -> None:
    source = _source()
    source.aggregate_group(0, ("1", "2"), exclude_trajectory_key="0001")

    with pytest.raises(ValueError, match="already built"):
        source.aggregate_group(0, ("1",), exclude_trajectory_key="0001")


def test_query_batch_sampler_length_matches_group_packing() -> None:
    pairs = [
        SimpleNamespace(
            query=SimpleNamespace(
                record=SimpleNamespace(sample_id=f"q{query}"), start=0
            ),
            realization=0,
        )
        for query in range(100)
        for _ in range(5)
    ]
    sampler = FEMQueryBatchSampler(
        SimpleNamespace(pairs=pairs), batch_size=32, seed=1, shuffle=False
    )

    batches = list(sampler)

    assert len(batches) == len(sampler)
    assert len(sampler) == 17


def test_group_aggregation_can_exclude_one_state() -> None:
    source = _source()

    result = source.aggregate_group(0, ("1", "2"), exclude_state_id="1")

    # Only state-2 records remain, regardless of the environment key.
    assert np.allclose(result["values"], 20.15)
    assert int(result["trajectory_count"]) == 2


def test_open_set_exclusion_removes_the_state_and_key_union_once() -> None:
    source = _source()

    result = source.aggregate_group(
        0,
        ("1", "2"),
        exclude_state_id="1",
        exclude_trajectory_key="0001",
    )

    # Remaining record: (state 2, trajectory 2) only.
    assert np.allclose(result["values"], 20.2)
    assert int(result["trajectory_count"]) == 1


def test_state_exclusion_is_a_noop_for_groups_without_that_state() -> None:
    source = _source()

    result = source.aggregate_group(1, ("3", "4"), exclude_state_id="1")

    assert np.allclose(result["values"], 35.15)
    assert int(result["trajectory_count"]) == 4


def test_state_aggregation_excludes_shared_environment_key() -> None:
    source = _source()

    result = source.aggregate_state("1", exclude_trajectory_key="0001")

    assert np.allclose(result["values"], 10.2)
    assert int(result["trajectory_count"]) == 1


def test_ranking_dataset_uses_frozen_gallery_for_unseen_query_states() -> None:
    from types import SimpleNamespace

    from KW51_training.tests.test_kw51_ad_groups import _index

    group_map = derive_ad_groups(_index(), "Ad_rel", expected_count=5)
    query_record = _index().records[0]
    query = {
        "values": np.full((4, 2), -1.0, dtype=np.float32),
        "valid_mask": np.ones((4, 2), dtype=bool),
        "quality": np.ones((4, 2), dtype=np.float32),
        "coverage": np.ones((4, 2), dtype=np.float32),
        "temperature": np.zeros(4, dtype=np.float32),
        "delta_days": np.zeros(4, dtype=np.float32),
    }
    source = SimpleNamespace(
        config=SimpleNamespace(input_mode="region"),
        observation_coordinates=np.zeros((2, 2), dtype=np.float32),
        point_present_mask=np.ones(2, dtype=bool),
        query_window=lambda ref, realization: query,
    )
    normalization = SimpleNamespace(
        transform_values=lambda values, valid: np.asarray(values, dtype=np.float32),
        transform_coordinates=lambda coordinates: np.asarray(coordinates, dtype=np.float32),
        transform_time_features=lambda temperature, delta: np.zeros((len(temperature), 2), dtype=np.float32),
    )

    def candidate(value: float) -> dict[str, np.ndarray]:
        return {
            "values": np.full((4, 2), value, dtype=np.float32),
            "valid_mask": np.ones((4, 2), dtype=bool),
            "quality": np.ones((4, 2), dtype=np.float32),
            "coverage": np.ones((4, 2), dtype=np.float32),
            "temperature": np.zeros(4, dtype=np.float32),
            "delta_days": np.zeros(4, dtype=np.float32),
            "trajectory_count": np.asarray(10),
        }

    base_gallery = {group.group_id: candidate(5.0) for group in group_map.groups}
    train_query_gallery = {group.group_id: candidate(3.0) for group in group_map.groups}
    dataset = FEMRankingDataset(
        make_fem_group_pairs([FEMWindowRef(query_record, 0, 4)], group_map),
        source,
        normalization,
        group_map=group_map,
        gallery_exclusion="query_state",
        candidate_galleries={None: base_gallery, query_record.structural_state_id: train_query_gallery},
    )

    train_item = dataset[0]
    unseen_state = next(record for record in _index().records if record.structural_state_id == "6")
    eval_dataset = FEMRankingDataset(
        make_fem_group_pairs([FEMWindowRef(unseen_state, 0, 4)], group_map),
        source,
        normalization,
        group_map=group_map,
        gallery_exclusion="query_state",
        candidate_galleries={None: base_gallery, query_record.structural_state_id: train_query_gallery},
    )

    assert np.allclose(train_item["candidate"]["values"], 3.0)
    assert np.allclose(eval_dataset[0]["candidate"]["values"], 5.0)
