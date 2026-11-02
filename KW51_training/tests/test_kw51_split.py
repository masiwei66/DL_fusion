from pathlib import Path

import pytest

from KW51_training.data.fem_insar import (
    FEMArchiveIndex,
    FEMArchiveRecord,
    derive_ad_groups,
    split_fem_records,
    split_fem_trajectory_keys,
)


AD_VALUES = (0.780, 0.805, 0.830, 0.855, 0.880)


def _index() -> FEMArchiveIndex:
    records = []
    for group_id, value in enumerate(AD_VALUES):
        for column in range(5):
            state = group_id * 5 + column + 1
            records.append(
                FEMArchiveRecord(
                    path=Path(f"state_{state}.h5"),
                    sample_id=f"sample_{state}",
                    structural_state_id=str(state),
                    trajectory_id=f"kw51_traj_{state:04d}_0001",
                    split_group=f"state_{state:04d}",
                    metadata={},
                    parameters={"Ad_rel": value},
                )
            )
    return FEMArchiveIndex(tuple(records), {})


def test_ad_group_stratified_split_keeps_every_group_in_eval() -> None:
    index = _index()
    group_map = derive_ad_groups(index, "Ad_rel", expected_count=5)

    split = split_fem_records(
        index,
        seed=20261015,
        train_fraction=0.7,
        validation_fraction=0.15,
        group_map=group_map,
        split_mode="ad_group_stratified",
    )

    assert {role: len(records) for role, records in split.items()} == {
        "train": 15,
        "validation": 5,
        "test": 5,
    }
    validation_states = {record.structural_state_id for record in split["validation"]}
    test_states = {record.structural_state_id for record in split["test"]}
    assert validation_states.isdisjoint(test_states)
    for group in group_map.groups:
        members = set(group.state_ids)
        assert len(validation_states & members) == 1
        assert len(test_states & members) == 1


def test_ad_group_stratified_split_requires_a_group_map() -> None:
    with pytest.raises(ValueError, match="requires an AdGroupMap"):
        split_fem_records(
            _index(),
            seed=1,
            train_fraction=0.7,
            validation_fraction=0.15,
            split_mode="ad_group_stratified",
        )


def test_legacy_random_split_still_partitions_every_state() -> None:
    split = split_fem_records(
        _index(),
        seed=1,
        train_fraction=0.7,
        validation_fraction=0.15,
    )

    total = sum(len(records) for records in split.values())
    assert total == 25
    assert split["validation"]
    assert split["test"]


def test_state_level_folds_cover_every_state_once() -> None:
    index = _index()
    group_map = derive_ad_groups(index, "Ad_rel", expected_count=5)
    all_test: set[str] = set()
    all_validation: set[str] = set()
    for fold in range(5):
        split = split_fem_records(
            index,
            seed=20261015,
            train_fraction=0.7,
            validation_fraction=0.15,
            group_map=group_map,
            split_mode="ad_group_stratified",
            fold=fold,
        )
        assert {role: len(records) for role, records in split.items()} == {
            "train": 15,
            "validation": 5,
            "test": 5,
        }
        test_states = {record.structural_state_id for record in split["test"]}
        validation_states = {
            record.structural_state_id for record in split["validation"]
        }
        assert len(test_states) == 5
        assert len(validation_states) == 5
        assert test_states.isdisjoint(validation_states)
        for group in group_map.groups:
            members = set(group.state_ids)
            assert len(test_states & members) == 1
            assert len(validation_states & members) == 1
        all_test.update(test_states)
        all_validation.update(validation_states)
    assert all_test == set(index.state_ids)
    assert all_validation == set(index.state_ids)


def test_state_level_fold_is_deterministic() -> None:
    index = _index()
    group_map = derive_ad_groups(index, "Ad_rel", expected_count=5)
    kwargs = dict(
        seed=20261015,
        train_fraction=0.7,
        validation_fraction=0.15,
        group_map=group_map,
        split_mode="ad_group_stratified",
        fold=2,
    )
    first = split_fem_records(index, **kwargs)
    second = split_fem_records(index, **kwargs)
    assert [record.sample_id for record in first["test"]] == [
        record.sample_id for record in second["test"]
    ]


def test_state_level_fold_out_of_range_is_rejected() -> None:
    index = _index()
    group_map = derive_ad_groups(index, "Ad_rel", expected_count=5)
    with pytest.raises(ValueError, match="out of range"):
        split_fem_records(
            index,
            seed=20261015,
            train_fraction=0.7,
            validation_fraction=0.15,
            group_map=group_map,
            split_mode="ad_group_stratified",
            fold=5,
        )


def _shared_trajectory_index(key_count: int = 20) -> FEMArchiveIndex:
    records = []
    for state in range(1, 4):
        for trajectory in range(1, key_count + 1):
            records.append(
                FEMArchiveRecord(
                    path=Path(f"state_{state}_traj_{trajectory}.h5"),
                    sample_id=f"sample_{state}_{trajectory}",
                    structural_state_id=str(state),
                    trajectory_id=f"kw51_traj_{state:04d}_{trajectory:04d}",
                    split_group=f"state_{state:04d}",
                    metadata={},
                    parameters={},
                )
            )
    return FEMArchiveIndex(tuple(records), {})


def test_trajectory_keys_are_disjoint_and_deterministic() -> None:
    index = _shared_trajectory_index()
    kwargs = dict(
        gallery_fraction=0.5,
        train_fraction=0.2,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=17,
    )

    first = split_fem_trajectory_keys(index, **kwargs)
    second = split_fem_trajectory_keys(index, **kwargs)

    parts = [set(first.gallery), set(first.train), set(first.validation), set(first.test)]
    assert first == second
    assert all(left.isdisjoint(right) for i, left in enumerate(parts) for right in parts[i + 1:])
    assert set.union(*parts) == {f"{key:04d}" for key in range(1, 21)}
    assert [len(part) for part in parts] == [10, 4, 3, 3]


def test_trajectory_split_rejects_state_specific_environment_keys() -> None:
    index = _shared_trajectory_index()
    records = tuple(
        record for record in index.records
        if not (record.structural_state_id == "3" and record.sample_id.endswith("20"))
    )
    with pytest.raises(ValueError, match="share the same trajectory-key set"):
        split_fem_trajectory_keys(
            FEMArchiveIndex(records, {}),
            gallery_fraction=0.5,
            train_fraction=0.2,
            validation_fraction=0.15,
            test_fraction=0.15,
            seed=17,
        )
