from pathlib import Path

import pytest

from KW51_training.data.fem_insar import (
    FEMArchiveIndex,
    FEMArchiveRecord,
    derive_ad_groups,
    trajectory_key,
)


AD_VALUES = (0.780, 0.805, 0.830, 0.855, 0.880)


def _record(state: int, trajectory: int, value: float) -> FEMArchiveRecord:
    state_id = str(state)
    return FEMArchiveRecord(
        path=Path(f"state_{state}_traj_{trajectory}.h5"),
        sample_id=f"sample_{state}_{trajectory}",
        structural_state_id=state_id,
        trajectory_id=f"kw51_traj_{state:04d}_{trajectory:04d}",
        split_group=f"state_{state:04d}",
        metadata={},
        parameters={"Ad_rel": float(value)},
    )


def _index() -> FEMArchiveIndex:
    records = []
    for group_id, value in enumerate(AD_VALUES):
        for column in range(5):
            state = group_id * 5 + column + 1
            for trajectory in range(1, 3):
                records.append(_record(state, trajectory, value))
    return FEMArchiveIndex(tuple(records), {})


def test_derive_ad_groups_orders_values_and_membership() -> None:
    group_map = derive_ad_groups(_index(), "Ad_rel", expected_count=5)

    assert group_map.group_count == 5
    assert group_map.parameter == "Ad_rel"
    assert [group.value for group in group_map.groups] == list(AD_VALUES)
    assert group_map.group_states(0) == ("1", "2", "3", "4", "5")
    assert group_map.group_states(4) == ("21", "22", "23", "24", "25")
    assert group_map.state_to_group["13"] == 2
    assert group_map.state_to_value["13"] == pytest.approx(0.830)


def test_derive_ad_groups_rejects_inconsistent_state_values() -> None:
    records = list(_index().records)
    records.append(_record(1, 3, 0.805))
    with pytest.raises(ValueError, match="inconsistent"):
        derive_ad_groups(FEMArchiveIndex(tuple(records), {}), "Ad_rel")


def test_derive_ad_groups_rejects_wrong_expected_count() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        derive_ad_groups(_index(), "Ad_rel", expected_count=4)


def test_derive_ad_groups_rejects_missing_parameter() -> None:
    record = FEMArchiveRecord(
        path=Path("missing.h5"),
        sample_id="missing",
        structural_state_id="1",
        trajectory_id="kw51_traj_0001_0001",
        split_group="state_0001",
        metadata={},
        parameters={},
    )
    with pytest.raises(ValueError, match="missing state/parameters/Ad_rel"):
        derive_ad_groups(FEMArchiveIndex((record,), {}), "Ad_rel")


def test_trajectory_key_uses_shared_environment_index() -> None:
    assert trajectory_key(_record(1, 7, 0.780)) == "0007"
    assert trajectory_key(_record(25, 7, 0.880)) == "0007"
