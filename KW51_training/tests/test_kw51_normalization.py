from pathlib import Path

from KW51_training.data.fem_insar import (
    FEMArchiveRecord,
    _sample_normalization_records,
)


def _records() -> list[FEMArchiveRecord]:
    return [
        FEMArchiveRecord(
            path=Path(f"state_{state}_record_{record}.h5"),
            sample_id=f"sample_{state}_{record}",
            structural_state_id=str(state),
            trajectory_id=f"trajectory_{state}_{record}",
            split_group=f"state_{state}",
            metadata={},
        )
        for state in range(1, 5)
        for record in range(3)
    ]


def test_normalization_sampling_covers_all_states_before_filling_limit() -> None:
    selected = _sample_normalization_records(_records(), max_records=8)

    assert len(selected) == 8
    assert {record.structural_state_id for record in selected} == {"1", "2", "3", "4"}
    assert [record.sample_id for record in selected[:4]] == [
        "sample_1_0",
        "sample_2_0",
        "sample_3_0",
        "sample_4_0",
    ]


def test_normalization_sampling_does_not_duplicate_records() -> None:
    selected = _sample_normalization_records(_records(), max_records=99)

    assert len(selected) == 12
    assert len({record.sample_id for record in selected}) == 12
