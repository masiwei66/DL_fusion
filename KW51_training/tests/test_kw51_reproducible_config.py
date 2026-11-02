from __future__ import annotations

from types import SimpleNamespace

from KW51_training.config import load_experiment_config
from KW51_training.data.fem_insar import FEMArchiveRecord, make_fem_window_refs
from KW51_training.training.kw51_runner import _archive_provenance


def test_kw51_local_smoke_config_uses_cpu_and_distinct_eval_stride() -> None:
    config = load_experiment_config("configs/kw51_candidate_ranking_region_smoke.yaml")

    assert config.task == "kw51_candidate_ranking"
    assert config.input_mode == "region"
    assert config.train.device == "cpu"
    assert config.smoke.enabled
    assert config.data.eval_stride == 8


def test_kw51_evaluation_stride_controls_window_references(tmp_path) -> None:
    record = FEMArchiveRecord(
        path=tmp_path / "fixture.h5",
        sample_id="sample-1",
        structural_state_id="state-1",
        trajectory_id="trajectory-1",
        split_group="state-1",
        metadata={"acquisition_count": 26},
    )

    refs = make_fem_window_refs([record], length=8, stride=8)

    assert [ref.start for ref in refs] == [0, 8, 16]


def test_kw51_archive_provenance_points_to_audit_when_run_manifest_is_missing(tmp_path) -> None:
    audit_path = tmp_path / "fem_archive_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    record = FEMArchiveRecord(
        path=tmp_path / "fixture.h5",
        sample_id="sample-1",
        structural_state_id="state-1",
        trajectory_id="trajectory-1",
        split_group="state-1",
        metadata={},
    )
    index = SimpleNamespace(
        run_manifest={
            "run_manifest_missing": True,
            "audit_report": {"status": "PASS_FEM_ARCHIVE", "blockers": []},
        },
        records=(record,),
    )

    provenance = _archive_provenance(index)

    assert provenance["audit_report_file"] == "fem_archive_audit.json"
    assert provenance["audit_status"] == "PASS_FEM_ARCHIVE"
