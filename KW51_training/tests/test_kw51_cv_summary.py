import json
from pathlib import Path

import pytest

from KW51_training.cli.summarize_cv import summarize


def _write_fold(
    base: Path,
    fold: int,
    *,
    top1: float,
    mrr: float,
    states: list[str],
) -> None:
    report = {
        "best_epoch": 1,
        "run_status": "FORMAL",
        "diagnostic_only": False,
        "evaluation_setting": "open_set_query_state_held_out",
        "gallery_exclusion": "query_state",
        "split_states": {"test": states},
        "metrics": {
            "test": {
                "top1_hit_rate": top1,
                "mrr": mrr,
                "pairwise_accuracy": 0.98,
                "ndcg": 0.97,
                "per_query_state": {
                    state: {
                        "top1_hit_rate": top1,
                        "mrr": mrr,
                        "query_groups": 100,
                    }
                    for state in states
                },
            }
        },
    }
    path = base.parent / f"{base.name}_fold{fold}"
    path.mkdir()
    (path / "training_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_summarize_cv_fold_and_state_levels(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "cv"
    base.mkdir(parents=True)
    _write_fold(base, 0, top1=0.9, mrr=0.95, states=["1", "2"])
    _write_fold(base, 1, top1=0.8, mrr=0.9, states=["3", "4"])

    summary = summarize(base, folds=2, repeats=100)

    assert summary["all_runs_formal"] is True
    assert summary["any_diagnostic_only"] is False
    assert summary["evaluation_setting"] == "open_set_query_state_held_out"
    assert summary["fold_level"]["top1_hit_rate"]["mean"] == pytest.approx(0.85)
    assert summary["state_level"]["top1_hit_rate"]["mean"] == pytest.approx(0.85)
    assert set(summary["state_level"]["states"]) == {"1", "2", "3", "4"}
    assert summary["folds"][0]["run_status"] == "FORMAL"
