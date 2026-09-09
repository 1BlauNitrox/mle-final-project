"""Integrity and retrospective-diagnostic tests for Issue #103."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.analyze_issue103_dqn_task2_reward_shaping import (
    PLAN_IDS,
    _criteria,
    rows_from_evidence,
    verify_from_evidence,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "2026-09-07-dqn-task2-reward-shaping"
)


def _timing_row() -> dict[str, object]:
    return {"decision_time_p95_ms": 1.0, "decision_time_max_ms": 2.0}


def test_retrospective_diagnostics_require_every_scenario_and_classic_survival() -> None:
    passing_collection = {
        "classic": {"ci_lower": -0.04},
        "coin-heaven": {"ci_lower": -0.03},
        "loot-crate": {"ci_lower": -0.02},
    }
    passing_survival = {"classic": {"ci_lower": 0.01}}

    assert all(
        _criteria([_timing_row()], passing_collection, passing_survival, True).values()
    )


def test_a_single_scenario_regression_rejects_collection_non_regression() -> None:
    collection = {
        "classic": {"ci_lower": -0.06},
        "coin-heaven": {"ci_lower": -0.01},
        "loot-crate": {"ci_lower": -0.01},
    }
    survival = {"classic": {"ci_lower": 0.01}}

    criteria = _criteria([_timing_row()], collection, survival, True)

    assert not criteria["collection_non_regression"]


def test_a_negative_classic_survival_bound_rejects_the_primary_gate() -> None:
    collection = {
        scenario: {"ci_lower": 0.0} for scenario in ("classic", "coin-heaven", "loot-crate")
    }
    survival = {"classic": {"ci_lower": -0.01}}

    criteria = _criteria([_timing_row()], collection, survival, True)

    assert not criteria["classic_survival_improved"]


def test_timing_gate_requires_both_limits_across_every_row() -> None:
    collection = {
        scenario: {"ci_lower": 0.0} for scenario in ("classic", "coin-heaven", "loot-crate")
    }
    survival = {"classic": {"ci_lower": 0.0}}
    slow_row = {"decision_time_p95_ms": 60.0, "decision_time_max_ms": 2.0}

    criteria = _criteria([_timing_row(), slow_row], collection, survival, True)

    assert not criteria["timing_within_limits"]


def test_committed_evidence_verifies_from_a_clean_checkout() -> None:
    _, matches = verify_from_evidence(
        EXPERIMENT_ROOT / "evidence",
        EXPERIMENT_ROOT / "result.json",
    )

    assert matches


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("size_bytes", 0, "Evidence size mismatch"),
        ("sha256", "0" * 64, "Evidence checksum mismatch"),
    ],
)
def test_evidence_verification_rejects_manifest_integrity_mismatch(
    tmp_path: Path,
    field: str,
    invalid_value: int | str,
    message: str,
) -> None:
    plan_id = PLAN_IDS["control"]
    plan_directory = tmp_path / plan_id
    plan_directory.mkdir(parents=True)
    (plan_directory / "training-episodes.csv.gz").write_bytes(b"training")
    (plan_directory / "evaluation-episodes.csv").write_bytes(b"evaluation")
    evidence_files = {}
    for filename in ("training-episodes.csv.gz", "evaluation-episodes.csv"):
        path = plan_directory / filename
        evidence_files[filename] = {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "plan_id": plan_id,
        "reward_variant": "control",
        "evidence_files": evidence_files,
    }
    manifest["evidence_files"]["evaluation-episodes.csv"][field] = invalid_value
    (plan_directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        rows_from_evidence(tmp_path)
