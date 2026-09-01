"""Tests for the registered Issue #41 DQN evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import training.evaluate_dqn_task1_baseline as evaluation


def test_registered_evaluation_matrix_matches_issue_41() -> None:
    assert tuple(range(31001, 31041)) == evaluation.DEVELOPMENT_SEEDS
    assert evaluation.EVALUATION_PASSES == ("primary", "repeat")
    assert [
        (model.run, model.agent_seed, model.artifact_name)
        for model in evaluation.MODELS
    ] == [
        (1, 22001, "run-01-final-checkpoint.pt"),
        (2, 22002, "run-02-final-checkpoint.pt"),
        (3, 22003, "run-03-final-checkpoint.pt"),
        (4, 22004, "run-04-final-checkpoint.pt"),
        (5, 22005, "run-05-final-checkpoint.pt"),
    ]


def test_remove_known_checkpoint_rejects_unknown_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"unknown")
    monkeypatch.setattr(evaluation, "CHECKPOINT_PATH", checkpoint)

    with pytest.raises(RuntimeError, match="unrecognized"):
        evaluation._remove_known_checkpoint(
            {"run-01": {"sha256": "0" * 64}}
        )


def test_remove_known_checkpoint_deletes_only_matching_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"known")
    digest = evaluation._sha256(checkpoint)
    monkeypatch.setattr(evaluation, "CHECKPOINT_PATH", checkpoint)

    evaluation._remove_known_checkpoint(
        {"run-01": {"sha256": digest}}
    )

    assert not checkpoint.exists()


def test_existing_manifest_must_match_artifacts_and_seeds(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "development_seeds": list(evaluation.DEVELOPMENT_SEEDS),
                "artifacts": {"run-01": {"sha256": "a" * 64}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact mismatch"):
        evaluation._load_or_create_manifest(
            manifest_path=manifest_path,
            series_directory=tmp_path,
            artifact_records={"run-01": {"sha256": "b" * 64}},
        )


def test_completed_job_requires_all_runner_outputs(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    job = {"status": "completed", "run_directory": "run"}

    assert not evaluation._job_is_complete(job, tmp_path)
    for name in ("metadata.json", "episodes.csv", "summary.json"):
        (run_directory / name).write_text("data", encoding="utf-8")
    assert evaluation._job_is_complete(job, tmp_path)
