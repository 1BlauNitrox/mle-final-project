"""Tests for the registered Issue #41 DQN evaluation workflow."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import training.evaluate_dqn_task1_baseline as evaluation
from training.metrics import CSV_COLUMNS


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
    monkeypatch.setattr(evaluation, "STAGED_CHECKPOINT_PATH", checkpoint)

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
    monkeypatch.setattr(evaluation, "STAGED_CHECKPOINT_PATH", checkpoint)

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
    for name in ("metadata.json", "summary.json"):
        (run_directory / name).write_text("data", encoding="utf-8")
    row = _deterministic_row(digest="a" * 64)
    row.update({"schema_version": 1, "round": 1, "agent": evaluation.AGENT, "mode": "evaluation"})
    with (run_directory / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    assert evaluation._job_is_complete(job, tmp_path)


def test_completed_job_without_action_digest_is_repeated(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    for name in ("metadata.json", "summary.json"):
        (run_directory / name).write_text("data", encoding="utf-8")
    row = _deterministic_row(digest=None)
    row.update({"schema_version": 1, "round": 1, "agent": evaluation.AGENT, "mode": "evaluation"})
    with (run_directory / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    assert not evaluation._job_is_complete(
        {"status": "completed", "run_directory": "run"}, tmp_path
    )


def test_executed_action_sequence_digest_depends_on_order() -> None:
    from environment import GenericWorld

    world = GenericWorld.__new__(GenericWorld)
    world.replay = {
        "actions": {
            "forward": ["UP", "RIGHT"],
            "reversed": ["RIGHT", "UP"],
            "repeat": ["UP", "RIGHT"],
        }
    }

    forward = world.executed_action_sequence_digest("forward")

    assert forward != world.executed_action_sequence_digest("reversed")
    assert forward == world.executed_action_sequence_digest("repeat")
    assert world.executed_action_sequence_digest("absent") is None


def test_executed_action_sequence_digest_normalizes_any_non_action_value() -> None:
    """act() can return anything; agents.py already tolerates this by bucketing
    non-conforming values into "action_unknown" rather than rejecting them.

    rule_based_agent returns None in some states, so a multi-agent round would
    otherwise raise while merely recording statistics. The same is true for any
    other value a misbehaving or future agent might return, not only None.
    """
    from environment import GenericWorld

    world = GenericWorld.__new__(GenericWorld)
    world.replay = {
        "actions": {
            "with_none": ["UP", None, "RIGHT"],
            "with_int": ["UP", 42, "RIGHT"],
            "with_list": ["UP", ["UP"], "RIGHT"],
            "with_bad_string": ["UP", "nonsense", "RIGHT"],
            "with_wait": ["UP", "WAIT", "RIGHT"],
            "repeat_of_none": ["UP", None, "RIGHT"],
        }
    }

    none_digest = world.executed_action_sequence_digest("with_none")

    assert isinstance(none_digest, str) and len(none_digest) == 64
    # None, an int, a list and an invalid string all normalize to the same
    # <unknown> token, so they collapse onto one digest.
    assert none_digest == world.executed_action_sequence_digest("with_int")
    assert none_digest == world.executed_action_sequence_digest("with_list")
    assert none_digest == world.executed_action_sequence_digest(
        "with_bad_string"
    )
    # WAIT is a real action and must stay distinguishable from "unknown".
    assert none_digest != world.executed_action_sequence_digest("with_wait")
    assert none_digest == world.executed_action_sequence_digest(
        "repeat_of_none"
    )


def test_executed_action_sequence_digest_passes_through_every_valid_action() -> None:
    from environment import GenericWorld

    world = GenericWorld.__new__(GenericWorld)
    sequence = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
    world.replay = {"actions": {"agent": list(sequence)}}

    digest = world.executed_action_sequence_digest("agent")

    assert digest == hashlib.sha256(
        "\n".join(sequence).encode("utf-8")
    ).hexdigest()


def test_executed_action_sequence_digest_is_none_before_a_round_starts() -> None:
    from environment import GenericWorld

    world = GenericWorld.__new__(GenericWorld)

    assert world.executed_action_sequence_digest("any") is None


def _deterministic_row(*, digest: str | None, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        evaluation.ACTION_SEQUENCE_COLUMN: digest,
        "episode_steps": 100,
        "survival_steps": 100,
        "score": 50,
        "coins_collected": 50,
        "invalid_actions": 0,
        "attempted_actions": 100,
        "invalid_action_rate": 0.0,
        "survived": True,
        "termination_reason": "step_limit",
        "action_up": 25,
        "action_right": 25,
        "action_down": 25,
        "action_left": 25,
        "action_wait": 0,
        "action_bomb": 0,
        "action_unknown": 0,
    }
    row.update(overrides)
    return row


def _check_determinism(
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary: dict[str, object],
    repeat: dict[str, object],
) -> bool:
    monkeypatch.setattr(evaluation, "DEVELOPMENT_SEEDS", (31_001,))

    def fake_read_job_row(*, manifest, series_directory, job_key):  # noqa: ANN001
        return primary if ":primary:" in job_key else repeat

    monkeypatch.setattr(evaluation, "_read_job_row", fake_read_job_row)

    return evaluation._model_is_deterministic(
        manifest={},
        series_directory=Path("."),
        model=evaluation.MODELS[0],
    )


def test_identical_evaluations_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _deterministic_row(digest="a" * 64)

    assert _check_determinism(monkeypatch, primary=row, repeat=dict(row))


def test_reordered_actions_fail_determinism_despite_equal_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action totals cannot distinguish UP,RIGHT from RIGHT,UP."""
    primary = _deterministic_row(digest="a" * 64)
    repeat = _deterministic_row(digest="b" * 64)

    # Every non-sequence field is identical, so only the digest can catch this.
    assert all(
        primary[column] == repeat[column]
        for column in evaluation.DETERMINISTIC_COLUMNS
        if column != evaluation.ACTION_SEQUENCE_COLUMN
    )
    assert not _check_determinism(monkeypatch, primary=primary, repeat=repeat)


@pytest.mark.parametrize("missing", [None, ""])
def test_absent_action_sequence_digest_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
    missing: str | None,
) -> None:
    row = _deterministic_row(digest=missing)

    with pytest.raises(ValueError, match="determinism cannot be verified"):
        _check_determinism(monkeypatch, primary=row, repeat=dict(row))


def test_manifest_write_retries_transient_windows_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "manifest.json"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(evaluation, "sleep", lambda _seconds: None)

    evaluation._write_json(destination, {"status": "completed"})

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "completed"
    }
