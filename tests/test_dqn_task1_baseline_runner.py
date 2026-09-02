"""Tests for the preregistered Issue #41 training series."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import training.run_dqn_task1_baseline as baseline


def test_registered_seed_pairs_match_issue_41() -> None:
    assert [
        (run.run, run.world_seed, run.agent_seed)
        for run in baseline.REGISTERED_RUNS
    ] == [
        (1, 12001, 22001),
        (2, 12002, 22002),
        (3, 12003, 22003),
        (4, 12004, 22004),
        (5, 12005, 22005),
    ]
    assert baseline.EPISODES_PER_RUN == 10_000


def test_preflight_rejects_dirty_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline, "_git_is_dirty", lambda: True)

    with pytest.raises(RuntimeError, match="clean committed worktree"):
        baseline._preflight(tmp_path / "outputs" / "issue-41")


def test_preflight_rejects_existing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "agent" / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"existing")
    monkeypatch.setattr(baseline, "CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(baseline, "_git_is_dirty", lambda: False)

    with pytest.raises(FileExistsError, match="Refusing to resume"):
        baseline._preflight(tmp_path / "outputs" / "issue-41")


def test_seed_collision_is_detected_from_runner_metadata(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs" / "issue-41"
    metadata_path = tmp_path / "outputs" / "old-run" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps({"world_seed": 12003, "agent_seed": 99}),
        encoding="utf-8",
    )

    assert baseline._find_seed_collisions(output_root) == [metadata_path]


def test_series_runs_serially_and_moves_each_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "agent" / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    output_root = tmp_path / "outputs" / "issue-41"
    calls: list[tuple[int, int, bool]] = []

    def fake_run_experiment(**kwargs: object) -> Path:
        world_seed = int(kwargs["world_seed"])
        agent_seed = int(kwargs["agent_seed"])
        calls.append((world_seed, agent_seed, checkpoint_path.exists()))
        run_directory = Path(kwargs["output_root"]) / f"run-{len(calls)}"
        run_directory.mkdir()
        checkpoint_path.write_bytes(f"model-{world_seed}".encode())
        return run_directory

    source_hash = "a" * 64
    monkeypatch.setattr(baseline, "CHECKPOINT_PATH", checkpoint_path)
    profile = replace(
        baseline.DEFAULT_PROFILE,
        expected_agent_source_sha256=source_hash,
    )
    monkeypatch.setattr(baseline, "_git_is_dirty", lambda: False)
    monkeypatch.setattr(baseline, "_git_commit", lambda: "b" * 40)
    monkeypatch.setattr(
        baseline,
        "_agent_configuration_reference",
        lambda _agent: {
            "path": "agent_code/DagobertDuckDQN",
            "sha256": source_hash,
            "snapshot_path": None,
        },
    )
    monkeypatch.setattr(baseline, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(baseline, "plot_run", lambda _path: [])

    series_directory = baseline.run_registered_series(output_root, profile)
    manifest = json.loads(
        (series_directory / "series.json").read_text(encoding="utf-8")
    )

    assert calls == [
        (12001, 22001, False),
        (12002, 22002, False),
        (12003, 22003, False),
        (12004, 22004, False),
        (12005, 22005, False),
    ]
    assert manifest["status"] == "completed"
    assert [run["status"] for run in manifest["runs"]] == [
        "completed"
    ] * 5
    assert all(
        len(run["artifact"]["sha256"]) == 64
        for run in manifest["runs"]
    )
    assert not checkpoint_path.exists()


def test_failed_run_is_retained_and_later_runs_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "agent" / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    output_root = tmp_path / "outputs" / "issue-41"
    call_count = 0

    def fake_run_experiment(**kwargs: object) -> Path:
        nonlocal call_count
        call_count += 1
        run_directory = Path(kwargs["output_root"]) / f"run-{call_count}"
        run_directory.mkdir()
        checkpoint_path.write_bytes(f"model-{call_count}".encode())
        if call_count == 2:
            raise RuntimeError("interrupted")
        return run_directory

    source_hash = "c" * 64
    monkeypatch.setattr(baseline, "CHECKPOINT_PATH", checkpoint_path)
    profile = replace(
        baseline.DEFAULT_PROFILE,
        expected_agent_source_sha256=source_hash,
    )
    monkeypatch.setattr(baseline, "_git_is_dirty", lambda: False)
    monkeypatch.setattr(baseline, "_git_commit", lambda: "d" * 40)
    monkeypatch.setattr(
        baseline,
        "_agent_configuration_reference",
        lambda _agent: {
            "path": "agent_code/DagobertDuckDQN",
            "sha256": source_hash,
            "snapshot_path": None,
        },
    )
    monkeypatch.setattr(baseline, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(baseline, "plot_run", lambda _path: [])

    with pytest.raises(RuntimeError, match="1 of 5 training runs failed"):
        baseline.run_registered_series(output_root, profile)

    series_directory = next(output_root.iterdir())
    manifest = json.loads(
        (series_directory / "series.json").read_text(encoding="utf-8")
    )
    failed = manifest["runs"][1]
    assert call_count == 5
    assert failed["status"] == "failed"
    assert "interrupted" in failed["error"]
    assert failed["artifact"]["path"].endswith("failed-checkpoint.pt")
