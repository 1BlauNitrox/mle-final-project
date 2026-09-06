"""Focused contracts for staged experiment run plans."""

from __future__ import annotations

import json
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

import training.run_plan as run_plan


def _plan_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "plan_id": "test-matrix",
        "agent": "DerKleineSprengstoffkapitalist",
        "artifact_path": "model.npz",
        "max_parallel_training": 2,
        "replicas": [
            {"id": "r1", "world_seed": 101, "agent_seed": 201},
            {"id": "r2", "world_seed": 102, "agent_seed": 202},
        ],
        "training_stages": [
            {"id": "coins", "scenario": "coin-heaven", "rounds": 2},
            {"id": "crates", "scenario": "loot-crate", "rounds": 3},
        ],
        "evaluation_suites": [
            {
                "id": "task1-regression",
                "population": "development",
                "scenario": "coin-heaven",
                "rounds": 1,
                "world_seeds": [301],
                "agent_seeds": [401],
            },
            {
                "id": "task3",
                "population": "confirmation",
                "scenario": "classic",
                "rounds": 1,
                "world_seeds": [501],
                "agent_seeds": [601],
                "opponents": ["peaceful_agent", "coin_collector_agent"],
            },
        ],
    }


def _write_plan(tmp_path: Path, data: dict[str, object]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_schema_expands_deterministic_ordered_isolated_matrix(tmp_path: Path) -> None:
    path = _write_plan(tmp_path, _plan_data())

    first = run_plan.load_plan(path)
    second = run_plan.load_plan(path)

    assert first == second
    assert [job.run_id for job in first.jobs] == [
        "train-r1-coins",
        "train-r1-crates",
        "train-r2-coins",
        "train-r2-crates",
        "eval-r1-task1-regression-seed-001",
        "eval-r1-task3-seed-001",
        "eval-r2-task1-regression-seed-001",
        "eval-r2-task3-seed-001",
    ]
    assert first.episode_budget == 14
    assert first.jobs[5].opponents == ("peaceful_agent", "coin_collector_agent")
    assert first.jobs[0].world_seed != first.jobs[2].world_seed
    assert first.jobs[0].agent_seed != first.jobs[2].agent_seed


def test_reward_variant_defaults_to_control_and_can_be_overridden(tmp_path: Path) -> None:
    default_plan = run_plan.load_plan(_write_plan(tmp_path / "default", _plan_data()))
    assert default_plan.reward_variant == "control"

    overridden = _plan_data()
    overridden["reward_variant"] = "safety_bomb"
    overridden_plan = run_plan.load_plan(_write_plan(tmp_path / "overridden", overridden))
    assert overridden_plan.reward_variant == "safety_bomb"


def test_agent_fingerprint_ignores_runtime_logs_and_staged_checkpoint(
    tmp_path: Path,
) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "callbacks.py").write_text("# policy\n", encoding="utf-8")
    before = run_plan._fingerprint_directory(agent)
    (agent / "logs").mkdir()
    (agent / "logs" / "agent.log").write_text("runtime\n", encoding="utf-8")
    (agent / run_plan.STAGED_EVALUATION_CHECKPOINT_NAME).write_bytes(b"staged")
    assert run_plan._fingerprint_directory(agent) == before


def test_schema_rejects_invalid_plans_before_execution(tmp_path: Path) -> None:
    mutations = {
        "schema version": (lambda plan: plan.update(schema_version=2), "schema version"),
        "duplicate IDs": (
            lambda plan: plan["replicas"].append(plan["replicas"][0].copy()),
            "Duplicate replica IDs",
        ),
        "seed overlap": (
            lambda plan: plan["evaluation_suites"][0]["world_seeds"].__setitem__(0, 101),
            "overlap",
        ),
        "scenario": (
            lambda plan: plan["training_stages"][0].update(scenario="unknown"),
            "Unsupported scenario",
        ),
        "opponent": (
            lambda plan: plan["evaluation_suites"][0].update(opponents=["unknown"]),
            "Unsupported supplied opponents",
        ),
        "budget": (
            lambda plan: plan["training_stages"][0].update(rounds=0),
            "rounds must be positive",
        ),
        "artifact path": (
            lambda plan: plan.update(artifact_path="../model.npz"),
            "unambiguous path",
        ),
        "reward variant": (
            lambda plan: plan.update(reward_variant="unknown"),
            "reward_variant must be one of",
        ),
    }
    for name, (mutate, message) in mutations.items():
        data = _plan_data()
        mutate(data)
        with pytest.raises(ValueError, match=message):
            run_plan.load_plan(_write_plan(tmp_path / name, data))


def test_execution_preserves_failures_and_resumes_exactly(tmp_path: Path) -> None:
    data = _plan_data()
    data["max_parallel_training"] = 1
    data["replicas"] = [data["replicas"][0]]
    data["training_stages"] = [data["training_stages"][0]]
    data["evaluation_suites"] = [data["evaluation_suites"][0]]
    plan = run_plan.load_plan(_write_plan(tmp_path, data))
    output_root = tmp_path / "outputs"
    calls: list[dict[str, object]] = []

    def fake_runner(**kwargs: object) -> Path:
        calls.append(kwargs)
        run_directory = Path(kwargs["output_root"]) / str(kwargs["run_id"])
        run_directory.mkdir(parents=True)
        (run_directory / "metadata.json").write_text("{}\n", encoding="utf-8")
        alias = run_plan.REPOSITORY_ROOT / "agent_code" / str(kwargs["agent"])
        if len(calls) == 1:
            (alias / "model.npz").write_bytes(b"partial")
            raise RuntimeError("planned failure")
        if kwargs["mode"] == "training":
            (alias / "model.npz").write_bytes(b"completed")
        else:
            assert not (alias / "model.npz").stat().st_mode & stat.S_IWRITE
        return run_directory

    with patch.object(run_plan, "run_experiment", side_effect=fake_runner):
        with pytest.raises(RuntimeError, match="planned failure"):
            run_plan.execute_plan(plan, output_root=output_root)

        plan_directory = output_root / plan.plan_id
        failed_status = json.loads((plan_directory / "status.json").read_text())
        failed_job = failed_status["jobs"]["train-r1-coins"]
        assert failed_job["status"] == "failed"
        assert len(failed_job["attempts"]) == 1
        assert (plan_directory / "jobs/train-r1-coins/attempt-001-failed-agent").is_dir()
        assert not run_plan._alias_directory(plan, plan.replicas[0]).exists()

        run_plan.execute_plan(plan, output_root=output_root, resume=True)

    status = json.loads((plan_directory / "status.json").read_text())
    assert status["status"] == "completed"
    assert len(status["jobs"]["train-r1-coins"]["attempts"]) == 2
    assert status["jobs"]["eval-r1-task1-regression-seed-001"]["status"] == "completed"
    assert (plan_directory / "artifacts/r1/coins/model.npz").read_bytes() == b"completed"
    assert calls[-1]["mode"] == "evaluation"
    assert calls[-1]["metadata_extra"]["run_plan"]["processes"] == 1
    assert calls[-1]["metadata_extra"]["run_plan"]["artifact_writable"] is False
    assert calls[-1]["environment_overrides"] == {
        "BOMBERMAN_DQN_ACTION_MASKING": "none",
        "BOMBERMAN_DQN_REWARD_VARIANT": "control",
        "BOMBERMAN_EVALUATION_CHECKPOINT": "model.npz",
    }


def test_resume_rejects_every_protected_fingerprint(tmp_path: Path) -> None:
    data = _plan_data()
    data["training_stages"] = []
    data["evaluation_suites"] = []
    plan = run_plan.load_plan(_write_plan(tmp_path, data))
    output_root = tmp_path / "outputs"

    with patch.object(run_plan, "_remove_staging_aliases"):
        run_plan.execute_plan(plan, output_root=output_root)

    for key in (
        "source",
        "configuration",
        "framework",
        "agent",
        "dependencies_sha256",
        "parent_artifacts",
    ):
        changed = dict(plan.fingerprints)
        changed[key] = "changed"
        mismatched = replace(plan, fingerprints=changed)
        with pytest.raises(ValueError, match=key):
            run_plan.execute_plan(mismatched, output_root=output_root, resume=True)


def test_parallel_replicas_keep_status_and_artifacts_isolated(tmp_path: Path) -> None:
    data = _plan_data()
    data["evaluation_suites"] = []
    data["training_stages"] = [data["training_stages"][0]]
    plan = run_plan.load_plan(_write_plan(tmp_path, data))
    calls: list[str] = []

    def fake_runner(**kwargs: object) -> Path:
        alias = str(kwargs["agent"])
        calls.append(alias)
        run_directory = Path(kwargs["output_root"]) / str(kwargs["run_id"])
        run_directory.mkdir(parents=True)
        (run_directory / "metadata.json").write_text("{}\n", encoding="utf-8")
        (run_plan.REPOSITORY_ROOT / "agent_code" / alias / "model.npz").write_bytes(
            alias.encode("utf-8")
        )
        return run_directory

    with patch.object(run_plan, "run_experiment", side_effect=fake_runner):
        plan_directory = run_plan.execute_plan(plan, output_root=tmp_path / "outputs")

    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert {
        (plan_directory / "artifacts" / replica.replica_id / "coins" / "model.npz").read_bytes()
        for replica in plan.replicas
    } == {alias.encode("utf-8") for alias in calls}
    status = json.loads((plan_directory / "status.json").read_text())
    assert {job["status"] for job in status["jobs"].values()} == {"completed"}


def test_existing_single_run_accepts_deterministic_id_and_plan_metadata(
    tmp_path: Path,
) -> None:
    def fake_process(command: list[str], **_: object) -> SimpleNamespace:
        path = Path(command[command.index("--save-stats") + 1])
        path.write_text('{"by_round": {}}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    with (
        patch("training.run_experiment._git_commit", return_value="a" * 40),
        patch("training.run_experiment._git_is_dirty", return_value=False),
        patch("training.run_experiment.subprocess.run", side_effect=fake_process),
        patch("training.run_experiment.normalize_framework_statistics", return_value=[{}]),
        patch("training.run_experiment.aggregate_episodes_csv"),
    ):
        from training.run_experiment import run_experiment

        directory = run_experiment(
            agent="random_agent",
            mode="evaluation",
            scenario="coin-heaven",
            rounds=1,
            world_seed=1,
            agent_seed=2,
            opponents=[],
            output_root=tmp_path,
            run_id="attempt-001",
            metadata_extra={"run_plan": {"job_id": "example"}},
        )

    metadata = json.loads((directory / "metadata.json").read_text())
    assert directory.name == "attempt-001"
    assert metadata["run_plan"] == {"job_id": "example"}
