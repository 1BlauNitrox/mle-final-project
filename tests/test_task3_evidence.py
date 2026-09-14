"""Small complete filesystem campaign checks evidence validation, not game strength."""

import json
from dataclasses import replace

import pytest

from tests.test_experiment_metrics import make_agent_statistics
from training import analyze_task3_campaign as analysis
from training import run_task3_campaign as campaign
from training.metrics import normalize_episode_rows, write_episodes_csv


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    config, source_plans, _ = campaign.validate_protocol()
    plans = {}
    root, binding = tmp_path / "campaign", tmp_path / "binding"
    write(binding / "binding.json", {"synthetic_test": True})
    for name, plan in source_plans.items():
        replica = plan.replicas[0]
        jobs = [
            replace(j, rounds=1)
            for j in plan.jobs
            if j.replica == replica.replica_id
            and (
                j.kind == "training"
                or (j.stage_or_suite.startswith("classic-peaceful") and j.world_seed == 1091101)
            )
        ]
        plans[name] = replace(plan, replicas=(replica,), jobs=tuple(jobs))
    for name, plan in plans.items():
        directory = root / "plans" / plan.plan_id
        directory.mkdir(parents=True)
        artifact = directory / "model.pt"
        artifact.write_bytes(name.encode())
        replicas = (replace(plan.replicas[0], parent_artifact_sha256=campaign.sha256(artifact)),)
        plans[name] = replace(plan, replicas=replicas)
    auth = {
        "identity": {
            "limits": config["resources"],
            "opponent_sources": {},
            "protocol_sha256": campaign.sha256(campaign.CONFIG),
            "binding_sha256": campaign.sha256(binding / "binding.json"),
            "plans": {n: campaign.portable_plan(p) for n, p in plans.items()},
            "reviewed_commit": "a" * 40,
        },
        "authorized_at": "2026-09-11T00:00:00+00:00",
    }
    write(root / "authorization.json", auth)
    write(
        root / "resources.json",
        {
            "limit_reached": None,
            "active_root_pids": [],
            "authorized_at": auth["authorized_at"],
            "limits": {"cpu_seconds": 86400, "wall_seconds": 54000, "memory_bytes": 8 * 1024**3},
            "cpu_seconds_consumed": 1,
            "wall_seconds_elapsed": 1,
            "peak_memory_bytes": 1024,
        },
    )
    for plan in plans.values():
        directory = root / "plans" / plan.plan_id
        write(directory / "resolved_plan.json", plan.to_dict())
        status = {"status": "completed", "jobs": {}}
        for j in plan.jobs:
            run = directory / "jobs" / j.run_id / "attempt-001"
            stats = make_agent_statistics(
                kills=0,
                executed_action_sequence_sha256="b" * 64,
                decision_time_median_ms=1.5,
                decision_time_p95_ms=1.95,
                decision_time_max_ms=2,
                decision_times_ms=[1, 2],
            )
            raw = {"by_round": {"1": {"agents": {"observed": stats, "opponent": dict(stats)}}}}
            write(run / "framework_stats.json", raw)
            write_episodes_csv(normalize_episode_rows(raw, j.kind), run / "episodes.csv")
            meta = {
                "observed_agent": "observed",
                "opponent_seed_policy": "task3_per_slot_v1",
                "status": "completed",
                "return_code": 0,
                "git_dirty": False,
                "git_commit": "a" * 40,
                "mode": j.kind,
                "world_seed": j.world_seed,
                "agent_seed": j.agent_seed,
                "scenario": j.scenario,
                "rounds": 1,
                "opponents": list(j.opponents),
                "run_plan": {
                    "campaign": {
                        "issue": 109,
                        "opponent_seed_policy": "task3_per_slot_v1",
                        "authorization_sha256": campaign.sha256(root / "authorization.json"),
                        "reviewed_commit": "a" * 40,
                    },
                    "job_id": j.run_id,
                    "logical_agent": plan.agent,
                    "processes": 1,
                    "artifact_writable": j.kind == "training",
                    "action_masking": plan.action_masking,
                    "escape_continuations": plan.escape_continuations,
                    "replay_treatment": plan.replay_treatment,
                },
            }
            write(run / "metadata.json", meta)
            status["jobs"][j.run_id] = {
                "status": "completed",
                "attempts": [{"status": "completed", "output": str(run.relative_to(directory))}],
                "artifact": {"path": "model.pt", "sha256": campaign.sha256(directory / "model.pt")},
            }
        write(directory / "status.json", status)
    monkeypatch.setattr(analysis, "validate_protocol", lambda *_: (config, plans, {}))
    return root, binding


def test_complete_evidence_filters_opponents_and_keeps_repeats(evidence):
    root, binding = evidence
    _, rows, training, manifest, _, _ = analysis.load_evidence(root, binding)
    assert len(rows) == 4 and len(training) == 1
    assert all(row["agent"] == "observed" for row in rows)
    assert len(manifest) == 15


@pytest.mark.parametrize(
    "field,value",
    [("observed_agent", "missing"), ("opponents", []), ("git_dirty", True), ("world_seed", 999)],
)
def test_tampered_job_provenance_is_rejected(evidence, field, value):
    root, binding = evidence
    path = next(root.glob("plans/*/jobs/*/attempt-001/metadata.json"))
    meta = campaign.read_json(path)
    meta[field] = value
    write(path, meta)
    with pytest.raises(ValueError):
        analysis.load_evidence(root, binding)


def test_changed_model_bytes_are_rejected(evidence):
    root, binding = evidence
    next(root.glob("plans/*/model.pt")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Artifact bytes"):
        analysis.load_evidence(root, binding)


def test_changed_repeat_is_rejected_even_when_csv_matches_raw(evidence):
    root, binding = evidence
    path = next(root.glob("plans/*/jobs/*repeat*/attempt-001/framework_stats.json"))
    raw = campaign.read_json(path)
    raw["by_round"]["1"]["agents"]["observed"]["executed_action_sequence_sha256"] = "c" * 64
    write(path, raw)
    write_episodes_csv(normalize_episode_rows(raw, "evaluation"), path.parent / "episodes.csv")
    with pytest.raises(ValueError, match="Deterministic"):
        analysis.load_evidence(root, binding)
