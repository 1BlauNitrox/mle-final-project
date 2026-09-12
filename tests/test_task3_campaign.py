"""Prospective budgets, pairing, uncertainty and conjunctive Task 3 decisions."""

from copy import deepcopy

import numpy as np
import pytest

from training import run_task3_campaign as campaign
from training.analyze_task3_campaign import SUITES, crossed_interval, decide


def passing_rows():
    rows = []
    for suite in SUITES:
        for arm, models in (("candidate", range(1, 6)), ("reference", [0])):
            for model in models:
                for seed in range(40):
                    for suffix in ("primary", "repeat"):
                        rows.append(
                            {
                                "arm": arm,
                                "replica": f"r{model}",
                                "suite": f"{suite}-{suffix}",
                                "world_seed": seed,
                                "agent_seed": seed + 100,
                                "opponent_count": int(suite == "classic-peaceful"),
                                "opponents_eliminated": int(arm == "candidate"),
                                "score": 9 if arm == "candidate" else 4,
                                "score_margin": 5 if arm == "candidate" else -5,
                                "first_place": int(arm == "candidate"),
                                "tied_first": 0,
                                "coins_collected": 4,
                                "initially_available_coins": 9,
                                "self_kills": 0,
                                "survived": True,
                                "crates_destroyed": 2,
                                "bombs_dropped": 1,
                                "survival_steps": 400,
                                "episode_steps": 400,
                                "attempted_actions": 400,
                                "invalid_actions": 0,
                                "action_wait": 400,
                                "decision_time_p95_ms": 1,
                                "decision_time_max_ms": 2,
                                "decision_times_ms": [1, 2],
                            }
                        )
    return rows


def test_registered_protocol_has_approved_budgets_and_no_seed_collisions():
    config, plans, report = campaign.validate_protocol()
    assert config["gates"]["elimination_min"] == 0.6
    assert config["gates"]["first_place_min"] == 0.6
    assert report["training_episodes"] == 50000
    assert report["evaluation_episodes"] == 1920
    assert not report["compute_authorized"] and not report["task2_complete"]
    assert len(plans["candidate"].replicas) == 5


def test_crossed_bootstrap_preserves_pairing_and_reference_is_not_replicated():
    baseline = np.arange(40, dtype=float)
    value = crossed_interval(np.tile(baseline + 0.25, (5, 1)), baseline)
    assert value == {"mean": 0.25, "lower": 0.25, "upper": 0.25}
    with pytest.raises(ValueError, match="Unpaired"):
        crossed_interval(np.zeros((5, 40)), np.zeros((5, 40)))


def test_all_gates_required_and_median_selection_is_mechanical():
    config = campaign.validate_protocol()[0]
    result = decide(passing_rows(), config)
    assert result["status"] == "exploratory_pass"
    assert result["selected_replica"] == "r3"
    assert not result["task2_complete"]


@pytest.mark.parametrize(
    "mutation, failed_gate",
    [
        ({"opponents_eliminated": 0}, "elimination_absolute"),
        ({"first_place": 0, "tied_first": 1}, "first_place_absolute"),
        ({"self_kills": 1}, "self_kill_absolute"),
        ({"decision_time_max_ms": 100}, "runtime"),
        ({"coins_collected": 0}, "coin-heaven-retention/collection"),
        ({"crates_destroyed": 0}, "classic-retention/crates"),
        ({"invalid_actions": 4}, "classic-peaceful/invalid_actions"),
    ],
)
def test_failed_gate_never_selects_a_fallback(mutation, failed_gate):
    config = deepcopy(campaign.validate_protocol()[0])
    config["bootstrap"]["samples"] = 100
    rows = passing_rows()
    for row in rows:
        if row["arm"] == "candidate":
            row.update(mutation)
    result = decide(rows, config)
    assert not result["gates"][failed_gate]
    assert result["selected_replica"] is None


def test_missing_candidate_seed_is_rejected():
    config = campaign.validate_protocol()[0]
    with pytest.raises(ValueError, match="Unpaired"):
        decide(passing_rows()[1:], config)


def test_no_authorization_never_executes(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "execute_plan", lambda *a, **k: pytest.fail("game started"))
    with pytest.raises(ValueError, match="authorize-compute"):
        campaign.main(["--output-root", str(tmp_path / "not-created")])
    assert not (tmp_path / "not-created").exists()


def test_evidence_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        campaign.relative_file(tmp_path, "../outside")


def test_serial_plans_share_monitor_and_resume_authorization(tmp_path, monkeypatch):
    config, plans, report = campaign.validate_protocol()
    binding = tmp_path / "binding"
    binding.mkdir()
    (binding / "binding.json").write_text("{}")
    monkeypatch.setattr(campaign, "validate_protocol", lambda *_: (config, plans, report))
    monkeypatch.setattr(campaign, "git", lambda *args: "a" * 40 if args[0] == "rev-parse" else "")
    monitors, calls = [], []

    class Monitor:
        def __init__(self, **kwargs):
            monitors.append(self)
            self.campaign_metadata = kwargs["campaign_metadata"]
            kwargs["state_path"].write_text("{}")

        def check(self):
            pass

    monkeypatch.setattr(campaign, "CampaignResourceMonitor", Monitor)
    monkeypatch.setattr(campaign, "execute_plan", lambda p, **kwargs: calls.append((p, kwargs)))
    argv = [
        "--binding-dir",
        str(binding),
        "--output-root",
        str(tmp_path / "campaign"),
        "--reviewed-commit",
        "a" * 40,
        "--authorized-by",
        "owner",
        "--hardware-description",
        "synthetic test",
        "--available-memory-gib",
        "8",
        "--authorize-compute",
    ]
    campaign.main(argv)
    assert [p for p, _ in calls] == [plans["reference"], plans["candidate"]]
    assert all(kwargs["process_monitor"] is monitors[0] for _, kwargs in calls)
    auth = (tmp_path / "campaign/authorization.json").read_bytes()
    campaign.main([*argv, "--resume"])
    assert (tmp_path / "campaign/authorization.json").read_bytes() == auth
    assert not (tmp_path / "campaign/.task3-campaign.lock").exists()
    with pytest.raises(ValueError, match="Authorization/resume"):
        campaign.main(argv)


def test_compact_export_recomputes_and_rejects_tampering(tmp_path, monkeypatch):
    from training import analyze_task3_campaign as analysis

    config = campaign.validate_protocol()[0]
    rows = passing_rows()
    for row in rows:
        row["artifact_sha256"] = "a" * 64
    auth = {"identity": {"protocol_sha256": campaign.sha256(campaign.CONFIG)}}
    monkeypatch.setattr(analysis, "load_evidence", lambda *_: (config, rows, [], {}, auth, {}))
    output = tmp_path / "analysis"
    analysis.analyze(tmp_path, tmp_path, output)
    assert analysis.verify_compact(output)["verified"]
    result_path = output / "result.json"
    result = campaign.read_json(result_path)
    result["selected_artifact_sha256"] = "b" * 64
    result_path.write_text(__import__("json").dumps(result))
    with pytest.raises(ValueError, match="Selected artifact"):
        analysis.verify_compact(output)
    result["selected_artifact_sha256"] = "a" * 64
    result_path.write_text(__import__("json").dumps(result))
    path = output / "observations.json.gz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash/size"):
        analysis.verify_compact(output)
