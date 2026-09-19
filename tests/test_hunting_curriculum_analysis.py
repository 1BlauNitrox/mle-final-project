"""Complete three-replica evidence fixtures exercise actual eligibility decisions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.analyze_hunting_curriculum import analyze, diagnostic_summary
from scripts.curriculum_io import read, sha, write
from scripts.run_hunting_curriculum import CONFIG
from training.hunting_curriculum import BombCredits


@pytest.fixture
def complete_evidence(tmp_path):
    cfg = read(CONFIG)
    cfg["updates"] = 2
    cfg["bootstrap_resamples"] = 20
    for s in cfg["evaluation"]["final"].values():
        s["seeds"][1] = s["seeds"][0] + 3
    write(tmp_path / "config.json", cfg)
    for replica in range(1, 4):
        pair = tmp_path / "pairs" / f"r{replica}"
        write(pair / "decision.json", {"status": "training_complete"})
        for arm in ("control", "curriculum", "reference"):
            snapshot = pair / arm / "snapshots/final"
            if arm == "reference":
                directory = pair / "reference/final"
                digest = cfg["reference_sha256"]
            else:
                snapshot.mkdir(parents=True)
                (snapshot / "checkpoint.pt").write_bytes(f"{arm}-{replica}".encode())
                digest = sha(snapshot / "checkpoint.pt")
                directory = snapshot / "evaluation"
            for suite, setting in cfg["evaluation"]["final"].items():
                observations = []
                for i, seed in enumerate(range(setting["seeds"][0], setting["seeds"][1] + 1)):
                    kills = int(arm == "curriculum" and bool(setting["opponents"]))
                    row = {
                        "world_seed": seed,
                        "slot": i % (len(setting["opponents"]) + 1),
                        "opponents": setting["opponents"],
                        "scenario": setting["scenario"],
                        "epsilon": 0,
                        "native": {
                            "score": 4 + 5 * kills,
                            "kills": kills,
                            "coins": 4,
                            "initially_available_coins": 10,
                            "invalid": 0,
                            "survived": 1,
                            "self_kills": 0,
                            "decision_times_ms": [2.0] * 10,
                        },
                        "loop_windows": {
                            "eligible": 2 if i == 0 else 8,
                            "looping": 2 if i == 0 else 0,
                        },
                        "initial_exposure": {"reachable_attack_position": i == 0},
                        "attack_exposure": {
                            "safe_attack_steps": 3 if i == 0 else 0,
                            "safe_attack_bombs": int(i == 0),
                            "threatening_steps": 4,
                        },
                        "bomb_credits": [
                            {
                                "safe_attack": i == 0,
                                "detonated": True,
                                "kill_credits": kills,
                                "self_kill_credits": 0,
                            }
                        ],
                        "kind": "hunting" if arm == "curriculum" else "classic",
                    }
                    observations.append(row)
                write(
                    directory / f"{suite}.json", {"checkpoint_sha256": digest, "rows": observations}
                )
            if arm != "reference":
                # Four retained observations, two actual updates and128 sampled items.
                training = deepcopy(read(directory / "classic.json")["rows"])
                write(
                    snapshot / "state.json",
                    {
                        "updates": 2,
                        "episodes": 4,
                        "rows": training,
                        "tracker": {
                            "transitions": 8,
                            "generated": {"classic": 4, "hunting": 4},
                            "sampled": {"parent": 96, "hunting": 32},
                            "origins": ["parent", "classic", "hunting", "hunting"],
                        },
                    },
                )
    return tmp_path


def change_invalid(root, replica, suite, count):
    path = (
        root / "pairs" / f"r{replica}" / "curriculum/snapshots/final/evaluation" / f"{suite}.json"
    )
    evidence = read(path)
    evidence["rows"][0]["native"]["invalid"] = count
    write(path, evidence)


def test_complete_analysis_passes_and_reports_all_registered_diagnostics(complete_evidence):
    result = analyze([complete_evidence])
    assert result["complete"] and result["eligible"]
    assert result["candidate_for_confirmation"] == "curriculum-r1"
    summary = result["training_diagnostics"]["r1"]["curriculum"]["final"]
    assert summary["all"]["loops"]["eligible_windows"] == 26
    assert summary["all"]["loops"]["window_rate"] == pytest.approx(2 / 26)
    assert summary["all"]["attack"]["reachable_at_start_episodes"] == 1
    assert summary["all"]["attack"]["safe_attack_bombs"] == 1
    assert summary["all"]["attack"]["safe_bomb_kill_credits"] == 1
    assert summary["all"]["attack"]["native_kills"] == 4
    assert summary["replay"]["generated"]["fractions"]["hunting"] == 0.5
    assert summary["replay"]["sampled"]["fractions"]["hunting"] == 0.25
    assert summary["replay"]["resident"]["counts"]["hunting"] == 2
    assert result["evaluation_diagnostics"]["r1"]["curriculum"]["classic"]["invalid"]["count"] == 0


def test_single_bad_replica_rejects_even_when_pooled_increase_passes(complete_evidence):
    change_invalid(complete_evidence, 1, "classic", 3)
    result = analyze([complete_evidence])
    gate = result["invalid_actions"]["suites"]["classic"]["control"]
    assert gate["pooled_increase"] == 0.25
    assert gate["replica_increases"] == [0.75, 0.0, 0.0]
    assert not result["eligible"] and result["candidate_for_confirmation"] is None


def test_pooled_gate_rejects_even_when_each_replica_passes(complete_evidence):
    for r in range(1, 4):
        change_invalid(complete_evidence, r, "mixed", 2)
    result = analyze([complete_evidence])
    assert not result["invalid_actions"]["suites"]["mixed"]["reference"]["passed"]
    assert not result["eligible"]


@pytest.mark.parametrize("suite", ["coins", "crates"])
def test_any_solo_invalid_rejects(complete_evidence, suite):
    change_invalid(complete_evidence, 2, suite, 1)
    assert not analyze([complete_evidence])["eligible"]


def test_latency_worlds_do_not_decide_behavioral_gate(complete_evidence):
    change_invalid(complete_evidence, 1, "latency", 10)
    result = analyze([complete_evidence])
    assert result["eligible"]
    assert "latency" not in result["invalid_actions"]["suites"]


def test_stopped_pair_keeps_diagnostics(complete_evidence):
    write(complete_evidence / "pairs/r2/decision.json", {"status": "stopped_safety"})
    result = analyze([complete_evidence])
    assert not result["eligible"] and not result["complete"]
    assert result["training_diagnostics"]["r2"]["control"]["final"]["episodes"] == 4


def test_no_loop_exposure_and_missing_attribution_are_not_zero(complete_evidence):
    sample = read(complete_evidence / "pairs/r1/curriculum/snapshots/final/state.json")["rows"][0]
    sample["loop_windows"] = {"eligible": 0, "looping": 0}
    sample.pop("bomb_credits")
    result = diagnostic_summary([sample])
    assert result["loops"]["window_rate"] is None
    assert result["attack"]["safe_bomb_kill_credits"] is None
    assert result["attack"]["episode_kills_when_safe_attack_used"] == 1


def test_corrupt_checkpoint_and_replay_counts_fail_closed(complete_evidence):
    snapshot = complete_evidence / "pairs/r1/control/snapshots/final"
    path = snapshot / "state.json"
    state = read(path)
    state["tracker"]["transitions"] += 1
    write(path, state)
    with pytest.raises(ValueError, match="generated-count"):
        analyze([complete_evidence])
    state["tracker"]["transitions"] -= 1
    write(path, state)
    (snapshot / "checkpoint.pt").write_bytes(b"wrong-model")
    with pytest.raises(ValueError, match="Wrong checkpoint"):
        analyze([complete_evidence])


def test_native_bomb_attribution_handles_smoke_and_dead_agents():
    tracker = BombCredits()
    owner = SimpleNamespace(x=1, y=1, dead=False)
    victim = SimpleNamespace(x=2, y=1, dead=False)

    class Explosion:
        def __init__(self):
            self.owner, self.blast_coords, self.dangerous = owner, [(1, 1), (2, 1)], True

        def is_dangerous(self):
            return self.dangerous

    bomb, explosion = object(), Explosion()
    tracker.placed(bomb, step=4, safe_attack=True)
    tracker.detonated(bomb, explosion)
    tracker.observe_hits([explosion], [owner, victim])
    owner.dead = victim.dead = True
    tracker.observe_hits([explosion], [owner, victim])
    explosion.dangerous = False
    victim.dead = False
    tracker.observe_hits([explosion], [victim])
    assert tracker.snapshot()[0]["kill_credits"] == 1
    assert tracker.snapshot()[0]["self_kill_credits"] == 1


def test_amendment_and_resume_preserve_consumed_budget(tmp_path):
    from scripts.run_hunting_curriculum import initial_usage

    cfg = {"prior_usage": {"pc": {"cpu_seconds": 1900.0, "first_start": 100.0}}}
    assert initial_usage(tmp_path, cfg, "pc")["cpu_seconds"] == 1900.0
    write(tmp_path / "inherited-resources.json", {"cpu_seconds": 2000.0, "first_start": 90.0})
    state = initial_usage(tmp_path, cfg, "pc")
    assert state["cpu_seconds"] == 2000.0 and state["first_start"] == 90.0
    write(tmp_path / "resources.json", {**state, "cpu_seconds": 2500.0})
    assert initial_usage(tmp_path, cfg, "pc")["cpu_seconds"] == 2500.0
