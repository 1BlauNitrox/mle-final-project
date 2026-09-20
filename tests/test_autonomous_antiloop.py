"""Contracts for the bounded autonomous anti-loop campaign."""

from copy import deepcopy

from scripts.analyze_autonomous_antiloop import analyze
from scripts.curriculum_io import read, write
from scripts.run_autonomous_antiloop import CONFIG, resource_breach, sources


def observation(seed, slot, setting, *, guarded, loops):
    return {
        "world_seed": seed,
        "slot": slot,
        "opponents": setting["opponents"],
        "scenario": setting["scenario"],
        "epsilon": 0,
        "native": {
            "score": 3,
            "kills": 0,
            "survived": 1,
            "self_kills": 0,
            "coins": 3,
            "initially_available_coins": 9,
            "invalid": 0,
            "decision_times_ms": [1.0, 2.0],
        },
        "loop_windows": {"eligible": 100, "looping": loops},
        "broad_loop_windows": {"eligible": 100, "looping": loops},
        "narrow_loop_guard": {
            "eligible": int(guarded),
            "overrides": int(guarded),
            "rejected_contested": 0,
            "rejected_no_candidate": 0,
        },
    }


def fixture(root):
    cfg = read(CONFIG)
    cfg["bootstrap_resamples"] = 20
    write(root / "config.json", cfg)
    for stage in ("pilot_r1", "pilot_r2"):
        for arm in cfg["arms"]:
            for suite, setting in cfg["evaluation"][stage].items():
                rows = []
                for index, seed in enumerate(range(setting["seeds"][0], setting["seeds"][1] + 1)):
                    rows.append(
                        observation(
                            seed,
                            index % (len(setting["opponents"]) + 1),
                            setting,
                            guarded=arm == "narrow_guard"
                            and suite in ("classic", "mixed")
                            and index < 5,
                            loops=50 if arm == "narrow_guard" else 100,
                        )
                    )
                write(root / "results" / stage / arm / f"{suite}.json", {"rows": rows})
    return cfg


def test_complete_pilot_enforces_primary_and_safety_gates(tmp_path):
    fixture(tmp_path)
    result = analyze(tmp_path)
    # Ten guarded games per block across the two multiplayer suites.
    assert result["complete"] and result["eligible"]
    assert result["pilot"]["effects"]["guarded_games"] == 20
    assert not result["submission_promotion"]

    path = tmp_path / "results/pilot_r1/narrow_guard/classic.json"
    original = read(path)
    changed = deepcopy(original)
    for row in changed["rows"]:
        row["native"]["invalid"] = 1
    write(path, changed)
    assert not analyze(tmp_path)["eligible"]

    write(path, original)
    for stage in ("pilot_r1", "pilot_r2"):
        for suite in ("classic", "mixed"):
            loop_path = tmp_path / "results" / stage / "narrow_guard" / f"{suite}.json"
            changed = read(loop_path)
            for row in changed["rows"]:
                row["loop_windows"] = {"eligible": 100, "looping": 80}
            write(loop_path, changed)
    assert not analyze(tmp_path)["eligible"]


def test_campaign_binds_guard_and_enforces_budget_deadline():
    cfg = read(CONFIG)
    assert "agent_code/DagobertDuckDQNAntiLoop/narrow_loop_guard.py" in sources()
    assert cfg["devices"]["pc"]["cpu_seconds"] == 144000
    assert (
        resource_breach(
            {"cpu_seconds": 0, "wall_seconds": 0},
            cfg,
            cfg["devices"]["pc"],
            10**20,
            0,
            10**10,
        )
        == "Pipeline absolute stop reached"
    )


def test_three_arm_screen_requires_registered_invalid_improvement(tmp_path):
    cfg = fixture(tmp_path)
    cfg["arms"].append("cooldown_guard")
    cfg["candidate_arm"] = "cooldown_guard"
    cfg["invalid_improvement_vs"] = {
        "arm": "narrow_guard",
        "minimum_reduction_per_game": 0.2,
    }
    for stage in ("pilot_r1", "pilot_r2"):
        for suite in cfg["evaluation"][stage]:
            source_path = tmp_path / "results" / stage / "narrow_guard" / f"{suite}.json"
            source = read(source_path)
            candidate = deepcopy(source)
            if suite in ("classic", "mixed"):
                for row in source["rows"]:
                    row["native"]["invalid"] = 1
            write(source_path, source)
            write(tmp_path / "results" / stage / "cooldown_guard" / f"{suite}.json", candidate)
    write(tmp_path / "config.json", cfg)
    result = analyze(tmp_path)
    assert result["eligible"]
    assert result["pilot"]["gates"]["invalid_vs_narrow_guard"]

    path = tmp_path / "results/pilot_r1/cooldown_guard/classic.json"
    changed = read(path)
    for row in changed["rows"]:
        row["native"]["invalid"] = 1
    write(path, changed)
    assert not analyze(tmp_path)["eligible"]


def test_zero_loop_baseline_requires_zero_candidate_loops(tmp_path):
    cfg = fixture(tmp_path)
    for stage in ("pilot_r1", "pilot_r2"):
        for arm in cfg["arms"]:
            for suite in ("classic", "mixed"):
                path = tmp_path / "results" / stage / arm / f"{suite}.json"
                data = read(path)
                for row in data["rows"]:
                    row["loop_windows"] = {"eligible": 100, "looping": 0}
                write(path, data)
    assert analyze(tmp_path)["eligible"]

    path = tmp_path / "results/pilot_r1/narrow_guard/classic.json"
    data = read(path)
    data["rows"][0]["loop_windows"]["looping"] = 1
    write(path, data)
    assert not analyze(tmp_path)["eligible"]


def test_registered_broad_loop_metric_is_used_for_selection(tmp_path):
    cfg = fixture(tmp_path)
    cfg["screen"]["loop_metric"] = "broad_loop_windows"
    for stage in ("pilot_r1", "pilot_r2"):
        for arm in cfg["arms"]:
            for suite in ("classic", "mixed"):
                path = tmp_path / "results" / stage / arm / f"{suite}.json"
                data = read(path)
                for row in data["rows"]:
                    row["loop_windows"] = {"eligible": 100, "looping": 0}
                    row["broad_loop_windows"] = {
                        "eligible": 100,
                        "looping": 80 if arm == "narrow_guard" else 100,
                    }
                write(path, data)
    write(tmp_path / "config.json", cfg)
    assert not analyze(tmp_path)["eligible"]


def test_registered_attack_override_exposure_is_a_hard_gate(tmp_path):
    cfg = fixture(tmp_path)
    cfg["screen"]["minimum_attack_overrides"] = 1
    write(tmp_path / "config.json", cfg)
    assert not analyze(tmp_path)["eligible"]

    path = tmp_path / "results/pilot_r1/narrow_guard/classic.json"
    data = read(path)
    data["rows"][0]["safe_attack_guard"] = {"overrides": 1}
    write(path, data)
    result = analyze(tmp_path)
    assert result["eligible"]
    assert result["pilot"]["effects"]["attack_overrides"] == 1
