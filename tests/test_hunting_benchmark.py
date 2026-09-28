import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


runner = module("run_hunting_benchmark")
analyze = module("analyze_hunting_benchmark")


def test_rotation_is_cyclic_and_preserves_members():
    lineup = ["a", "b", "c", "d"]
    assert runner.rotated(lineup, 0) == lineup
    assert runner.rotated(lineup, 2) == ["c", "d", "a", "b"]
    assert sorted(runner.rotated(lineup, 3)) == sorted(lineup)


def test_bootstrap_resamples_world_clusters_deterministically():
    values = analyze.np.asarray([1.0, 3.0, 5.0])
    first = analyze.interval(values, resamples=1000, seed=7)
    second = analyze.interval(values, resamples=1000, seed=7)
    assert first == second
    assert first["estimate"] == pytest.approx(3.0)
    assert first["ci95"][0] <= 3.0 <= first["ci95"][1]


def test_registered_config_has_exact_game_budget_and_distinct_seeds():
    path = Path(__file__).parents[1] / "experiments/2026-09-19-hunting-agent-screen/config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    seeds = config["design"]["world_seeds"]
    assert len(seeds) == len(set(seeds)) == 20
    assert config["design"]["smoke_seed"] not in seeds
    assert config["design"]["serial_timing_seed"] not in seeds
    assert config["design"]["serial_timing_seed"] != config["design"]["smoke_seed"]
    assert len(config["design"]["lineups"]) * len(seeds) * config["design"]["rotations"] == 240
    assert config["runtime"]["maximum_concurrent_games"] == 2
    assert config["runtime"]["minimum_available_ram_gib"] == 3
    assert config["runtime"]["timing_limits_ms"] == {
        "decision_time_p95": 50,
        "decision_time_maximum": 100,
    }


def test_timing_failure_blocks_external_training_admission():
    passing = {"external": {"passes": True}}
    failing = {"external": {"passes": False}}
    assert analyze.external_training_admitted("appears stronger", passing)
    assert not analyze.external_training_admitted("appears stronger", failing)
    assert not analyze.external_training_admitted("inconclusive", passing)


def test_rule_contrast_bootstrap_clusters_rotations_within_world():
    rows = []
    for world, difference in ((11, 1.0), (22, 3.0)):
        for rotation in range(4):
            common = {
                "lineup": "A",
                "world_seed": world,
                "rotation": rotation,
                "kills": 0.0,
                "coins": 0.0,
                "survived": 1.0,
                "self_kills": 0.0,
                "invalid": 0.0,
                "first_place_share": 0.0,
            }
            rows.append({**common, "identity": "benchmark_fallback", "score": difference})
            for _ in range(3):
                rows.append({**common, "identity": "rule_based_agent", "score": 0.0})
    config = {
        "design": {"world_seeds": [11, 22]},
        "uncertainty": {"resamples": 500, "seed": 9},
    }
    result = analyze.relative_to_rules(rows, "A", "benchmark_fallback", config)
    assert result["score"]["estimate"] == pytest.approx(2.0)
    assert result["score"]["ci95"] == [1.0, 3.0]


def test_first_place_share_splits_tied_credit(tmp_path):
    def agent(score):
        return {
            "score": score,
            "kills": 0,
            "coins": 0,
            "survived": True,
            "self_kills": 0,
            "invalid": 0,
            "attempted_actions": 1,
            "survival_steps": 1,
            "decision_time_median_ms": 1.0,
            "decision_time_p95_ms": 1.0,
            "decision_time_max_ms": 1.0,
        }

    path = tmp_path / "game.json"
    path.write_text(
        json.dumps(
            {
                "by_round": {
                    "round": {
                        "agents": {
                            "benchmark_fallback": agent(2),
                            "rule_based_agent": agent(2),
                            "rule_based_agent_1": agent(1),
                            "rule_based_agent_2": agent(0),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rows = analyze.game_observations(path, "A", 1, 0)
    shares = {row["agent_name"]: row["first_place_share"] for row in rows}
    assert shares["benchmark_fallback"] == 0.5
    assert shares["rule_based_agent"] == 0.5
    assert sum(shares.values()) == 1.0
