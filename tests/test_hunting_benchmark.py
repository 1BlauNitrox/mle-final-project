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
