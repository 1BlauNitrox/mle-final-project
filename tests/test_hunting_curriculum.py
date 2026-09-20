"""Mechanics and failure contracts; these are not policy efficacy experiments."""

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from agent_code.DagobertDuckDQNTask3.config import DQNConfig
from agent_code.DagobertDuckDQNTask3.model import DQNLearner
from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer
from scripts.analyze_hunting_curriculum import interval, latency_gate, safety_gate
from scripts.curriculum_io import read, write
from scripts.run_antiloop_pilot import resource_breach, restore, save_generation
from training.hunting_curriculum import UpdateTracker, distances, hunting_layout, training_setting


@pytest.mark.parametrize("index", [1, 3, 5, 7, 9, 11, 13, 15])
def test_reproducible_reachable_layout_preserves_official_walls(index):
    field, positions, meta = hunting_layout(123456 + index, index)
    again, positions2, meta2 = hunting_layout(123456 + index, index)
    assert np.array_equal(field, again) and positions == positions2 and meta == meta2
    assert meta["reachable_attack_position"]
    assert distances(field, positions[0])[positions[1]] == meta["opponent_path_distance"]
    assert np.all(field[0] == -1) and np.all(field[-1] == -1)
    assert np.all(field[:, 0] == -1) and np.all(field[:, -1] == -1)
    assert np.all(field[2:16:2, 2:16:2] == -1)
    assert all(field[p] == 0 for p in positions)
    assert (field == 1).any() == (meta["layout"] == "corridor")


def test_fixed_mixture_and_opponent_progression():
    assert sum(training_setting("curriculum", i)["kind"] == "hunting" for i in range(6000)) == 3000
    assert all(training_setting("control", i)["kind"] == "classic" for i in range(6000))
    assert training_setting("curriculum", 99)["opponents"] == ["peaceful_agent"] * 3
    assert training_setting("curriculum", 101)["opponents"] == ["coin_collector_agent"] * 3
    assert training_setting("curriculum", 501)["opponents"] == ["rule_based_agent"] * 3


def policy():
    config = DQNConfig()
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=42)
    transition = {
        "state": np.zeros(39, dtype=np.float32),
        "action_index": 0,
        "reward": 1.0,
        "next_state": None,
        "terminal": True,
    }
    for _ in range(config.replay_warmup):
        replay.add(**transition)
    learner = DQNLearner(config=config, seed=42)
    return SimpleNamespace(
        config=config,
        replay_buffer=replay,
        learner=learner,
        episode_reward=0.0,
        losses=[],
        absolute_td_errors=[],
        episode_target_synchronizations=0,
    ), transition


def test_cadence_cap_provenance_and_exact_resume():
    a, transition = policy()
    b = deepcopy(a)
    state = {
        "transitions": 0,
        "origins": ["parent"] * len(a.replay_buffer),
        "generated": {},
        "sampled": {},
    }
    continuous = UpdateTracker(state, capacity=10000, every=4, maximum_updates=3, origin="hunting")
    resumed = UpdateTracker(state, capacity=10000, every=4, maximum_updates=3, origin="hunting")
    for _ in range(7):
        continuous.record(a, **transition)
        resumed.record(b, **transition)
    resumed = UpdateTracker(
        resumed.snapshot(), capacity=10000, every=4, maximum_updates=3, origin="hunting"
    )
    for _ in range(13):
        continuous.record(a, **transition)
        resumed.record(b, **transition)
    assert a.learner.update_steps == b.learner.update_steps == 3
    assert continuous.snapshot() == resumed.snapshot()
    assert sum(continuous.state["sampled"].values()) == 3 * a.config.batch_size
    assert continuous.state["generated"] == {"hunting": 20}
    for x, y in zip(
        a.learner.online_network.parameters(), b.learner.online_network.parameters(), strict=True
    ):
        assert (x == y).all()


def test_atomic_generation_rejects_mixed_state(tmp_path):
    checkpoint = tmp_path / "working.pt"
    checkpoint.write_bytes(b"model1")
    state = {"episodes": 10, "updates": 20, "tracker": {"transitions": 80}}
    generation = save_generation(tmp_path, checkpoint, state)
    restored, loaded = restore(tmp_path, None, None)
    assert restored == generation / "checkpoint.pt" and loaded == state
    write(generation / "state.json", {"episodes": 11})
    with pytest.raises(ValueError, match="Corrupt"):
        restore(tmp_path, None, None)


def evidence(path, self_kills=0, latency=2.0):
    row = {
        "world_seed": 1,
        "slot": 0,
        "opponents": [],
        "native": {
            "score": 3,
            "kills": 0.1,
            "survived": 1 - self_kills,
            "self_kills": self_kills,
            "coins": 4,
            "initially_available_coins": 10,
            "decision_times_ms": [latency] * 100,
        },
    }
    for suite in ("classic", "coins", "crates", "latency"):
        write(path / f"{suite}.json", {"rows": [row]})


def test_safety_and_latency_cannot_hide_bad_candidate(tmp_path):
    a, b = tmp_path / "bad", tmp_path / "reference"
    evidence(a, self_kills=0.5, latency=60)
    evidence(b)
    limits = {
        "score_loss": 0.5,
        "survival_loss": 0.1,
        "self_kills_increase": 0.1,
        "collection_loss": 0.1,
    }
    assert not safety_gate(a, b, limits)["passed"]
    assert not latency_gate(a, {"latency_p95_ms": 50, "latency_max_ms": 100})["passed"]
    assert latency_gate(b, {"latency_p95_ms": 50, "latency_max_ms": 100})["passed"]


def test_resource_limits_and_resume_usage():
    limits = {
        "wall_seconds": 100,
        "stop_utc": "2026-09-20T18:00:00+00:00",
        "minimum_available_bytes": 2,
    }
    device = {"cpu_seconds": 200, "rss_bytes": 10}
    assert (
        resource_breach(
            {"cpu_seconds": 201, "wall_seconds": 1}, limits, device, now=0, rss=1, available=3
        )
        == "CPU budget exhausted"
    )
    now = datetime(2026, 9, 20, 18, tzinfo=timezone.utc).timestamp()
    assert (
        resource_breach(
            {"cpu_seconds": 1, "wall_seconds": 1}, limits, device, now=now, rss=1, available=3
        )
        == "Sunday absolute stop reached"
    )


def test_crossed_interval_requires_all_replicas():
    result = interval(np.ones((3, 20)), 1, 100)
    assert result["ci95"] == [1.0, 1.0]
    with pytest.raises(ValueError, match="Three"):
        interval(np.ones((2, 20)), 1, 100)


def test_registered_protocol_has_disjoint_worlds_and_budgets():
    from scripts.run_antiloop_pilot import CONFIG

    cfg = read(CONFIG)
    groups = [set(range(a, b + 1)) for a, b in cfg["train_ranges"]]
    for suites in cfg["evaluation"].values():
        groups += [set(range(s["seeds"][0], s["seeds"][1] + 1)) for s in suites.values()]
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert cfg["devices"]["pc"]["replicas"] == [1, 2]
    assert set(cfg["devices"]) == {"pc"}
    assert cfg["devices"]["pc"]["cpu_seconds"] == 7200
    assert cfg["limits"]["stop_utc"] == "2026-09-20T12:00:00+00:00"


def test_venv_descendant_cpu_and_memory_are_counted(monkeypatch):
    import scripts.run_antiloop_pilot as runner

    child = SimpleNamespace(
        pid=2,
        create_time=lambda: 22.0,
        cpu_times=lambda: (50.0, 10.0),
        memory_info=lambda: SimpleNamespace(rss=900),
    )
    parent = SimpleNamespace(
        pid=1,
        create_time=lambda: 11.0,
        cpu_times=lambda: (0.1, 0.2),
        memory_info=lambda: SimpleNamespace(rss=10),
        children=lambda recursive: [child],
    )
    monkeypatch.setattr(runner.psutil, "Process", lambda pid: parent)
    observed = {}
    assert runner.process_usage(SimpleNamespace(pid=1), observed) == 910
    assert sum(observed.values()) == pytest.approx(60.3)
    runner.process_usage(SimpleNamespace(pid=1), observed)
    assert sum(observed.values()) == pytest.approx(60.3)


def test_attack_escape_matches_existing_policy_feature():
    from agent_code.DagobertDuckDQNTask3.features import state_to_features
    from training.hunting_curriculum import attack_exposure

    for i in range(1, 16, 2):
        field, positions, _ = hunting_layout(700 + i, i)
        state = {
            "field": field,
            "self": ("learner", 0, True, positions[0]),
            "others": [(str(j), 0, True, p) for j, p in enumerate(positions[1:])],
            "bombs": [],
            "explosion_map": np.zeros_like(field),
            "coins": [],
            "step": 1,
        }
        features = state_to_features(state, include_continuation_features=True)
        assert attack_exposure(state)["safe_attack_position"] == bool(features[31])


def test_failed_pilot_stops_pair_before_next_checkpoint(tmp_path, monkeypatch):
    import scripts.analyze_antiloop_pilot as analyzer
    import scripts.run_antiloop_pilot as runner

    cfg = {
        "evaluation": {"pilot": {"classic": {}}},
        "arms": ["control", "curriculum"],
        "pilot": {},
        "checks": [{"name": "first"}, {"name": "must-not-run"}],
    }

    def fake_evaluate(root, cfg, checkpoint, directory, stage):
        write(directory / "classic.json", {"rows": [{"actions_sha256": "unchanged"}]})

    visited = []

    def fake_advance(root, cfg, replica, arm, check):
        visited.append((arm, check["name"]))
        directory = root / arm / check["name"]
        write(directory / "state.json", {"rows": []})
        return directory

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner, "advance", fake_advance)
    monkeypatch.setattr(analyzer, "safety_gate", lambda *args: {"passed": False, "gates": {}})
    runner.pair(tmp_path, cfg, 1)
    assert visited == [("control", "first"), ("curriculum", "first")]
    assert read(tmp_path / "pairs/r1/decision.json")["status"] == "stopped_safety"
