"""History, transition, migration and decision contracts for the bounded pilot."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNAntiLoop import train
from agent_code.DagobertDuckDQNAntiLoop.config import DQNConfig
from agent_code.DagobertDuckDQNAntiLoop.loop_reward import LoopReward
from agent_code.DagobertDuckDQNAntiLoop.observations import ObservationHistory
from scripts.analyze_antiloop_pilot import analyze, loop_rate, safety_gate
from scripts.antiloop_migration import initialize
from scripts.curriculum_io import read, write
from scripts.run_antiloop_pilot import CONFIG, resource_breach, restore, save_generation, sources


def state(step=1, position=(3, 3), round_id=1):
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2:16:2, 2:16:2] = -1
    return {
        "round": round_id,
        "step": step,
        "field": field,
        "self": ("test", 0, True, position),
        "others": [],
        "coins": [],
        "bombs": [],
        "explosion_map": np.zeros_like(field),
    }


def positions(step):
    return (3, 3) if step % 2 else (3, 4)


def primed_reward():
    reward = LoopReward()
    for step in range(1, 25):
        reward.observe(state(step, positions(step)))
    return reward


def test_reward_only_after_persistent_calm_successful_return():
    old, new = state(24, positions(24)), state(25, positions(25))
    assert primed_reward().penalty(old, new, "UP", []) == -0.1
    assert LoopReward().penalty(old, new, "UP", []) == 0
    assert primed_reward().penalty(old, state(25, (3, 5)), "DOWN", []) == 0


@pytest.mark.parametrize(
    "exclusion",
    [
        "WAIT",
        "BOMB",
        "INVALID_ACTION",
        "COIN_COLLECTED",
        "CRATE_DESTROYED",
        "KILLED_OPPONENT",
        "bomb",
        "fire",
        "score",
        "coins",
        "field",
        "opponent",
        "blocked",
    ],
)
def test_reward_exclusions(exclusion):
    old, new = state(24, positions(24)), state(25, positions(25))
    action, events = "UP", []
    if exclusion in {"WAIT", "BOMB"}:
        action = exclusion
    elif exclusion in {"INVALID_ACTION", "COIN_COLLECTED", "CRATE_DESTROYED", "KILLED_OPPONENT"}:
        events = [exclusion]
    elif exclusion == "bomb":
        new["bombs"] = [((9, 9), 3)]
    elif exclusion == "fire":
        new["explosion_map"][9, 9] = 1
    elif exclusion == "score":
        new["self"] = ("test", 1, True, positions(25))
    elif exclusion == "coins":
        new["coins"] = [(5, 5)]
    elif exclusion == "field":
        new["field"][5, 5] = 1
    elif exclusion == "opponent":
        new["others"] = [("other", 0, True, (7, 7))]
    elif exclusion == "blocked":
        new["self"] = old["self"]
    assert primed_reward().penalty(old, new, action, events) == 0


def test_penalty_resets_on_new_round_or_observation_gap():
    reward = primed_reward()
    assert reward.penalty(state(1, round_id=2), state(2, (3, 4), 2), "DOWN", []) == 0
    reward = primed_reward()
    assert reward.penalty(state(30), state(31, (3, 4)), "DOWN", []) == 0


def test_memory_is_causal_cached_and_round_scoped():
    history = ObservationHistory()
    config = DQNConfig(observation_mode="memory", escape_continuation_features=True)
    first = history.encode(state(), config)
    assert np.all(first[39:] == 0)
    second = history.encode(state(2, (3, 4)), config)
    assert second[52] == 1 / 16  # previous tile is UP; current occupies suffix51
    assert np.array_equal(history.encode(state(1), config), first)
    assert np.array_equal(history.encode(state(2, (3, 4)), config), second)
    assert len(history.positions) == 2
    assert np.all(history.encode(state(1, round_id=2), config)[51:] == 0)


def test_terminal_duplicate_keeps_penalty_exactly_once(tmp_path):
    policy = SimpleNamespace(
        config=DQNConfig(observation_mode="memory"),
        logger=Mock(),
        epsilon=0.0,
        completed_episodes=0,
        replay_buffer=[],
        learner=SimpleNamespace(update_steps=0),
        action_rng=None,
        agent_seed=1,
    )
    train.setup_training(policy)
    policy.loop_reward = primed_reward()
    old, new = state(24, positions(24)), state(25, positions(25))
    captured = []
    with (
        patch.object(train, "_record_transition", lambda self, **kw: captured.append(kw)),
        patch.object(train, "save_checkpoint"),
    ):
        train.game_events_occurred(policy, old, "UP", new, [])
        assert policy.pending_transition.loop_penalty == -0.1
        train.end_of_round(policy, old, "UP", ["SURVIVED_ROUND"])
    assert len(captured) == 1 and captured[0]["terminal"]
    assert captured[0]["reward"] == pytest.approx(4.9)


def test_migration_preserves_adam_replay_and_q_values(tmp_path):
    from agent_code.DagobertDuckDQNAntiLoop.persistence import load_training_checkpoint
    from agent_code.DagobertDuckDQNTask3.config import DQNConfig as ParentConfig
    from agent_code.DagobertDuckDQNTask3.model import DQNLearner
    from agent_code.DagobertDuckDQNTask3.persistence import save_checkpoint
    from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer

    config = ParentConfig(batch_size=2, replay_warmup=2)
    learner = DQNLearner(config=config, seed=9)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=9)
    for _ in range(2):
        replay.add(
            state=np.ones(39, dtype=np.float32),
            action_index=0,
            reward=1.0,
            next_state=None,
            terminal=True,
        )
    learner.train_batch(replay.sample(2))
    path = tmp_path / "parent.pt"
    save_checkpoint(
        learner=learner,
        replay_buffer=replay,
        action_rng=np.random.default_rng(9),
        epsilon=0.0,
        completed_episodes=80,
        agent_seed=9,
        path=path,
    )
    for mode in ("control", "memory"):
        destination = tmp_path / f"{mode}.pt"
        initialize(path, destination, mode, 8, 0.0001)
        child = load_training_checkpoint(destination)
        assert child.completed_episodes == 80 and child.learner.update_steps == 1
        assert len(child.replay_buffer) == 2
        assert np.all(child.replay_buffer.state_dict()["states"][:, 39:] == 0)
        before = learner.optimizer.state_dict()["state"][0]
        after = child.learner.optimizer.state_dict()["state"][0]
        assert torch.equal(before["exp_avg"], after["exp_avg"][:, :39])
        assert torch.count_nonzero(after["exp_avg"][:, 39:]) == 0
        assert torch.equal(before["step"], after["step"])
        child.learner.train_batch(child.replay_buffer.sample(2))


def test_runtime_sources_and_deadline_are_bound():
    cfg = read(CONFIG)
    assert "agent_code/DagobertDuckDQNAntiLoop/loop_reward.py" in sources()
    assert cfg["devices"]["pc"]["cpu_seconds"] == 7200
    assert resource_breach(
        {"cpu_seconds": 0, "wall_seconds": 0},
        cfg["limits"],
        cfg["devices"]["pc"],
        now=1789905600,
        rss=0,
        available=10**10,
    )


def test_no_exposure_is_unavailable_not_zero():
    assert loop_rate([])["rate"] is None


def test_incomplete_cannot_pass(tmp_path):
    write(tmp_path / "config.json", read(CONFIG))
    assert not analyze(tmp_path)["eligible"]


def test_safety_gate_includes_invalid_actions(tmp_path):
    def row(invalid):
        return {
            "world_seed": 1,
            "slot": 0,
            "opponents": [],
            "native": {
                "score": 3,
                "survived": 1,
                "self_kills": 0,
                "coins": 3,
                "initially_available_coins": 9,
                "invalid": invalid,
            },
        }

    for name in ("candidate", "reference"):
        for suite in ("classic", "coins", "crates"):
            write(tmp_path / name / f"{suite}.json", {"rows": [row(int(name == "candidate"))]})
    result = safety_gate(tmp_path / "candidate", tmp_path / "reference", read(CONFIG)["pilot"])
    assert not result["passed"] and not result["gates"]["invalid"]


def test_saved_generation_rejects_corrupt_checkpoint(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"model")
    directory = tmp_path / "arm"
    directory.mkdir()
    saved = save_generation(directory, checkpoint, {"episodes": 1})
    assert restore(directory, None, None)[1]["episodes"] == 1
    (saved / "checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        restore(directory, None, None)


def test_complete_analyzer_enforces_loops_invalid_and_both_baselines(tmp_path):
    cfg = read(CONFIG)
    cfg["bootstrap_resamples"] = 50
    write(tmp_path / "config.json", cfg)

    def observation(seed, slot, opponents, memory):
        return {
            "world_seed": seed,
            "slot": slot,
            "opponents": opponents,
            "epsilon": 0,
            "scenario": "classic",
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
            "loop_windows": {"eligible": 100, "looping": 50 if memory else 100},
            "attack_exposure": {
                "safe_attack_steps": 0,
                "threatening_steps": 0,
                "safe_attack_bombs": 0,
            },
            "initial_exposure": {"reachable_attack_position": False},
            "bomb_credits": [],
        }

    paths = {}
    for replica in (1, 2):
        pair = tmp_path / "pairs" / f"r{replica}"
        write(pair / "decision.json", {"status": "training_complete"})
        for arm in ("control", "memory", "reference"):
            directory = (
                pair / "reference/final"
                if arm == "reference"
                else pair / arm / "snapshots/final/final-evaluation"
            )
            paths[replica, arm] = directory
            for suite, setting in cfg["evaluation"]["final"].items():
                observations = [
                    observation(
                        seed,
                        i % (len(setting["opponents"]) + 1),
                        setting["opponents"],
                        arm == "memory",
                    )
                    for i, seed in enumerate(range(setting["seeds"][0], setting["seeds"][1] + 1))
                ]
                for row in observations:
                    row["scenario"] = setting["scenario"]
                write(directory / f"{suite}.json", {"rows": observations})
    result = analyze(tmp_path)
    assert result["complete"] and result["eligible"] and not result["submission_promotion"]
    path = paths[1, "memory"] / "classic.json"
    original = read(path)
    changed = deepcopy(original)
    for row in changed["rows"]:
        row["native"]["invalid"] = 1
    write(path, changed)
    assert not analyze(tmp_path)["eligible"]
    write(path, original)
    for replica in (1, 2):
        path = paths[replica, "memory"] / "classic.json"
        changed = read(path)
        for row in changed["rows"]:
            row["loop_windows"]["eligible"] = 0
            row["loop_windows"]["looping"] = 0
        write(path, changed)
    assert not analyze(tmp_path)["eligible"]
