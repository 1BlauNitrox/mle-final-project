"""Observation, migration, resume and decision contracts for issue 211."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNObservation.config import DQNConfig
from agent_code.DagobertDuckDQNObservation.features import normalize_features, state_to_features
from agent_code.DagobertDuckDQNObservation.observations import ObservationHistory, hunting_geometry
from agent_code.DagobertDuckDQNObservation.persistence import load_training_checkpoint
from scripts.analyze_observation_screen import (
    LINE_ENDING_HASH_PAIRS,
    classify,
    equivalent_code_hashes,
    interval,
    loops,
)
from scripts.observation_migration import migrate
from scripts.run_observation_screen import read, resume, save_generation


def state(position=(5, 5), step=1, round_id=1):
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2:16:2, 2:16:2] = -1
    return {
        "round": round_id,
        "step": step,
        "field": field,
        "self": ("me", 0, True, position),
        "others": [("opponent", 0, True, (13, 13))],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def test_geometry_distinguishes_the_aliased_example_and_preserves_prefix():
    first, second = state(), state((7, 5), step=2)
    assert state_to_features(first) == state_to_features(second)
    assert hunting_geometry(first)[0] > hunting_geometry(second)[0]
    history = ObservationHistory()
    value = history.encode(first, DQNConfig(observation_mode="geometry"))
    np.testing.assert_array_equal(
        value[:39],
        normalize_features(state_to_features(first, include_continuation_features=False)),
    )
    assert value.shape == (56,)
    assert np.all(value[51:] == 0)


def test_code_hash_equivalence_accepts_only_registered_lf_crlf_pairs():
    path, pair = next(iter(LINE_ENDING_HASH_PAIRS.items()))
    first, second = sorted(pair)
    bindings = [
        {"code_hashes": {path: first, "same.py": "same"}},
        {"code_hashes": {path: second, "same.py": "same"}},
        {"code_hashes": {path: first, "same.py": "same"}},
    ]
    assert equivalent_code_hashes(bindings)
    bindings[2]["code_hashes"][path] = "different-source"
    assert not equivalent_code_hashes(bindings)
    bindings[2]["code_hashes"][path] = first
    bindings[2]["code_hashes"]["extra.py"] = "extra"
    assert not equivalent_code_hashes(bindings)


def test_route_geometry_respects_obstacles_and_no_opponents():
    blocked = state()
    blocked["field"][6, :] = -1
    values = hunting_geometry(blocked)
    assert values[0] == 1 and values[5] == 0
    blocked["others"] = []
    np.testing.assert_array_equal(hunting_geometry(blocked), np.zeros(12))


def test_geometry_describes_safe_escape_counts_without_action_override():
    value = state()
    value["others"] = [("other", 0, True, (7, 5))]
    observation = hunting_geometry(value)
    assert np.all(np.isfinite(observation))
    assert 0 <= observation[-1] <= observation[-2] <= 1
    value["bombs"] = [(value["self"][3], 3)]
    assert hunting_geometry(value)[5] == 1  # Leaving an occupied own-bomb tile is possible.


def test_memory_is_causal_idempotent_bounded_and_resets():
    history = ObservationHistory()
    config = DQNConfig(observation_mode="memory")
    first = history.encode(state(), config)
    assert first[51] == 0
    np.testing.assert_array_equal(first, history.encode(state(), config))
    for step in range(2, 22):
        result = history.encode(state(step=step), config)
    assert result[51] == 1.0
    assert len(history.positions) == 16
    assert history.encode(state(round_id=2), config)[51] == 0


def test_training_replay_uses_same_observation_as_act():
    from agent_code.DagobertDuckDQNObservation import callbacks, train

    agent = SimpleNamespace(
        config=DQNConfig(observation_mode="memory"),
        train=True,
        epsilon=0.0,
        action_rng=np.random.default_rng(2),
        policy_network=SimpleNamespace(),
        pending_transition=None,
        episode_event_counts=__import__("collections").Counter(),
    )
    captured = []
    from unittest.mock import patch

    with patch.object(
        callbacks,
        "select_action",
        side_effect=lambda **kw: captured.append(kw["state"].copy()) or "RIGHT",
    ):
        old, new = state(), state((6, 5), step=2)
        callbacks.act(agent, old)
        train.game_events_occurred(agent, old, "RIGHT", new, ["MOVED_RIGHT"])
        pending = agent.pending_transition
        callbacks.act(agent, new)
    np.testing.assert_array_equal(pending.state, captured[0])
    np.testing.assert_array_equal(pending.next_state, captured[1])
    assert pending.next_state[55] == 1 / 16  # LEFT neighbour was the previous tile
    assert len(agent.observation_history.positions) == 2


def test_equal_migration_resets_training_but_preserves_both_networks(tmp_path):
    from agent_code.DagobertDuckDQNTask3.config import DQNConfig as ParentConfig
    from agent_code.DagobertDuckDQNTask3.model import DQNLearner
    from agent_code.DagobertDuckDQNTask3.persistence import save_checkpoint
    from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer

    parent = tmp_path / "parent.pt"
    config = replace(ParentConfig(), learning_rate=0.0002)
    learner = DQNLearner(config=config, seed=3)
    with torch.no_grad():
        next(learner.target_network.parameters()).add_(0.1)
    save_checkpoint(
        learner=learner,
        replay_buffer=ReplayBuffer(capacity=10000, seed=3),
        action_rng=np.random.default_rng(3),
        epsilon=0.1,
        completed_episodes=8000,
        agent_seed=3,
        path=parent,
    )
    for mode in ("control", "geometry", "memory"):
        destination = tmp_path / f"{mode}.pt"
        migrate(parent, destination, mode, 7)
        loaded = load_training_checkpoint(destination)
        assert loaded.completed_episodes == 0 and len(loaded.replay_buffer) == 0
        assert loaded.learner.update_steps == 0 and not loaded.learner.optimizer.state
        for name in ("online_network", "target_network"):
            network = getattr(loaded.learner, name)
            before = getattr(learner, name)
            torch.testing.assert_close(network.layers[0].weight[:, :39], before.layers[0].weight)
            assert torch.count_nonzero(network.layers[0].weight[:, 39:]) == 0


def test_resume_keeps_matched_generation_when_incomplete_generation_exists(tmp_path):
    directory = tmp_path / "training"
    directory.mkdir()
    checkpoint = directory / "working.pt"
    checkpoint.write_bytes(b"model25")
    save_generation(directory, checkpoint, [{"i": i} for i in range(25)])
    old = read(directory / "resume.json")
    checkpoint.write_bytes(b"partial26")
    (directory / "generation-incomplete").mkdir()
    restored, rows = resume(directory, tmp_path / "unused")
    assert restored.read_bytes() == b"model25" and len(rows) == 25
    assert read(directory / "resume.json") == old


def test_loop_denominator_does_not_reward_missing_late_game_exposure():
    assert loops([{"late_steps": []}])["rate"] is None
    steps = [
        {
            "position": [i % 2, 0],
            "crates_left": 0,
            "coins_visible": 0,
            "hazards": False,
            "progress": False,
            "board_coins_sha256": "same",
        }
        for i in range(30)
    ]
    assert loops([{"late_steps": steps}])["rate"] == 1
    steps[15]["hazards"] = True
    assert loops([{"late_steps": steps}])["eligible_windows"] == 0


def test_uncertainty_and_decisions_do_not_conflate_negative_with_invalid():
    assert interval(np.zeros((3, 8)), resamples=20, seed=1)["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="three"):
        interval(np.zeros((2, 8)), resamples=20, seed=1)
    assert classify({"score_ci": False}, True, False) == "promising_but_unresolved"
    assert classify({"score_ci": False}, False, False) == "inconclusive"
    assert classify({"safety": False}, True, True) == "reject_for_this_deadline"
    assert classify({"safety": True}, True, False) == "eligible_for_confirmation"


def test_complete_analyzer_and_evidence_corruption(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    from scripts.analyze_observation_screen import analyze
    from scripts.run_observation_screen import CONFIG, rows_write, write

    cfg = json.loads(Path(CONFIG).read_text())
    cfg["uncertainty"]["resamples"] = 20
    for setting in [*cfg["evaluation_suites"].values(), cfg["latency"]]:
        setting["world_seed_range_inclusive"][1] = setting["world_seed_range_inclusive"][0] + 3
    roots = []
    for replica in (1, 2, 3):
        root = tmp_path / f"r{replica}"
        roots.append(root)
        write(root / "config.json", cfg)
        write(
            root / "binding.json",
            {
                "replica": replica,
                "code_hashes": {"source": "same"},
                "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
                "initial_hashes": {"reference.pt": "reference-model"},
            },
        )
        for stage in ("training", "evaluation", "latency"):
            write(root / f"{stage}-summary.json", {"complete": True})
        for arm in (*cfg["arms"], "reference"):
            model_hash = f"{arm}-model"
            if arm != "reference":
                write(
                    root / "training" / arm / "result.json",
                    {
                        "complete": True,
                        "episodes": 1000,
                        "checkpoint_sha256": model_hash,
                    },
                )
            for suite, setting in {**cfg["evaluation_suites"], "latency": cfg["latency"]}.items():
                opponents = (
                    cfg["evaluation_suites"]["primary-classic-rule-based"]["opponents"]
                    if suite == "latency"
                    else setting["opponents"]
                )
                first, last = setting["world_seed_range_inclusive"]
                rows = [
                    {
                        "world_seed": seed,
                        "slot": i % (len(opponents) + 1),
                        "opponents": opponents,
                        "late_steps": [],
                        "native": {
                            "score": [-8.0, 3.5, 3.5, 3.5][i] if arm == "geometry" else 0.0,
                            "kills": 0.1 if arm == "geometry" else 0.0,
                            "survived": 1.0,
                            "self_kills": 0.0,
                            "invalid": 0.0,
                            "coins": 1.0,
                            "crates_destroyed": 0.0,
                            "initially_available_coins": 2,
                            "decision_times_ms": [1.0, 2.0],
                        },
                    }
                    for i, seed in enumerate(range(first, last + 1))
                ]
                directory = root / ("latency" if suite == "latency" else "evaluation") / arm
                if suite != "latency":
                    directory /= suite
                directory.mkdir(parents=True)
                rows_write(directory / "episodes.json.gz", rows)
                write(
                    directory / "result.json",
                    {
                        "checkpoint_sha256": model_hash,
                        "rows_sha256": hashlib.sha256(
                            (directory / "episodes.json.gz").read_bytes()
                        ).hexdigest(),
                    },
                )
    result = analyze(roots)
    assert result["valid_complete_screen"]
    assert result["next_confirmation_arm"] == "geometry"
    assert result["arms"]["geometry"]["warnings"]
    assert not result["arms"]["geometry"]["diagnostic_gates"]["control:score_interval"]
    assert "control:score_interval" not in result["arms"]["geometry"]["gates"]
    assert result["automatic_promotion"] is False
    assert result["selected_submission_artifact"] is None
    rowfile = roots[0] / "evaluation/geometry/primary-classic-rule-based/episodes.json.gz"
    rowfile.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Corrupt rows"):
        analyze(roots)


def test_new_inputs_can_receive_learning_updates():
    from agent_code.DagobertDuckDQNObservation.model import DQNLearner
    from agent_code.DagobertDuckDQNObservation.replay import ReplayBuffer

    config = DQNConfig(observation_mode="memory")
    learner = DQNLearner(config=config, seed=2)
    with torch.no_grad():
        learner.online_network.layers[0].weight[:, 39:].zero_()
    replay = ReplayBuffer(capacity=10000, seed=2)
    for _ in range(64):
        replay.add(
            state=np.ones(56, dtype=np.float32),
            action_index=0,
            reward=5.0,
            next_state=None,
            terminal=True,
        )
    learner.train_batch(replay.sample(64))
    assert torch.count_nonzero(learner.online_network.layers[0].weight[:, 51:]) > 0


def test_detached_launcher_requires_authorization_and_returns_working_pid(tmp_path, monkeypatch):
    import contextlib
    import sys
    import time

    import psutil

    from scripts import run_observation_screen as runner

    runner.write(tmp_path / "smoke/report.json", {"passed": True})
    monkeypatch.setattr(
        runner, "bound", lambda root: ({}, {"replica": 1, "config_sha256": "mechanics"})
    )
    sentinel = tmp_path / "detached-ok.txt"
    command = [
        sys.executable,
        "-c",
        "import pathlib,time; time.sleep(1); "
        f"pathlib.Path({str(sentinel)!r}).write_text('mechanics only'); time.sleep(1)",
    ]
    monkeypatch.setattr(runner, "child_command", lambda *args, **kwargs: command)
    args = SimpleNamespace(root=tmp_path, workers=1, authorize_compute=False)
    with pytest.raises(RuntimeError, match="authorize-compute"):
        runner.launch(args)
    args.authorize_compute = True
    runner.launch(args)
    record = runner.read(tmp_path / "supervisor.json")
    deadline = time.monotonic() + 15
    while not sentinel.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert sentinel.read_text() == "mechanics only"
    with contextlib.suppress(psutil.NoSuchProcess):
        psutil.Process(record["pid"]).wait(timeout=10)


def test_pc_queue_requires_training_evaluation_and_latency_completion(tmp_path):
    from scripts.queue_observation_pc import previous_complete
    from scripts.run_observation_screen import write

    assert not previous_complete(tmp_path)
    write(tmp_path / "recovery-chain.json", {"status": "completed"})
    write(tmp_path / "training-state.json", {"status": "completed"})
    write(tmp_path / "evaluation-state.json", {"status": "completed"})
    assert not previous_complete(tmp_path)
    write(tmp_path / "latency-state.json", {"status": "completed"})
    assert not previous_complete(tmp_path)
    write(tmp_path / "task4-competition-evidence.tar.gz.manifest.json", {})
    assert previous_complete(tmp_path)


def test_latency_and_evaluation_share_one_compute_budget(tmp_path, monkeypatch):
    from scripts import run_observation_screen as runner

    cfg = runner.read(runner.CONFIG)
    runner.write(
        tmp_path / "evaluate-resources.json", {"cpu_seconds": 14400.0, "wall_seconds": 5.0}
    )
    monkeypatch.setattr(runner, "bound", lambda root: (cfg, {}))
    monkeypatch.setattr(
        runner.psutil, "virtual_memory", lambda: SimpleNamespace(available=100 * 1024**3)
    )
    monkeypatch.setattr(runner, "child_command", lambda *a, **kw: pytest.fail("Budget exhausted"))
    args = SimpleNamespace(root=tmp_path, workers=3)
    assert not runner.supervise(args, [("timing", "_evaluate", {})], "latency")
    assert runner.read(tmp_path / "stop-request.json")["reason"] == "registered resource limit"
