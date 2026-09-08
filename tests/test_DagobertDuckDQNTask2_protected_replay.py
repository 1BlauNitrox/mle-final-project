"""Contracts for the Issue #88 protected replay treatment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from agent_code.DagobertDuckDQNTask2.config import (
    DEFAULT_CONFIG,
    FEATURE_COUNT,
)
from agent_code.DagobertDuckDQNTask2.model import DQNLearner
from agent_code.DagobertDuckDQNTask2.persistence import (
    export_evaluation_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from agent_code.DagobertDuckDQNTask2.replay import ReplayBuffer


def _add(buffer: ReplayBuffer, value: int, *, partition: str | None = None) -> None:
    state = np.full(FEATURE_COUNT, value, dtype=np.float32)
    next_state = np.full(FEATURE_COUNT, value + 0.5, dtype=np.float32)
    buffer.add(
        state=state,
        action_index=value % 6,
        reward=float(value),
        next_state=next_state,
        terminal=False,
        partition=partition,
    )


def test_protected_replay_has_fixed_capacity_and_fifo_replacement() -> None:
    buffer = ReplayBuffer(capacity=10_000, seed=1, mode="protected_task1")

    for value in range(2_005):
        _add(buffer, value, partition="protected")
    buffer.close_protected_collection()
    for value in range(8_005):
        _add(buffer, value + 10_000, partition="other")

    assert len(buffer) == 10_000
    assert buffer.protected_size == 2_000
    assert buffer.other_size == 8_000
    state = buffer.state_dict()
    assert state["protected"]["rewards"].tolist() == list(
        map(float, range(5, 2_005))
    )
    assert state["other"]["rewards"].tolist() == list(
        map(float, range(10_005, 18_005))
    )


def test_protected_and_other_membership_is_disjoint() -> None:
    buffer = ReplayBuffer(capacity=10_000, seed=2, mode="protected_task1")
    for value in range(20):
        _add(buffer, value, partition="protected")
    buffer.close_protected_collection()
    for value in range(20):
        _add(buffer, value + 100, partition="other")

    state = buffer.state_dict()
    protected = set(state["protected"]["rewards"].tolist())
    other = set(state["other"]["rewards"].tolist())
    assert protected.isdisjoint(other)
    assert state["mode"] == "protected_task1"
    assert state["protected_capacity"] == 2_000
    assert state["other_capacity"] == 8_000
    assert state["protected_source_episode_limit"] == 2_000
    assert state["protected_source_scenario"] == "coin-heaven"


def test_closed_protected_replay_uses_exact_16_48_quota() -> None:
    buffer = ReplayBuffer(capacity=10_000, seed=3, mode="protected_task1")
    for value in range(32):
        _add(buffer, value, partition="protected")
    buffer.close_protected_collection()
    for value in range(64):
        _add(buffer, value + 100, partition="other")

    batch = buffer.sample(64)
    assert int(batch.protected_flags.sum()) == 16
    assert int((~batch.protected_flags).sum()) == 48


def test_insufficient_partition_quota_is_filled_without_replacement() -> None:
    protected_short = ReplayBuffer(
        capacity=10_000, seed=4, mode="protected_task1"
    )
    for value in range(4):
        _add(protected_short, value, partition="protected")
    protected_short.close_protected_collection()
    for value in range(60):
        _add(protected_short, value + 100, partition="other")
    batch = protected_short.sample(64)
    assert int(batch.protected_flags.sum()) == 4
    assert len(set(batch.rewards.tolist())) == 64

    other_short = ReplayBuffer(capacity=10_000, seed=5, mode="protected_task1")
    for value in range(60):
        _add(other_short, value, partition="protected")
    other_short.close_protected_collection()
    for value in range(4):
        _add(other_short, value + 100, partition="other")
    batch = other_short.sample(64)
    assert int(batch.protected_flags.sum()) == 60
    assert len(set(batch.rewards.tolist())) == 64


def test_protected_save_resume_reproduces_samples_and_updates(tmp_path: Path) -> None:
    config = replace(
        DEFAULT_CONFIG,
        batch_size=1,
        replay_warmup=1,
        replay_treatment="protected_task1",
    )
    learner = DQNLearner(config=config, seed=11)
    buffer = ReplayBuffer(capacity=config.replay_capacity, seed=12, mode="protected_task1")
    for value in range(80):
        _add(buffer, value, partition="protected")
    buffer.close_protected_collection()
    for value in range(80):
        _add(buffer, value + 100, partition="other")

    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        learner=learner,
        replay_buffer=buffer,
        action_rng=np.random.default_rng(13),
        epsilon=0.5,
        completed_episodes=2_000,
        agent_seed=11,
        path=checkpoint,
    )
    restored = load_training_checkpoint(checkpoint)
    first_batch = buffer.sample(1)
    second_batch = restored.replay_buffer.sample(1)
    np.testing.assert_array_equal(first_batch.states, second_batch.states)
    assert restored.learner.update_steps == learner.update_steps

    learner.train_batch(first_batch)
    restored.learner.train_batch(second_batch)
    for first, second in zip(
        learner.online_network.parameters(),
        restored.learner.online_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(first, second)


def test_evaluation_export_contains_no_training_or_replay_state(tmp_path: Path) -> None:
    config = replace(DEFAULT_CONFIG, replay_treatment="protected_task1")
    learner = DQNLearner(config=config, seed=21)
    buffer = ReplayBuffer(capacity=config.replay_capacity, seed=22, mode="protected_task1")
    _add(buffer, 1, partition="protected")
    source = tmp_path / "training.pt"
    exported = tmp_path / "evaluation.pt"
    save_checkpoint(
        learner=learner,
        replay_buffer=buffer,
        action_rng=np.random.default_rng(23),
        epsilon=1.0,
        completed_episodes=0,
        agent_seed=21,
        path=source,
    )

    export_evaluation_checkpoint(source, exported)
    payload = torch.load(exported, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "artifact_schema_version",
        "model_schema_version",
        "feature_schema_version",
        "actions",
        "rewards",
        "config",
        "completed_episodes",
        "network_state",
    }


def test_protected_replay_composes_with_feature_on_configuration(monkeypatch) -> None:
    import agent_code.DagobertDuckDQNTask2.callbacks as callbacks

    monkeypatch.setenv("BOMBERMAN_DQN_REPLAY_TREATMENT", "protected_task1")
    monkeypatch.setenv("BOMBERMAN_DQN_ESCAPE_CONTINUATIONS", "on")
    monkeypatch.setenv("BOMBERMAN_DQN_ACTION_MASKING", "none")

    config = callbacks._configured_training_config()

    assert config.replay_treatment == "protected_task1"
    assert config.escape_continuation_features is True
    assert config.input_dim == 26


def test_closed_stage_preserves_loaded_protected_partition(
    monkeypatch, tmp_path: Path
) -> None:
    """A stage boundary closes collection without moving protected entries."""
    config = replace(DEFAULT_CONFIG, replay_treatment="protected_task1")
    learner = DQNLearner(config=config, seed=31)
    buffer = ReplayBuffer(capacity=config.replay_capacity, seed=32, mode="protected_task1")
    for value in range(20):
        _add(buffer, value, partition="protected")
    source = tmp_path / "checkpoint.pt"
    save_checkpoint(
        learner=learner,
        replay_buffer=buffer,
        action_rng=np.random.default_rng(33),
        epsilon=1.0,
        completed_episodes=20,
        agent_seed=31,
        path=source,
    )

    from types import SimpleNamespace
    from unittest.mock import Mock

    import agent_code.DagobertDuckDQNTask2.callbacks as callbacks

    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", source)
    monkeypatch.setenv("BOMBERMAN_DQN_REPLAY_TREATMENT", "protected_task1")
    monkeypatch.setenv("BOMBERMAN_DQN_REPLAY_COLLECTION", "closed")
    monkeypatch.setenv("BOMBERMAN_DQN_ESCAPE_CONTINUATIONS", "off")
    monkeypatch.setenv("BOMBERMAN_DQN_ACTION_MASKING", "none")
    restored = SimpleNamespace(logger=Mock())
    callbacks._setup_training_policy(restored, 31)

    assert restored.replay_buffer.protected_size == 20
    assert restored.replay_buffer.other_size == 0
    assert restored.replay_buffer.collection_open is False


def test_uniform_mode_remains_one_fifo() -> None:
    buffer = ReplayBuffer(capacity=3, seed=34, mode="uniform")
    for value in range(5):
        _add(buffer, value)
    assert buffer.protected_size == 0
    assert buffer.other_size == 3
    assert set(buffer.sample(3).rewards.tolist()) == {2.0, 3.0, 4.0}
