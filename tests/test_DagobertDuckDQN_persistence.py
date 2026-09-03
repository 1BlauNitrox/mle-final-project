"""Tests for DagobertDuckDQN checkpoint persistence."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import agent_code.DagobertDuckDQN.persistence as persistence
from agent_code.DagobertDuckDQN.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQN.model import DQNLearner
from agent_code.DagobertDuckDQN.persistence import (
    CHECKPOINT_PATH,
    load_evaluation_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from agent_code.DagobertDuckDQN.replay import (
    ReplayBatch,
    ReplayBuffer,
)


def make_config():
    """Create a small persistence-test configuration."""
    return replace(
        DEFAULT_CONFIG,
        batch_size=2,
        replay_warmup=2,
        replay_capacity=8,
        target_update_interval=10,
    )


def make_batch() -> ReplayBatch:
    """Create one deterministic training batch."""
    return ReplayBatch(
        states=np.array(
            [
                [0.0] * 8,
                [1.0] * 8,
            ],
            dtype=np.float32,
        ),
        action_indices=np.array([0, 1], dtype=np.int64),
        rewards=np.array([1.0, -1.0], dtype=np.float32),
        next_states=np.array(
            [
                [0.5] * 8,
                [0.0] * 8,
            ],
            dtype=np.float32,
        ),
        terminals=np.array([False, True], dtype=np.bool_),
    )


def make_training_state():
    """Create learner, replay and action-RNG state."""
    config = make_config()
    learner = DQNLearner(config=config, seed=123)
    learner.train_batch(make_batch())

    replay_buffer = ReplayBuffer(
        capacity=config.replay_capacity,
        seed=456,
    )
    replay_buffer.add(
        state=np.zeros(8, dtype=np.float32),
        action_index=0,
        reward=1.0,
        next_state=np.full(8, 0.5, dtype=np.float32),
        terminal=False,
    )
    replay_buffer.add(
        state=np.ones(8, dtype=np.float32),
        action_index=1,
        reward=-1.0,
        next_state=None,
        terminal=True,
    )

    action_rng = np.random.default_rng(789)
    action_rng.random()

    return config, learner, replay_buffer, action_rng


def test_checkpoint_path_is_relative_to_agent_module() -> None:
    assert CHECKPOINT_PATH.parent == Path(
        __file__
    ).resolve().parents[1] / "agent_code" / "DagobertDuckDQN"


def test_training_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    config, learner, replay_buffer, action_rng = make_training_state()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        learner=learner,
        replay_buffer=replay_buffer,
        action_rng=action_rng,
        epsilon=0.5,
        completed_episodes=3,
        agent_seed=123,
        path=path,
    )

    loaded = load_training_checkpoint(path)

    assert loaded.config == config
    assert loaded.epsilon == pytest.approx(0.5)
    assert loaded.completed_episodes == 3
    assert loaded.agent_seed == 123
    assert loaded.learner.update_steps == learner.update_steps
    assert len(loaded.replay_buffer) == len(replay_buffer)

    for original, restored in zip(
        learner.online_network.parameters(),
        loaded.learner.online_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original, restored)

    for original, restored in zip(
        learner.target_network.parameters(),
        loaded.learner.target_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original, restored)


def test_checkpoint_restores_replay_sampling_and_action_rng(
    tmp_path: Path,
) -> None:
    _, learner, replay_buffer, action_rng = make_training_state()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        learner=learner,
        replay_buffer=replay_buffer,
        action_rng=action_rng,
        epsilon=0.5,
        completed_episodes=3,
        agent_seed=123,
        path=path,
    )
    loaded = load_training_checkpoint(path)

    original_batch = replay_buffer.sample(2)
    restored_batch = loaded.replay_buffer.sample(2)

    np.testing.assert_array_equal(
        original_batch.states,
        restored_batch.states,
    )
    np.testing.assert_array_equal(
        original_batch.action_indices,
        restored_batch.action_indices,
    )
    np.testing.assert_array_equal(
        original_batch.rewards,
        restored_batch.rewards,
    )
    np.testing.assert_array_equal(
        original_batch.next_states,
        restored_batch.next_states,
    )
    np.testing.assert_array_equal(
        original_batch.terminals,
        restored_batch.terminals,
    )

    assert action_rng.random() == pytest.approx(
        loaded.action_rng.random()
    )


def test_checkpoint_restores_optimizer_for_exact_next_update(
    tmp_path: Path,
) -> None:
    _, learner, replay_buffer, action_rng = make_training_state()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        learner=learner,
        replay_buffer=replay_buffer,
        action_rng=action_rng,
        epsilon=0.5,
        completed_episodes=3,
        agent_seed=123,
        path=path,
    )
    loaded = load_training_checkpoint(path)

    original_result = learner.train_batch(make_batch())
    restored_result = loaded.learner.train_batch(make_batch())

    assert restored_result.loss == pytest.approx(original_result.loss)

    for original, restored in zip(
        learner.online_network.parameters(),
        loaded.learner.online_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original, restored)


def test_evaluation_checkpoint_loads_frozen_online_network(
    tmp_path: Path,
) -> None:
    _, learner, replay_buffer, action_rng = make_training_state()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        learner=learner,
        replay_buffer=replay_buffer,
        action_rng=action_rng,
        epsilon=0.5,
        completed_episodes=3,
        agent_seed=123,
        path=path,
    )
    loaded = load_evaluation_checkpoint(path)

    assert not loaded.network.training
    assert all(
        not parameter.requires_grad
        for parameter in loaded.network.parameters()
    )

    for original, restored in zip(
        learner.online_network.parameters(),
        loaded.network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original, restored)


def test_evaluation_loading_does_not_modify_checkpoint(
    tmp_path: Path,
) -> None:
    _, learner, replay_buffer, action_rng = make_training_state()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        learner=learner,
        replay_buffer=replay_buffer,
        action_rng=action_rng,
        epsilon=0.5,
        completed_episodes=3,
        agent_seed=123,
        path=path,
    )
    bytes_before = path.read_bytes()

    load_evaluation_checkpoint(path)

    assert path.read_bytes() == bytes_before


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_training_checkpoint(tmp_path / "missing.pt")


def test_incompatible_schema_is_rejected(tmp_path: Path) -> None:
    _, learner, replay_buffer, action_rng = make_training_state()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        learner=learner,
        replay_buffer=replay_buffer,
        action_rng=action_rng,
        epsilon=0.5,
        completed_episodes=3,
        agent_seed=123,
        path=path,
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    payload["checkpoint_schema_version"] = 999
    torch.save(payload, path)

    with pytest.raises(ValueError):
        load_training_checkpoint(path)


def test_save_rejects_inconsistent_replay_capacity(
    tmp_path: Path,
) -> None:
    config, learner, _, action_rng = make_training_state()
    wrong_replay = ReplayBuffer(
        capacity=config.replay_capacity + 1,
        seed=456,
    )

    with pytest.raises(ValueError):
        save_checkpoint(
            learner=learner,
            replay_buffer=wrong_replay,
            action_rng=action_rng,
            epsilon=0.5,
            completed_episodes=3,
            agent_seed=123,
            path=tmp_path / "checkpoint.pt",
        )

def test_failed_save_preserves_existing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, learner, replay_buffer, action_rng = make_training_state()
    path = tmp_path / "checkpoint.pt"
    original_bytes = b"existing-checkpoint"
    path.write_bytes(original_bytes)

    def fail_during_serialization(*args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(
        "agent_code.DagobertDuckDQN.persistence.torch.save",
        fail_during_serialization
    )

    with pytest.raises(RuntimeError, match="simulated"):
        save_checkpoint(
            learner=learner,
            replay_buffer=replay_buffer,
            action_rng=action_rng,
            epsilon=0.5,
            completed_episodes=3,
            agent_seed=123,
            path=path,
        )

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_replace_retries_transient_windows_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A brief antivirus lock must not end a multi-hour training run."""
    source = tmp_path / "source.tmp"
    destination = tmp_path / "checkpoint.pt"
    source.write_bytes(b"payload")
    attempts = 0
    original_replace = os.replace

    def flaky_replace(a, b):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        return original_replace(a, b)

    monkeypatch.setattr(persistence.os, "replace", flaky_replace)
    monkeypatch.setattr(persistence, "sleep", lambda _seconds: None)

    persistence._replace_with_retry(source, destination)

    assert attempts == 3
    assert destination.read_bytes() == b"payload"


def test_checkpoint_replace_gives_up_after_the_registered_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts = 0

    def always_locked(a, b):
        nonlocal attempts
        attempts += 1
        raise PermissionError("locked")

    monkeypatch.setattr(persistence.os, "replace", always_locked)
    monkeypatch.setattr(persistence, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError):
        persistence._replace_with_retry(tmp_path / "a", tmp_path / "b")

    assert attempts == persistence.CHECKPOINT_REPLACE_ATTEMPTS
