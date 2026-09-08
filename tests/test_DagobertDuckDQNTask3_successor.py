"""Task 2-prefix, migration, persistence, and reward-contract tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import agent_code.DagobertDuckDQNTask3.train as training
from agent_code.DagobertDuckDQNTask2.config import LEGACY_FEATURE_COUNT
from agent_code.DagobertDuckDQNTask2.config import DQNConfig as Task2Config
from agent_code.DagobertDuckDQNTask2.features import (
    normalize_features as normalize_task2,
)
from agent_code.DagobertDuckDQNTask2.model import build_q_network as build_task2_network
from agent_code.DagobertDuckDQNTask3.config import (
    ACTIONS,
    DEFAULT_CONFIG,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
)
from agent_code.DagobertDuckDQNTask3.features import (
    normalize_features,
    state_to_features,
)
from agent_code.DagobertDuckDQNTask3.migration import (
    INHERITED_Q_VALUE_TOLERANCE,
    migrate_online_network,
)
from agent_code.DagobertDuckDQNTask3.model import DQNLearner
from agent_code.DagobertDuckDQNTask3.persistence import (
    CHECKPOINT_PATH,
    load_evaluation_checkpoint,
    load_training_checkpoint,
)
from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer
from agent_code.DagobertDuckDQNTask3.rewards import reward_from_events

LEGACY_TASK2_CONFIG = Task2Config(
    input_dim=LEGACY_FEATURE_COUNT, escape_continuation_features=False
)


def make_state(step: int = 1, *, others: list[tuple] | None = None) -> dict:
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("task3", 0, True, (3, 3)),
        "coins": [(5, 3)],
        "bombs": [],
        "others": [] if others is None else others,
        "explosion_map": np.zeros_like(field),
    }


def make_agent() -> SimpleNamespace:
    config = replace(
        DEFAULT_CONFIG,
        batch_size=1,
        replay_warmup=1,
        replay_capacity=8,
        target_update_interval=10,
    )
    learner = DQNLearner(config=config, seed=123)
    return SimpleNamespace(
        train=True,
        logger=Mock(),
        config=config,
        learner=learner,
        replay_buffer=ReplayBuffer(capacity=config.replay_capacity, seed=456),
        action_rng=np.random.default_rng(789),
        epsilon=config.initial_epsilon,
        completed_episodes=0,
        agent_seed=123,
        policy_network=learner.online_network,
    )


def test_schema_and_action_order_extend_task2_without_reordering() -> None:
    assert FEATURE_COUNT == 34
    assert FEATURE_SCHEMA_VERSION == 3
    assert ACTIONS == ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def test_committed_checkpoint_round_trips_and_evaluation_does_not_mutate() -> None:
    before = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).digest()
    evaluation = load_evaluation_checkpoint(CHECKPOINT_PATH)
    training = load_training_checkpoint(CHECKPOINT_PATH)

    assert evaluation.config.input_dim == FEATURE_COUNT
    assert evaluation.config.output_dim == len(ACTIONS)
    assert evaluation.network.training is False
    assert all(parameter.requires_grad is False for parameter in evaluation.network.parameters())
    assert training.completed_episodes == 0
    assert len(training.replay_buffer) == 0
    assert hashlib.sha256(CHECKPOINT_PATH.read_bytes()).digest() == before


def test_task1_and_task2_prefix_is_preserved() -> None:
    state = make_state()
    task1_features = __import__(
        "agent_code.DagobertDuckDQN.features",
        fromlist=["state_to_features"],
    ).state_to_features(state)
    task2_features = __import__(
        "agent_code.DagobertDuckDQNTask2.features",
        fromlist=["state_to_features"],
    ).state_to_features(state)
    task3_features = state_to_features(state)

    assert task1_features is not None
    assert task2_features is not None
    assert task3_features is not None
    assert task3_features[:8] == task1_features
    # Task2's own feature vector is now permanently 26-wide (issue #87's
    # continuation columns are always present, zero-gated rather than
    # shape-gated) but Task3 was migrated from the frozen pre-#87 21-wide
    # checkpoint, so only the shared 21-element prefix is comparable.
    assert task3_features[:21] == task2_features[:21]
    assert len(task3_features) == FEATURE_COUNT


def test_migration_zeroes_new_columns_and_preserves_all_outputs() -> None:
    parent = build_task2_network(config=LEGACY_TASK2_CONFIG, seed=7)
    migrated = migrate_online_network(parent, seed=44)
    parent_layers = [layer for layer in parent.layers if hasattr(layer, "weight")]
    migrated_layers = [layer for layer in migrated.layers if hasattr(layer, "weight")]

    assert torch.equal(
        migrated_layers[0].weight[:, :21], parent_layers[0].weight
    )
    assert torch.equal(
        migrated_layers[0].weight[:, 21:],
        torch.zeros_like(migrated_layers[0].weight[:, 21:]),
    )
    for parent_layer, migrated_layer in zip(
        parent_layers[1:], migrated_layers[1:], strict=True
    ):
        assert torch.equal(parent_layer.weight, migrated_layer.weight)
        assert torch.equal(parent_layer.bias, migrated_layer.bias)


def test_migration_preserves_q_values_and_new_columns_receive_gradients() -> None:
    parent = build_task2_network(config=LEGACY_TASK2_CONFIG, seed=8)
    migrated = migrate_online_network(parent, seed=44)
    state = make_state(others=[("opponent", 0, True, (5, 3))])
    task2_features = __import__(
        "agent_code.DagobertDuckDQNTask2.features",
        fromlist=["state_to_features"],
    ).state_to_features(state)
    task3_features = state_to_features(state)
    assert task2_features is not None
    assert task3_features is not None

    with torch.no_grad():
        parent_q = parent(torch.from_numpy(normalize_task2(task2_features[:21])))
        migrated_q = migrated(torch.from_numpy(normalize_features(task3_features)))

    torch.testing.assert_close(
        parent_q,
        migrated_q,
        rtol=0.0,
        atol=INHERITED_Q_VALUE_TOLERANCE,
    )

    migrated(torch.ones(FEATURE_COUNT, dtype=torch.float32)).sum().backward()
    input_layer = [layer for layer in migrated.layers if hasattr(layer, "weight")][0]
    assert bool(input_layer.weight.grad[:, 21:].abs().any())


def test_reward_counts_each_attributed_elimination_and_ignores_third_party_event() -> None:
    assert reward_from_events(
        ["KILLED_OPPONENT", "KILLED_OPPONENT", "OPPONENT_ELIMINATED"]
    ) == pytest.approx(10.0)


def test_terminal_elimination_is_recorded_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    training.setup_training(agent)
    monkeypatch.setattr(training, "CHECKPOINT_PATH", tmp_path / "checkpoint.pt")
    old_state = make_state()
    new_state = make_state(step=2)

    training.game_events_occurred(
        agent,
        old_state,
        "BOMB",
        new_state,
        ["KILLED_OPPONENT"],
    )
    metrics = training.end_of_round(
        agent,
        old_state,
        "BOMB",
        ["KILLED_OPPONENT", "SURVIVED_ROUND"],
    )

    assert metrics["shaped_reward"] == pytest.approx(10.0)
    assert metrics["event_count_killed_opponent"] == pytest.approx(1.0)
    assert metrics["update_count"] == 1
    assert agent.learner.update_steps == 1


def test_terminal_elimination_preserves_genuine_event_multiplicity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    training.setup_training(agent)
    monkeypatch.setattr(training, "CHECKPOINT_PATH", tmp_path / "checkpoint.pt")
    old_state = make_state()
    new_state = make_state(step=2)
    kill_events = ["KILLED_OPPONENT", "KILLED_OPPONENT"]

    training.game_events_occurred(
        agent,
        old_state,
        "BOMB",
        new_state,
        kill_events,
    )
    metrics = training.end_of_round(
        agent,
        old_state,
        "BOMB",
        [*kill_events, "SURVIVED_ROUND"],
    )

    assert metrics["event_count_killed_opponent"] == pytest.approx(2.0)
    assert metrics["shaped_reward"] == pytest.approx(15.0)
    assert metrics["update_count"] == 1


def test_task3_normalization_preserves_binary_and_signed_domains() -> None:
    raw = tuple(range(21)) + (1, -1, 1, 3, 1, 0, 1, 0, 1, 0, -1, 1, 3)
    normalized = normalize_features(raw)

    assert normalized.shape == (FEATURE_COUNT,)
    assert normalized[22] == -1.0
    assert normalized[24] == 1.0
    assert normalized[27] == 1.0
    assert normalized[28] == 0.0
    assert normalized[31] == -1.0
    assert normalized[33] == 1.0
