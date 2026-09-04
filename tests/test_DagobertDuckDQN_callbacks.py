"""Tests for DagobertDuckDQN evaluation and action callbacks."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

import agent_code.DagobertDuckDQN.callbacks as callbacks
from agent_code.DagobertDuckDQN.config import ACTIONS, DEFAULT_CONFIG
from agent_code.DagobertDuckDQN.model import QNetwork, build_q_network
from agent_code.DagobertDuckDQN.persistence import save_evaluation_artifact


def make_agent(*, training: bool) -> SimpleNamespace:
    """Create a minimal framework-like callback object."""
    return SimpleNamespace(
        train=training,
        logger=Mock(),
    )


def make_game_state() -> dict:
    """Create a small open Task 1 game state."""
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("DagobertDuckDQN", 0, False, (3, 3)),
        "coins": [(5, 3)],
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros_like(field),
    }


def set_fixed_outputs(
    network: QNetwork,
    values: list[float],
) -> None:
    """Configure the network to return fixed Q-values."""
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()

        output_layer = network.layers[-1]
        assert isinstance(output_layer, nn.Linear)
        output_layer.bias.copy_(torch.tensor(values, dtype=torch.float32))


def create_checkpoint(
    path: Path,
    *,
    seed: int = 123,
    tied_outputs: bool = False,
) -> None:
    """Create a valid evaluation-only checkpoint for callback tests."""
    network = build_q_network(DEFAULT_CONFIG, seed=seed)

    if tied_outputs:
        set_fixed_outputs(
            network,
            [1.0, 1.0, 1.0, 1.0, 1.0],
        )
    else:
        set_fixed_outputs(
            network,
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )

    save_evaluation_artifact(
        network=network,
        config=DEFAULT_CONFIG,
        completed_episodes=3,
        path=path,
    )


@pytest.mark.parametrize("training", [True, False])
def test_setup_without_checkpoint_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
) -> None:
    """Setup requires the frozen evaluation checkpoint regardless of train mode."""
    monkeypatch.setattr(
        callbacks,
        "CHECKPOINT_PATH",
        tmp_path / "missing.pt",
    )
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "123")
    agent = make_agent(training=training)

    with pytest.raises(FileNotFoundError):
        callbacks.setup(agent)


def test_evaluation_checkpoint_override_stays_in_agent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_path = tmp_path / "checkpoint.pt"
    staged_path = tmp_path / ".evaluation-checkpoint.pt"
    create_checkpoint(staged_path)
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", default_path)
    monkeypatch.setenv(
        callbacks.EVALUATION_CHECKPOINT_ENV,
        staged_path.name,
    )
    agent = make_agent(training=False)

    callbacks.setup(agent)

    assert not default_path.exists()
    assert agent.completed_episodes == 3


@pytest.mark.parametrize("file_name", ["", "../checkpoint.pt", "sub/checkpoint.pt"])
def test_evaluation_checkpoint_override_rejects_non_file_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
) -> None:
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", tmp_path / "checkpoint.pt")
    monkeypatch.setenv(callbacks.EVALUATION_CHECKPOINT_ENV, file_name)

    with pytest.raises(ValueError, match="must contain one file name"):
        callbacks._evaluation_checkpoint_path()


def test_evaluation_loads_only_frozen_policy_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    create_checkpoint(path)
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", path)
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "123")
    agent = make_agent(training=False)

    callbacks.setup(agent)

    assert not hasattr(agent, "learner")
    assert not hasattr(agent, "replay_buffer")
    assert not agent.policy_network.training
    assert all(not parameter.requires_grad for parameter in agent.policy_network.parameters())
    assert agent.epsilon == 0.0


def test_evaluation_selects_greedy_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    create_checkpoint(path)
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", path)
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "123")
    agent = make_agent(training=False)
    callbacks.setup(agent)

    action = callbacks.act(agent, make_game_state())

    assert action == "WAIT"


def test_evaluation_is_deterministic_for_equal_agent_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    create_checkpoint(path, tied_outputs=True)
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", path)
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "456")

    first = make_agent(training=False)
    second = make_agent(training=False)
    callbacks.setup(first)
    callbacks.setup(second)

    first_actions = [callbacks.act(first, make_game_state()) for _ in range(20)]
    second_actions = [callbacks.act(second, make_game_state()) for _ in range(20)]

    assert first_actions == second_actions


def test_evaluation_does_not_modify_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    create_checkpoint(path)
    bytes_before = path.read_bytes()
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", path)
    agent = make_agent(training=False)

    callbacks.setup(agent)
    callbacks.act(agent, make_game_state())

    assert path.read_bytes() == bytes_before


def test_actions_always_respect_task1_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    create_checkpoint(path)
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", path)
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "123")
    agent = make_agent(training=True)
    callbacks.setup(agent)

    selected = {callbacks.act(agent, make_game_state()) for _ in range(100)}

    assert selected <= set(ACTIONS)
    assert "BOMB" not in selected


def test_none_game_state_falls_back_to_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    create_checkpoint(path)
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", path)
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "123")
    agent = make_agent(training=True)
    callbacks.setup(agent)

    assert callbacks.act(agent, None) == "WAIT"


def test_invalid_agent_seed_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        callbacks,
        "CHECKPOINT_PATH",
        tmp_path / "missing.pt",
    )
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "not-an-integer")
    agent = make_agent(training=True)

    with pytest.raises(ValueError, match="integer"):
        callbacks.setup(agent)


def test_evaluation_uses_one_torch_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    create_checkpoint(path)
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", path)
    agent = make_agent(training=False)

    callbacks.setup(agent)

    assert torch.get_num_threads() == 1
