"""Double DQN selects online, evaluates target, masks legality and preserves old artifacts."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNTask3.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQNTask3.model import compute_bellman_targets
from agent_code.DagobertDuckDQNTask3.persistence import CHECKPOINT_PATH, load_training_checkpoint


def test_online_selection_target_evaluation_legality_terminal_and_no_gradient():
    target = torch.tensor([[1.0, 10.0, 3.0, 4.0, 5.0, 6.0]] * 3, requires_grad=True)
    online = torch.tensor([[9.0, 2.0, 1.0, 0.0, 0.0, 0.0]] * 3, requires_grad=True)
    mask = torch.ones((3, 6), dtype=torch.bool)
    mask[1, 0] = False
    kwargs = dict(
        rewards=torch.tensor([2.0, 2.0, 2.0]),
        next_q_values=target,
        terminals=torch.tensor([False, False, True]),
        discount_factor=0.9,
        next_action_masks=mask,
    )
    assert torch.allclose(compute_bellman_targets(**kwargs), torch.tensor([11.0, 11.0, 2.0]))
    values = compute_bellman_targets(**kwargs, online_next_q_values=online)
    assert torch.allclose(values, torch.tensor([2.9, 11.0, 2.0]))
    assert not values.requires_grad


def test_tie_uses_lowest_legal_action_index():
    result = compute_bellman_targets(
        rewards=torch.zeros(1),
        next_q_values=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        online_next_q_values=torch.zeros((1, 6)),
        terminals=torch.tensor([False]),
        discount_factor=1.0,
        next_action_masks=torch.tensor([[False, True, True, True, True, True]]),
    )
    assert result.item() == 2.0


def test_historical_checkpoint_is_standard_dqn_and_bool_is_strict():
    assert load_training_checkpoint(CHECKPOINT_PATH).config.double_dqn is False
    with pytest.raises(ValueError, match="boolean"):
        replace(DEFAULT_CONFIG, double_dqn="yes")


def test_training_uses_online_selection_and_persists_mode(tmp_path, monkeypatch):
    from agent_code.DagobertDuckDQNTask3 import model
    from agent_code.DagobertDuckDQNTask3.persistence import save_checkpoint
    from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer

    config = replace(DEFAULT_CONFIG, double_dqn=True, batch_size=2, replay_warmup=2)
    learner = model.DQNLearner(config=config, seed=1)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=2)
    for terminal in (False, True):
        replay.add(
            state=np.zeros(39, dtype=np.float32),
            action_index=0,
            reward=1.0,
            next_state=None if terminal else np.ones(39, dtype=np.float32),
            terminal=terminal,
        )
    calls = []
    original = model.compute_bellman_targets

    def observe(**kwargs):
        calls.append(kwargs["online_next_q_values"])
        return original(**kwargs)

    monkeypatch.setattr(model, "compute_bellman_targets", observe)
    learner.train_batch(replay.sample(2))
    assert len(calls) == 1 and calls[0].shape == (2, 6)
    path = tmp_path / "double.pt"
    save_checkpoint(
        learner=learner,
        replay_buffer=replay,
        action_rng=np.random.default_rng(3),
        epsilon=0.1,
        completed_episodes=1,
        agent_seed=1,
        path=path,
    )
    loaded = load_training_checkpoint(path)
    assert loaded.config.double_dqn and loaded.learner.update_steps == 1
