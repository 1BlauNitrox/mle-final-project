"""Contracts for Issue #86 framework-legal action masking."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNTask2.config import DQNConfig
from agent_code.DagobertDuckDQNTask2.legality import framework_legal_action_mask
from agent_code.DagobertDuckDQNTask2.model import (
    build_q_network,
    compute_bellman_targets,
    select_action,
)


def _state() -> dict:
    field = np.zeros((7, 7), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[3, 2] = 1
    return {
        "field": field,
        "self": ("agent", 0, False, (3, 3)),
        "bombs": [((4, 3), 3)],
        "others": [("other", 0, True, (2, 3))],
    }


def test_mask_matches_framework_obstacles_bomb_inventory_and_wait() -> None:
    assert framework_legal_action_mask(_state()).tolist() == [
        False,
        False,
        True,
        False,
        True,
        False,
    ]


def test_masked_exploration_and_greedy_selection_never_choose_ineligible_action() -> None:
    network = build_q_network(DQNConfig(), seed=1)
    with torch.no_grad():
        network.layers[-1].bias.copy_(torch.arange(6, dtype=torch.float32))
    state = np.zeros(21, dtype=np.float32)
    mask = np.asarray([True, False, False, False, True, False], dtype=np.bool_)
    assert (
        select_action(
            network=network,
            state=state,
            epsilon=0.0,
            rng=np.random.default_rng(1),
            action_mask=mask,
        )
        == "WAIT"
    )
    assert {
        select_action(
            network=network,
            state=state,
            epsilon=1.0,
            rng=np.random.default_rng(seed),
            action_mask=mask,
        )
        for seed in range(20)
    } <= {"UP", "WAIT"}


def test_masked_bellman_target_excludes_an_invalid_high_q_value() -> None:
    target = compute_bellman_targets(
        rewards=torch.tensor([1.0]),
        next_q_values=torch.tensor([[1.0, 99.0, 2.0, 3.0, 4.0, 5.0]]),
        terminals=torch.tensor([False]),
        discount_factor=0.5,
        next_action_masks=torch.tensor([[True, False, True, True, True, True]]),
    )
    torch.testing.assert_close(target, torch.tensor([3.5]))


def test_empty_mask_is_rejected() -> None:
    network = build_q_network(DQNConfig(), seed=1)
    with pytest.raises(ValueError, match="legal"):
        select_action(
            network=network,
            state=np.zeros(21, dtype=np.float32),
            epsilon=0.0,
            rng=np.random.default_rng(1),
            action_mask=np.zeros(6, dtype=np.bool_),
        )
