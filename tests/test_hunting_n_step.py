import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.hunting_n_step import NStepAccumulator, RawTransition

model = importlib.import_module("agent_code.Bomb-omb-nstep.model")
ROOT = Path(__file__).resolve().parents[1]


def raw(index, reward, *, terminal=False):
    state = np.full(39, index, dtype=np.float32)
    return RawTransition(
        state=state,
        action_index=index % 6,
        reward=reward,
        next_state=None if terminal else np.full(39, index + 1, dtype=np.float32),
        next_action_mask=None if terminal else np.ones(6, dtype=np.bool_),
        terminal=terminal,
    )


def test_five_step_discounted_return_and_bootstrap_discount():
    accumulator = NStepAccumulator(5, 0.9)
    emitted = []
    for index in range(5):
        emitted += accumulator.push(raw(index, index + 1))
    assert len(emitted) == 1
    assert emitted[0].reward == pytest.approx(sum(0.9**i * (i + 1) for i in range(5)))
    assert emitted[0].bootstrap_discount == pytest.approx(0.9**5)
    assert emitted[0].horizon == 5


def test_terminal_flushes_short_tails_without_bootstrap_or_boundary_leakage():
    accumulator = NStepAccumulator(5, 0.9)
    assert accumulator.push(raw(0, 1.0)) == []
    assert accumulator.push(raw(1, 2.0)) == []
    emitted = accumulator.push(raw(2, 3.0, terminal=True))
    assert [item.horizon for item in emitted] == [3, 2, 1]
    assert [item.reward for item in emitted] == pytest.approx(
        [1 + 0.9 * 2 + 0.9**2 * 3, 2 + 0.9 * 3, 3]
    )
    assert all(item.terminal and item.bootstrap_discount == 0 for item in emitted)
    assert not accumulator.pending
    assert accumulator.push(raw(10, 4.0)) == []


def test_n_equals_one_is_one_raw_transition_per_identical_replay_item():
    accumulator = NStepAccumulator(1, 0.9)
    for transition in (raw(0, 2.5), raw(1, -1.0, terminal=True)):
        emitted = accumulator.push(transition)
        assert len(emitted) == 1
        item = emitted[0]
        assert item.state is transition.state
        assert item.action_index == transition.action_index
        assert item.reward == transition.reward
        assert item.next_state is transition.next_state
        assert item.terminal is transition.terminal
        assert item.bootstrap_discount == (0.0 if transition.terminal else 0.9)


def test_per_transition_discount_controls_bootstrap_and_terminal_death():
    targets = model.compute_bellman_targets(
        rewards=torch.tensor([1.0, 2.0]),
        next_q_values=torch.tensor([[1.0, 4.0, 2.0, 0.0, 3.0, -1.0]] * 2),
        terminals=torch.tensor([False, True]),
        discount_factor=0.9,
        bootstrap_discounts=torch.tensor([0.9**5, 0.0]),
    )
    assert targets.tolist() == pytest.approx([1 + 0.9**5 * 4, 2.0])


def test_registered_comparison_changes_only_return_horizon():
    config = json.loads(
        (ROOT / "experiments/2026-09-19-five-step-dqn/config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["arms"] == {"one-step": {"n_step": 1}, "five-step": {"n_step": 5}}
    assert config["shared_training"]["opponents"] == [
        "RUEHL_BASED_AGENT",
        "rule_based_agent",
        "peaceful_agent",
    ]
    assert config["shared_training"]["decision_checkpoint"] == 1000
    assert config["shared_training"]["total_episodes"] == 6000
