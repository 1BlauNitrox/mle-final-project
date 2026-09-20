from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNAntiLoop.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQNAntiLoop.model import DQNLearner
from agent_code.DagobertDuckDQNAntiLoop.persistence import save_checkpoint
from agent_code.DagobertDuckDQNAntiLoop.replay import ReplayBuffer
from training.bomb_head_finetune import (
    BOMB_INDEX,
    HeadDataset,
    build_dataset,
    install_online_state,
    train_snapshots,
)


def record(value, *, trapped=False, kills=0, self_kills=0):
    features = np.full(DEFAULT_CONFIG.input_dim, value, dtype=np.float32)
    return {
        "features": features.tolist(),
        "q_values": [0.0, 0.1, 0.2, 0.3, 0.4, -0.1],
        "trapped_attack": trapped,
        "kill_credits": kills,
        "self_kill_credits": self_kills,
    }


def test_build_dataset_requires_empirical_kill_and_keeps_teacher_examples():
    rows = [
        {
            "head_training_trace": {
                "anchors": [record(0.1), record(0.2, trapped=True)],
                "bombs": [record(0.3, kills=1), record(0.4, self_kills=1)],
            }
        }
    ]
    dataset = build_dataset(rows, input_dim=DEFAULT_CONFIG.input_dim, minimum_empirical_positives=1)
    assert dataset.positive.shape == (2, DEFAULT_CONFIG.input_dim)
    assert dataset.negative.shape == (1, DEFAULT_CONFIG.input_dim)
    assert dataset.anchors.shape == (2, DEFAULT_CONFIG.input_dim)
    assert dataset.empirical_positive_count == 1
    assert dataset.trapped_teacher_count == 1

    with pytest.raises(ValueError, match="empirical positive"):
        build_dataset(rows, input_dim=DEFAULT_CONFIG.input_dim, minimum_empirical_positives=2)


def test_training_changes_only_bomb_row_and_checkpoint_round_trips(tmp_path):
    learner = DQNLearner(config=DEFAULT_CONFIG, seed=17)
    replay = ReplayBuffer(capacity=DEFAULT_CONFIG.replay_capacity, seed=18)
    parent = tmp_path / "parent.pt"
    save_checkpoint(
        learner=learner,
        replay_buffer=replay,
        action_rng=np.random.default_rng(19),
        epsilon=0.0,
        completed_episodes=0,
        agent_seed=17,
        path=parent,
    )
    original = deepcopy(learner.online_network.state_dict())
    generator = torch.Generator().manual_seed(20)
    positive = torch.rand((8, DEFAULT_CONFIG.input_dim), generator=generator)
    negative = torch.rand((6, DEFAULT_CONFIG.input_dim), generator=generator)
    anchors = torch.rand((12, DEFAULT_CONFIG.input_dim), generator=generator)
    with torch.no_grad():
        anchor_values = learner.online_network(anchors)[:, BOMB_INDEX].clone()
    dataset = HeadDataset(positive, negative, anchors, anchor_values, 8, 0, 6)

    snapshots, history = train_snapshots(
        parent,
        dataset,
        snapshot_steps=[2],
        learning_rate=0.001,
        margin=0.05,
        anchor_weight=1.0,
        negative_weight=1.0,
        l2_weight=0.1,
        batch_size=4,
        seed=21,
    )
    state = snapshots[2]
    assert len(history) == 2
    assert torch.equal(
        state["layers.4.weight"][:BOMB_INDEX], original["layers.4.weight"][:BOMB_INDEX]
    )
    assert torch.equal(state["layers.4.bias"][:BOMB_INDEX], original["layers.4.bias"][:BOMB_INDEX])
    assert not torch.equal(
        state["layers.4.weight"][BOMB_INDEX], original["layers.4.weight"][BOMB_INDEX]
    )
    for name in original:
        if name not in {"layers.4.weight", "layers.4.bias"}:
            assert torch.equal(state[name], original[name])

    destination = tmp_path / "trained.pt"
    install_online_state(parent, destination, state)
    assert destination.is_file()
