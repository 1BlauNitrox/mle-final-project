"""Preserve pretrained networks, target phase, replay and Adam on widening."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

from agent_code.DagobertDuckDQNAntiLoop.config import DQNConfig
from agent_code.DagobertDuckDQNAntiLoop.model import DQNLearner
from agent_code.DagobertDuckDQNAntiLoop.persistence import save_checkpoint
from agent_code.DagobertDuckDQNAntiLoop.replay import ReplayBuffer
from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint


def initialize(parent_path, destination, mode, seed, learning_rate):
    parent = load_training_checkpoint(parent_path)
    if parent.config.input_dim != 39 or mode not in {"control", "memory"}:
        raise ValueError("Wrong parent or arm")
    config = DQNConfig(
        **{
            **asdict(parent.config),
            "input_dim": 56,
            "observation_mode": mode,
            "learning_rate": learning_rate,
        }
    )
    learner = DQNLearner(config=config, seed=seed)
    for name in ("online_network", "target_network"):
        state = deepcopy(getattr(parent.learner, name).state_dict())
        state["layers.0.weight"] = torch.nn.functional.pad(state["layers.0.weight"], (0, 17))
        getattr(learner, name).load_state_dict(state)
    optimizer = deepcopy(parent.learner.optimizer.state_dict())
    first = optimizer["param_groups"][0]["params"][0]
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key in optimizer["state"].get(first, {}):
            optimizer["state"][first][key] = torch.nn.functional.pad(
                optimizer["state"][first][key], (0, 17)
            )
    learner.optimizer.load_state_dict(optimizer)
    for group in learner.optimizer.param_groups:
        group["lr"] = learning_rate
    learner.update_steps = parent.learner.update_steps
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=seed)
    state = parent.replay_buffer.state_dict()
    for key in ("states", "next_states"):
        state[key] = np.pad(state[key], ((0, 0), (0, 17)))
    state["rng_state"] = replay.state_dict()["rng_state"]
    replay.load_state_dict(state)
    save_checkpoint(
        learner=learner,
        replay_buffer=replay,
        action_rng=np.random.default_rng(seed),
        epsilon=parent.epsilon,
        completed_episodes=parent.completed_episodes,
        agent_seed=seed,
        path=destination,
    )
    generator = torch.Generator().manual_seed(221)
    original = torch.rand(256, 39, generator=generator)
    widened = torch.cat([original, torch.rand(256, 17, generator=generator)], dim=1)
    with torch.no_grad():
        for name in ("online_network", "target_network"):
            torch.testing.assert_close(
                getattr(parent.learner, name)(original),
                getattr(learner, name)(widened),
                rtol=1e-5,
                atol=1e-5,
            )
    return {
        "episodes": parent.completed_episodes,
        "updates": learner.update_steps,
        "replay_size": len(replay),
        "optimizer_entries": len(learner.optimizer.state),
        "historical_memory": "unavailable; zero-padded, not reconstructed",
    }
