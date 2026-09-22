"""Weight-preserving, replay-free initialization for all observation arms."""

from __future__ import annotations

from dataclasses import asdict

import torch

from agent_code.DagobertDuckDQNObservation.callbacks import _initial_random_streams
from agent_code.DagobertDuckDQNObservation.config import DQNConfig
from agent_code.DagobertDuckDQNObservation.model import DQNLearner
from agent_code.DagobertDuckDQNObservation.persistence import save_checkpoint
from agent_code.DagobertDuckDQNObservation.replay import ReplayBuffer
from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint


def migrate(reference, destination, mode, seed):
    parent = load_training_checkpoint(reference)
    settings = asdict(parent.config)
    settings.update(input_dim=56, observation_mode=mode, learning_rate=0.0002)
    config = DQNConfig(**settings)
    learner = DQNLearner(config=config, seed=seed)
    for name in ("online_network", "target_network"):
        before = getattr(parent.learner, name).state_dict()
        after = getattr(learner, name).state_dict()
        for key, value in before.items():
            if key == "layers.0.weight":
                after[key].zero_()
                after[key][:, :39].copy_(value)
            else:
                after[key].copy_(value)
        getattr(learner, name).load_state_dict(after)
    action_rng, replay_seed = _initial_random_streams(seed)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=replay_seed)
    save_checkpoint(
        learner=learner,
        replay_buffer=replay,
        action_rng=action_rng,
        epsilon=0.0,
        completed_episodes=0,
        agent_seed=seed,
        path=destination,
    )
    # Test the migration on a fixed synthetic mechanics sample; no game outcome
    # or holdout data is used. All arms receive the same widened architecture.
    generator = torch.Generator().manual_seed(211)
    original = torch.rand(256, 39, generator=generator)
    augmented = torch.cat((original, torch.rand(256, 17, generator=generator)), dim=1)
    with torch.no_grad():
        for name in ("online_network", "target_network"):
            left = getattr(parent.learner, name)(original)
            right = getattr(learner, name)(augmented)
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
            if not torch.equal(left.argmax(dim=1), right.argmax(dim=1)):
                raise ValueError("Migration changed greedy actions on mechanics states")
    return config
