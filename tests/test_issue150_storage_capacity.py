"""Exercise storage envelopes with a full replay and maximum-length episode data."""

import json

import numpy as np
import torch

from agent_code.DagobertDuckDQNTask3.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQNTask3.model import DQNLearner
from agent_code.DagobertDuckDQNTask3.persistence import save_checkpoint
from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer
from training.issue150_storage import EVALUATION_BYTES, SNAPSHOT_BYTES, TRAINING_BYTES


def test_full_replay_optimizer_and_atomic_checkpoint_fit_envelope(tmp_path):
    torch.set_num_threads(1)
    config = DEFAULT_CONFIG
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=150)
    state = np.ones(config.input_dim, dtype=np.float32)
    for index in range(config.replay_capacity):
        replay.add(
            state=state,
            action_index=index % 6,
            reward=1.0,
            next_state=state,
            terminal=False,
            next_action_mask=np.ones(6, dtype=np.bool_),
        )
    learner = DQNLearner(config=config, seed=150)
    learner.train_batch(replay.sample(config.batch_size))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        learner=learner,
        replay_buffer=replay,
        action_rng=np.random.default_rng(150),
        epsilon=0.1,
        completed_episodes=10000,
        agent_seed=150,
        path=path,
    )
    # Allow simultaneous old/new checkpoints during atomic saving, plus one MiB
    # for source files and the inherited evaluation fixture.
    assert 2 * path.stat().st_size + 1024**2 < SNAPSHOT_BYTES


def test_two_agent_400_step_records_fit_registered_episode_budget():
    # JSON's long finite-number spelling and indented raw timings for both agents.
    agent = {
        "decision_times_ms": [1.2345678901234567e-123] * 400,
        **{f"diagnostic_{index}": 1.2345678901234567e123 for index in range(100)},
    }
    episode = {"agents": {"observed_agent": agent, "peaceful_agent": agent}}
    per_episode = len(json.dumps(episode, indent=4).encode())
    assert per_episode + 64 * 1024 < EVALUATION_BYTES
    # Include one KiB per episode for normalized CSV and 1 MiB fixed metadata.
    assert 10000 * (per_episode + 1024) + 1024**2 < TRAINING_BYTES
