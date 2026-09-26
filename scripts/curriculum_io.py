"""Shared atomic I/O and pinned DQN initialization, adapted from issue 213."""

import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def initialize(parent_path, destination, arm, seed, learning_rate):
    import numpy as np

    from agent_code.DagobertDuckDQNTask3.model import DQNLearner
    from agent_code.DagobertDuckDQNTask3.persistence import (
        load_training_checkpoint,
        save_checkpoint,
    )
    from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer

    parent = load_training_checkpoint(parent_path)
    require(parent.config.input_dim == 39, "Expected original 39-input checkpoint")
    require(arm == "preserve", "Curriculum requires preserved training state")
    config = replace(parent.config, learning_rate=learning_rate)
    learner = DQNLearner(config=config, seed=seed)
    learner.online_network.load_state_dict(parent.learner.online_network.state_dict())
    learner.target_network.load_state_dict(parent.learner.target_network.state_dict())
    learner.update_steps = parent.learner.update_steps
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=seed)
    if arm == "preserve":
        learner.optimizer.load_state_dict(deepcopy(parent.learner.optimizer.state_dict()))
        for group in learner.optimizer.param_groups:
            group["lr"] = learning_rate
        state = parent.replay_buffer.state_dict()
        state["rng_state"] = deepcopy(replay.state_dict()["rng_state"])
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
    return {
        "episodes": parent.completed_episodes,
        "updates": parent.learner.update_steps,
        "replay_size": len(replay),
        "optimizer_entries": len(learner.optimizer.state),
    }
