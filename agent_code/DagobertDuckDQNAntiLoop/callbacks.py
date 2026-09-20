"""Framework setup and action callbacks for the Task 3 DQN successor."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import ACTIONS, DEFAULT_CONFIG, DQNConfig
from .legality import framework_legal_action_mask
from .model import CPU_DEVICE, DQNLearner, select_action
from .narrow_loop_guard import NarrowLoopGuard
from .observations import observe
from .persistence import (
    CHECKPOINT_PATH,
    load_evaluation_checkpoint,
    load_training_checkpoint,
)
from .replay import ReplayBuffer

EVALUATION_CHECKPOINT_ENV = "BOMBERMAN_EVALUATION_CHECKPOINT"
ACTION_MASKING_ENV = "BOMBERMAN_DQN_ACTION_MASKING"
ESCAPE_CONTINUATIONS_ENV = "BOMBERMAN_DQN_ESCAPE_CONTINUATIONS"
NARROW_LOOP_GUARD_ENV = "BOMBERMAN_NARROW_LOOP_GUARD"
DEFAULT_NARROW_LOOP_GUARD_MODE = "mid_persistent"


def setup(self) -> None:
    """Initialize a new training state or load a checkpoint."""
    agent_seed = _read_agent_seed()
    self.agent_seed = agent_seed

    if self.train:
        _setup_training_policy(self, agent_seed)
    else:
        _setup_evaluation_policy(self, agent_seed)

    torch.set_num_threads(self.config.torch_num_threads)


def act(self, game_state: dict | None) -> str:
    """Select one seeded epsilon-greedy Task 2 action."""
    features = observe(self, game_state)

    if features is None:
        return "WAIT"

    state = features

    epsilon = self.epsilon if self.train else 0.0

    action_mask = framework_legal_action_mask(game_state) if self.config.action_masking else None
    action = select_action(
        network=self.policy_network,
        state=state,
        epsilon=epsilon,
        rng=self.action_rng,
        action_mask=action_mask,
    )
    guard_mode = os.environ.get(NARROW_LOOP_GUARD_ENV, DEFAULT_NARROW_LOOP_GUARD_MODE)
    if self.train or guard_mode == "off":
        return action
    if guard_mode not in {
        "on",
        "cooldown",
        "persistent",
        "early_persistent",
        "mid_persistent",
    }:
        raise ValueError(f"Invalid {NARROW_LOOP_GUARD_ENV} mode")
    guard = getattr(self, "narrow_loop_guard", None)
    if guard is None:
        window = _loop_guard_window(guard_mode)
        guard = self.narrow_loop_guard = NarrowLoopGuard(window=window)
    guard.observe(game_state)

    cached_q_values = None

    def q_values():
        nonlocal cached_q_values
        if cached_q_values is None:
            with torch.no_grad():
                cached_q_values = (
                    self.policy_network(torch.from_numpy(state).to(CPU_DEVICE)).cpu().numpy()
                )
        return cached_q_values

    legal = action_mask if action_mask is not None else np.ones(len(ACTIONS), dtype=bool)
    if guard_mode in {"cooldown", "persistent", "early_persistent", "mid_persistent"}:
        action = guard.redirect_contested(game_state, action, q_values, legal)
    return guard.choose(
        game_state,
        action,
        q_values,
        legal,
        followup_steps={
            "on": 0,
            "cooldown": 4,
            "persistent": 400,
            "early_persistent": 400,
            "mid_persistent": 400,
        }[guard_mode],
    )


def _loop_guard_window(mode: str) -> int:
    return {"early_persistent": 12, "mid_persistent": 16}.get(mode, 24)


def _setup_training_policy(self, agent_seed: int) -> None:
    """Restore resumable training state or initialize a new one."""
    if CHECKPOINT_PATH.is_file():
        loaded = load_training_checkpoint(CHECKPOINT_PATH)
        configured = _configured_training_config(loaded.config)
        is_fresh_migration = loaded.completed_episodes == 0 and len(loaded.replay_buffer) == 0
        config = loaded.config
        replay_buffer = loaded.replay_buffer
        action_rng = loaded.action_rng
        if (
            loaded.config.action_masking != configured.action_masking
            or loaded.config.escape_continuation_features != configured.escape_continuation_features
        ):
            raise ValueError("Configured mask/escape mode does not match parent checkpoint")

        if loaded.agent_seed != agent_seed:
            if not is_fresh_migration:
                raise ValueError("BOMBERMAN_AGENT_SEED does not match checkpoint seed.")
            action_rng, replay_seed = _initial_random_streams(agent_seed)
            replay_buffer = ReplayBuffer(capacity=config.replay_capacity, seed=replay_seed)

        self.config = config
        self.learner = loaded.learner
        self.replay_buffer = replay_buffer
        self.action_rng = action_rng
        self.epsilon = loaded.epsilon
        self.completed_episodes = loaded.completed_episodes
        self.agent_seed = agent_seed
        self.policy_network = self.learner.online_network

        self.logger.info(
            "Resumed DQN training after %d episodes and %d updates",
            self.completed_episodes,
            self.learner.update_steps,
        )
        return

    self.config = _configured_training_config()
    action_rng, replay_seed = _initial_random_streams(agent_seed)

    self.learner = DQNLearner(
        config=self.config,
        seed=agent_seed,
    )
    self.replay_buffer = ReplayBuffer(
        capacity=self.config.replay_capacity,
        seed=replay_seed,
    )
    self.action_rng = action_rng
    self.epsilon = self.config.initial_epsilon
    self.completed_episodes = 0
    self.policy_network = self.learner.online_network

    self.logger.info(
        "Initialized new DQN training state with seed %d",
        agent_seed,
    )


def _setup_evaluation_policy(self, agent_seed: int) -> None:
    """Load a frozen evaluation network without training objects."""
    loaded = load_evaluation_checkpoint(_evaluation_checkpoint_path())

    if _configured_training_config(loaded.config) != loaded.config:
        raise ValueError("Evaluation mask/escape mode does not match checkpoint")
    self.config = loaded.config
    self.policy_network = loaded.network
    self.action_rng, _unused_replay_seed = _initial_random_streams(agent_seed)
    self.epsilon = 0.0
    self.completed_episodes = loaded.completed_episodes

    self.logger.info(
        "Loaded frozen DQN policy after %d training episodes",
        self.completed_episodes,
    )


def _evaluation_checkpoint_path() -> Path:
    """Resolve an optional repository-managed evaluation artifact locally."""
    file_name = os.environ.get(EVALUATION_CHECKPOINT_ENV)
    if file_name is None:
        return CHECKPOINT_PATH
    if not file_name or file_name != os.path.basename(file_name):
        raise ValueError(f"{EVALUATION_CHECKPOINT_ENV} must contain one file name.")
    return CHECKPOINT_PATH.with_name(file_name)


def _initial_random_streams(
    agent_seed: int,
) -> tuple[np.random.Generator, int]:
    """Derive independent action and replay streams from one seed."""
    if agent_seed < 0:
        raise ValueError("BOMBERMAN_AGENT_SEED must be non-negative.")

    root_sequence = np.random.SeedSequence(agent_seed)
    action_sequence, replay_sequence = root_sequence.spawn(2)

    action_rng = np.random.default_rng(action_sequence)
    replay_seed = int(
        replay_sequence.generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )

    return action_rng, replay_seed


def _read_agent_seed() -> int:
    """Read the explicit agent seed from the environment."""
    raw_seed = os.environ.get("BOMBERMAN_AGENT_SEED")

    if raw_seed is None:
        return DEFAULT_CONFIG.default_seed

    try:
        seed = int(raw_seed)
    except ValueError as error:
        raise ValueError("BOMBERMAN_AGENT_SEED must be an integer.") from error

    if seed < 0:
        raise ValueError("BOMBERMAN_AGENT_SEED must be non-negative.")

    return seed


def _configured_training_config(base: DQNConfig = DEFAULT_CONFIG) -> DQNConfig:
    """Absent selectors preserve persisted parent modes, including during training."""
    mode = os.environ.get(ACTION_MASKING_ENV, "framework_legal" if base.action_masking else "none")
    escape = os.environ.get(
        ESCAPE_CONTINUATIONS_ENV, "on" if base.escape_continuation_features else "off"
    )
    if mode not in {"none", "framework_legal"} or escape not in {"off", "on"}:
        raise ValueError("Invalid action mask or escape selector")
    return replace(
        base, action_masking=mode == "framework_legal", escape_continuation_features=escape == "on"
    )
