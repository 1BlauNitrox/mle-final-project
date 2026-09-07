"""Framework setup and action callbacks for the Task 2 successor scaffold."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CONFIG, PROTECTED_SOURCE_EPISODES, DQNConfig
from .features import normalize_features, state_to_features
from .legality import framework_legal_action_mask
from .model import DQNLearner, select_action
from .persistence import (
    CHECKPOINT_PATH,
    load_evaluation_checkpoint,
    load_training_checkpoint,
)
from .replay import ReplayBuffer

EVALUATION_CHECKPOINT_ENV = "BOMBERMAN_EVALUATION_CHECKPOINT"
ACTION_MASKING_ENV = "BOMBERMAN_DQN_ACTION_MASKING"
ESCAPE_CONTINUATIONS_ENV = "BOMBERMAN_DQN_ESCAPE_CONTINUATIONS"


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
    features = state_to_features(
        game_state,
        include_continuation_features=self.config.escape_continuation_features,
    )

    if features is None:
        return "WAIT"

    if self.config.input_dim < len(features):
        features = features[: self.config.input_dim]

    state = normalize_features(features)

    epsilon = self.epsilon if self.train else 0.0

    return select_action(
        network=self.policy_network,
        state=state,
        epsilon=epsilon,
        rng=self.action_rng,
        action_mask=(
            framework_legal_action_mask(game_state) if self.config.action_masking else None
        ),
    )


def _setup_training_policy(self, agent_seed: int) -> None:
    """Restore resumable training state or initialize a new one."""
    if CHECKPOINT_PATH.is_file():
        loaded = load_training_checkpoint(CHECKPOINT_PATH)
        configured = _configured_training_config()
        is_fresh_migration = (
            loaded.completed_episodes == 0 and len(loaded.replay_buffer) == 0
        )
        config = loaded.config
        replay_buffer = loaded.replay_buffer
        action_rng = loaded.action_rng
        treatment_mismatch = (
            loaded.config.action_masking != configured.action_masking
            or loaded.config.escape_continuation_features
            != configured.escape_continuation_features
        )
        reset_fresh_state = treatment_mismatch or loaded.agent_seed != agent_seed
        if treatment_mismatch:
            if not is_fresh_migration:
                raise ValueError(
                    "Configured DQN treatment does not match the checkpoint "
                    "(action masking, escape-continuation, or replay mode)."
                )
            loaded.learner.config = configured
            loaded.learner.online_network.config = configured
            loaded.learner.target_network.config = configured
            config = configured

        if loaded.agent_seed != agent_seed:
            if not is_fresh_migration:
                raise ValueError("BOMBERMAN_AGENT_SEED does not match checkpoint seed.")
            action_rng, replay_seed = _initial_random_streams(agent_seed)
        if reset_fresh_state:
            if loaded.agent_seed == agent_seed:
                action_rng, replay_seed = _initial_random_streams(agent_seed)
            replay_buffer = ReplayBuffer(
                capacity=config.replay_capacity,
                seed=replay_seed,
            )

        self.config = config
        self.learner = loaded.learner
        self.replay_buffer = replay_buffer
        self.action_rng = action_rng
        self.epsilon = loaded.epsilon
        self.completed_episodes = loaded.completed_episodes
        _apply_collection_phase(self, _collection_is_open())
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
        mode=self.config.replay_treatment,
        collection_open=_collection_is_open(),
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

    self.config = loaded.config
    _validate_evaluation_treatment(self.config)
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
        raise ValueError(
            f"{EVALUATION_CHECKPOINT_ENV} must contain one file name."
        )
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


def _configured_training_config() -> DQNConfig:
    """Read the run-plan's explicit treatment selectors for a fresh run."""
    mode = os.environ.get(ACTION_MASKING_ENV, "none")
    if mode not in {"none", "framework_legal"}:
        raise ValueError(f"{ACTION_MASKING_ENV} must be 'none' or 'framework_legal'.")

    escape_mode = os.environ.get(ESCAPE_CONTINUATIONS_ENV, "off")
    if escape_mode not in {"off", "on"}:
        raise ValueError(f"{ESCAPE_CONTINUATIONS_ENV} must be 'off' or 'on'.")

    if mode == "none" and escape_mode == "off":
        return DEFAULT_CONFIG

    return DQNConfig(
        action_masking=mode == "framework_legal",
        escape_continuation_features=escape_mode == "on",
    )


def _validate_evaluation_treatment(config: DQNConfig) -> None:
    """Reject a run-plan selector that disagrees with persisted artifact mode."""
    action_mode = os.environ.get(ACTION_MASKING_ENV)
    if action_mode is not None:
        expected = action_mode == "framework_legal"
        if action_mode not in {"none", "framework_legal"}:
            raise ValueError(
                f"{ACTION_MASKING_ENV} must be 'none' or 'framework_legal'."
            )
        if config.action_masking != expected:
            raise ValueError("Evaluation action-masking mode does not match checkpoint.")

    escape_mode = os.environ.get(ESCAPE_CONTINUATIONS_ENV)
    if escape_mode is not None:
        if escape_mode not in {"off", "on"}:
            raise ValueError(f"{ESCAPE_CONTINUATIONS_ENV} must be 'off' or 'on'.")
        expected = escape_mode == "on"
        if config.escape_continuation_features != expected:
            raise ValueError(
                "Evaluation escape-continuation mode does not match checkpoint."
            )
