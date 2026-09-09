"""Behavior-preserving callbacks for the tabular Task 2 successor."""

from __future__ import annotations

import os

import numpy as np

from .config import DEFAULT_SEED, INITIAL_EPSILON
from .features import state_to_features
from .legality import framework_legal_action_mask
from .migration import load_parent_prior
from .model import QTable
from .persistence import MODEL_PATH, load_model

USEFUL_BOMB_REWARD_ENV = "BOMBERMAN_TABULAR_USEFUL_BOMB_REWARD"
VALID_USEFUL_BOMB_REWARDS = (0.0, 1.0)
ACTION_MASKING_ENV = "BOMBERMAN_TABULAR_ACTION_MASKING"
VALID_ACTION_MASKING = {"none", "framework_legal"}


def setup(self) -> None:
    """Initialize the Qtable, random generator and exploration state."""

    agent_seed = _read_agent_seed()
    configured_reward = _read_useful_bomb_reward()
    self.useful_bomb_reward = configured_reward

    configured_action_masking = _read_action_masking()
    self.action_masking = configured_action_masking

    self.rng = np.random.default_rng(agent_seed)

    if MODEL_PATH.is_file():
        loaded = load_model(MODEL_PATH)

        is_fresh_model = loaded.completed_episodes == 0 and len(loaded.q_table) == 0

        if loaded.action_masking != configured_action_masking and not is_fresh_model:
            raise ValueError(f"{ACTION_MASKING_ENV} does not match the stored model mode.")

        self.q_table = loaded.q_table
        self.completed_episodes = loaded.completed_episodes

        is_fresh_model = loaded.completed_episodes == 0 and len(loaded.q_table) == 0
        if loaded.useful_bomb_reward != configured_reward and not is_fresh_model:
            raise ValueError(f"{USEFUL_BOMB_REWARD_ENV} does not match the stored model treatment.")

        if self.train:
            self.epsilon = loaded.epsilon
        else:
            self.epsilon = 0.0

        self.logger.info(
            "Loaded model with %d states after %d episodes",
            len(self.q_table),
            self.completed_episodes,
        )
        return

    if not self.train:
        raise FileNotFoundError(f"Evaluation model does not exist: {MODEL_PATH}")

    parent_prior = load_parent_prior()
    self.q_table = QTable(parent_values=parent_prior.values)
    self.completed_episodes = 0
    self.epsilon = INITIAL_EPSILON

    self.logger.info("Initialized new model with seed %d", agent_seed)


def act(self, game_state: dict) -> str:
    """Choose an action using epsilon-greedy exploration during training."""

    state = state_to_features(game_state)

    if state is None:
        return "WAIT"

    epsilon = self.epsilon if self.train else 0.0

    action_mask = (
        framework_legal_action_mask(game_state)
        if self.action_masking == "framework_legal"
        else None
    )

    return self.q_table.select_action(
        state,
        epsilon=epsilon,
        rng=self.rng,
        action_mask=action_mask,
    )


def _read_agent_seed() -> int:
    """Read the agent seed from the environment."""

    raw_seed = os.environ.get("BOMBERMAN_AGENT_SEED")

    if raw_seed is None:
        return DEFAULT_SEED

    try:
        return int(raw_seed)
    except ValueError as error:
        raise ValueError("BOMBERMAN_AGENT_SEED must be an integer") from error


def _read_useful_bomb_reward() -> float:
    """Read the prospectively registered reward treatment."""

    raw_reward = os.environ.get(USEFUL_BOMB_REWARD_ENV, "0.0")
    try:
        reward = float(raw_reward)
    except ValueError as error:
        raise ValueError(f"{USEFUL_BOMB_REWARD_ENV} must be numeric.") from error

    if reward not in VALID_USEFUL_BOMB_REWARDS:
        raise ValueError(
            f"{USEFUL_BOMB_REWARD_ENV} must be one of {list(VALID_USEFUL_BOMB_REWARDS)}."
        )
    return reward

def _read_action_masking() -> str:
    """Read and validate the action-masking treatment."""

    mode = os.environ.get(ACTION_MASKING_ENV, "none")

    if mode not in VALID_ACTION_MASKING:
        raise ValueError(f"{ACTION_MASKING_ENV} must be 'none' or 'framework_legal'.")

    return mode
