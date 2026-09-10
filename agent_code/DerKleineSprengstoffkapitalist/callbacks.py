"""Behavior-preserving callbacks for the tabular Task 2 successor."""

from __future__ import annotations

import os

import numpy as np

from .config import DEFAULT_SEED, INITIAL_EPSILON
from .features import (
    BASELINE_STATE_REPRESENTATION,
    VALID_STATE_REPRESENTATIONS,
    encode_state,
    get_state_representation,
)
from .legality import framework_legal_action_mask
from .migration import load_parent_prior
from .model import (
    PARENT_PRIOR_INITIALIZATION,
    VALID_INITIALIZATIONS,
    QTable,
)
from .persistence import MODEL_PATH, load_model
from .potential_shaping import NO_POTENTIAL_SHAPING

USEFUL_BOMB_REWARD_ENV = "BOMBERMAN_TABULAR_USEFUL_BOMB_REWARD"
VALID_USEFUL_BOMB_REWARDS = (0.0, 1.0)
ACTION_MASKING_ENV = "BOMBERMAN_TABULAR_ACTION_MASKING"
VALID_ACTION_MASKING = {"none", "framework_legal"}
STATE_REPRESENTATION_ENV = "BOMBERMAN_TABULAR_STATE_REPRESENTATION"
INITIALIZATION_ENV = "BOMBERMAN_TABULAR_INITIALIZATION"


def setup(self) -> None:
    """Initialize the Q-table, random generator and exploration state."""

    agent_seed = _read_agent_seed()

    self.useful_bomb_reward = _read_useful_bomb_reward()
    self.action_masking = _read_action_masking()
    self.state_representation = _read_state_representation()
    self.initialization = _read_initialization()
    self.potential_shaping = NO_POTENTIAL_SHAPING

    representation = get_state_representation(
        self.state_representation
    )

    self.rng = np.random.default_rng(agent_seed)

    self.evaluation_decisions = 0
    self.evaluation_unseen_decisions = 0

    if MODEL_PATH.is_file():
        loaded = load_model(MODEL_PATH)
        is_fresh_model = (
            loaded.completed_episodes == 0
            and len(loaded.q_table) == 0
        )

        mismatches = []

        if loaded.action_masking != self.action_masking:
            mismatches.append(ACTION_MASKING_ENV)

        if loaded.useful_bomb_reward != self.useful_bomb_reward:
            mismatches.append(USEFUL_BOMB_REWARD_ENV)

        if loaded.state_representation != self.state_representation:
            mismatches.append(STATE_REPRESENTATION_ENV)

        if loaded.initialization != self.initialization:
            mismatches.append(INITIALIZATION_ENV)

        if mismatches and (not is_fresh_model or not self.train):
            raise ValueError(
                "Configured treatment does not match the stored model: "
                + ", ".join(mismatches)
            )

        if not mismatches:
            self.q_table = loaded.q_table
            self.completed_episodes = loaded.completed_episodes
            self.epsilon = loaded.epsilon if self.train else 0.0

            self.logger.info(
                "Loaded model with %d states after %d episodes",
                len(self.q_table),
                self.completed_episodes,
            )
            return

    if not self.train:
        raise FileNotFoundError(
            f"Evaluation model does not exist: {MODEL_PATH}"
        )

    parent_prior = load_parent_prior()

    self.q_table = QTable(
        parent_values=(
            parent_prior.values
            if self.initialization == PARENT_PRIOR_INITIALIZATION
            else None
        ),
        feature_count=representation.feature_count,
        initialization=self.initialization,
    )
    self.completed_episodes = 0
    self.epsilon = INITIAL_EPSILON

    self.logger.info(
        "Initialized %s model with %s initialization and seed %d",
        self.state_representation,
        self.initialization,
        agent_seed,
    )


def act(self, game_state: dict) -> str:
    """Choose an action using epsilon-greedy exploration during training."""

    state = encode_state(
        game_state,
        self.state_representation,
    )

    if state is None:
        return "WAIT"

    if not self.train:
        self.evaluation_decisions += 1

        if not self.q_table.contains_state(state):
            self.evaluation_unseen_decisions += 1

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


def end_of_round(
    self,
    last_game_state: dict | None,
    last_action: str | None,
    events: list[str],
) -> dict[str, float | int | None]:
    """Return state-coverage diagnostics for one evaluation episode."""

    del last_game_state, last_action, events

    decisions = self.evaluation_decisions
    unseen_decisions = self.evaluation_unseen_decisions

    metrics: dict[str, float | int | None] = {
        "evaluation_decisions": decisions,
        "evaluation_unseen_decisions": unseen_decisions,
        "evaluation_unseen_state_rate": (
            unseen_decisions / decisions
            if decisions > 0
            else None
        ),
    }

    self.evaluation_decisions = 0
    self.evaluation_unseen_decisions = 0

    return metrics


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


def _read_state_representation() -> str:
    """Read and validate the selected state representation."""

    representation = os.environ.get(
        STATE_REPRESENTATION_ENV,
        BASELINE_STATE_REPRESENTATION,
    )

    if representation not in VALID_STATE_REPRESENTATIONS:
        raise ValueError(
            f"{STATE_REPRESENTATION_ENV} must be one of "
            f"{list(VALID_STATE_REPRESENTATIONS)}."
        )

    return representation


def _read_initialization() -> str:
    """Read and validate the Q-table initialization mode."""

    initialization = os.environ.get(
        INITIALIZATION_ENV,
        PARENT_PRIOR_INITIALIZATION,
    )

    if initialization not in VALID_INITIALIZATIONS:
        raise ValueError(
            f"{INITIALIZATION_ENV} must be one of "
            f"{list(VALID_INITIALIZATIONS)}."
        )

    return initialization
