"""Minimal training callbacks for the runnable team-agent template."""

from collections.abc import Iterable

import events as e

from .callbacks import ACTIONS, MODEL_PATH

REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.INVALID_ACTION: -0.5,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.SURVIVED_ROUND: 1.0,
}


def setup_training(self) -> None:
    """Initialize training-only state."""
    self.logger.info("Training the template action-value baseline")


def game_events_occurred(
    self,
    old_game_state: dict | None,
    self_action: str,
    new_game_state: dict,
    events: list[str],
) -> None:
    """Update the selected action's incremental mean reward."""
    del old_game_state, new_game_state
    _update_action_value(self, self_action, events)


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list[str],
) -> None:
    """Apply the final reward and persist the learned action values."""
    del last_game_state
    _update_action_value(self, last_action, events)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np

    np.savez(
        MODEL_PATH,
        action_values=self.action_values,
        action_counts=self.action_counts,
    )


def reward_from_events(events: Iterable[str]) -> float:
    """Convert framework events into a scalar baseline reward."""
    return sum(REWARDS.get(event, 0.0) for event in events)


def _update_action_value(self, action: str, events: Iterable[str]) -> None:
    if action not in ACTIONS:
        return

    index = ACTIONS.index(action)
    reward = reward_from_events(events)
    self.action_counts[index] += 1
    count = self.action_counts[index]
    self.action_values[index] += (reward - self.action_values[index]) / count
