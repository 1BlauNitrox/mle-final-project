"""Minimal training callbacks for the runnable team-agent template."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .config import ACTIONS, EPSILON_DECAY, MINIMUM_EPSILON
from .features import StateFeatures, state_to_features
from .features.bombs_and_crates import crates_destroyed_by_bomb_at
from .persistence import MODEL_PATH, save_model
from .rewards import reward_from_events

DIAGNOSTIC_EVENT_METRICS = {
    "COIN_COLLECTED": "coins_collected",
    "COIN_FOUND": "coins_found",
    "CRATE_DESTROYED": "crates_destroyed",
    "BOMB_DROPPED": "bombs_dropped",
    "USEFUL_BOMB_PLACED": "useful_bombs",
    "KILLED_SELF": "self_kills",
    "SURVIVED_ROUND": "survived_round",
    "INVALID_ACTION": "invalid_actions",
}


@dataclass(frozen=True)
class PendingTransition:
    """A surviving transition whose terminal status is not yet knwown."""

    identity: tuple[Any, Any] | None
    state: StateFeatures
    action: str
    next_state: StateFeatures
    reward: float


def setup_training(self) -> None:
    """Initialize the Task 2 training state."""
    _initialize_training_state(self)


def _initialize_training_state(self) -> None:
    """Initialize state used by unit tests of the frozen learning logic."""

    self.episode_reward = 0.0
    self.absolute_td_errors: list[float] = []
    self.episode_event_counts: Counter[str] = Counter()
    self.pending_transition: PendingTransition | None = None

    self.logger.info("Training initialzied with epsilon=%.4f", self.epsilon)


def game_events_occurred(
    self,
    old_game_state: dict | None,
    self_action: str | None,
    new_game_state: dict,
    events: list[str],
) -> None:
    """store current transition and finalice previous."""

    _finalize_pending_transition(self)

    _count_diagnostic_events(self, events)

    if _is_useful_bomb(
        old_game_state,
        self_action,
        events,
    ):
        self.episode_event_counts["USEFUL_BOMB_PLACED"] += 1

    if old_game_state is None or self_action is None:
        return

    if self_action not in ACTIONS:
        self.logger.warning(
            "Selected action %r is not in the action space %r",
            self_action,
            ACTIONS,
        )
        return

    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    if old_state is None or new_state is None:
        return

    training_events = list(events)

    movement_event = _coin_movement_event(
        old_game_state,
        new_game_state,
        self_action,
    )

    if movement_event is not None:
        training_events.append(movement_event)

    self.pending_transition = PendingTransition(
        identity=_transition_identity(old_game_state),
        state=old_state,
        action=self_action,
        next_state=new_state,
        reward=reward_from_events(training_events),
    )


def end_of_round(
    self,
    last_game_state: dict | None,
    last_action: str | None,
    events: list[str],
) -> dict[str, float]:
    """Apply the terminal update and return episode diagnostics."""

    _count_diagnostic_events(self, events)

    callback_identity = _transition_identity(last_game_state)
    pending = self.pending_transition

    callback_matches_pending = (
        pending is not None
        and callback_identity is not None
        and callback_identity == pending.identity
        and last_action == pending.action
    )

    if callback_matches_pending:
        assert pending is not None

        _apply_update(
            self,
            state=pending.state,
            action=pending.action,
            reward=reward_from_events(events),
            next_state=None,
            terminal=True,
        )
    else:
        _finalize_pending_transition(self)

        if last_game_state is not None and last_action in ACTIONS:
            last_state = state_to_features(last_game_state)

            if last_state is not None:
                _apply_update(
                    self,
                    state=last_state,
                    action=last_action,
                    reward=reward_from_events(events),
                    next_state=None,
                    terminal=True,
                )

    completed_episode_epsilon = float(self.epsilon)

    mean_abs_td_error = fmean(self.absolute_td_errors) if self.absolute_td_errors else 0.0

    metrics: dict[str, float | int] = {
        "shaped_reward": float(self.episode_reward),
        "epsilon": completed_episode_epsilon,
        "q_table_size": len(self.q_table),
        "mean_abs_td_error": float(mean_abs_td_error),
    }

    for event_name, metric_name in DIAGNOSTIC_EVENT_METRICS.items():
        metrics[metric_name] = float(self.episode_event_counts.get(event_name, 0))

    self.epsilon = max(MINIMUM_EPSILON, self.epsilon * EPSILON_DECAY)
    self.completed_episodes += 1

    save_model(
        self.q_table,
        epsilon=self.epsilon,
        completed_episodes=self.completed_episodes,
        path=MODEL_PATH,
    )

    self.episode_reward = 0.0
    self.absolute_td_errors = []
    self.episode_event_counts = Counter()
    self.pending_transition = None

    return metrics


def _finalize_pending_transition(self) -> None:
    """Apply the pending transition if it exists."""

    pending = self.pending_transition

    if pending is None:
        return

    _apply_update(
        self,
        state=pending.state,
        action=pending.action,
        reward=pending.reward,
        next_state=pending.next_state,
        terminal=False,
    )

    self.pending_transition = None


def _apply_update(
    self,
    *,
    state: StateFeatures,
    action: str,
    reward: float,
    next_state: StateFeatures | None,
    terminal: bool,
) -> None:
    """Update the Q-table and record diagnostics."""

    td_error = self.q_table.update(
        state=state, action=action, reward=reward, next_state=next_state, terminal=terminal
    )

    self.episode_reward += reward
    self.absolute_td_errors.append(abs(td_error))


def _transition_identity(game_state: dict | None) -> tuple[Any, Any] | None:
    """Return a unique identity for the transition based on the game state."""

    if game_state is None:
        return None

    round_number = game_state.get("round")
    step_number = game_state.get("step")

    if round_number is None or step_number is None:
        return None

    return (round_number, step_number)


def _coin_movement_event(
    old_game_state: dict,
    new_game_state: dict,
    action: str,
) -> str | None:
    """Return a reward-shaping event based on distance to the nearest old coin."""

    if action not in {"UP", "RIGHT", "DOWN", "LEFT"}:
        return None

    coins = old_game_state.get("coins", [])

    if not coins:
        return None

    old_position = old_game_state["self"][3]
    new_position = new_game_state["self"][3]

    old_distance = min(_manhattan_distance(old_position, coin) for coin in coins)
    new_distance = min(_manhattan_distance(new_position, coin) for coin in coins)

    if new_distance < old_distance:
        return "MOVED_TOWARDS_COIN"

    if new_distance > old_distance:
        return "MOVED_AWAY_FROM_COIN"

    return None


def _manhattan_distance(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _count_diagnostic_events(
    self,
    events: list[str],
) -> None:
    """Count relevant framework events for the current episode."""

    self.episode_event_counts.update(event for event in events if event in DIAGNOSTIC_EVENT_METRICS)


def _is_useful_bomb(
    old_game_state: dict | None,
    action: str | None,
    events: list[str],
) -> bool:
    """Return whether a successfully placed bomb can destroy a crate."""

    if old_game_state is None:
        return False

    if action != "BOMB":
        return False

    if "BOMB_DROPPED" not in events:
        return False

    position = old_game_state["self"][3]
    field = old_game_state["field"]

    return crates_destroyed_by_bomb_at(position, field) > 0
