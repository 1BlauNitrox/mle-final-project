"""Minimal training callbacks for the runnable team-agent template."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .config import ACTIONS, EPSILON_DECAY, MINIMUM_EPSILON
from .features import StateFeatures, state_to_features
from .persistence import MODEL_PATH, save_model
from .rewards import reward_from_events


@dataclass(frozen=True)
class PendingTransition:
    """A surviving transition whose terminal status is not yet knwown."""

    identity: tuple[Any, Any] | None
    state: StateFeatures
    action: str
    next_state: StateFeatures
    reward: float

def setup_training(self) -> None:
    """Initialize training-only state."""

    self.episode_reward = 0.0
    self.absolute_td_errors: list[float] = []
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

    _finalize_pennding_transition(self)

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

    self.pending_transition = PendingTransition(
        identity=_transition_identity(old_game_state),
        state=old_state,
        action=self_action,
        next_state=new_state,
        reward=reward_from_events(events),
    )

def end_of_round(
    self,
    last_game_state: dict | None,
    last_action: str | None,
    events: list[str],
) -> dict[str, float]:
    """Apply the terminal update and return episode diagnostics."""

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
            terminal=True
        )
    else:
        _finalize_pennding_transition(self)

        if last_game_state is not None and last_action in ACTIONS:
            last_state = state_to_features(last_game_state)

            if last_state is not None:
               _apply_update(
                    self,
                    state=last_state,
                    action=last_action,
                    reward=reward_from_events(events),
                    next_state=None,
                    terminal=True
                )

    completed_episode_epsilon = float(self.epsilon)

    mean_abs_td_error = fmean(self.absolute_td_errors) if self.absolute_td_errors else 0.0

    metrics: dict[str, float | int] = {
        "shaped_reward": float(self.episode_reward),
        "epsilon": completed_episode_epsilon,
        "q_table_size": len(self.q_table),
        "mean_abs_td_error": float(mean_abs_td_error),
    }

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
    self.pending_transition = None

    return metrics

def _finalize_pennding_transition(self) -> None:
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
        terminal=False
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
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        terminal=terminal
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