"""Framework training callbacks for DagobertDuckDQN."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

import numpy as np

from .config import ACTION_TO_INDEX
from .features import normalize_features, state_to_features
from .persistence import CHECKPOINT_PATH, save_checkpoint
from .rewards import reward_from_events


@dataclass(frozen=True)
class PendingTransition:
    """A surviving transition whose terminal status is not yet known."""

    identity: tuple[Any, Any] | None
    state: np.ndarray
    action: str
    action_index: int
    reward: float
    next_state: np.ndarray


def setup_training(self) -> None:
    """Initialize per-episode training diagnostics."""
    self.episode_reward = 0.0
    self.absolute_td_errors: list[float] = []
    self.pending_transition: PendingTransition | None = None

    self.logger.info(
        "DQN training initialized with epsilon %.4f",
        self.epsilon,
    )


def game_events_occurred(
    self,
    old_game_state: dict | None,
    self_action: str | None,
    new_game_state: dict | None,
    events: list[str],
) -> None:
    """Finalize the previous transition and retain the current one."""
    _finalize_pending_transition(self)

    if old_game_state is None or new_game_state is None or self_action not in ACTION_TO_INDEX:
        return

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    if old_features is None or new_features is None:
        return

    self.pending_transition = PendingTransition(
        identity=_transition_identity(old_game_state),
        state=normalize_features(old_features),
        action=self_action,
        action_index=ACTION_TO_INDEX[self_action],
        reward=reward_from_events(events),
        next_state=normalize_features(new_features),
    )


def end_of_round(
    self,
    last_game_state: dict | None,
    last_action: str | None,
    events: list[str],
) -> dict[str, float | None]:
    """Finalize the episode exactly once and save resumable state."""
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

        _record_transition(
            self,
            state=pending.state,
            action_index=pending.action_index,
            reward=reward_from_events(events),
            next_state=None,
            terminal=True,
        )
        self.pending_transition = None
    else:
        _finalize_pending_transition(self)

        if last_game_state is not None and last_action in ACTION_TO_INDEX:
            last_features = state_to_features(last_game_state)

            if last_features is not None:
                _record_transition(
                    self,
                    state=normalize_features(last_features),
                    action_index=ACTION_TO_INDEX[last_action],
                    reward=reward_from_events(events),
                    next_state=None,
                    terminal=True,
                )

    completed_episode_epsilon = float(self.epsilon)
    mean_abs_td_error = float(fmean(self.absolute_td_errors)) if self.absolute_td_errors else None

    metrics: dict[str, float | None] = {
        "shaped_reward": float(self.episode_reward),
        "epsilon": completed_episode_epsilon,
        "mean_abs_td_error": mean_abs_td_error,
    }

    self.epsilon = max(
        self.config.minimum_epsilon,
        self.epsilon * self.config.epsilon_decay,
    )
    self.completed_episodes += 1

    save_checkpoint(
        learner=self.learner,
        replay_buffer=self.replay_buffer,
        action_rng=self.action_rng,
        epsilon=self.epsilon,
        completed_episodes=self.completed_episodes,
        agent_seed=self.agent_seed,
        path=CHECKPOINT_PATH,
    )

    self.episode_reward = 0.0
    self.absolute_td_errors = []
    self.pending_transition = None

    return metrics


def _finalize_pending_transition(self) -> None:
    """Add the pending transition as an ordinary transition."""
    pending = self.pending_transition

    if pending is None:
        return

    _record_transition(
        self,
        state=pending.state,
        action_index=pending.action_index,
        reward=pending.reward,
        next_state=pending.next_state,
        terminal=False,
    )
    self.pending_transition = None


def _record_transition(
    self,
    *,
    state: np.ndarray,
    action_index: int,
    reward: float,
    next_state: np.ndarray | None,
    terminal: bool,
) -> None:
    """Store one transition and perform at most one DQN update."""
    self.replay_buffer.add(
        state=state,
        action_index=action_index,
        reward=reward,
        next_state=next_state,
        terminal=terminal,
    )
    self.episode_reward += reward

    if len(self.replay_buffer) < self.config.replay_warmup:
        return

    batch = self.replay_buffer.sample(self.config.batch_size)
    result = self.learner.train_batch(batch)

    self.absolute_td_errors.append(result.mean_abs_td_error)


def _transition_identity(
    game_state: dict | None,
) -> tuple[Any, Any] | None:
    """Identify the pre-action state for duplicate prevention."""
    if game_state is None:
        return None

    round_number = game_state.get("round")
    step_number = game_state.get("step")

    if round_number is None or step_number is None:
        return None

    return round_number, step_number
