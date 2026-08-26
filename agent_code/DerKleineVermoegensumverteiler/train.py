"""Minimal training callbacks for the runnable team-agent template."""

from __future__ import annotations

from statistics import fmean

from .config import ACTIONS, EPSILON_DECAY, MINIMUM_EPSILON
from .features import state_to_features
from .persistence import MODEL_PATH, save_model
from .rewards import reward_from_events


def setup_training(self) -> None:
    """Initialize training-only state."""

    self.episode_reward = 0.0
    self.absolute_td_errors: list[float] = []

    self.logger.info("Training initialzied with epsilon=%.4f", self.epsilon)


def game_events_occurred(
    self,
    old_game_state: dict | None,
    self_action: str | None,
    new_game_state: dict,
    events: list[str],
) -> None:
    """Update the selected action's incremental mean reward."""
    
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
    
    reward = reward_from_events(events)

    td_error = self.q_table.update(
        state=old_state,
        action=self_action,
        reward=reward,
        next_state=new_state,
        terminal=False
    )

    self.episode_reward += reward
    self.absolute_td_errors.append(abs(td_error))


def end_of_round(
    self,
    last_game_state: dict | None,
    last_action: str | None,
    events: list[str],
) -> dict[str, float]:
    """Apply the terminal update and return episode diagnostics."""
    if (
        last_game_state is not None
        and last_action in ACTIONS
    ):
        last_state = state_to_features(last_game_state)

        if last_state is not None:
            reward = reward_from_events(events)

            td_error = self.q_table.update(
                state=last_state,
                action=last_action,
                reward=reward,
                next_state=None,
                terminal=True
            )

            self.episode_reward += reward
            self.absolute_td_errors.append(abs(td_error))

    completed_episode_epsilon = float(self.epsilon)

    mean_abs_td_error = (
        fmean(self.absolute_td_errors)
        if self.absolute_td_errors
        else 0.0
    )

    metrics = {
        "shaped_reward": float(self.episode_reward),
        "epsilon": completed_episode_epsilon,
        "q_table_size": len(self.q_table),
        "mean_abs_td_error": float(mean_abs_td_error)
    }

    self.epsilon = max(MINIMUM_EPSILON, self.epsilon * EPSILON_DECAY)
    self.completed_episodes += 1

    save_model(
        self.q_table, 
        epsilon=self.epsilon, 
        completed_episodes=self.completed_episodes, 
        path=MODEL_PATH
        )

    self.episode_reward = 0.0
    self.absolute_td_errors = []

    return metrics



