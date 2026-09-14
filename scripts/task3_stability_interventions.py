"""Training-only interventions; evaluation remains the archived self-contained agent."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from scripts.frozen_opponent_inputs import FIRST, PREFIX, FrozenOpponentInputs


class RegularizedOpponentInputs(FrozenOpponentInputs):
    """Gradient of L_DQN + lambda/2 * sum(W_opponent**2), before clipping."""

    def __init__(self, learner, anchor, strength=0.0):
        if strength not in (0.0, 0.01):
            raise ValueError("Unregistered L2 coefficient")
        self.strength = strength
        super().__init__(learner, anchor)

    def mask_gradient(self, gradient):
        result = FrozenOpponentInputs.mask_gradient(gradient)
        if self.strength:
            result[:, PREFIX:] += self.strength * self.parameters[FIRST].detach()[:, PREFIX:]
        return result


@contextmanager
def update_frequency(train, every):
    """Count post-warmup transitions per episode; skipped transitions still enter replay."""
    if every not in (1, 8):
        raise ValueError("Unregistered update frequency")
    if every == 1:
        yield
        return

    def record(self, *, state, action_index, reward, next_state, next_action_mask=None, terminal):
        self.replay_buffer.add(
            state=state,
            action_index=action_index,
            reward=reward,
            next_state=next_state,
            next_action_mask=next_action_mask,
            terminal=terminal,
        )
        self.episode_reward += reward
        if len(self.replay_buffer) < self.config.replay_warmup:
            return
        self.stability_eligible = getattr(self, "stability_eligible", 0) + 1
        if self.stability_eligible % every:
            return
        result = self.learner.train_batch(self.replay_buffer.sample(self.config.batch_size))
        self.losses.append(result.loss)
        self.absolute_td_errors.append(result.mean_abs_td_error)
        if result.target_synchronized:
            self.episode_target_synchronizations += 1

    with patch.object(train, "_record_transition", record):
        yield
