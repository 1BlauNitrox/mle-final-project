"""Training-only fixed-horizon return construction for Issue #207."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

REGISTERED_HORIZONS = (1, 5)


@dataclass(frozen=True)
class RawTransition:
    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray | None
    next_action_mask: np.ndarray | None
    terminal: bool


@dataclass(frozen=True)
class AggregatedTransition:
    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray | None
    next_action_mask: np.ndarray | None
    terminal: bool
    bootstrap_discount: float
    horizon: int


class NStepAccumulator:
    """Emit one replay item per raw item without crossing episode boundaries."""

    def __init__(self, horizon: int, discount_factor: float):
        if horizon not in REGISTERED_HORIZONS:
            raise ValueError("Unregistered n-step horizon")
        if not 0.0 < discount_factor <= 1.0:
            raise ValueError("discount_factor must be in (0, 1]")
        self.horizon = horizon
        self.discount_factor = discount_factor
        self.pending: deque[RawTransition] = deque()

    def push(self, transition: RawTransition) -> list[AggregatedTransition]:
        if self.pending and self.pending[-1].terminal:
            raise ValueError("An episode boundary was not flushed")
        self.pending.append(transition)
        emitted = []
        if transition.terminal:
            while self.pending:
                emitted.append(self._emit_one())
        elif len(self.pending) >= self.horizon:
            emitted.append(self._emit_one())
        return emitted

    def _emit_one(self) -> AggregatedTransition:
        count = min(self.horizon, len(self.pending))
        window = list(self.pending)[:count]
        terminal_indices = [index for index, item in enumerate(window) if item.terminal]
        if terminal_indices:
            window = window[: terminal_indices[0] + 1]
        last = window[-1]
        reward = sum(self.discount_factor**index * item.reward for index, item in enumerate(window))
        first = self.pending.popleft()
        return AggregatedTransition(
            state=first.state,
            action_index=first.action_index,
            reward=float(reward),
            next_state=None if last.terminal else last.next_state,
            next_action_mask=None if last.terminal else last.next_action_mask,
            terminal=last.terminal,
            bootstrap_discount=(0.0 if last.terminal else self.discount_factor ** len(window)),
            horizon=len(window),
        )


@contextmanager
def n_step_training(train_module, *, horizon: int, discount_factor: float):
    """Replace only transition construction while retaining the update budget."""
    if horizon not in REGISTERED_HORIZONS:
        raise ValueError("Unregistered n-step horizon")
    original = train_module._record_transition

    def record(
        self,
        *,
        state,
        action_index,
        reward,
        next_state,
        next_action_mask=None,
        terminal,
    ):
        accumulator = getattr(self, "issue207_n_step", None)
        if accumulator is None:
            accumulator = NStepAccumulator(horizon, discount_factor)
            self.issue207_n_step = accumulator
        self.episode_reward += reward
        emitted = accumulator.push(
            RawTransition(
                state=state,
                action_index=action_index,
                reward=float(reward),
                next_state=next_state,
                next_action_mask=next_action_mask,
                terminal=terminal,
            )
        )
        for item in emitted:
            self.replay_buffer.add(
                state=item.state,
                action_index=item.action_index,
                reward=item.reward,
                next_state=item.next_state,
                terminal=item.terminal,
                next_action_mask=item.next_action_mask,
                bootstrap_discount=item.bootstrap_discount,
            )
            if len(self.replay_buffer) < self.config.replay_warmup:
                continue
            result = self.learner.train_batch(self.replay_buffer.sample(self.config.batch_size))
            self.losses.append(result.loss)
            self.absolute_td_errors.append(result.mean_abs_td_error)
            if result.target_synchronized:
                self.episode_target_synchronizations += 1
        if terminal and accumulator.pending:
            raise RuntimeError("Terminal n-step tail did not flush")

    train_module._record_transition = record
    try:
        yield
    finally:
        train_module._record_transition = original
