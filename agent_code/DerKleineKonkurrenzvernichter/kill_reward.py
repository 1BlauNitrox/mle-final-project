"""Opponent-kill reward timing treatments for Task 3 training."""

from __future__ import annotations

from collections.abc import Iterable

NATIVE_KILL_REWARD = "native"
CAUSAL_BOMB_KILL_REWARD = "causal_bomb"
VALID_KILL_REWARD_MODES = (
    NATIVE_KILL_REWARD,
    CAUSAL_BOMB_KILL_REWARD,
)


def without_native_kill_reward(events: Iterable[str]) -> list[str]:
    """Return events without opponent kills while preserving diagnostics elsewhere."""

    return [event for event in events if event != "KILLED_OPPONENT"]


def redistributed_kill_reward(
    *,
    kill_count: int,
    delay: int,
    discount_factor: float,
    native_reward: float,
) -> float:
    """Return discounted reward assigned to a causal bomb transition."""

    if type(kill_count) is not int or kill_count < 0:
        raise ValueError("kill_count must be a non-negative integer")
    if type(delay) is not int or delay < 0:
        raise ValueError("delay must be a non-negative integer")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1]")

    return float(kill_count * native_reward * discount_factor**delay)
