"""events -> reward mapping for DerKleineVermögensumverteiler."""

from __future__ import annotations

from collections.abc import Iterable

from .config import REWARDS


def reward_from_events(
    events: Iterable[str],
    *,
    useful_bomb_reward: float = 0.0,
) -> float:
    """Calculate the reward from the events that occurred during a step."""

    return float(
        sum(
            useful_bomb_reward if event == "USEFUL_BOMB_PLACED" else REWARDS.get(event, 0.0)
            for event in events
        )
    )
