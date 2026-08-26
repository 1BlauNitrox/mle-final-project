"""events -> reward mapping for DerKleineVermögensumverteiler."""

from __future__ import annotations

from collections.abc import Iterable

from .config import REWARDS


def reward_from_events(events: Iterable[str]) -> float:
    """Calculate the reward from the events that occurred during a step."""

    return float(sum(REWARDS.get(event, 0.0) for event in events))
