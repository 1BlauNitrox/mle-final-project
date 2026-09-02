"""Reward calculation for the DagobertDuckDQN agent"""

from __future__ import annotations

from collections.abc import Iterable

from .config import REWARDS


def reward_from_events(events: Iterable[str]) -> float:
    """Return the sum of all configured event rewards"""
    return float(sum(REWARDS.get(event, 0.0) for event in events))