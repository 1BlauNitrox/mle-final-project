"""Causal, training-only shaping for persistent movement cycles."""

from collections import deque

import numpy as np


class LoopReward:
    """Require 24 calm unchanged observations on at most three distinct tiles."""

    def __init__(self):
        self.round = None
        self.window = deque(maxlen=24)
        self.last_step = None

    def observe(self, state):
        if state["round"] != self.round:
            self.round = state["round"]
            self.window.clear()
            self.last_step = None
        if state["step"] == self.last_step:
            return
        if self.last_step is not None and state["step"] != self.last_step + 1:
            self.window.clear()
        self.last_step = state["step"]
        self.window.append(
            (
                tuple(state["self"][3]),
                state["field"].tobytes(),
                tuple(sorted(state["coins"])),
                state["self"][1],
                len(state["others"]),
                bool(state["bombs"] or np.any(state["explosion_map"])),
            )
        )

    def penalty(self, old, new, action, events):
        self.observe(old)
        history = list(self.window)
        self.observe(new)
        if action not in {"UP", "RIGHT", "DOWN", "LEFT"}:
            return 0.0
        if tuple(old["self"][3]) == tuple(new["self"][3]):
            return 0.0
        if {"COIN_COLLECTED", "CRATE_DESTROYED", "KILLED_OPPONENT", "INVALID_ACTION"} & set(events):
            return 0.0
        if len(history) < 24 or len({row[0] for row in history}) > 3:
            return 0.0
        combined = [*history, self.window[-1]]
        if any(row[-1] for row in combined) or len({row[1:5] for row in combined}) != 1:
            return 0.0
        return -0.1 if tuple(new["self"][3]) in {row[0] for row in history} else 0.0


def loop_penalty(agent, old, new, action, events):
    if agent.config.observation_mode != "memory":
        return 0.0
    if not hasattr(agent, "loop_reward"):
        agent.loop_reward = LoopReward()
    value = agent.loop_reward.penalty(old, new, action, events)
    agent.loop_penalty_count = getattr(agent, "loop_penalty_count", 0) + int(value < 0)
    return value
