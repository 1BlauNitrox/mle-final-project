"""Append descriptive geometry or causal visit history to the unchanged prefix."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from .features import normalize_features, state_to_features
from .features.bombs_and_crates import (
    BOMB_TIMER,
    blast_footprint,
    build_danger_map,
    surviving_continuation_after_action,
)
from .features.navigation import DIRECTIONS


def hunting_geometry(state):
    """Five route distances/reachability flags and two bounded escape counts.

    Attack positions are currently unoccupied walkable tiles whose static blast
    reaches any living opponent. Distances treat crates, bombs and opponents as
    obstacles. Escape counts describe the most constrained currently threatened
    opponent, before/after our hypothetical bomb; they are not kill predictions.
    """
    field = state["field"]
    position = tuple(state["self"][3])
    opponents = {tuple(other[3]) for other in state["others"]}
    if not opponents:
        return np.zeros(12, dtype=np.float32)
    bombs = state.get("bombs", [])
    blocked = opponents | {tuple(p) for p, _timer in bombs}
    free = {tuple(map(int, p)) for p in np.argwhere(field == 0)} - blocked
    targets = {p for p in free if opponents.intersection(blast_footprint(p, field))}
    distances = dict.fromkeys(targets, 0)
    queue = deque(sorted(targets))
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            neighbour = x + dx, y + dy
            if neighbour in free and neighbour not in distances:
                distances[neighbour] = distances[x, y] + 1
                queue.append(neighbour)
    sites = [position, *((position[0] + dx, position[1] + dy) for dx, dy in DIRECTIONS)]
    route = [distances.get(p, field.size) / field.size for p in sites]
    reachable = [float(p in distances) for p in sites]
    threatened = opponents.intersection(blast_footprint(position, field))
    escape_counts = []
    if state["self"][2] and threatened:
        hypothetical = [*bombs, (position, BOMB_TIMER - 1)]
        for opponent in sorted(threatened):
            counts = []
            for active_bombs in (bombs, hypothetical):
                danger = build_danger_map(field, active_bombs, state["explosion_map"])
                obstacles = (opponents - {opponent}) | {position}
                obstacles |= {tuple(p) for p, _timer in active_bombs}
                counts.append(
                    sum(
                        surviving_continuation_after_action(
                            field, danger, obstacles, active_bombs, opponent, direction
                        )
                        for direction in (*DIRECTIONS, (0, 0))
                    )
                    / 5.0
                )
            escape_counts.append(counts)
    before, after = min(escape_counts, key=lambda counts: (counts[1], counts[0]), default=(0, 0))
    return np.asarray([*route, *reachable, before, after], dtype=np.float32)


class ObservationHistory:
    """Cache immutable observations so act and replay see exactly the same history."""

    def __init__(self):
        self.round = None
        self.positions = deque(maxlen=16)
        self.cache = {}

    def encode(self, state, config):
        if state is None:
            return None
        round_id, step = state["round"], state["step"]
        if round_id != self.round:
            self.round = round_id
            self.positions.clear()
            self.cache.clear()
        if step in self.cache:
            return self.cache[step].copy()
        if self.cache and step < max(self.cache):
            raise ValueError("Observation history requested an uncached past state")
        position = tuple(state["self"][3])
        # Counts are previous decisions, excluding this arrival. Event callbacks
        # may observe t+1 before act; the cache prevents counting it twice.
        counts = Counter(self.positions)
        sites = [position, *((position[0] + dx, position[1] + dy) for dx, dy in DIRECTIONS)]
        prefix = normalize_features(
            state_to_features(
                state, include_continuation_features=config.escape_continuation_features
            )
        )
        geometry = (
            hunting_geometry(state) if config.observation_mode == "geometry" else np.zeros(12)
        )
        memory = [counts[p] / 16 for p in sites] if config.observation_mode == "memory" else [0] * 5
        value = np.asarray([*prefix, *geometry, *memory], dtype=np.float32)
        self.positions.append(position)
        self.cache[step] = value.copy()
        # Training only revisits the preceding state and terminal pending state.
        for previous in sorted(self.cache)[:-3]:
            del self.cache[previous]
        return value


def observe(agent, state):
    if not hasattr(agent, "observation_history"):
        agent.observation_history = ObservationHistory()
    return agent.observation_history.encode(state, agent.config)
