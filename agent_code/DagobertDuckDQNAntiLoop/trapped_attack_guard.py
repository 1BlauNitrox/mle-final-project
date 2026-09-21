"""Conservative endgame bomb override for a geometrically trapped opponent."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .config import ACTIONS
from .features.bombs_and_crates import blast_footprint, build_danger_map, safe_escape_exists
from .features.navigation import DIRECTIONS, _blocked_positions

BOMB_INDEX = ACTIONS.index("BOMB")
NEW_BOMB_TIMER = 3
DETONATION_STEPS = NEW_BOMB_TIMER + 1

Position = tuple[int, int]


def opponent_can_escape(field: np.ndarray, bomb: Position, opponent: Position) -> bool:
    """Give the opponent every unobstructed route out before detonation."""
    threatened = set(blast_footprint(bomb, field))
    if opponent not in threatened:
        return True

    queue = deque([(opponent, 0)])
    visited = {(opponent, 0)}
    while queue:
        position, elapsed = queue.popleft()
        if elapsed >= DETONATION_STEPS:
            continue
        for dx, dy in DIRECTIONS:
            target = position[0] + dx, position[1] + dy
            next_time = elapsed + 1
            if target == bomb or field[target] != 0:
                continue
            if target not in threatened:
                return True
            state = target, next_time
            if state not in visited:
                visited.add(state)
                queue.append(state)
    return False


@dataclass
class TrappedAttackGuard:
    """Place at most one safe bomb against a sole opponent with no escape route."""

    round_id: int | None = None
    used: bool = False
    eligible: int = 0
    overrides: int = 0
    rejected_opponent_escape: int = 0
    rejected_self_safety: int = 0

    def choose(
        self,
        game_state: dict,
        chosen: str,
        _q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
    ) -> str:
        round_id = int(game_state["round"])
        if round_id != self.round_id:
            self.round_id = round_id
            self.used = False
        if self.used or chosen == "BOMB" or not legal_mask[BOMB_INDEX]:
            return chosen
        if game_state.get("bombs") or np.any(np.asarray(game_state["explosion_map"])):
            return chosen
        if len(game_state.get("others", ())) != 1:
            return chosen

        field = np.asarray(game_state["field"])
        position = tuple(int(value) for value in game_state["self"][3])
        opponent = tuple(int(value) for value in game_state["others"][0][3])
        if opponent not in blast_footprint(position, field):
            return chosen
        if opponent_can_escape(field, position, opponent):
            self.rejected_opponent_escape += 1
            return chosen

        proposed_bombs = [(position, NEW_BOMB_TIMER)]
        danger = build_danger_map(field, proposed_bombs, np.asarray(game_state["explosion_map"]))
        if not safe_escape_exists(
            field,
            danger,
            _blocked_positions(game_state),
            proposed_bombs,
            position,
        ):
            self.rejected_self_safety += 1
            return chosen

        self.eligible += 1
        self.used = True
        self.overrides += 1
        return "BOMB"

    def snapshot(self) -> dict[str, int]:
        return {
            "eligible": self.eligible,
            "overrides": self.overrides,
            "rejected_opponent_escape": self.rejected_opponent_escape,
            "rejected_self_safety": self.rejected_self_safety,
        }
