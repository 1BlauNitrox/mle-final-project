"""Break pacing loops at play time without overriding the agent's own safety model.

Trained Bomb-omb policies can settle into stepping between two or three tiles
for the rest of a round once nothing nearby rewards moving on. On an
opponent-free board that forfeits most of the coins, and the same pacing appears
in the tournament once the opponents are dead.

The guard intervenes only when all of these hold:
  * stuck - the last WINDOW positions cover at most MAX_DISTINCT tiles;
  * calm - no bomb or active explosion within CALM_RADIUS tiles, so dodging a
    blast by stepping away and back is never second-guessed;
  * the policy's chosen move would step back onto a recently visited tile.

It then takes the move the policy itself values most among legal moves that the
agent's danger model says have a surviving continuation, preferring tiles not
visited recently. If no such move exists the policy's choice stands. It never
chooses WAIT or BOMB, and it is never active while training.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .config import ACTIONS
from .features.bombs_and_crates import build_danger_map, surviving_continuation_after_action
from .features.navigation import _blocked_positions

WINDOW = 8
MAX_DISTINCT = 3
CALM_RADIUS = 5
MOVES = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}

Position = tuple[int, int]


class LoopGuard:
    def __init__(self) -> None:
        self.positions: list[Position] = []
        self.overrides = 0

    def observe(self, game_state: dict) -> None:
        if game_state["step"] == 1:
            self.positions.clear()
        self.positions.append(tuple(int(v) for v in game_state["self"][3]))

    def stuck(self) -> bool:
        recent = self.positions[-WINDOW:]
        return len(recent) == WINDOW and len(set(recent)) <= MAX_DISTINCT

    def choose(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
    ) -> str:
        if chosen not in MOVES or not self.stuck() or not calm(game_state):
            return chosen
        here = self.positions[-1]
        recent = set(self.positions[-WINDOW:])
        if step(here, chosen) not in recent:
            return chosen

        field = np.asarray(game_state["field"])
        bombs = [(tuple(p), int(t)) for p, t in game_state.get("bombs", [])]
        danger = build_danger_map(field, bombs, np.asarray(game_state["explosion_map"]))
        blocked = _blocked_positions(game_state)
        previous = self.positions[-2] if len(self.positions) > 1 else None

        values = None
        unvisited, non_reversing = [], []
        for index, action in enumerate(ACTIONS):
            if action not in MOVES or action == chosen or not legal_mask[index]:
                continue
            target = step(here, action)
            if not surviving_continuation_after_action(field, danger, blocked, bombs, here, MOVES[action]):
                continue
            if values is None:
                values = q_values()
            entry = (float(values[index]), action)
            if target not in recent:
                unvisited.append(entry)
            elif target != previous:
                non_reversing.append(entry)

        pool = unvisited or non_reversing
        if not pool:
            return chosen
        self.overrides += 1
        return max(pool)[1]


def step(position: Position, action: str) -> Position:
    dx, dy = MOVES[action]
    return position[0] + dx, position[1] + dy


def calm(game_state: dict) -> bool:
    x, y = game_state["self"][3]
    for (bx, by), _timer in game_state.get("bombs", []):
        if abs(bx - x) + abs(by - y) <= CALM_RADIUS:
            return False
    explosions = np.argwhere(np.asarray(game_state["explosion_map"]) > 0)
    return not any(abs(int(ex) - x) + abs(int(ey) - y) <= CALM_RADIUS for ex, ey in explosions)
