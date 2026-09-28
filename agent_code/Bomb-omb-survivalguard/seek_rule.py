"""Go hunting when the board has nothing left to offer.

Measured on twenty games at episode 8,000: while crates are standing the agent
paces on 16.9% of its steps, and once they are gone it paces on 76.3% of them.
828 of the 1,272 stalled steps had the same board in front of them - no crates,
no coins visible, and an opponent still alive. That is about a fifth of the
agent's life spent stepping between two tiles while someone is still out there.

The loop guard tried to fix this by breaking the cycle, and lost its A/B: on a
cleared board, stepping somewhere else is worth exactly as little as stepping
back. The missing behaviour is not "stop looping", it is "go and do something".

So when the board is cleared, no coin is visible and an opponent is alive, this
walks toward the nearest tile from which a bomb would reach one - which is
precisely where the attack rule takes over. Each step is checked against the
agent's own danger model, so seeking never walks into a blast. It never
overrides a bomb, and never fires while training.

It is off unless BOMBERMAN_SEEK_RULE=on.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

from .features.bombs_and_crates import blast_footprint, build_danger_map, surviving_continuation_after_action
from .features.navigation import DIRECTIONS, _blocked_positions

SEEK_ENV = "BOMBERMAN_SEEK_RULE"
MOVES = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
MAX_SEEK_DISTANCE = 30


def seek_mode(environ: dict[str, str] | None = None) -> bool:
    """Read the switch, rejecting anything but "on" and "off"."""
    environ = os.environ if environ is None else environ
    value = environ.get(SEEK_ENV, "off")
    if value not in {"on", "off"}:
        raise ValueError(f"{SEEK_ENV} must be 'on' or 'off', not {value!r}")
    return value == "on"


class SeekRule:
    """Walk toward a firing position once the board is empty."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.steps_taken = 0
        self.no_route = 0

    @property
    def active(self) -> bool:
        return self.enabled

    def board_is_empty(self, game_state: dict) -> bool:
        field = np.asarray(game_state["field"])
        if int((field == 1).sum()):
            return False
        if game_state.get("coins"):
            return False
        return bool(game_state.get("others"))

    def first_step_toward_a_firing_position(self, game_state: dict) -> str | None:
        """Breadth-first over free tiles to the nearest tile that could bomb someone."""
        field = np.asarray(game_state["field"])
        blocked = _blocked_positions(game_state)
        start = tuple(int(v) for v in game_state["self"][3])
        opponents = {tuple(int(v) for v in other[3]) for other in game_state.get("others", [])}

        # Standing on a firing position already: the attack rule owns this step,
        # and walking on would only give the opponent time to leave.
        if opponents & set(blast_footprint(start, field)):
            return None

        queue = deque([(start, None, 0)])
        seen = {start}
        while queue:
            position, first, distance = queue.popleft()
            if distance > MAX_SEEK_DISTANCE:
                break
            if first is not None and opponents & set(blast_footprint(position, field)):
                return first
            for action, (dx, dy) in MOVES.items():
                nxt = (position[0] + dx, position[1] + dy)
                if nxt in seen or not (0 <= nxt[0] < field.shape[0] and 0 <= nxt[1] < field.shape[1]):
                    continue
                if field[nxt] != 0 or nxt in blocked:
                    continue
                seen.add(nxt)
                queue.append((nxt, first if first is not None else action, distance + 1))
        return None

    def choose(self, game_state: dict, chosen: str) -> str:
        if not self.enabled or chosen == "BOMB":
            return chosen
        if not self.board_is_empty(game_state):
            return chosen

        action = self.first_step_toward_a_firing_position(game_state)
        if action is None or action == chosen:
            self.no_route += 1
            return chosen

        field = np.asarray(game_state["field"])
        bombs = [(tuple(int(v) for v in bomb), int(timer))
                 for bomb, timer in game_state.get("bombs", [])]
        danger = build_danger_map(field, bombs, np.asarray(game_state["explosion_map"]))
        start = tuple(int(v) for v in game_state["self"][3])
        if not surviving_continuation_after_action(
            field, danger, _blocked_positions(game_state), bombs, start, MOVES[action]
        ):
            self.no_route += 1
            return chosen

        self.steps_taken += 1
        return action
