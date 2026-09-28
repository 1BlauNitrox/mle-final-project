"""Refuse actions our own danger model says we cannot survive.

Bomb-omb kills itself in about half of all classic rounds, and that number has
not moved since roughly episode 1,500 of the final run. The information it would
need is already in the state it sees: `features/assemble.py` feeds the network
five flags for "this move has a surviving continuation" plus one for "a bomb
placed here still leaves an escape". Nothing enforces them. `legality.py` masks
framework legality only, and says so itself.

This guard enforces them at play time. It never invents a move: it walks the
policy's own action ranking downwards and takes the best action the agent's
danger model considers survivable. If nothing is survivable the policy's choice
stands, because then we know nothing better.

Two factors switch independently, so an experiment can tell apart the two ways
the agent dies:

  BOMBERMAN_SURVIVAL_GUARD_BOMB=on   refuse BOMB when placing it here leaves no
                                     escape (bombing yourself into a dead end)
  BOMBERMAN_SURVIVAL_GUARD_MOVE=on   refuse a move or WAIT with no surviving
                                     continuation (walking into a blast, or
                                     standing still in one)

Both default to off, and with both off this file changes nothing: the agent
plays exactly as the shipped Bomb-omb does. The guard is never active while
training, so it cannot affect a training run.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import numpy as np

from .config import ACTIONS
from .features.bombs_and_crates import (
    BOMB_TIMER,
    build_danger_map,
    safe_escape_exists,
    surviving_continuation_after_action,
)
from .features.navigation import DIRECTIONS, _blocked_positions

BOMB_ENV = "BOMBERMAN_SURVIVAL_GUARD_BOMB"
MOVE_ENV = "BOMBERMAN_SURVIVAL_GUARD_MOVE"
BOMB_INDEX = ACTIONS.index("BOMB")
WAIT_INDEX = ACTIONS.index("WAIT")
# ACTIONS is UP, RIGHT, DOWN, LEFT, WAIT, BOMB and DIRECTIONS is UP, RIGHT,
# DOWN, LEFT, so one offset table covers every action a direction can describe.
ACTION_DIRECTIONS = (*DIRECTIONS, (0, 0))


def guard_modes(environ: dict[str, str] | None = None) -> tuple[bool, bool]:
    """Read the two switches, rejecting anything but "on" and "off"."""
    environ = os.environ if environ is None else environ
    modes = []
    for name in (BOMB_ENV, MOVE_ENV):
        value = environ.get(name, "off")
        if value not in {"on", "off"}:
            raise ValueError(f"{name} must be 'on' or 'off', not {value!r}")
        modes.append(value == "on")
    return modes[0], modes[1]


class SurvivalGuard:
    """Veto unsurvivable actions, keeping the policy's preference order."""

    def __init__(self, veto_bomb: bool, veto_move: bool) -> None:
        self.veto_bomb = veto_bomb
        self.veto_move = veto_move
        self.vetoes = 0
        self.fallbacks = 0

    @property
    def active(self) -> bool:
        return self.veto_bomb or self.veto_move

    def survivable(self, game_state: dict, index: int, cache: dict) -> bool:
        """Does the agent's own danger model give this action a future?"""
        if index == BOMB_INDEX:
            if not self.veto_bomb:
                return True
            return self._bomb_has_escape(game_state, cache)
        if not self.veto_move:
            return True
        return self._move_has_continuation(game_state, index, cache)

    def choose(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
    ) -> str:
        if not self.active:
            return chosen
        cache: dict = {}
        chosen_index = ACTIONS.index(chosen)
        if self.survivable(game_state, chosen_index, cache):
            return chosen

        values = q_values()
        ranked = sorted(
            (index for index in range(len(ACTIONS))
             if index != chosen_index and legal_mask[index]),
            key=lambda index: float(values[index]),
            reverse=True,
        )
        for index in ranked:
            if self.survivable(game_state, index, cache):
                self.vetoes += 1
                return ACTIONS[index]
        self.fallbacks += 1
        return chosen

    def _geometry(self, game_state: dict, cache: dict) -> tuple:
        if "geometry" not in cache:
            field = np.asarray(game_state["field"])
            explosions = np.asarray(game_state["explosion_map"])
            bombs = [(tuple(int(v) for v in position), int(timer))
                     for position, timer in game_state.get("bombs", [])]
            position = tuple(int(v) for v in game_state["self"][3])
            cache["geometry"] = (
                field,
                explosions,
                bombs,
                position,
                _blocked_positions(game_state),
            )
        return cache["geometry"]

    def _move_has_continuation(self, game_state: dict, index: int, cache: dict) -> bool:
        field, explosions, bombs, position, blocked = self._geometry(game_state, cache)
        if "danger" not in cache:
            cache["danger"] = build_danger_map(field, bombs, explosions)
        return surviving_continuation_after_action(
            field, cache["danger"], blocked, bombs, position, ACTION_DIRECTIONS[index]
        )

    def _bomb_has_escape(self, game_state: dict, cache: dict) -> bool:
        field, explosions, bombs, position, blocked = self._geometry(game_state, cache)
        if "hypothetical" not in cache:
            # The framework decrements a bomb placed by this action once before
            # the next observable state, exactly as assemble.py models it.
            hypothetical = [*bombs, (position, BOMB_TIMER - 1)]
            cache["hypothetical"] = (
                build_danger_map(field, hypothetical, explosions),
                hypothetical,
            )
        danger, hypothetical = cache["hypothetical"]
        return safe_escape_exists(field, danger, blocked, hypothetical, position)
