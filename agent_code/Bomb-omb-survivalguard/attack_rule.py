"""Take the attack the policy keeps declining.

Our agent kills 0.105 opponents per game against three rule_based agents, which
take 0.23 from us. Score is coins plus five times kills, so that deficit costs
0.63 points and is the whole gap between us and the heuristic.

Six registered experiments tried to teach the network to attack and all six came
back empty. The hunting diagnosis shows why they could not work: a bomb aimed at
an opponent with no crate in its blast books the wasteful-bomb penalty of -0.5
immediately and with certainty, while the +5 for a kill arrives four steps later,
discounted by gamma^4 to about two thirds, and only if the opponent fails to walk
away. At our measured conversion, roughly one kill per hundred attack steps, the
expected value of attacking is negative no matter how large the kill reward is.

That is a credit-assignment problem, not a knowledge problem, so this rule sits
outside the network entirely. When an opponent is standing inside the blast
footprint of a bomb placed where we are, and our own danger model says we could
still escape that bomb, we place it. Nothing else changes: the rule never fires
without a bomb in hand, never fires when the policy already chose BOMB, and never
fires while training.

It is off unless BOMBERMAN_ATTACK_RULE=on.
"""

from __future__ import annotations

import os

import numpy as np

from .features.bombs_and_crates import (
    BOMB_TIMER,
    blast_footprint,
    build_danger_map,
    safe_escape_exists,
)
from .features.navigation import _blocked_positions

ATTACK_ENV = "BOMBERMAN_ATTACK_RULE"
MODES = ("off", "on", "selective")


def attack_mode(environ: dict[str, str] | None = None) -> str:
    """Read the switch: off, on (bomb anyone in range), or selective.

    `selective` additionally asks whether the opponent could walk out of the
    blast, and only fires when they cannot. It trades how often the rule fires
    for how often it kills; which of those matters more is what the third cell
    of the registration measures.
    """
    environ = os.environ if environ is None else environ
    value = environ.get(ATTACK_ENV, "off")
    if value not in MODES:
        raise ValueError(f"{ATTACK_ENV} must be one of {MODES}, not {value!r}")
    return value


class AttackRule:
    """Place a bomb when an opponent is in range and we can still get out."""

    def __init__(self, enabled: bool | str) -> None:
        # Accepts the mode string, and True/False so older callers keep working.
        mode = {True: "on", False: "off"}.get(enabled, enabled)
        if mode not in MODES:
            raise ValueError(f"unknown attack mode {enabled!r}")
        self.mode = mode
        self.attacks = 0
        self.declined_no_escape = 0
        self.declined_opponent_escapes = 0

    @property
    def active(self) -> bool:
        return self.mode != "off"

    @property
    def selective(self) -> bool:
        return self.mode == "selective"

    def opponents_in_range(self, game_state: dict) -> list[tuple[int, int]]:
        field = np.asarray(game_state["field"])
        position = tuple(int(v) for v in game_state["self"][3])
        footprint = set(blast_footprint(position, field))
        others = {tuple(int(v) for v in other[3]) for other in game_state.get("others", [])}
        return sorted(footprint & others)

    def trapped(self, game_state, target, field, danger, hypothetical) -> bool:
        """Could this opponent walk out of the blast we are about to create?

        The same search we use on ourselves, run from their tile. Their own tile
        is not an obstacle to them, and ours is, because agents cannot share a
        tile. Opponents are not modelled forward in time, so this is the same
        approximation the agent's own escape check makes.
        """
        blocked = set(_blocked_positions(game_state))
        blocked.discard(target)
        blocked.add(tuple(int(v) for v in game_state["self"][3]))
        return not safe_escape_exists(field, danger, blocked, hypothetical, target)

    def choose(self, game_state: dict, chosen: str) -> str:
        if not self.active or chosen == "BOMB":
            return chosen
        if not game_state["self"][2]:  # no bomb in hand
            return chosen
        targets = self.opponents_in_range(game_state)
        if not targets:
            return chosen

        field = np.asarray(game_state["field"])
        explosions = np.asarray(game_state["explosion_map"])
        bombs = [(tuple(int(v) for v in bomb), int(timer))
                 for bomb, timer in game_state.get("bombs", [])]
        position = tuple(int(v) for v in game_state["self"][3])
        # The framework decrements a bomb placed by this action once before the
        # next observable state, exactly as assemble.py models it.
        hypothetical = [*bombs, (position, BOMB_TIMER - 1)]
        danger = build_danger_map(field, hypothetical, explosions)
        if not safe_escape_exists(field, danger, _blocked_positions(game_state), hypothetical, position):
            self.declined_no_escape += 1
            return chosen

        if self.selective and not any(
            self.trapped(game_state, target, field, danger, hypothetical) for target in targets
        ):
            self.declined_opponent_escapes += 1
            return chosen

        self.attacks += 1
        return "BOMB"
