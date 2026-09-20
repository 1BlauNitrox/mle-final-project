"""Conservative play-time escape from persistent, progress-free late-game loops."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import ACTIONS
from .features.bombs_and_crates import build_danger_map, surviving_continuation_after_action
from .features.navigation import _blocked_positions

WINDOW = 24
MAX_DISTINCT = 3
MOVES = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}

Position = tuple[int, int]


def step(position: Position, action: str) -> Position:
    dx, dy = MOVES[action]
    return position[0] + dx, position[1] + dy


def _signature(game_state: dict) -> tuple:
    """State elements whose change constitutes progress or a new objective."""
    return (
        np.asarray(game_state["field"]).tobytes(),
        tuple(sorted(tuple(int(v) for v in coin) for coin in game_state.get("coins", []))),
        int(game_state["self"][1]),
        len(game_state.get("others", [])),
    )


def _hazard_free(game_state: dict) -> bool:
    return not game_state.get("bombs") and not np.any(np.asarray(game_state["explosion_map"]))


def _contested_targets(game_state: dict) -> set[Position]:
    """Tiles another player can occupy before our next decision."""
    field = np.asarray(game_state["field"])
    occupied = {tuple(int(v) for v in other[3]) for other in game_state.get("others", [])}
    contested = set(occupied)
    for x, y in occupied:
        for dx, dy in MOVES.values():
            target = x + dx, y + dy
            if field[target] == 0:
                contested.add(target)
    return contested


@dataclass
class NarrowLoopGuard:
    """Redirect only a confirmed late-game loop to the policy's safest novel move."""

    observations: list[tuple[int, Position, tuple, bool]] = field(default_factory=list)
    overrides: int = 0
    eligible: int = 0
    rejected_contested: int = 0
    rejected_no_candidate: int = 0
    followup_steps_remaining: int = 0
    followup_redirects: int = 0

    def observe(self, game_state: dict) -> None:
        if int(game_state["step"]) == 1:
            self.observations.clear()
            self.followup_steps_remaining = 0
        self.observations.append(
            (
                int(game_state["step"]),
                tuple(int(v) for v in game_state["self"][3]),
                _signature(game_state),
                _hazard_free(game_state),
            )
        )
        del self.observations[:-WINDOW]

    def stuck(self, game_state: dict) -> bool:
        recent = self.observations[-WINDOW:]
        if len(recent) != WINDOW:
            return False
        steps = [entry[0] for entry in recent]
        positions = [entry[1] for entry in recent]
        signatures = [entry[2] for entry in recent]
        return bool(
            steps == list(range(steps[0], steps[0] + WINDOW))
            and len(set(positions)) <= MAX_DISTINCT
            and len(set(signatures)) == 1
            and all(entry[3] for entry in recent)
            and not np.any(np.asarray(game_state["field"]) == 1)
            and (game_state.get("coins") or game_state.get("others"))
        )

    def choose(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
        followup_steps: int = 0,
    ) -> str:
        if chosen not in MOVES or not self.stuck(game_state):
            return chosen
        recent = {entry[1] for entry in self.observations[-WINDOW:]}
        here = self.observations[-1][1]
        if step(here, chosen) not in recent:
            return chosen
        self.eligible += 1

        field = np.asarray(game_state["field"])
        bombs = [(tuple(position), int(timer)) for position, timer in game_state.get("bombs", [])]
        danger = build_danger_map(field, bombs, np.asarray(game_state["explosion_map"]))
        blocked = _blocked_positions(game_state)
        contested = _contested_targets(game_state)
        candidates = []
        for index, action in enumerate(ACTIONS):
            if action not in MOVES or action == chosen or not legal_mask[index]:
                continue
            target = step(here, action)
            if target in recent:
                continue
            if target in contested:
                self.rejected_contested += 1
                continue
            if not surviving_continuation_after_action(
                field, danger, blocked, bombs, here, MOVES[action]
            ):
                continue
            candidates.append((float(q_values()[index]), action))
        if not candidates:
            self.rejected_no_candidate += 1
            return chosen
        self.overrides += 1
        self.followup_steps_remaining = followup_steps
        return max(candidates)[1]

    def redirect_contested(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
    ) -> str:
        """Avoid a simultaneous-move collision briefly after an override."""
        if self.followup_steps_remaining <= 0:
            return chosen
        self.followup_steps_remaining -= 1
        if chosen not in MOVES:
            return chosen
        here = tuple(int(v) for v in game_state["self"][3])
        contested = _contested_targets(game_state)
        if step(here, chosen) not in contested:
            return chosen
        candidates = []
        for index, action in enumerate(ACTIONS):
            if not legal_mask[index] or action == chosen:
                continue
            if action in MOVES and step(here, action) in contested:
                continue
            candidates.append((float(q_values()[index]), action))
        self.followup_redirects += 1
        return max(candidates)[1] if candidates else "WAIT"

    def snapshot(self) -> dict[str, int]:
        return {
            "eligible": self.eligible,
            "overrides": self.overrides,
            "rejected_contested": self.rejected_contested,
            "rejected_no_candidate": self.rejected_no_candidate,
            "followup_redirects": self.followup_redirects,
            "followup_steps_remaining": self.followup_steps_remaining,
        }
