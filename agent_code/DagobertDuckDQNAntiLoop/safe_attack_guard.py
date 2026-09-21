"""A narrow learned-policy tie-break for safe opponent-facing bombs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .config import ACTIONS
from .features.bombs_and_crates import blast_footprint, build_danger_map, safe_escape_exists
from .features.navigation import _blocked_positions

BOMB_INDEX = ACTIONS.index("BOMB")


@dataclass
class SafeAttackGuard:
    """Force at most one safe bomb when the DQN already ranks it second."""

    round_id: int | None = None
    used: bool = False
    eligible: int = 0
    overrides: int = 0
    rejected_rank: int = 0

    def choose(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
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

        field = np.asarray(game_state["field"])
        position = tuple(int(value) for value in game_state["self"][3])
        opponents = {tuple(int(value) for value in other[3]) for other in game_state["others"]}
        if not opponents.intersection(blast_footprint(position, field)):
            return chosen

        proposed_bombs = [(position, 3)]
        danger = build_danger_map(field, proposed_bombs, np.asarray(game_state["explosion_map"]))
        if not safe_escape_exists(
            field,
            danger,
            _blocked_positions(game_state),
            proposed_bombs,
            position,
        ):
            return chosen
        self.eligible += 1

        values = np.asarray(q_values(), dtype=np.float64)
        better_legal_actions = np.count_nonzero(
            legal_mask & (values > values[BOMB_INDEX] + 1e-12)
        )
        if better_legal_actions > 1:
            self.rejected_rank += 1
            return chosen
        self.used = True
        self.overrides += 1
        return "BOMB"

    def snapshot(self) -> dict[str, int]:
        return {
            "eligible": self.eligible,
            "overrides": self.overrides,
            "rejected_rank": self.rejected_rank,
        }
