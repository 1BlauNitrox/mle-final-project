"""Framework-level legal-action masking for the Task 2 tabular agent."""

from __future__ import annotations

from typing import Any

import numpy as np


def framework_legal_action_mask(game_state: dict[str, Any]) -> np.ndarray:
    """Return legal actions in UP, RIGHT, DOWN, LEFT, WAIT, BOMB order.

    Only actions rejected directly by the game framework are excluded.
    Tactical risks such as explosions are deliberately ignored.
    """

    field = np.asarray(game_state["field"])
    position = tuple(game_state["self"][3])

    if field.ndim != 2 or len(position) != 2:
        raise ValueError("Game state has invalid action-legality geometry.")

    x, y = position

    occupied = {tuple(bomb_position) for bomb_position, _timer in game_state.get("bombs", [])}
    occupied.update(tuple(other[3]) for other in game_state.get("others", []))

    def tile_is_free(candidate: tuple[int, int]) -> bool:
        candidate_x, candidate_y = candidate

        return (
            0 <= candidate_x < field.shape[0]
            and 0 <= candidate_y < field.shape[1]
            and field[candidate_x, candidate_y] == 0
            and candidate not in occupied
        )

    return np.asarray(
        (
            tile_is_free((x, y - 1)),  # UP
            tile_is_free((x + 1, y)),  # RIGHT
            tile_is_free((x, y + 1)),  # DOWN
            tile_is_free((x - 1, y)),  # LEFT
            True,  # WAIT
            bool(game_state["self"][2]),  # BOMB
        ),
        dtype=np.bool_,
    )
