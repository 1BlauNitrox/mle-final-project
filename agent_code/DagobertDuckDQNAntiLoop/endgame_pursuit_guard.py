"""Learned pursuit policy gated to hazard-free, empty-board endgames."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ACTIONS
from .features.bombs_and_crates import (
    blast_footprint,
    build_danger_map,
    safe_escape_exists,
    surviving_continuation_after_action,
)
from .features.navigation import _blocked_positions
from .observations import hunting_geometry

BOMB_INDEX = ACTIONS.index("BOMB")
ESCAPE_ACTIONS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
}


def pursuit_eligible(game_state: dict) -> bool:
    """Return whether changing the fallback is allowed in this state."""
    field = np.asarray(game_state["field"])
    return bool(
        game_state.get("others")
        and not game_state.get("coins")
        and not np.any(field == 1)
        and not game_state.get("bombs")
        and not np.any(np.asarray(game_state["explosion_map"]))
    )


def pursuit_input(game_state: dict, state: np.ndarray, q_values: Sequence[float]) -> np.ndarray:
    """Combine frozen policy state, explicit pursuit geometry, and Q-values."""
    value = np.asarray(
        [*state, *hunting_geometry(game_state), *q_values],
        dtype=np.float32,
    )
    if value.shape != (74,) or not np.all(np.isfinite(value)):
        raise ValueError("Invalid learned pursuit input")
    return value


@dataclass
class EndgamePursuitGuard:
    """Use a supervised policy only in the registered empty-board context."""

    mean: np.ndarray
    scale: np.ndarray
    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: np.ndarray
    threshold: float
    allow_bomb_override: bool = True
    allow_movement_override: bool = True
    maximum_better_bomb_actions: int | None = None
    maximum_bomb_overrides_per_round: int | None = None
    maximum_immediate_target_escapes: int | None = None
    stop_after_opponent_elimination: bool = False
    escape_followup_steps: int = 0
    eligible: int = 0
    overrides: int = 0
    bomb_overrides: int = 0
    rejected_confidence: int = 0
    rejected_bomb_safety: int = 0
    rejected_bomb_override: int = 0
    rejected_movement_override: int = 0
    rejected_bomb_rank: int = 0
    rejected_bomb_limit: int = 0
    rejected_target_mobility: int = 0
    rejected_post_kill: int = 0
    escape_checks: int = 0
    escape_redirects: int = 0
    escape_rejected_no_candidate: int = 0
    round_id: int | None = None
    bomb_overrides_this_round: int = 0
    opponents_at_first_bomb_override: int | None = None
    escape_steps_remaining: int = 0
    escape_armed_step: int | None = None

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        allow_bomb_override: bool = True,
        allow_movement_override: bool = True,
        maximum_better_bomb_actions: int | None = None,
        maximum_bomb_overrides_per_round: int | None = None,
        maximum_immediate_target_escapes: int | None = None,
        stop_after_opponent_elimination: bool = False,
        escape_followup_steps: int = 0,
    ) -> EndgamePursuitGuard:
        """Load a strictly shaped, training-produced pursuit artifact."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "schema",
            "mean",
            "scale",
            "weight1",
            "bias1",
            "weight2",
            "bias2",
            "threshold",
            "parent_checkpoint_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != 1:
            raise ValueError("Learned pursuit artifact has an invalid schema")
        arrays = {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in ("mean", "scale", "weight1", "bias1", "weight2", "bias2")
        }
        if (
            arrays["mean"].shape != (74,)
            or arrays["scale"].shape != (74,)
            or arrays["weight1"].shape != (32, 74)
            or arrays["bias1"].shape != (32,)
            or arrays["weight2"].shape != (len(ACTIONS), 32)
            or arrays["bias2"].shape != (len(ACTIONS),)
        ):
            raise ValueError("Learned pursuit artifact has incompatible dimensions")
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("Learned pursuit artifact contains non-finite parameters")
        threshold = float(payload["threshold"])
        if np.any(arrays["scale"] <= 0) or not 0.0 < threshold < 1.0:
            raise ValueError("Learned pursuit artifact has invalid controls")
        if maximum_better_bomb_actions is not None and maximum_better_bomb_actions < 0:
            raise ValueError("maximum_better_bomb_actions must be non-negative")
        if (
            maximum_bomb_overrides_per_round is not None
            and maximum_bomb_overrides_per_round < 1
        ):
            raise ValueError("maximum_bomb_overrides_per_round must be positive")
        if (
            maximum_immediate_target_escapes is not None
            and maximum_immediate_target_escapes < 0
        ):
            raise ValueError("maximum_immediate_target_escapes must be non-negative")
        if escape_followup_steps < 0:
            raise ValueError("escape_followup_steps must be non-negative")
        return cls(
            **arrays,
            threshold=threshold,
            allow_bomb_override=allow_bomb_override,
            allow_movement_override=allow_movement_override,
            maximum_better_bomb_actions=maximum_better_bomb_actions,
            maximum_bomb_overrides_per_round=maximum_bomb_overrides_per_round,
            maximum_immediate_target_escapes=maximum_immediate_target_escapes,
            stop_after_opponent_elimination=stop_after_opponent_elimination,
            escape_followup_steps=escape_followup_steps,
        )

    def _probabilities(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        hidden = np.maximum(0.0, self.weight1 @ normalized + self.bias1)
        logits = self.weight2 @ hidden + self.bias2
        logits -= np.max(logits)
        probabilities = np.exp(np.clip(logits, -30.0, 0.0))
        return probabilities / probabilities.sum()

    @staticmethod
    def _safe_bomb(game_state: dict) -> bool:
        field = np.asarray(game_state["field"])
        position = tuple(int(value) for value in game_state["self"][3])
        opponents = {tuple(int(value) for value in other[3]) for other in game_state["others"]}
        if not opponents.intersection(blast_footprint(position, field)):
            return False
        bombs = [(position, 3)]
        danger = build_danger_map(field, bombs, np.asarray(game_state["explosion_map"]))
        return safe_escape_exists(
            field,
            danger,
            _blocked_positions(game_state),
            bombs,
            position,
        )

    @staticmethod
    def _target_escape_options(game_state: dict) -> list[int]:
        """Count immediate open moves out of the proposed blast for each target."""
        field = np.asarray(game_state["field"])
        position = tuple(int(value) for value in game_state["self"][3])
        footprint = set(blast_footprint(position, field))
        targets = [
            tuple(int(value) for value in other[3])
            for other in game_state["others"]
            if tuple(int(value) for value in other[3]) in footprint
        ]
        occupied = _blocked_positions(game_state) | {position}
        counts = []
        for target in targets:
            count = 0
            for dx, dy in ESCAPE_ACTIONS.values():
                if (dx, dy) == (0, 0):
                    continue
                neighbor = target[0] + dx, target[1] + dy
                if (
                    0 <= neighbor[0] < field.shape[0]
                    and 0 <= neighbor[1] < field.shape[1]
                    and field[neighbor] == 0
                    and neighbor not in occupied
                    and neighbor not in footprint
                ):
                    count += 1
            counts.append(count)
        return counts

    def choose(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
        state: np.ndarray,
    ) -> str:
        """Select the learned endgame action or preserve the fallback exactly."""
        round_id = int(game_state["round"])
        if round_id != self.round_id:
            self.round_id = round_id
            self.bomb_overrides_this_round = 0
            self.opponents_at_first_bomb_override = None
            self.escape_steps_remaining = 0
            self.escape_armed_step = None
        if not pursuit_eligible(game_state):
            return chosen
        self.eligible += 1
        values = np.asarray(q_values(), dtype=np.float64)
        probabilities = self._probabilities(pursuit_input(game_state, state, values))
        masked = np.where(legal_mask, probabilities, -1.0)
        index = int(np.argmax(masked))
        if masked[index] < self.threshold:
            self.rejected_confidence += 1
            return chosen
        proposed = ACTIONS[index]
        if proposed != "BOMB" and not self.allow_movement_override:
            self.rejected_movement_override += 1
            return chosen
        if proposed == "BOMB" and not self.allow_bomb_override:
            self.rejected_bomb_override += 1
            return chosen
        if proposed == "BOMB" and not self._safe_bomb(game_state):
            self.rejected_bomb_safety += 1
            return chosen
        if proposed == "BOMB" and chosen != "BOMB":
            target_escapes = self._target_escape_options(game_state)
            if (
                self.maximum_immediate_target_escapes is not None
                and (
                    not target_escapes
                    or min(target_escapes) > self.maximum_immediate_target_escapes
                )
            ):
                self.rejected_target_mobility += 1
                return chosen
            better_actions = int(
                np.count_nonzero(legal_mask & (values > values[BOMB_INDEX] + 1e-12))
            )
            if (
                self.maximum_better_bomb_actions is not None
                and better_actions > self.maximum_better_bomb_actions
            ):
                self.rejected_bomb_rank += 1
                return chosen
            if (
                self.stop_after_opponent_elimination
                and self.opponents_at_first_bomb_override is not None
                and len(game_state["others"]) < self.opponents_at_first_bomb_override
            ):
                self.rejected_post_kill += 1
                return chosen
            if (
                self.maximum_bomb_overrides_per_round is not None
                and self.bomb_overrides_this_round
                >= self.maximum_bomb_overrides_per_round
            ):
                self.rejected_bomb_limit += 1
                return chosen
            self.bomb_overrides_this_round += 1
            if self.opponents_at_first_bomb_override is None:
                self.opponents_at_first_bomb_override = len(game_state["others"])
            self.bomb_overrides += 1
            self.escape_steps_remaining = self.escape_followup_steps
            self.escape_armed_step = int(game_state["step"])
        if proposed != chosen:
            self.overrides += 1
        return proposed

    def redirect_unsafe_escape(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
    ) -> str:
        """Preserve the DQN unless a learned-bomb follow-up cannot survive."""
        if self.escape_steps_remaining <= 0:
            return chosen
        step = int(game_state["step"])
        if step == self.escape_armed_step:
            if chosen != "BOMB":
                self.escape_steps_remaining = 0
                self.escape_armed_step = None
            return chosen
        bombs = [(tuple(position), int(timer)) for position, timer in game_state.get("bombs", [])]
        explosion_map = np.asarray(game_state["explosion_map"])
        if not bombs and not np.any(explosion_map):
            self.escape_steps_remaining = 0
            self.escape_armed_step = None
            return chosen
        self.escape_steps_remaining -= 1
        self.escape_checks += 1
        field = np.asarray(game_state["field"])
        danger = build_danger_map(field, bombs, explosion_map)
        blocked = _blocked_positions(game_state)
        position = tuple(int(value) for value in game_state["self"][3])

        def survives(action: str) -> bool:
            return action in ESCAPE_ACTIONS and surviving_continuation_after_action(
                field,
                danger,
                blocked,
                bombs,
                position,
                ESCAPE_ACTIONS[action],
            )

        if survives(chosen):
            return chosen
        values = np.asarray(q_values(), dtype=np.float64)
        candidates = [
            (float(values[index]), action)
            for index, action in enumerate(ACTIONS)
            if legal_mask[index] and survives(action)
        ]
        if not candidates:
            self.escape_rejected_no_candidate += 1
            return chosen
        self.escape_redirects += 1
        return max(candidates)[1]

    def snapshot(self) -> dict[str, int | float]:
        return {
            "eligible": self.eligible,
            "overrides": self.overrides,
            "bomb_overrides": self.bomb_overrides,
            "rejected_confidence": self.rejected_confidence,
            "rejected_bomb_safety": self.rejected_bomb_safety,
            "rejected_bomb_override": self.rejected_bomb_override,
            "rejected_movement_override": self.rejected_movement_override,
            "rejected_bomb_rank": self.rejected_bomb_rank,
            "rejected_bomb_limit": self.rejected_bomb_limit,
            "rejected_target_mobility": self.rejected_target_mobility,
            "rejected_post_kill": self.rejected_post_kill,
            "escape_checks": self.escape_checks,
            "escape_redirects": self.escape_redirects,
            "escape_rejected_no_candidate": self.escape_rejected_no_candidate,
            "threshold": self.threshold,
        }
