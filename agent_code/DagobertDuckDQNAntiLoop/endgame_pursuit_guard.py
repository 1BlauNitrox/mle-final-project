"""Learned pursuit policy gated to hazard-free, empty-board endgames."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ACTIONS
from .features.bombs_and_crates import blast_footprint, build_danger_map, safe_escape_exists
from .features.navigation import _blocked_positions
from .observations import hunting_geometry

BOMB_INDEX = ACTIONS.index("BOMB")


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
    eligible: int = 0
    overrides: int = 0
    rejected_confidence: int = 0
    rejected_bomb_safety: int = 0
    rejected_bomb_override: int = 0

    @classmethod
    def load(cls, path: Path, *, allow_bomb_override: bool = True) -> EndgamePursuitGuard:
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
        return cls(
            **arrays,
            threshold=threshold,
            allow_bomb_override=allow_bomb_override,
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

    def choose(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
        state: np.ndarray,
    ) -> str:
        """Select the learned endgame action or preserve the fallback exactly."""
        if not pursuit_eligible(game_state):
            return chosen
        self.eligible += 1
        probabilities = self._probabilities(pursuit_input(game_state, state, q_values()))
        masked = np.where(legal_mask, probabilities, -1.0)
        index = int(np.argmax(masked))
        if masked[index] < self.threshold:
            self.rejected_confidence += 1
            return chosen
        proposed = ACTIONS[index]
        if proposed == "BOMB" and not self.allow_bomb_override:
            self.rejected_bomb_override += 1
            return chosen
        if proposed == "BOMB" and not self._safe_bomb(game_state):
            self.rejected_bomb_safety += 1
            return chosen
        if proposed != chosen:
            self.overrides += 1
        return proposed

    def snapshot(self) -> dict[str, int | float]:
        return {
            "eligible": self.eligible,
            "overrides": self.overrides,
            "rejected_confidence": self.rejected_confidence,
            "rejected_bomb_safety": self.rejected_bomb_safety,
            "rejected_bomb_override": self.rejected_bomb_override,
            "threshold": self.threshold,
        }
