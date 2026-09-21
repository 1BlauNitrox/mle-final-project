"""Learned, high-confidence gate for rare safe opponent-facing bombs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ACTIONS
from .features.bombs_and_crates import blast_footprint, build_danger_map, safe_escape_exists
from .features.navigation import _blocked_positions
from .model import CPU_DEVICE, QNetwork

BOMB_INDEX = ACTIONS.index("BOMB")


@dataclass
class LearnedAttackGuard:
    """Override once per round only when a learned kill gate is confident."""

    network: QNetwork
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    threshold: float
    max_better_actions: int
    round_id: int | None = None
    used: bool = False
    eligible: int = 0
    overrides: int = 0
    rejected_rank: int = 0
    rejected_confidence: int = 0

    @classmethod
    def load(cls, path: Path, network: QNetwork) -> LearnedAttackGuard:
        """Load a strictly validated training-only linear-head artifact."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "schema",
            "mean",
            "scale",
            "weight",
            "bias",
            "threshold",
            "max_better_actions",
            "parent_checkpoint_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != 1:
            raise ValueError("Learned attack artifact has an invalid schema")
        mean = np.asarray(payload["mean"], dtype=np.float32)
        scale = np.asarray(payload["scale"], dtype=np.float32)
        weight = np.asarray(payload["weight"], dtype=np.float32)
        if mean.ndim != 1 or scale.shape != mean.shape or weight.shape != mean.shape:
            raise ValueError("Learned attack artifact has incompatible dimensions")
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(weight))
        ):
            raise ValueError("Learned attack artifact contains non-finite parameters")
        threshold = float(payload["threshold"])
        max_better = int(payload["max_better_actions"])
        if np.any(scale <= 0) or not 0.0 < threshold < 1.0 or max_better not in {1, 2}:
            raise ValueError("Learned attack artifact contains invalid controls")
        return cls(
            network=network,
            mean=mean,
            scale=scale,
            weight=weight,
            bias=float(payload["bias"]),
            threshold=threshold,
            max_better_actions=max_better,
        )

    def _probability(self, state: np.ndarray) -> float:
        tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(CPU_DEVICE)
        with torch.no_grad():
            hidden = self.network.layers[:-1](tensor).cpu().numpy()
        logit = float(np.dot((hidden - self.mean) / self.scale, self.weight) + self.bias)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))

    def choose(
        self,
        game_state: dict,
        chosen: str,
        q_values: Callable[[], Sequence[float]],
        legal_mask: np.ndarray,
        state: np.ndarray,
    ) -> str:
        """Return BOMB only inside the geometric, rank, and learned gates."""
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
        better = int(np.count_nonzero(legal_mask & (values > values[BOMB_INDEX] + 1e-12)))
        if better > self.max_better_actions:
            self.rejected_rank += 1
            return chosen
        if self._probability(state) < self.threshold:
            self.rejected_confidence += 1
            return chosen
        self.used = True
        self.overrides += 1
        return "BOMB"

    def snapshot(self) -> dict[str, int | float]:
        return {
            "eligible": self.eligible,
            "overrides": self.overrides,
            "rejected_rank": self.rejected_rank,
            "rejected_confidence": self.rejected_confidence,
            "threshold": self.threshold,
        }
