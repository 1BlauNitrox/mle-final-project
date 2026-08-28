"""Bounded and reproducible experience replay for DagobertDuckDQN"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from .config import ACTIONS, FEATURE_COUNT


@dataclass(frozen=True)
class Transition:
    """One immutable replay buffer entry."""

    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray | None
    terminal: bool

@dataclass(frozen=True)
class ReplayBatch:
    """A sampled batch represented as NumPy arrays."""

    states: np.ndarray
    action_indices: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminals: np.ndarray

class ReplayBuffer:
    """Fixed-Capacity replay buffer with seeded random sampling"""

    def __init__(self, *, capacity: int, seed: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.capacity = capacity
        self._transitions: deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of stored transitions"""
        return len(self._transitions)
    
    def add(
        self,
        *,
        state: np.ndarray,
        action_index: int,
        reward: float,
        next_state: np.ndarray | None,
        terminal: bool
    ) -> None:
        """Validate, copy and append one transition"""
        checked_state = _copy_state(state, name="state")

        if(
            isinstance(action_index, bool)
            or not isinstance(action_index, (int, np.integer))
            or not 0 <= int(action_index) < len(ACTIONS)
        ):
            raise ValueError("action_index is outside the action space")
        
        if (
            isinstance(reward, bool)
            or not isinstance(reward, Real)
            or not np.isfinite(float(reward))
        ): 
            raise ValueError("reward must be a finite number")
        
        if not isinstance(terminal, bool):
            raise ValueError("terminal must be a bool")
        
        if terminal:
            if next_state is not None:
                raise ValueError("A terminal must not have a next state")
            checked_next_state = None
        else:
            if next_state is None:
                raise ValueError("A non-terminal transition requires a next state")
            checked_next_state = _copy_state(next_state, name="next_state")

        self._transitions.append(
            Transition(
                state=checked_state,
                action_index=int(action_index),
                reward=float(reward),
                next_state=checked_next_state,
                terminal=terminal
            )
        )
    
    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample transitions uniformly without replacement"""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if batch_size > len(self._transitions):
            raise ValueError("Not enough transitions for the requested batch")
        
        indices = self._rng.choice(
            len(self._transitions),
            size=batch_size,
            replace=False
        )
        selected = [self._transitions[int(index)] for index in indices]

        return _transitions_to_batch(selected)
    
    def state_dict(self) -> dict[str, Any]:
        """Export replay data and RNG state using validated array types"""
        batch = _transitions_to_batch(list(self._transitions))

        return {
            "capacity": self.capacity,
            "states": batch.states.copy(),
            "action_indices": batch.action_indices.copy(),
            "rewards": batch.rewards.copy(),
            "next_states": batch.next_states.copy(),
            "terminals": batch.terminals.copy(),
            "rng_state": deepcopy(self._rng.bit_generator.state)
        }
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay contents and sampling state atomically"""
        required_fields = {
            "capacity",
            "states",
            "action_indices",
            "rewards",
            "next_states",
            "terminals",
            "rng_state"
        }

        if not isinstance(state, dict) or set(state) != required_fields:
            raise ValueError("Replay state has unexpected fields")
        
        capacity = state["capacity"]

        if type(capacity) is not int or capacity != self.capacity:
            raise ValueError("Replay capacity does not match configuration")
        
        states = _require_array(
            state["states"],
            name="states",
            dtype=np.float32
        )
        action_indices = _require_array(
            state["action_indices"],
            name="action_indices",
            dtype=np.int64
        )
        rewards = _require_array(
            state["rewards"],
            name="rewards",
            dtype=np.float32
        )
        next_states = _require_array(
            state["next_states"],
            name="next_states",
            dtype=np.float32
        )
        terminals = _require_array(
            state["terminals"],
            name="terminals",
            dtype=np.bool_
        )

        count = states.shape[0]

        if states.shape != (count, FEATURE_COUNT):
            raise ValueError("Replay states have an incompatible shape.")

        if action_indices.shape != (count,):
            raise ValueError("Replay action indices have an incompatible shape.")

        if rewards.shape != (count,):
            raise ValueError("Replay rewards have an incompatible shape.")

        if next_states.shape != (count, FEATURE_COUNT):
            raise ValueError("Replay next states have an incompatible shape.")

        if terminals.shape != (count,):
            raise ValueError("Replay terminal flags have an incompatible shape.")

        if count > self.capacity:
            raise ValueError("Replay state exceeds configured capacity.")

        if not np.all(np.isfinite(states)):
            raise ValueError("Replay states must be finite.")

        if not np.all(np.isfinite(next_states)):
            raise ValueError("Replay next states must be finite.")

        if not np.all(np.isfinite(rewards)):
            raise ValueError("Replay rewards must be finite.")

        if np.any(action_indices < 0) or np.any(
            action_indices >= len(ACTIONS)
        ):
            raise ValueError("Replay contains an invalid action index.")

        if count and np.any(next_states[terminals] != 0.0):
            raise ValueError(
                "Terminal replay entries require zero next-state placeholders."
            )

        restored_transitions = [
            Transition(
                state=states[index].copy(),
                action_index=int(action_indices[index]),
                reward=float(rewards[index]),
                next_state=(
                    None
                    if bool(terminals[index])
                    else next_states[index].copy()
                ),
                terminal=bool(terminals[index]),
            )
            for index in range(count)
        ]

        rng_state = state["rng_state"]

        if not isinstance(rng_state, dict):
            raise ValueError("Replay RNG state must be a dictionary.")

        restored_rng = np.random.default_rng()

        try:
            restored_rng.bit_generator.state = deepcopy(rng_state)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Replay RNG state is invalid.") from error

        self._transitions.clear()
        self._transitions.extend(restored_transitions)
        self._rng = restored_rng


def _copy_state(state: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and defensively copy one normalized feature vector"""
    values = np.asarray(state, dtype=np.float32)

    if values.shape != (FEATURE_COUNT, ):
        raise ValueError(
            f"{name} must have shape ({FEATURE_COUNT}, ), "
            f"got {values.shape}"
        )
    
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    
    return values.copy()

def _transitions_to_batch(
    transitions: list[Transition],
) -> ReplayBatch:
    """Convert an ordered transition collection into array form"""
    if not transitions:
        return ReplayBatch(
            states=np.empty((0, FEATURE_COUNT), dtype=np.float32),
            action_indices=np.empty(0, dtype=np.int64), 
            rewards=np.empty(0, dtype=np.float32),
            next_states=np.empty((0, FEATURE_COUNT), dtype=np.float32),
            terminals=np.empty(0, dtype=np.bool_)
        )
    
    return ReplayBatch(
        states=np.stack(
            [transition.state for transition in transitions]
        ).astype(np.float32, copy=False),
        action_indices=np.asarray(
            [transition.action_index for transition in transitions],
            dtype=np.int64,
        ),
        rewards=np.asarray(
            [transition.reward for transition in transitions],
            dtype=np.float32,
        ),
        next_states=np.stack(
            [
                (
                    np.zeros(FEATURE_COUNT, dtype=np.float32)
                    if transition.next_state is None
                    else transition.next_state
                )
                for transition in transitions
            ]
        ).astype(np.float32, copy=False),
        terminals=np.asarray(
            [transition.terminal for transition in transitions],
            dtype=np.bool_,
        ),
    )

def _require_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype,
) -> np.ndarray:
    """Require an array with an exact persistence dtype"""
    if not isinstance(value, np.ndarray):
        raise ValueError(f"Replay {name} must be a NumPy array")
    
    if value.dtype != dtype:
        raise ValueError(f"Replay {name} has an incompatible dtype")
    
    return value


