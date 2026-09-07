"""Deterministic uniform and protected experience replay for Task 2."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from .config import (
    ACTIONS,
    FEATURE_COUNT,
    OTHER_REPLAY_CAPACITY,
    PROTECTED_REPLAY_BATCH_SIZE,
    PROTECTED_REPLAY_CAPACITY,
    PROTECTED_SOURCE_EPISODES,
    PROTECTED_SOURCE_SCENARIO,
    REPLAY_TREATMENTS,
)


@dataclass(frozen=True)
class Transition:
    """One immutable replay buffer entry."""

    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray | None
    terminal: bool
    next_action_mask: np.ndarray | None


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled batch represented as NumPy arrays."""

    states: np.ndarray
    action_indices: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminals: np.ndarray
    next_action_masks: np.ndarray
    protected_flags: np.ndarray


class ReplayBuffer:
    """A bounded replay buffer with an optional protected Task 1 partition.

    ``uniform`` is the historical one-deque FIFO and keeps its sampling
    behavior unchanged. ``protected_task1`` uses two disjoint FIFO deques:
    2,000 entries for the initial Task 1 collection and 8,000 entries for all
    other experience. Once collection is closed, a full batch draws 16 and 48
    entries from those partitions, respectively. If a partition is too small,
    its missing quota is filled from the other partition without replacement.
    """

    def __init__(
        self,
        *,
        capacity: int,
        seed: int,
        mode: str = "uniform",
        collection_open: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if mode not in REPLAY_TREATMENTS:
            raise ValueError(f"Unknown replay treatment: {mode!r}")
        if mode == "protected_task1" and capacity != (
            PROTECTED_REPLAY_CAPACITY + OTHER_REPLAY_CAPACITY
        ):
            raise ValueError(
                "Protected replay requires total capacity 10000 so its "
                "2,000/8,000 partition is unambiguous."
            )
        if not isinstance(collection_open, bool):
            raise ValueError("collection_open must be a bool")

        self.capacity = capacity
        self.mode = mode
        self._collection_open = (
            collection_open if mode == "protected_task1" else False
        )
        self._transitions: deque[Transition] = deque(maxlen=capacity)
        self._protected_transitions: deque[Transition] = deque(
            maxlen=PROTECTED_REPLAY_CAPACITY
        )
        self._other_transitions: deque[Transition] = deque(
            maxlen=OTHER_REPLAY_CAPACITY
        )
        self._rng = np.random.default_rng(seed)

    @property
    def collection_open(self) -> bool:
        """Whether new entries may be assigned to the protected partition."""
        return self._collection_open

    @property
    def protected_size(self) -> int:
        """Return the number of retained protected entries."""
        return (
            len(self._protected_transitions)
            if self.mode == "protected_task1"
            else 0
        )

    @property
    def other_size(self) -> int:
        """Return the number of retained non-protected entries."""
        if self.mode == "uniform":
            return len(self._transitions)
        return len(self._other_transitions)

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        if self.mode == "uniform":
            return len(self._transitions)
        return self.protected_size + self.other_size

    def close_protected_collection(self) -> None:
        """Close Task 1 collection permanently for this training lineage."""
        if self.mode == "protected_task1":
            self._collection_open = False

    def add(
        self,
        *,
        state: np.ndarray,
        action_index: int,
        reward: float,
        next_state: np.ndarray | None,
        terminal: bool,
        next_action_mask: np.ndarray | None = None,
        partition: str | None = None,
    ) -> None:
        """Validate, copy and append one transition to its FIFO partition."""
        checked_state = _copy_state(state, name="state")

        if (
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
            checked_next_action_mask = None
        else:
            if next_state is None:
                raise ValueError("A non-terminal transition requires a next state")
            checked_next_state = _copy_state(next_state, name="next_state")
            checked_next_action_mask = _copy_action_mask(next_action_mask)

        transition = Transition(
            state=checked_state,
            action_index=int(action_index),
            reward=float(reward),
            next_state=checked_next_state,
            terminal=terminal,
            next_action_mask=checked_next_action_mask,
        )

        if self.mode == "uniform":
            if partition not in (None, "other"):
                raise ValueError("Uniform replay has no protected partition")
            self._transitions.append(transition)
            return

        selected_partition = partition
        if selected_partition is None:
            selected_partition = "protected" if self.collection_open else "other"
        if selected_partition == "protected":
            if not self.collection_open:
                raise ValueError("Protected Task 1 collection is already closed")
            self._protected_transitions.append(transition)
        elif selected_partition == "other":
            self._other_transitions.append(transition)
        else:
            raise ValueError("Replay partition must be 'protected' or 'other'")

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample without replacement using the persisted replay RNG."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self):
            raise ValueError("Not enough transitions for the requested batch")

        if self.mode == "uniform" or self.collection_open:
            selected = (
                list(self._transitions)
                if self.mode == "uniform"
                else [*self._protected_transitions, *self._other_transitions]
            )
            indices = self._rng.choice(len(selected), size=batch_size, replace=False)
            flags = (
                [False] * batch_size
                if self.mode == "uniform"
                else [
                    int(index) < len(self._protected_transitions)
                    for index in indices
                ]
            )
            return _transitions_to_batch(
                [selected[int(index)] for index in indices],
                protected_flags=flags,
            )

        protected_count = min(
            PROTECTED_REPLAY_BATCH_SIZE,
            batch_size,
            self.protected_size,
        )
        other_count = min(batch_size - protected_count, self.other_size)
        remaining = batch_size - protected_count - other_count

        if remaining:
            extra_other = min(remaining, self.other_size - other_count)
            other_count += extra_other
            remaining -= extra_other
        if remaining:
            extra_protected = min(remaining, self.protected_size - protected_count)
            protected_count += extra_protected
            remaining -= extra_protected
        if remaining:
            raise ValueError("Replay partitions cannot satisfy the requested batch")

        protected_indices = self._rng.choice(
            self.protected_size, size=protected_count, replace=False
        )
        other_indices = self._rng.choice(
            self.other_size, size=other_count, replace=False
        )
        selected = [
            self._protected_transitions[int(index)] for index in protected_indices
        ] + [self._other_transitions[int(index)] for index in other_indices]
        return _transitions_to_batch(
            selected,
            protected_flags=[True] * protected_count + [False] * other_count,
        )

    def state_dict(self) -> dict[str, Any]:
        """Export FIFO partitions and the exact sampling RNG state."""
        if self.mode == "uniform":
            batch = _transitions_to_batch(list(self._transitions))
            return {
                "capacity": self.capacity,
                "states": batch.states.copy(),
                "action_indices": batch.action_indices.copy(),
                "rewards": batch.rewards.copy(),
                "next_states": batch.next_states.copy(),
                "terminals": batch.terminals.copy(),
                "next_action_masks": batch.next_action_masks.copy(),
                "rng_state": deepcopy(self._rng.bit_generator.state),
            }

        return {
            "mode": self.mode,
            "capacity": self.capacity,
            "protected_capacity": PROTECTED_REPLAY_CAPACITY,
            "other_capacity": OTHER_REPLAY_CAPACITY,
            "protected_batch_size": PROTECTED_REPLAY_BATCH_SIZE,
            "protected_source_episode_limit": PROTECTED_SOURCE_EPISODES,
            "protected_source_scenario": PROTECTED_SOURCE_SCENARIO,
            "collection_open": self.collection_open,
            "protected": _partition_state(self._protected_transitions),
            "other": _partition_state(self._other_transitions),
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore contents and RNG state atomically after strict validation."""
        if not isinstance(state, dict):
            raise ValueError("Replay state must be a dictionary")

        stored_mode = state.get("mode", "uniform")
        if stored_mode != self.mode:
            raise ValueError("Replay treatment does not match the checkpoint")

        if self.mode == "uniform":
            required_fields = {
                "mode", "capacity", "states", "action_indices", "rewards",
                "next_states", "terminals", "next_action_masks", "rng_state",
            }
            legacy_fields = required_fields - {"mode"}
            if set(state) not in (required_fields, legacy_fields):
                raise ValueError("Replay state has unexpected fields")
            transitions = _transitions_from_arrays(
                {
                    "states": state["states"],
                    "action_indices": state["action_indices"],
                    "rewards": state["rewards"],
                    "next_states": state["next_states"],
                    "terminals": state["terminals"],
                    "next_action_masks": state["next_action_masks"],
                },
                capacity=state["capacity"],
            )
            restored_rng = _restore_rng(state["rng_state"])
            self._transitions.clear()
            self._transitions.extend(transitions)
            self._rng = restored_rng
            return

        required_fields = {
            "mode", "capacity", "protected_capacity", "other_capacity",
            "protected_batch_size", "protected_source_episode_limit",
            "protected_source_scenario", "collection_open", "protected",
            "other", "rng_state",
        }
        if set(state) != required_fields:
            raise ValueError("Protected replay state has unexpected fields")
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity does not match configuration")
        if state["protected_capacity"] != PROTECTED_REPLAY_CAPACITY:
            raise ValueError("Protected replay capacity is incompatible")
        if state["other_capacity"] != OTHER_REPLAY_CAPACITY:
            raise ValueError("Other replay capacity is incompatible")
        if state["protected_batch_size"] != PROTECTED_REPLAY_BATCH_SIZE:
            raise ValueError("Protected replay quota is incompatible")
        if state["protected_source_episode_limit"] != PROTECTED_SOURCE_EPISODES:
            raise ValueError("Protected replay source limit is incompatible")
        if state["protected_source_scenario"] != PROTECTED_SOURCE_SCENARIO:
            raise ValueError("Protected replay source scenario is incompatible")
        if not isinstance(state["collection_open"], bool):
            raise ValueError("Protected replay collection state must be a bool")

        protected = _transitions_from_arrays(
            state["protected"], capacity=PROTECTED_REPLAY_CAPACITY
        )
        other = _transitions_from_arrays(
            state["other"], capacity=OTHER_REPLAY_CAPACITY
        )
        restored_rng = _restore_rng(state["rng_state"])
        self._protected_transitions.clear()
        self._protected_transitions.extend(protected)
        self._other_transitions.clear()
        self._other_transitions.extend(other)
        self._collection_open = state["collection_open"]
        self._rng = restored_rng


def _partition_state(transitions: deque[Transition]) -> dict[str, Any]:
    batch = _transitions_to_batch(list(transitions))
    return {
        "states": batch.states.copy(),
        "action_indices": batch.action_indices.copy(),
        "rewards": batch.rewards.copy(),
        "next_states": batch.next_states.copy(),
        "terminals": batch.terminals.copy(),
        "next_action_masks": batch.next_action_masks.copy(),
    }


def _transitions_from_arrays(value: Any, *, capacity: int) -> list[Transition]:
    if not isinstance(value, dict):
        raise ValueError("Replay partition must be a dictionary")
    required = {
        "states", "action_indices", "rewards", "next_states", "terminals",
        "next_action_masks",
    }
    if set(value) != required:
        raise ValueError("Replay partition has unexpected fields")

    states = _require_array(value["states"], name="states", dtype=np.float32)
    action_indices = _require_array(
        value["action_indices"], name="action_indices", dtype=np.int64
    )
    rewards = _require_array(value["rewards"], name="rewards", dtype=np.float32)
    next_states = _require_array(
        value["next_states"], name="next_states", dtype=np.float32
    )
    terminals = _require_array(value["terminals"], name="terminals", dtype=np.bool_)
    next_action_masks = _require_array(
        value["next_action_masks"], name="next_action_masks", dtype=np.bool_
    )
    count = states.shape[0]
    if states.shape != (count, FEATURE_COUNT):
        raise ValueError("Replay states have an incompatible shape")
    if action_indices.shape != (count,) or rewards.shape != (count,):
        raise ValueError("Replay scalar arrays have an incompatible shape")
    if next_states.shape != (count, FEATURE_COUNT):
        raise ValueError("Replay next states have an incompatible shape")
    if terminals.shape != (count,):
        raise ValueError("Replay terminal flags have an incompatible shape")
    if next_action_masks.shape != (count, len(ACTIONS)):
        raise ValueError("Replay next action masks have an incompatible shape")
    if count > capacity:
        raise ValueError("Replay partition exceeds configured capacity")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(next_states)):
        raise ValueError("Replay states must be finite")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Replay rewards must be finite")
    if np.any(action_indices < 0) or np.any(action_indices >= len(ACTIONS)):
        raise ValueError("Replay contains an invalid action index")
    if count and np.any(next_states[terminals] != 0.0):
        raise ValueError("Terminal replay entries require zero next-state placeholders")
    if count and not np.all(next_action_masks.any(axis=1)):
        raise ValueError("Replay next action masks require a legal action")

    return [
        Transition(
            state=states[index].copy(),
            action_index=int(action_indices[index]),
            reward=float(rewards[index]),
            next_state=None if bool(terminals[index]) else next_states[index].copy(),
            terminal=bool(terminals[index]),
            next_action_mask=(
                None if bool(terminals[index]) else next_action_masks[index].copy()
            ),
        )
        for index in range(count)
    ]


def _copy_state(state: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and defensively copy one normalized feature vector."""
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (FEATURE_COUNT,):
        raise ValueError(f"{name} must have shape ({FEATURE_COUNT},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values.copy()


def _copy_action_mask(mask: np.ndarray | None) -> np.ndarray:
    """Copy an optional next-state mask; absent masks retain all actions."""
    if mask is None:
        return np.ones(len(ACTIONS), dtype=np.bool_)
    values = np.asarray(mask)
    if values.shape != (len(ACTIONS),) or values.dtype != np.bool_:
        raise ValueError("next_action_mask must be a boolean action vector")
    if not np.any(values):
        raise ValueError("next_action_mask must retain a legal action")
    return values.copy()


def _transitions_to_batch(
    transitions: list[Transition],
    *,
    protected_flags: list[bool] | np.ndarray | None = None,
) -> ReplayBatch:
    """Convert an ordered transition collection into array form."""
    if protected_flags is None:
        protected_flags = [False] * len(transitions)
    if len(protected_flags) != len(transitions):
        raise ValueError("Protected flags must match transition count")
    if not transitions:
        return ReplayBatch(
            states=np.empty((0, FEATURE_COUNT), dtype=np.float32),
            action_indices=np.empty(0, dtype=np.int64),
            rewards=np.empty(0, dtype=np.float32),
            next_states=np.empty((0, FEATURE_COUNT), dtype=np.float32),
            terminals=np.empty(0, dtype=np.bool_),
            next_action_masks=np.empty((0, len(ACTIONS)), dtype=np.bool_),
            protected_flags=np.empty(0, dtype=np.bool_),
        )
    return ReplayBatch(
        states=np.stack([item.state for item in transitions]).astype(np.float32, copy=False),
        action_indices=np.asarray([item.action_index for item in transitions], dtype=np.int64),
        rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
        next_states=np.stack(
            [
                np.zeros(FEATURE_COUNT, dtype=np.float32)
                if item.next_state is None
                else item.next_state
                for item in transitions
            ]
        ).astype(np.float32, copy=False),
        terminals=np.asarray([item.terminal for item in transitions], dtype=np.bool_),
        next_action_masks=np.stack(
            [
                np.ones(len(ACTIONS), dtype=np.bool_)
                if item.next_action_mask is None
                else item.next_action_mask
                for item in transitions
            ]
        ).astype(np.bool_, copy=False),
        protected_flags=np.asarray(protected_flags, dtype=np.bool_),
    )


def _require_array(value: Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    """Require an array with an exact persistence dtype."""
    if not isinstance(value, np.ndarray):
        raise ValueError(f"Replay {name} must be a NumPy array")
    if value.dtype != dtype:
        raise ValueError(f"Replay {name} has an incompatible dtype")
    return value


def _restore_rng(value: Any) -> np.random.Generator:
    if not isinstance(value, dict):
        raise ValueError("Replay RNG state must be a dictionary")
    rng = np.random.default_rng()
    try:
        rng.bit_generator.state = deepcopy(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Replay RNG state is invalid") from error
    return rng
