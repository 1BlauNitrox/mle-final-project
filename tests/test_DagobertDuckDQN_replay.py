"""Tests for the DagobertDuckDQN replay buffer"""

import numpy as np
import pytest

from agent_code.DagobertDuckDQN.replay import ReplayBuffer


def make_state(value: float) -> np.ndarray:
    """Create one synthetic normalized state"""
    return np.full(8, value, dtype=np.float32)

def add_transition(
    buffer: ReplayBuffer,
    value: float,
    *,
    terminal: bool = False,
) -> None:
    """Add one distinguishable transition"""
    buffer.add(
        state=make_state(value),
        action_index=int(value) % 5,
        reward=value,
        next_state=None if terminal else make_state(value + 0.1),
        terminal=terminal,
    )

def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0, seed=1)

def test_new_buffer_is_empty() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)

    assert len(buffer) == 0
    assert buffer.capacity == 3

def test_add_increases_buffer_size() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)

    add_transition(buffer, 0.0)

    assert len(buffer) == 1

def test_capacity_discards_oldest_transitions() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)

    for value in range(5):
        add_transition(buffer, float(value))

    batch = buffer.sample(3)

    assert len(buffer) == 3
    assert set(batch.rewards.tolist()) == {2.0, 3.0, 4.0}

def test_sample_has_expected_shapes_and_dtypes() -> None:
    buffer = ReplayBuffer(capacity=5, seed=1)

    add_transition(buffer, 0.0)
    add_transition(buffer, 1.0, terminal=True)

    batch = buffer.sample(2)

    assert batch.states.shape == (2,8)
    assert batch.states.dtype == np.float32

    assert batch.action_indices.shape == (2,)
    assert batch.action_indices.dtype == np.int64

    assert batch.rewards.shape == (2,)
    assert batch.rewards.dtype == np.float32

    assert batch.next_states.shape == (2, 8)
    assert batch.next_states.dtype == np.float32

    assert batch.terminals.shape == (2,)
    assert batch.terminals.dtype == np.bool_

def test_terminal_next_state_is_represented_by_zeros_in_batch() -> None:
    buffer = ReplayBuffer(capacity=2, seed=1)
    add_transition(buffer, 1.0, terminal=True)

    batch = buffer.sample(1)

    assert batch.terminals[0]
    np.testing.assert_array_equal(
        batch.next_states[0],
        np.zeros(8, dtype=np.float32),
    )

def test_sampling_is_reproducible_for_equal_seeds() -> None:
    first = ReplayBuffer(capacity=10, seed=42069)
    second = ReplayBuffer(capacity=10, seed=42069)

    for value in range(10):
        add_transition(first, float(value))
        add_transition(second, float(value))

    first_batch = first.sample(4)
    second_batch = second.sample(4)

    np.testing.assert_array_equal(first_batch.states, second_batch.states)
    np.testing.assert_array_equal(first_batch.action_indices, second_batch.action_indices)
    np.testing.assert_array_equal(first_batch.rewards, second_batch.rewards)

def test_sampling_too_many_transitions_is_rejected() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)
    add_transition(buffer, 0.0)

    with pytest.raises(ValueError):
        buffer.sample(2)

def test_non_terminal_transition_requires_next_state() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)

    with pytest.raises(ValueError):
        buffer.add(
            state=make_state(0.0),
            action_index=0,
            reward=0.0,
            next_state=None,
            terminal=False
        )

def test_terminal_transition_rejects_next_state() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)

    with pytest.raises(ValueError):
        buffer.add(
            state=make_state(0.0),
            action_index=0,
            reward=0.0,
            next_state=make_state(0.1),
            terminal=True
        )

@pytest.mark.parametrize("action_index", [-1, 5])
def test_invalid_action_index_is_rejected(action_index: int) -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)

    with pytest.raises(ValueError):
        buffer.add(
            state=make_state(0.0),
            action_index=action_index,
            reward=0.0,
            next_state=make_state(0.1),
            terminal=False
        )

def test_wrong_state_shape_is_rejected() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)

    with pytest.raises(ValueError):
        buffer.add(
            state=np.zeros(7, dtype=np.float32),
            action_index=0,
            reward=0.0,
            next_state=make_state(0.1),
            terminal=False
        )

def test_non_finite_values_are_rejected() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)
    state = make_state(0.0)
    state[3] = np.nan

    with pytest.raises(ValueError):
        buffer.add(
            state=state,
            action_index=0,
            reward=0.0,
            next_state=make_state(0.1),
            terminal=False
        )

def test_added_states_are_copied() -> None:
    buffer = ReplayBuffer(capacity=3, seed=1)
    state = make_state(0.0)
    next_state = make_state(0.1)

    buffer.add(
        state=state,
        action_index=0,
        reward=0.0,
        next_state=next_state,
        terminal=False
    )

    state[:] = 99.0
    next_state[:] = 99.0

    batch = buffer.sample(1)

    np.testing.assert_array_equal(
        batch.states[0],
        make_state(0.0)
    )
    np.testing.assert_array_equal(
        batch.next_states[0],
        make_state(0.1)
    )

def assert_batches_equal(first, second) -> None:
    """Assert two sampled replay batches are identical"""
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.action_indices, second.action_indices)
    np.testing.assert_array_equal(first.rewards, second.rewards)
    np.testing.assert_array_equal(first.next_states, second.next_states)
    np.testing.assert_array_equal(first.terminals, second.terminals)

def test_state_dict_round_trip_restores_data_and_rng() -> None:
    original = ReplayBuffer(capacity=10, seed=43069)

    for value in range(6):
        add_transition(
            original,
            float(value),
            terminal=value==5,
        )

    saved_state = original.state_dict()

    restored = ReplayBuffer(capacity=10, seed=999)
    restored.load_state_dict(saved_state)

    assert len(restored) == len(original)

    expected_batch = original.sample(4)
    restored_batch = restored.sample(4)

    assert_batches_equal(expected_batch, restored_batch)

def test_empty_buffer_round_trip() -> None:
    original = ReplayBuffer(capacity=5, seed=42069)
    restored = ReplayBuffer(capacity=5, seed=999)

    restored.load_state_dict(original.state_dict())

    assert len(restored) == 0

def test_state_dict_capacity_mismatch_is_rejected() -> None:
    original = ReplayBuffer(capacity=5, seed=42069)
    saved_state = original.state_dict()
    saved_state["capacity"] = 6

    restored = ReplayBuffer(capacity=5, seed=999)

    with pytest.raises(ValueError):
        restored.load_state_dict(saved_state)

def test_state_dict_with_unexpected_fields_is_rejected() -> None:
    buffer = ReplayBuffer(capacity=5, seed=42069)
    saved_state = buffer.state_dict()
    saved_state["unexpected"] = "value"

    with pytest.raises(ValueError):
        buffer.load_state_dict(saved_state)

def test_state_dict_returns_defensive_array_copies() -> None:
    buffer = ReplayBuffer(capacity=5, seed=42069)
    add_transition(buffer, 0.25)

    saved_state = buffer.state_dict()
    saved_states = saved_state["states"]

    assert isinstance(saved_states, np.ndarray)
    saved_states[:] = 99.0

    batch = buffer.sample(1)

    np.testing.assert_array_equal(
        batch.states[0],
        make_state(0.25)
    )

def test_terminal_state_dict_requires_zero_placeholder() -> None:
    source = ReplayBuffer(capacity=5, seed=42069)
    add_transition(source, 1.0, terminal=True)

    saved_state = source.state_dict()
    saved_next_states = saved_state["next_states"]

    assert isinstance(saved_next_states, np.ndarray)
    saved_next_states[0, 0] = 1.0

    restored = ReplayBuffer(capacity=5, seed=999)

    with pytest.raises(ValueError):
        restored.load_state_dict(saved_state)
