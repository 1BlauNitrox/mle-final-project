"""The proposed intervention affects only safe confirmed crate-free opponent bombs."""

from dataclasses import replace

import numpy as np
import pytest

from agent_code.DagobertDuckDQNTask3 import train
from agent_code.DagobertDuckDQNTask3.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQNTask3.persistence import CHECKPOINT_PATH, load_training_checkpoint
from tests.test_DagobertDuckDQNTask3_opponents import make_field, make_state
from tests.test_DagobertDuckDQNTask3_successor import make_agent


def event(state, events=("BOMB_DROPPED",), mode=True):
    return train._bomb_usefulness_event(state, list(events), neutral_safe_attack_bombs=mode)


def test_confirmed_safe_attack_exempts_penalty_but_default_does_not():
    state = make_state(others=[("target", 0, True, (5, 3))])
    assert event(state) is None
    assert event(state, mode=False) == "WASTEFUL_BOMB_PLACED"
    assert event(state, events=("INVALID_ACTION",)) is None


def test_crate_reward_and_solo_behavior_unchanged():
    state = make_state()
    assert event(state) == event(state, mode=False) == "WASTEFUL_BOMB_PLACED"
    state["field"][4, 3] = 1
    state["others"] = [("target", 0, True, (3, 5))]
    assert event(state) == event(state, mode=False) == "USEFUL_BOMB_PLACED"


def test_blocked_or_unsafe_attacks_still_penalized():
    field = make_field()
    field[4, 3] = -1
    assert (
        event(make_state(field=field, others=[("target", 0, True, (5, 3))]))
        == "WASTEFUL_BOMB_PLACED"
    )
    field = np.full((7, 7), -1)
    field[3, 3] = field[4, 3] = 0
    assert (
        event(make_state(field=field, others=[("target", 0, True, (4, 3))]))
        == "WASTEFUL_BOMB_PLACED"
    )


def test_mode_defaults_and_serialization(tmp_path):
    import torch

    assert load_training_checkpoint(CHECKPOINT_PATH).config.neutral_safe_attack_bombs is False
    with pytest.raises(ValueError, match="boolean"):
        replace(DEFAULT_CONFIG, neutral_safe_attack_bombs=1)
    payload = torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu")
    payload["config"]["neutral_safe_attack_bombs"] = True
    path = tmp_path / "neutral.pt"
    torch.save(payload, path)
    assert load_training_checkpoint(path).config.neutral_safe_attack_bombs is True


@pytest.mark.parametrize("mode,expected", [(False, -0.5), (True, 0.0)])
def test_callback_applies_mode_once_and_terminal_semantics_are_unchanged(
    monkeypatch, mode, expected
):
    agent = make_agent()
    agent.config = replace(agent.config, neutral_safe_attack_bombs=mode)
    train.setup_training(agent)
    state = make_state(others=[("target", 0, True, (5, 3))])
    after = make_state(others=state["others"])
    after["step"] = 2
    train.game_events_occurred(agent, state, "BOMB", after, ["BOMB_DROPPED"])
    assert agent.pending_transition.reward == expected
    records = []
    monkeypatch.setattr(train, "_record_transition", lambda *a, **kw: records.append(kw))
    monkeypatch.setattr(train, "save_checkpoint", lambda **kw: None)
    train.end_of_round(agent, state, "BOMB", ["BOMB_DROPPED", "SURVIVED_ROUND"])
    assert len(records) == 1
    assert records[0]["reward"] == 5.0  # Existing terminal branch has no placement shaping.
    assert records[0]["terminal"]
