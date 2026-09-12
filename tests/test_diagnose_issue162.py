"""Contract checks for the fixed post-Double-DQN diagnostic."""

import copy
import json

import pytest

from scripts.diagnose_issue162 import CONFIG, ROOT, instrument_source, summarize


def row():
    features = [0] * 39
    features[30] = features[31] = 1
    return {
        "features": features,
        "state": {"others": [["opponent", 0, True, [3, 1]]], "self": ["learner", 0, True, [1, 1]]},
        "next_position": [2, 1],
        "legal_mask": [True] * 6,
        "events": ["BOMB_DROPPED"],
        "reward_components": {"WASTEFUL_BOMB_PLACED": -0.5},
        "q_values": [2, 1, 0, 0, 0, 1],
    }


def test_attack_penalty_and_q_gap():
    result = summarize([row()])
    assert result["penalized_safe_attack_bombs"] == 1
    assert result["available_attack_bomb_q_gaps"] == [1]
    assert result["opponent_manhattan_distances"] == [2]
    assert result["approach_steps"] == 1
    assert result["crate_free_safe_attack_steps"] == 1


@pytest.mark.parametrize("change", ["unavailable", "illegal", "unsafe", "not_dropped"])
def test_distinguish_opportunity_availability_and_actual_event(change):
    value = row()
    if change == "unavailable":
        value["state"]["self"][2] = False
    elif change == "illegal":
        value["legal_mask"][5] = False
    elif change == "unsafe":
        value["features"][31] = 0
    else:
        value["events"] = []
    before = copy.deepcopy(value)
    result = summarize([value])
    assert value == before
    if change != "not_dropped":
        assert result["available_safe_attack_steps"] == 0
    if change in {"unsafe", "not_dropped"}:
        assert result["penalized_safe_attack_bombs"] == 0


def test_solo_and_schema():
    value = row()
    value["features"] = [0] * 39
    value["state"]["others"] = []
    result = summarize([value])
    assert result["opponent_steps"] == 0
    assert result["opponent_manhattan_distances"] == []
    value["features"] = [0] * 34
    with pytest.raises(ValueError, match="schema"):
        summarize([value])


def test_registered_matrix_and_pins(monkeypatch):
    config = json.loads(CONFIG.read_text())
    assert config["training_episodes"] == 0
    assert config["checkpoint_selection"] is None
    assert len(config["replicas"]) * len(config["cases"]) == config["max_episodes"] == 15
    assert [r["replica"] for r in config["replicas"]] == [f"r{i}" for i in range(1, 6)]
    assert [c["world_seed"] for c in config["cases"]] == [1501101, 1501102, 1501201]
    assert config["wall_seconds"] == config["cpu_seconds"] == 1800
    original = (
        b"world_seed in {1460001, 1460002} and agent_seed == world_seed + 1000000\n"
        b"Diagnostic altered checkpoint"
    )
    monkeypatch.setattr(
        "scripts.diagnose_issue162.subprocess.check_output", lambda *a, **k: original
    )
    source = instrument_source(ROOT)
    assert "world_seed in {1501101, 1501102, 1501201}" in source
    assert "Diagnostic altered checkpoint" in source
