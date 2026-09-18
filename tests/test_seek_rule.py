"""The seek rule only fires on an empty board, and only onto a safe step."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
module = importlib.import_module("agent_code.Bomb-omb-survivalguard.seek_rule")
SeekRule = module.SeekRule
seek_mode = module.seek_mode

WALL, FREE, CRATE = -1, 0, 1


def arena(width=15, height=15):
    field = np.full((width, height), FREE, dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = WALL
    return field


def state(field, position, others=(), coins=(), bombs=(), bomb_available=True):
    return {
        "field": field,
        "bombs": [(tuple(p), t) for p, t in bombs],
        "explosion_map": np.zeros_like(field),
        "coins": [tuple(c) for c in coins],
        "others": [("other", 0, True, tuple(p)) for p in others],
        "self": ("me", 0, bomb_available, tuple(position)),
        "step": 40,
        "round": 1,
        "user_input": None,
    }


def test_it_walks_toward_a_far_opponent_on_a_cleared_board():
    rule = SeekRule(enabled=True)
    game_state = state(arena(), (2, 2), others=[(12, 2)])
    assert rule.choose(game_state, "LEFT") == "RIGHT"
    assert rule.steps_taken == 1


def test_it_stays_out_of_the_way_while_crates_remain():
    field = arena()
    field[5, 5] = CRATE
    rule = SeekRule(enabled=True)
    game_state = state(field, (2, 2), others=[(12, 2)])
    assert rule.choose(game_state, "LEFT") == "LEFT"


def test_a_visible_coin_outranks_hunting():
    rule = SeekRule(enabled=True)
    game_state = state(arena(), (2, 2), others=[(12, 2)], coins=[(3, 3)])
    assert rule.choose(game_state, "LEFT") == "LEFT"


def test_it_does_nothing_with_no_opponent_left():
    rule = SeekRule(enabled=True)
    game_state = state(arena(), (2, 2))
    assert rule.choose(game_state, "LEFT") == "LEFT"


def test_it_stops_once_it_can_already_shoot():
    # Two tiles away in a straight line is inside the blast, so we are already
    # standing on a firing position and the attack rule owns the decision.
    rule = SeekRule(enabled=True)
    game_state = state(arena(), (2, 2), others=[(4, 2)])
    assert rule.choose(game_state, "LEFT") == "LEFT"
    assert rule.steps_taken == 0


def test_it_never_overrides_a_bomb():
    rule = SeekRule(enabled=True)
    game_state = state(arena(), (2, 2), others=[(12, 2)])
    assert rule.choose(game_state, "BOMB") == "BOMB"


def test_it_routes_around_a_bomb_rather_than_through_it():
    # A live bomb sits on the direct line to the opponent. Seeking must not walk
    # into it; going around is the right answer and also gets us out of its row.
    rule = SeekRule(enabled=True)
    game_state = state(arena(), (2, 2), others=[(12, 2)], bombs=[((4, 2), 0)])
    assert rule.choose(game_state, "LEFT") in {"UP", "DOWN"}


def test_it_gives_up_when_every_route_is_unsurvivable():
    # A one-tile corridor with a bomb about to go off in it: there is nowhere to
    # step that survives, so the policy's choice stands.
    field = np.full((15, 5), WALL, dtype=int)
    for x in range(1, 14):
        field[x, 2] = FREE
    rule = SeekRule(enabled=True)
    game_state = state(field, (2, 2), others=[(12, 2)], bombs=[((5, 2), 0)])
    assert rule.choose(game_state, "LEFT") == "LEFT"
    assert rule.steps_taken == 0
    assert rule.no_route == 1


def test_switched_off_it_changes_nothing():
    rule = SeekRule(enabled=False)
    game_state = state(arena(), (2, 2), others=[(12, 2)])
    assert rule.choose(game_state, "LEFT") == "LEFT"
    assert not rule.active


def test_an_unreachable_opponent_leaves_the_policy_alone():
    field = arena()
    field[7, :] = WALL  # a wall splitting the board in two
    rule = SeekRule(enabled=True)
    game_state = state(field, (2, 2), others=[(12, 2)])
    assert rule.choose(game_state, "LEFT") == "LEFT"


@pytest.mark.parametrize("value,expected", [({}, False), ({"BOMBERMAN_SEEK_RULE": "on"}, True),
                                            ({"BOMBERMAN_SEEK_RULE": "off"}, False)])
def test_the_switch_maps_to_the_cells(value, expected):
    assert seek_mode(value) is expected


@pytest.mark.parametrize("value", ["ON", "true", "1", ""])
def test_a_misspelled_switch_is_rejected(value):
    with pytest.raises(ValueError):
        seek_mode({"BOMBERMAN_SEEK_RULE": value})


def test_the_prototype_still_differs_from_the_shipped_agent_only_in_the_add_ons():
    shipped = ROOT / "agent_code/Bomb-omb"
    prototype = ROOT / "agent_code/Bomb-omb-survivalguard"
    allowed = {"callbacks.py", "survival_guard.py", "attack_rule.py", "seek_rule.py"}
    for path in prototype.rglob("*"):
        if path.is_dir() or {"__pycache__", "logs"} & set(path.relative_to(prototype).parts):
            continue
        relative = path.relative_to(prototype).as_posix()
        if relative in allowed or relative.startswith(("eval-", "watch-", "stall-")):
            continue
        assert path.read_bytes() == (shipped / relative).read_bytes(), relative
