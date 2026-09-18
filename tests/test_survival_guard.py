"""The survival guard refuses only what our danger model calls unsurvivable."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
guard_module = importlib.import_module("agent_code.Bomb-omb-survivalguard.survival_guard")
SurvivalGuard = guard_module.SurvivalGuard
guard_modes = guard_module.guard_modes
ACTIONS = guard_module.ACTIONS

WALL, FREE, CRATE = -1, 0, 1


def arena(width=7, height=7):
    """Open floor inside a solid border, the shape the framework builds."""
    field = np.full((width, height), FREE, dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = WALL
    return field


def dead_end(length=3):
    """A corridor of `length` open tiles against the left border, walled off.

    A bomb dropped at the closed end fills the whole corridor, because the blast
    reaches three tiles and the far wall stops it from venting. Walking out is
    still perfectly safe, which is exactly the situation the bomb veto is for.
    """
    field = np.full((length + 3, 5), WALL, dtype=int)
    for x in range(1, length + 1):
        field[x, 2] = FREE
    return field


def state(field, position, bombs=(), others=(), bomb_available=True, step=1):
    return {
        "field": field,
        "bombs": [(tuple(p), t) for p, t in bombs],
        "explosion_map": np.zeros_like(field),
        "coins": [],
        "others": [("other", 0, True, tuple(p)) for p in others],
        "self": ("me", 0, bomb_available, tuple(position)),
        "step": step,
        "round": 1,
        "user_input": None,
    }


def q_ranking(order):
    """Q values that rank the named actions highest, in the given order."""
    values = np.zeros(len(ACTIONS))
    for rank, action in enumerate(order):
        values[ACTIONS.index(action)] = 10.0 - rank
    return lambda: values


def legal(*actions):
    mask = np.zeros(len(ACTIONS), dtype=bool)
    for action in actions:
        mask[ACTIONS.index(action)] = True
    return mask


def test_both_switches_off_never_changes_the_action():
    field = arena()
    # A corridor with one entrance: bombing here is certain death.
    field[1, 2] = field[2, 1] = field[2, 2] = WALL
    guard = SurvivalGuard(veto_bomb=False, veto_move=False)
    assert not guard.active
    chosen = guard.choose(state(field, (1, 1)), "BOMB", q_ranking(["BOMB"]), legal("BOMB"))
    assert chosen == "BOMB"
    assert guard.vetoes == 0


def test_a_bomb_with_no_escape_is_refused_for_the_next_best_action():
    field = arena()
    field[1, 2] = field[2, 1] = field[2, 2] = WALL
    guard = SurvivalGuard(veto_bomb=True, veto_move=False)
    game_state = state(field, (1, 1))
    chosen = guard.choose(game_state, "BOMB", q_ranking(["BOMB", "WAIT"]), legal("BOMB", "WAIT"))
    assert chosen == "WAIT"
    assert guard.vetoes == 1


def test_a_bomb_with_an_escape_is_left_alone():
    field = arena(11, 11)
    guard = SurvivalGuard(veto_bomb=True, veto_move=False)
    game_state = state(field, (5, 5))
    chosen = guard.choose(game_state, "BOMB", q_ranking(["BOMB"]), legal("BOMB", "UP", "LEFT"))
    assert chosen == "BOMB"
    assert guard.vetoes == 0


def test_the_move_switch_alone_leaves_a_hopeless_bomb_alone():
    field = arena()
    field[1, 2] = field[2, 1] = field[2, 2] = WALL
    guard = SurvivalGuard(veto_bomb=False, veto_move=True)
    chosen = guard.choose(state(field, (1, 1)), "BOMB", q_ranking(["BOMB"]), legal("BOMB", "WAIT"))
    assert chosen == "BOMB"


def test_a_move_into_a_certain_blast_is_refused():
    field = arena(11, 11)
    # A bomb about to go off one tile to the right; stepping onto it is fatal
    # and so is standing in its blast, but the left corridor is clear.
    guard = SurvivalGuard(veto_bomb=False, veto_move=True)
    game_state = state(field, (5, 5), bombs=[((7, 5), 0)])
    chosen = guard.choose(game_state, "RIGHT", q_ranking(["RIGHT", "LEFT"]),
                          legal("RIGHT", "LEFT", "UP", "DOWN", "WAIT"))
    assert chosen != "RIGHT"


def test_the_policy_keeps_its_action_when_nothing_is_survivable():
    field = arena(5, 5)
    # Boxed in with a bomb on top of us: every action dies, so the guard must
    # not pretend it knows better.
    guard = SurvivalGuard(veto_bomb=True, veto_move=True)
    field[1, 2] = field[2, 1] = WALL
    game_state = state(field, (1, 1), bombs=[((1, 1), 0)], bomb_available=False)
    chosen = guard.choose(game_state, "WAIT", q_ranking(["WAIT", "UP"]), legal("WAIT"))
    assert chosen == "WAIT"
    assert guard.fallbacks == 1
    assert guard.vetoes == 0


def test_the_replacement_follows_the_policys_own_ranking():
    guard = SurvivalGuard(veto_bomb=True, veto_move=True)
    game_state = state(dead_end(), (1, 2))
    chosen = guard.choose(game_state, "BOMB", q_ranking(["BOMB", "RIGHT", "WAIT"]),
                          legal("BOMB", "RIGHT", "WAIT"))
    assert chosen == "RIGHT"
    assert guard.vetoes == 1


def test_an_illegal_action_is_never_chosen_as_the_replacement():
    guard = SurvivalGuard(veto_bomb=True, veto_move=True)
    game_state = state(dead_end(), (1, 2))
    chosen = guard.choose(game_state, "BOMB", q_ranking(["BOMB", "RIGHT", "WAIT"]),
                          legal("BOMB", "WAIT"))
    assert chosen == "WAIT"


@pytest.mark.parametrize(
    "environ,expected",
    [
        ({}, (False, False)),
        ({"BOMBERMAN_SURVIVAL_GUARD_BOMB": "on"}, (True, False)),
        ({"BOMBERMAN_SURVIVAL_GUARD_MOVE": "on"}, (False, True)),
        ({"BOMBERMAN_SURVIVAL_GUARD_BOMB": "on", "BOMBERMAN_SURVIVAL_GUARD_MOVE": "on"}, (True, True)),
        ({"BOMBERMAN_SURVIVAL_GUARD_BOMB": "off"}, (False, False)),
    ],
)
def test_the_switches_map_to_the_four_cells(environ, expected):
    assert guard_modes(environ) == expected


@pytest.mark.parametrize("value", ["ON", "true", "1", "yes", ""])
def test_a_misspelled_switch_is_rejected_instead_of_silently_off(value):
    with pytest.raises(ValueError):
        guard_modes({"BOMBERMAN_SURVIVAL_GUARD_BOMB": value})


def test_the_prototype_differs_from_the_shipped_agent_only_in_the_guard():
    # The control cell has to reproduce the agent we ship, or the whole
    # factorial measures the wrong difference.
    shipped = ROOT / "agent_code/Bomb-omb"
    prototype = ROOT / "agent_code/Bomb-omb-survivalguard"
    allowed = {"callbacks.py", "survival_guard.py", "attack_rule.py"}
    for path in prototype.rglob("*"):
        # logs/ and __pycache__/ are written by playing, and are gitignored.
        if path.is_dir() or {"__pycache__", "logs"} & set(path.relative_to(prototype).parts):
            continue
        relative = path.relative_to(prototype).as_posix()
        if relative in allowed or relative.startswith("eval-"):
            continue
        assert path.read_bytes() == (shipped / relative).read_bytes(), relative


def test_the_hook_returns_before_the_guard_when_both_switches_are_off():
    # The cheapest possible guarantee that the control cell costs nothing and
    # changes nothing: act() must not reach the guard, and training must not
    # reach either add-on.
    source = (ROOT / "agent_code/Bomb-omb-survivalguard/callbacks.py").read_text(encoding="utf-8")
    assert "if self.train:\n        return action" in source
    assert "if guard is None or not guard.active:\n        return action" in source
