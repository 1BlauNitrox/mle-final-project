"""The attack rule bombs an opponent in range, and only when we can get out."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
module = importlib.import_module("agent_code.Bomb-omb-survivalguard.attack_rule")
AttackRule = module.AttackRule
attack_mode = module.attack_mode

WALL, FREE, CRATE = -1, 0, 1


def arena(width=11, height=11):
    field = np.full((width, height), FREE, dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = WALL
    return field


def dead_end(length=3):
    """A corridor against the left border: a bomb here fills it completely."""
    field = np.full((length + 3, 5), WALL, dtype=int)
    for x in range(1, length + 1):
        field[x, 2] = FREE
    return field


def state(field, position, others=(), bombs=(), bomb_available=True):
    return {
        "field": field,
        "bombs": [(tuple(p), t) for p, t in bombs],
        "explosion_map": np.zeros_like(field),
        "coins": [],
        "others": [("other", 0, True, tuple(p)) for p in others],
        "self": ("me", 0, bomb_available, tuple(position)),
        "step": 1,
        "round": 1,
        "user_input": None,
    }


def test_it_bombs_an_opponent_standing_in_range():
    rule = AttackRule(enabled=True)
    game_state = state(arena(), (5, 5), others=[(5, 7)])
    assert rule.choose(game_state, "LEFT") == "BOMB"
    assert rule.attacks == 1


def test_it_does_nothing_when_no_opponent_is_in_the_blast():
    rule = AttackRule(enabled=True)
    # Four tiles away on an open board: outside the three-tile blast.
    game_state = state(arena(), (5, 5), others=[(5, 9)])
    assert rule.choose(game_state, "LEFT") == "LEFT"
    assert rule.attacks == 0


def test_a_wall_between_us_and_the_opponent_is_not_an_attack():
    field = arena()
    field[5, 6] = WALL
    rule = AttackRule(enabled=True)
    game_state = state(field, (5, 5), others=[(5, 7)])
    assert rule.choose(game_state, "LEFT") == "LEFT"


def test_it_refuses_an_attack_it_could_not_escape():
    # Opponent in the corridor with us: the bomb would fill the only way out.
    rule = AttackRule(enabled=True)
    game_state = state(dead_end(), (1, 2), others=[(3, 2)])
    assert rule.choose(game_state, "RIGHT") == "RIGHT"
    assert rule.attacks == 0
    assert rule.declined_no_escape == 1


def test_it_needs_a_bomb_in_hand():
    rule = AttackRule(enabled=True)
    game_state = state(arena(), (5, 5), others=[(5, 7)], bomb_available=False)
    assert rule.choose(game_state, "LEFT") == "LEFT"


def test_it_leaves_the_policys_own_bomb_alone():
    rule = AttackRule(enabled=True)
    game_state = state(arena(), (5, 5), others=[(5, 7)])
    assert rule.choose(game_state, "BOMB") == "BOMB"
    assert rule.attacks == 0  # not counted: the policy chose it, not us


def test_switched_off_it_changes_nothing():
    rule = AttackRule(enabled=False)
    game_state = state(arena(), (5, 5), others=[(5, 7)])
    assert rule.choose(game_state, "LEFT") == "LEFT"
    assert not rule.active


def test_a_crate_between_us_and_the_opponent_still_counts():
    # The framework's blast is stopped by walls only, so a crate in the way does
    # not save the opponent. blast_footprint reproduces that exactly.
    field = arena()
    field[5, 6] = CRATE
    rule = AttackRule(enabled=True)
    game_state = state(field, (5, 5), others=[(5, 7)])
    assert rule.choose(game_state, "LEFT") == "BOMB"


@pytest.mark.parametrize("value,expected", [({}, "off"), ({"BOMBERMAN_ATTACK_RULE": "on"}, "on"),
                                            ({"BOMBERMAN_ATTACK_RULE": "off"}, "off"),
                                            ({"BOMBERMAN_ATTACK_RULE": "selective"}, "selective")])
def test_the_switch_maps_to_the_cells(value, expected):
    assert attack_mode(value) == expected


@pytest.mark.parametrize("value", ["ON", "true", "1", "yes", "", "strict"])
def test_a_misspelled_switch_is_rejected(value):
    with pytest.raises(ValueError):
        attack_mode({"BOMBERMAN_ATTACK_RULE": value})


def trap():
    """A corridor with one side exit, placed so only we can reach it in time.

    Free tiles are the row (1..8, 2) plus the pocket (6, 1). A bomb at (4, 2)
    fills x = 1..7 of the row. From (4, 2) the pocket is three steps away; from
    (2, 2) it is five, which is more than the fuse allows.
    """
    field = np.full((10, 5), WALL, dtype=int)
    for x in range(1, 9):
        field[x, 2] = FREE
    field[6, 1] = FREE
    return field


def test_selective_fires_when_the_opponent_cannot_escape():
    rule = AttackRule(enabled="selective")
    game_state = state(trap(), (4, 2), others=[(2, 2)])
    assert rule.choose(game_state, "LEFT") == "BOMB"
    assert rule.attacks == 1


def test_permissive_and_selective_agree_when_the_opponent_is_trapped():
    permissive = AttackRule(enabled="on")
    assert permissive.choose(state(trap(), (4, 2), others=[(2, 2)]), "LEFT") == "BOMB"


def test_selective_holds_fire_when_the_opponent_can_walk_out():
    rule = AttackRule(enabled="selective")
    game_state = state(arena(), (5, 5), others=[(5, 7)])
    assert rule.choose(game_state, "LEFT") == "LEFT"
    assert rule.attacks == 0
    assert rule.declined_opponent_escapes == 1


def test_permissive_fires_on_the_same_escapable_opponent():
    # The difference between the two cells, on one board.
    rule = AttackRule(enabled="on")
    game_state = state(arena(), (5, 5), others=[(5, 7)])
    assert rule.choose(game_state, "LEFT") == "BOMB"


def test_selective_still_refuses_an_attack_it_could_not_escape_itself():
    rule = AttackRule(enabled="selective")
    game_state = state(dead_end(), (1, 2), others=[(3, 2)])
    assert rule.choose(game_state, "RIGHT") == "RIGHT"
    assert rule.declined_no_escape == 1


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


def test_training_never_reaches_either_add_on():
    source = (ROOT / "agent_code/Bomb-omb-survivalguard/callbacks.py").read_text(encoding="utf-8")
    assert "if self.train:\n        return action" in source
