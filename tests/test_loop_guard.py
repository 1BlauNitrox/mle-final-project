"""The play-time loop guard only redirects a stuck, calm agent, and only safely."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
guard_module = importlib.import_module("agent_code.Bomb-omb-loopguard.loop_guard")
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def open_board(size=9):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return field


def state(position, step, field=None, bombs=()):
    field = open_board() if field is None else field
    return {
        "field": field,
        "bombs": list(bombs),
        "explosion_map": np.zeros(field.shape),
        "self": ("me", 0, True, position),
        "others": [],
        "step": step,
    }


def pacing_guard(field=None, bombs=()):
    guard = guard_module.LoopGuard()
    for step in range(1, 9):
        guard.observe(state((4, 4) if step % 2 else (4, 5), step, field, bombs))
    return guard


def q(**values):
    return lambda: [values.get(action, 0.0) for action in ACTIONS]


LEGAL = np.ones(6, dtype=bool)


def test_a_moving_agent_is_left_alone():
    guard = guard_module.LoopGuard()
    for step in range(1, 9):
        guard.observe(state((1 + step % 7, 4), step))
    assert guard.choose(state((1, 4), 9), "RIGHT", q(LEFT=5), LEGAL) == "RIGHT"
    assert guard.overrides == 0


def test_a_pacing_agent_takes_its_best_valued_move_to_an_unvisited_tile():
    guard = pacing_guard()
    # At (4,5) having paced with (4,4): UP would continue the loop.
    chosen = guard.choose(state((4, 5), 9), "UP", q(UP=9, RIGHT=1, DOWN=3, LEFT=2), LEGAL)
    assert chosen == "DOWN"
    assert guard.overrides == 1


def test_waiting_and_bombing_are_never_overridden():
    guard = pacing_guard()
    assert guard.choose(state((4, 5), 9), "WAIT", q(RIGHT=9), LEGAL) == "WAIT"
    assert guard.choose(state((4, 5), 9), "BOMB", q(RIGHT=9), LEGAL) == "BOMB"


def test_a_nearby_bomb_means_dodging_is_left_to_the_policy():
    bombs = [((4, 8), 3)]
    guard = pacing_guard(bombs=bombs)
    assert guard.choose(state((4, 5), 9, bombs=bombs), "UP", q(DOWN=9), LEGAL) == "UP"
    assert guard.overrides == 0


def test_no_override_when_every_alternative_is_blocked():
    field = open_board()
    field[3, 5] = field[5, 5] = field[4, 6] = -1  # walls left, right and below (4,5)
    guard = pacing_guard(field=field)
    assert guard.choose(state((4, 5), 9, field=field), "UP", q(DOWN=9), LEGAL) == "UP"


def test_history_resets_at_the_start_of_a_round():
    guard = pacing_guard()
    guard.observe(state((2, 2), 1))
    assert not guard.stuck()


def test_the_prototype_differs_from_the_trained_agent_only_in_the_guard():
    trained = ROOT / "agent_code/DagobertDuckDQNTask3"
    prototype = ROOT / "agent_code/Bomb-omb-loopguard"
    assert not (prototype / "train.py").exists()
    allowed = {"callbacks.py", "loop_guard.py", "checkpoint.pt"}
    for path in prototype.rglob("*"):
        # logs/ and __pycache__/ are written by playing, and are gitignored.
        if path.is_dir() or {"__pycache__", "logs"} & set(path.relative_to(prototype).parts):
            continue
        relative = path.relative_to(prototype).as_posix()
        if relative in allowed:
            continue
        assert path.read_bytes() == (trained / relative).read_bytes(), relative
