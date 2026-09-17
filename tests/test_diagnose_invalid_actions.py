"""The invalid-action diagnosis names causes correctly and measures pacing honestly."""

from __future__ import annotations

import json

from scripts import diagnose_invalid_actions as diag

BASE = dict(action="RIGHT", allowed=True, start=(3, 3), occupant_start=None,
            occupant_acted_first=False, bomb_on_target=False, arena_blocked=False)


def test_an_opponent_stepping_into_a_free_tile_first_is_a_contested_tile():
    reason = diag.classify_invalid(**{**BASE, "occupant_start": (5, 3), "occupant_acted_first": True})
    assert reason == "contested_tile_opponent_moved_in_first"


def test_an_opponent_already_standing_there_is_not_a_contest():
    # The mask would have forbidden that move, so this must not be explained away.
    reason = diag.classify_invalid(**{**BASE, "occupant_start": (4, 3), "occupant_acted_first": True})
    assert reason == "unexplained"


def test_a_move_the_mask_forbade_is_reported_as_a_mask_failure_first():
    reason = diag.classify_invalid(**{**BASE, "allowed": False, "occupant_start": (5, 3),
                                      "occupant_acted_first": True})
    assert reason == "mask_disallowed_it"


def test_reversals_flag_only_moves_that_undo_the_previous_move():
    actions = ["UP", "DOWN", "UP", "LEFT", "RIGHT", "WAIT", "LEFT"]
    assert diag.reversals(actions) == [False, True, True, False, True, False, False]


def test_report_separates_pacing_alone_from_pacing_with_opponents(tmp_path, capsys):
    def step(seed, n, action, alive, invalid=False, reason=None):
        return {"seed": seed, "step": n, "action": action, "move": action != "WAIT",
                "invalid": invalid, "reason": reason, "nearest_opponent": 1 if alive else None,
                "opponents_alive": alive}

    steps = [step(1, 1, "UP", 2), step(1, 2, "DOWN", 2, True, "contested_tile_opponent_moved_in_first"),
             step(1, 3, "LEFT", 0), step(1, 4, "RIGHT", 0), step(1, 5, "LEFT", 0)]
    recording = tmp_path / "rec.json"
    recording.write_text(json.dumps({"suite": "classic-rule-based", "steps": steps}), encoding="utf-8")

    diag.report(type("Args", (), {"recordings": [str(recording)]})())
    row = json.loads(capsys.readouterr().out.strip())
    assert row["invalid_reasons"] == {"contested_tile_opponent_moved_in_first": 1}
    assert row["reversal_share_opponents_alive"] == 0.5   # UP then DOWN
    assert row["reversal_share_alone"] == round(2 / 3, 4)  # LEFT, RIGHT, LEFT
    assert row["invalid_followed_by_death_within_5"] == 1
