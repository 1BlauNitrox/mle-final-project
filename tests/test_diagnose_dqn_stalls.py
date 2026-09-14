"""Stall diagnosis excludes bomb waiting and world progress before classifying cycles."""

from training.diagnose_dqn_stalls import stall_windows


def rows():
    return [
        {
            "step": i + 1,
            "position": [i % 2, 1],
            "events": [],
            "hazards": False,
            "board_coins_sha256": "fixed",
        }
        for i in range(24)
    ]


def test_detects_repeated_positions_with_explicit_window_and_period():
    assert stall_windows(rows()) == [{"start": 1, "end": 24, "period": 2}]
    assert stall_windows(rows()[:23]) == []


def test_rejects_hazards_progress_and_changed_world():
    for change in (
        {"hazards": True},
        {"events": ["COIN_COLLECTED"]},
        {"board_coins_sha256": "changed"},
    ):
        sample = rows()
        sample[12].update(change)
        assert stall_windows(sample) == []


def test_rejects_nonperiodic_positions():
    sample = rows()
    sample[15]["position"] = [7, 7]
    assert stall_windows(sample) == []
