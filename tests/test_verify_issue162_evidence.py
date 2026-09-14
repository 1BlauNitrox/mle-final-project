"""Analysis-only regression and corruption checks for the retained diagnosis."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.verify_issue162_evidence import stall_windows, verify

RECORD = Path(__file__).resolve().parents[1] / "experiments/2026-09-12-task3-hunting-diagnosis"


def test_all_retained_diagnostic_evidence_reproduces_without_games():
    result = verify(RECORD)
    assert result["verified"] and result["new_games"] == 0
    assert len(result["episodes"]) == 15
    assert result["totals"]["available_safe_attack_steps"] == 2
    assert result["totals"]["penalized_safe_attack_bombs"] == 1
    assert result["episodes_with_stalls"] == 8
    assert result["checkpoint_selection"] is None


@pytest.mark.parametrize("kind", ["archive", "metric", "missing_job", "compact", "example"])
def test_evidence_corruption_is_rejected(tmp_path, kind):
    record = tmp_path / "record"
    shutil.copytree(RECORD, record)
    if kind == "archive":
        path = record / "trajectory-evidence.zip"
        data = bytearray(path.read_bytes())
        data[100] ^= 1
        path.write_bytes(data)
    elif kind in {"metric", "missing_job"}:
        path = record / "summary.json"
        data = json.loads(path.read_text())
        if kind == "metric":
            data["jobs"][0]["metrics"]["attack_steps"] += 1
        else:
            data["jobs"].pop()
        path.write_text(json.dumps(data))
    elif kind == "compact":
        path = record / "observations.csv"
        path.write_text(
            path.read_text().replace("r1-1501101,1,15,1,14,1", "r1-1501101,1,15,1,13,1", 1)
        )
    else:
        path = record / "attack-example.json"
        data = json.loads(path.read_text())
        data[0]["reward_components"]["WASTEFUL_BOMB_PLACED"] = 0
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify(record)


@pytest.mark.parametrize("change", ["short", "hazard", "progress", "board", "nonperiodic"])
def test_stall_exclusions_and_24_step_boundary(change):
    rows = [
        {
            "step": i + 1,
            "position": [i % 2, 0],
            "hazards": False,
            "events": [],
            "board_coins_sha256": "unchanged",
        }
        for i in range(24)
    ]
    assert stall_windows(rows) == [{"start": 1, "end": 24, "period": 2}]
    if change == "short":
        rows.pop()
    elif change == "hazard":
        rows[12]["hazards"] = True
    elif change == "progress":
        rows[12]["events"] = ["CRATE_DESTROYED"]
    elif change == "board":
        rows[12]["board_coins_sha256"] = "changed"
    else:
        rows[12]["position"] = [99, 99]
    assert not stall_windows(rows)
