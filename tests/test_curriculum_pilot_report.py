from pathlib import Path

import pytest

from scripts.curriculum_io import read
from scripts.report_curriculum_pilot import original_bytes, report


def test_original_bytes_rejects_corruption():
    with pytest.raises(ValueError, match="checksum"):
        original_bytes(b"corrupt evidence", "0" * 64)


def test_retained_pilot_reproduces_registered_rejections(tmp_path):
    directory = Path(__file__).resolve().parents[1] / (
        "experiments/2026-09-19-hunting-curriculum/evidence")
    result = report([directory / "pc", directory / "laptop"], tmp_path / "report")
    combined = read(tmp_path / "report/registered-analysis.json")
    assert set(combined["decisions"]) == {"r1", "r2", "r3"}
    assert not combined["eligible"]
    assert all(d["status"] == "stopped_safety" for d in combined["decisions"].values())
    assert result["r3/curriculum/e10/classic"]["games"] == 40
