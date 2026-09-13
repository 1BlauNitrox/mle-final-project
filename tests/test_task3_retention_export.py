from copy import deepcopy

import pytest

from scripts.export_task3_retention_results import arm_summary, write_bytes


def test_arm_means_preserve_equal_replica_weight_and_reference_count():
    data = {
        "summary": {
            "suite": {
                "reference": {"score": 9},
                "control-r1": {"score": 0},
                "control-r2": {"score": 3},
                "control-r3": {"score": 6},
                "treatment-r1": {"score": 1},
                "treatment-r2": {"score": 4},
                "treatment-r3": {"score": 7},
            }
        }
    }
    rows = arm_summary(data)
    assert [r["score"] for r in rows] == [9, 3, 4]
    assert [r["primary_episodes"] for r in rows] == [5, 15, 15]
    incomplete = deepcopy(data)
    del incomplete["summary"]["suite"]["treatment-r2"]
    with pytest.raises(ValueError, match="Incomplete"):
        arm_summary(incomplete)


def test_generated_results_never_overwrite_different_existing_bytes(tmp_path):
    path = tmp_path / "result.json"
    write_bytes(path, b"original")
    write_bytes(path, b"original")
    with pytest.raises(ValueError, match="overwrite"):
        write_bytes(path, b"changed")
    assert path.read_bytes() == b"original"
