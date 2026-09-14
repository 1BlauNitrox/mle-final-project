"""Task 4 preparation stays blocked without its parent; statistics retain regression gates."""

import pytest

from tests.test_task3_campaign import passing_rows
from training import task4_protocol as task4


def test_task4_matrix_is_matched_and_cannot_authorize_execution():
    _, plans, report = task4.validate()
    assert report["training_episodes"] == 50000
    assert report["evaluation_episodes"] == 7920
    assert not report["parent_bound"] and not report["compute_authorized"]
    assert len(report["launch_blockers"]) == 3
    assert len(plans["strong"].replicas) == 5


def observations():
    base = [r for r in passing_rows() if r["suite"] == "classic-peaceful-primary"]
    suites = (
        "strong-three",
        "strong-one",
        "mixed-three",
        "learned-one",
        "classic-peaceful",
        "classic-collector",
        "classic-retention",
        "coin-heaven-retention",
        "loot-crate-retention",
    )
    result = []
    for suite in suites:
        for row in base:
            if row["arm"] == "reference":
                result.append(dict(row, suite=suite))
            else:
                result.append(dict(row, suite=suite, arm="strong", score=8))
                result.append(dict(row, suite=suite, arm="mixture", score=10))
    return result


def test_proposed_task4_selection_requires_improvement_and_retention():
    config = task4.validate()[0]
    rows = observations()
    result = task4.decide_primary(rows, config)
    assert result["proposed_selected_replica"] == "r3"
    assert result["raw_integrity_required"] and not result["automatic_training_authorized"]
    for row in rows:
        if row["arm"] == "mixture" and row["suite"] == "coin-heaven-retention":
            row["coins_collected"] = 0
    assert task4.decide_primary(rows, config)["proposed_selected_replica"] is None
    with pytest.raises(ValueError, match="pair"):
        task4.decide_primary(rows[:-1], config)
