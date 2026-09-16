"""Tests for the Issue #186 baseline decision rules."""

from training.analyze_issue186_task3_baseline import evaluate_criteria


def test_issue186_criteria_pass_for_registered_boundary_values() -> None:
    rows = []
    for replica in ("r1", "r2", "r3", "r4", "r5"):
        for scenario, collection in (
            ("peaceful", 0.0),
            ("coincollector", 0.0),
            ("classic", 0.15),
            ("coin-heaven", 0.90),
            ("loot-crate", 0.20),
        ):
            rows.append(
                {
                    "replica": replica,
                    "scenario": scenario,
                    "mean_collection_fraction": collection,
                    "mean_opponents_eliminated": 0.20 if scenario == "peaceful" else 0.0,
                    "self_kill_rate": 0.15,
                    "decision_time_p95_ms": 1.0,
                    "decision_time_max_ms": 2.0,
                }
            )

    assert all(evaluate_criteria(rows, True, True).values())


def test_issue186_criteria_reject_no_hunting() -> None:
    rows = []
    for replica in ("r1", "r2", "r3", "r4", "r5"):
        for scenario in (
            "peaceful",
            "coincollector",
            "classic",
            "coin-heaven",
            "loot-crate",
        ):
            rows.append(
                {
                    "replica": replica,
                    "scenario": scenario,
                    "mean_collection_fraction": 1.0,
                    "mean_opponents_eliminated": 0.0,
                    "self_kill_rate": 0.0,
                    "decision_time_p95_ms": 1.0,
                    "decision_time_max_ms": 2.0,
                }
            )

    criteria = evaluate_criteria(rows, True, True)
    assert not criteria["mean_peaceful_eliminations_at_least_0_20"]
    assert not criteria["four_replicas_positive_peaceful_elimination"]
