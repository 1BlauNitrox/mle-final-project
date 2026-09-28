"""Tests for the Issue #234 head-to-head analysis."""

from __future__ import annotations

from scripts.run_final_agent_comparison import extract_row, game_plan, placement
from training.analyze_issue234_final_agent_comparison import (
    DQN,
    TABULAR,
    analyse,
    bootstrap_interval,
)

CONFIG = {
    "world_seeds": {"first": 1, "count": 3},
    "suites": {
        "head_to_head": {
            "lineup": [DQN, TABULAR, "rule_based_agent", "rule_based_agent"],
            "rotations": [0, 1, 2, 3],
        },
        "versus_rule_based": {
            "learned_agents": [DQN, TABULAR],
            "opponents": ["rule_based_agent"] * 3,
        },
    },
    "analysis": {"bootstrap_resamples": 500, "bootstrap_seed": 1},
}


def _agent(score: int, *, kills: int = 0, survived: bool = True) -> dict:
    return {
        "score": score,
        "coins": score - 5 * kills,
        "kills": kills,
        "self_kills": 0,
        "survived": survived,
        "invalid_actions": 0,
        "strict_first": False,
        "placement": 2,
        "decision_time_p95_ms": 1.0,
        "decision_time_max_ms": 2.0,
    }


def _rows() -> list[dict]:
    rows = []
    for game in game_plan(CONFIG):
        agents = {}
        rb = 0
        for name in game["lineup"]:
            if name == "rule_based_agent":
                agents[f"rule_based_agent_{rb}"] = _agent(2)
                rb += 1
            else:
                agents[name] = _agent(4 if name == DQN else 1)
        rows.append({**game, "agents": agents})
    return rows


def test_game_plan_rotates_all_seats_and_pairs_worlds():
    plan = game_plan(CONFIG)
    head = [g for g in plan if g["suite"] == "head_to_head"]
    alone = [g for g in plan if g["suite"] == "versus_rule_based"]
    assert len(head) == 12 and len(alone) == 6
    seats = {g["lineup"].index(DQN) for g in head if g["world_seed"] == 1}
    assert seats == {0, 1, 2, 3}
    for seed in (1, 2, 3):
        pair = [g for g in alone if g["world_seed"] == seed]
        assert {g["lineup"].index(a) for g in pair for a in (DQN, TABULAR) if a in g["lineup"]} == {
            (seed - 1) % 4
        }


def test_placement_counts_ties_as_no_strict_win():
    assert placement({"a": 3, "b": 3, "c": 1, "d": 0}, "a") == (1, False)
    assert placement({"a": 4, "b": 3, "c": 1, "d": 0}, "a") == (1, True)
    assert placement({"a": 1, "b": 3, "c": 1, "d": 0}, "a") == (2, False)


def test_extract_row_reads_framework_round_statistics():
    stats = {
        "by_round": {
            "r": {
                "agents": {
                    DQN: {
                        "score": 6,
                        "coins": 1,
                        "kills": 1,
                        "suicides": 0,
                        "survived": True,
                        "invalid": 2,
                    },
                    TABULAR: {"score": 1, "coins": 1, "survived": False, "invalid": 0},
                    "rule_based_agent_0": {"score": 0, "survived": False},
                    "rule_based_agent_1": {"score": 2, "coins": 2, "survived": True},
                }
            }
        }
    }
    row = extract_row(
        {
            "suite": "head_to_head",
            "world_seed": 1,
            "rotation": 0,
            "lineup": [DQN, TABULAR, "rule_based_agent", "rule_based_agent"],
        },
        stats,
    )
    assert row["agents"][DQN]["strict_first"] is True
    assert row["agents"][DQN]["invalid_actions"] == 2
    assert row["agents"][TABULAR]["kills"] == 0


def test_bootstrap_interval_contains_mean_and_constant_data_is_degenerate():
    est = bootstrap_interval([2.0, 2.0, 2.0], resamples=100, seed=0)
    assert est["low"] == est["mean"] == est["high"] == 2.0
    est = bootstrap_interval([0.0, 1.0, 2.0, 3.0], resamples=1000, seed=0)
    assert est["low"] <= est["mean"] <= est["high"]


def test_analyse_reports_primary_difference_and_rule_based_gap():
    result = analyse(_rows(), CONFIG)
    assert result["games"] == {"head_to_head": 12, "versus_rule_based": 6}
    assert result["primary"]["mean"] == 3.0 and result["primary"]["success"] is True
    gap = result["versus_rule_based"]["gap_to_rule_based"]
    assert gap[DQN]["mean"] == 2.0 and gap[TABULAR]["mean"] == -1.0
