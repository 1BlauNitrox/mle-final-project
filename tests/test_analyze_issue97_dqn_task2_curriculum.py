"""Tests for Issue #97's curriculum analysis, focused on its direct-arm-only
degradation path (the staged arm's raw evidence is an external archive not
always available locally -- see STAGED_ARM_MISSING_NOTE)."""

from __future__ import annotations

from training.analyze_issue97_dqn_task2_curriculum import _summaries


def _row(**overrides: object) -> dict[str, object]:
    base = {
        "scenario": "classic",
        "arm": "direct",
        "attempted_actions": 10,
        "coins_collected": 1,
        "collection_fraction": 0.1,
        "survival_steps": 20,
        "survived": True,
        "self_kills": 0,
        "invalid_actions": 1,
        "crates_destroyed": 2,
        "action_up": 1,
        "action_right": 1,
        "action_down": 1,
        "action_left": 1,
        "action_wait": 1,
        "action_bomb": 1,
        "action_unknown": 0,
        "decision_time_median_ms": 0.5,
        "decision_time_p95_ms": 1.0,
        "decision_time_max_ms": 2.0,
    }
    return {**base, **overrides}


def test_summaries_skip_the_absent_staged_arm_without_error() -> None:
    """With only "direct" rows present, the missing "staged" group is skipped
    rather than raising or producing an empty/NaN entry -- this is exactly
    what happens when the staged arm's raw archive is not available locally.
    """
    rows = [_row(), _row(scenario="coin-heaven"), _row(scenario="loot-crate")]

    summaries = _summaries(rows)

    assert {entry["scenario"] for entry in summaries} == {"classic", "coin-heaven", "loot-crate"}
    assert all(entry["arm"] == "direct" for entry in summaries)


def test_summaries_include_both_arms_once_staged_rows_exist() -> None:
    rows = [_row(), _row(arm="staged")]

    summaries = _summaries(rows)

    assert {entry["arm"] for entry in summaries} == {"direct", "staged"}
