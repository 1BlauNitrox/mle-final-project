"""Decision-rule tests for the preregistered Issue #86 analysis."""

from __future__ import annotations

from training.analyze_issue86_legal_action_masking import _criteria


def _row(treatment: str, *, invalid_actions: int = 0) -> dict[str, object]:
    return {
        "treatment": treatment,
        "invalid_actions": invalid_actions,
        "decision_time_p95_ms": 1.0,
        "decision_time_max_ms": 2.0,
    }


def test_registered_gates_require_every_performance_interval_to_pass() -> None:
    comparisons = {
        "classic": {"ci_lower": -0.04},
        "coin-heaven": {"ci_lower": -0.03},
        "loot-crate": {"ci_lower": -0.02},
    }
    assert all(_criteria([_row("masked")], comparisons, comparisons, True).values())

    rejected = _criteria(
        [_row("masked"), _row("unmasked")],
        {**comparisons, "classic": {"ci_lower": -0.051}},
        comparisons,
        True,
    )
    assert not rejected["collection_non_regression"]
    assert rejected["survival_non_regression"]


def test_structural_gate_rejects_framework_invalid_masked_action() -> None:
    comparisons = {
        scenario: {"ci_lower": 0.0}
        for scenario in ("classic", "coin-heaven", "loot-crate")
    }
    criteria = _criteria([_row("masked", invalid_actions=1)], comparisons, comparisons, True)
    assert not criteria["masked_primary_invalid_actions_zero"]
