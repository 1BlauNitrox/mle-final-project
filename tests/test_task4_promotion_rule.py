"""The re-specified Task 4 promotion rule can be passed, and still has to be earned."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_task4_competition import (
    blocking_gates,
    legality_not_worse,
    median_replica_by_score,
)

ROOT = Path(__file__).resolve().parents[1]
COMPLETED = (
    "experiments/2026-09-15-task4-trainable-scope/results/analysis.json",
    "experiments/2026-09-15-task4-opponent-mixture/results/analysis.json",
)


def test_a_gate_that_cannot_apply_cannot_veto():
    # Null is "not applicable to this programme"; False is still a veto.
    assert blocking_gates({"a": True, "frozen_online_weights": None}) == [True]
    assert not all(blocking_gates({"a": True, "frozen_online_weights": False}))


def test_legality_is_measured_against_the_untrained_reference():
    # The reference emits its own invalid actions, so the old global-zero rule
    # vetoed every arm. Matching the reference is allowed; exceeding it is not.
    assert legality_not_worse({"reference": 82, "treatment-r1": 82, "treatment-r2": 10})
    assert not legality_not_worse({"reference": 82, "treatment-r1": 83})
    # A perfect reference still admits a perfect arm, which the old rule did not.
    assert legality_not_worse({"reference": 0, "treatment-r1": 0})


def test_the_median_replica_is_selected_by_score_and_the_best_one_is_not():
    suite = {f"treatment-r{i}": {"score": s} for i, s in enumerate([5.0, 1.0, 3.0], start=1)}
    assert median_replica_by_score(suite, "treatment", 3) == 2  # zero-based: r3, score 3.0
    ordered = sorted(suite.values(), key=lambda v: v["score"])
    assert suite["treatment-r3"]["score"] == ordered[1]["score"]


@pytest.mark.parametrize("path", COMPLETED)
def test_the_respecified_rule_does_not_retrospectively_promote_a_completed_comparison(path):
    """The rule was re-specified because it could never pass, not to make ours pass.

    Both completed comparisons stay non-promotable under it, and the reasons are
    now substantive - no arm has significantly beaten the incumbent on score -
    rather than structural.
    """
    record = ROOT / path
    if not record.exists():
        pytest.skip(f"{path} lands with its own results branch")
    analysis = json.loads(record.read_text(encoding="utf-8"))
    invalid = analysis["invalid_actions_by_artifact"]
    assert not legality_not_worse(invalid), "a trained artifact exceeded the reference"
    for arm, differences in analysis["paired_differences"]["classic-rule-based"].items():
        if arm == "control":
            continue
        lower = differences["reference"]["score"]["paired_crossed_bootstrap"][0]
        assert lower <= 0, f"{arm} would now clear the score gate; re-check the disclosure"


def test_a_gate_defined_against_the_control_cannot_veto_the_control():
    # The control arm is a trained agent; only the reference is untrained. Gates
    # phrased "versus control" are undefined for it and are reported as null, so
    # they must not block it the way a real failure would.
    control_checks = {
        "score_versus_reference_ci": True,
        "self_kills_versus_control": None,
        "earlier_task_retention": True,
    }
    assert all(blocking_gates(control_checks))
    assert not all(blocking_gates({**control_checks, "earlier_task_retention": False}))


def test_the_multiplicity_correction_covers_every_arm_that_can_be_promoted():
    # Judging the control too adds a hypothesis, so the family grows and the
    # intervals widen. The correction has to follow the rule it protects.
    def percent(arm_count, registered=95.0):
        return 100.0 - (100.0 - registered) / max(arm_count, 1)

    assert percent(2) == 97.5
    assert percent(3) > percent(2), "a third promotable arm must widen the interval"
    assert round(percent(3), 4) == round(100.0 - 5.0 / 3, 4)
