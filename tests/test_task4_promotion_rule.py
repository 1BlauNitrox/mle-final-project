"""The re-specified Task 4 promotion rule can be passed, and still has to be earned."""

from __future__ import annotations

import pytest

from scripts.analyze_task4_competition_v2 import (
    blocking_gates,
    legality_not_worse,
    median_replica_by_score,
    require_versioned_registration,
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


def test_corrected_rule_requires_a_new_registered_protocol_version():
    with pytest.raises(ValueError, match="promotion_rule_version=2"):
        require_versioned_registration({"profile": "completed-v1"})
    registered = require_versioned_registration({"promotion_rule_version": 2})
    assert registered["promotion_rule_version"] == 2


def test_the_analyzer_follows_the_registration_not_the_newest_rule(monkeypatch):
    """A completed protocol must reproduce the decision it actually made.

    The corrected rule lives in a separate module so re-analysing an older
    campaign with the repository's current code cannot silently apply rules it
    was never registered under. Opting in is explicit.
    """
    import json
    from pathlib import Path

    from scripts import pilot_task4_competition as pilot

    for declared, expected in (
        (None, "scripts.analyze_task4_competition"),
        (1, "scripts.analyze_task4_competition"),
        (2, "scripts.analyze_task4_competition_v2"),
    ):
        registration = {} if declared is None else {"promotion_rule_version": declared}
        monkeypatch.setattr(pilot, "config", lambda registration=registration: registration)
        assert pilot.registered_analyzer().__module__ == expected

    # The five screens ran before the rule was corrected and must keep the rule
    # they executed under. The final training run opted in, and the warm-started
    # continuation inherits that choice because it is the same programme.
    root = Path(pilot.ROOT)
    for profile in pilot.PROFILES:
        shipped = json.loads(
            (root / f"experiments/{pilot.profile_dir(profile)}/config.json").read_text(
                encoding="utf-8"
            )
        )
        opted_in = shipped.get("promotion_rule_version") == 2
        assert opted_in == (profile in {"final-training", "warm-lineup"}), profile

    # Both analyzers are bound into the run's provenance, so neither can change
    # under a prepared campaign without the binding noticing.
    hashes = pilot.code_hashes()
    assert "scripts/analyze_task4_competition.py" in hashes
    assert "scripts/analyze_task4_competition_v2.py" in hashes


def test_the_corrected_rule_refuses_an_unversioned_registration():
    from scripts.analyze_task4_competition_v2 import require_versioned_registration

    require_versioned_registration({"promotion_rule_version": 2})
    for cfg in ({}, {"promotion_rule_version": 1}):
        with pytest.raises(ValueError, match="promotion_rule_version=2"):
            require_versioned_registration(cfg)
