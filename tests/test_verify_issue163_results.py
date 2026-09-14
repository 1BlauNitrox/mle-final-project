import hashlib

import pytest

from training.analyze_task3_campaign import check_repeat, repeat_difference
from training.verify_issue163_results import blocked_decision, fingerprint


def pair():
    first = {
        "suite": "loot-primary",
        "attempted_actions": 400,
        "action_wait": 390,
        "executed_action_sequence_sha256": "same",
        "score": 0,
        "decision_time_max_ms": 7.1,
        "decision_times_ms": [7.1],
    }
    repeat = {
        **first,
        "suite": "loot-repeat",
        "attempted_actions": 399,
        "action_wait": 389,
        "decision_time_max_ms": 1184.5,
        "decision_times_ms": [1184.5],
    }
    return first, repeat


def test_strict_default_keeps_counter_mismatch_failure():
    with pytest.raises(ValueError, match="Deterministic repeat mismatch"):
        check_repeat(*pair(), ("control", "r5", 1631403))


def test_diagnostics_retain_all_registered_counter_differences():
    failures = []
    check_repeat(*pair(), ("control", "r5", 1631403), failures)
    assert failures == [
        {
            "pair": ["control", "r5", 1631403],
            "differences": {"attempted_actions": [400, 399], "action_wait": [390, 389]},
        }
    ]


def test_latency_only_remains_excluded_from_repeat_gate():
    first, repeat = pair()
    repeat.update(attempted_actions=400, action_wait=390)
    assert not repeat_difference(first, repeat)
    check_repeat(first, repeat, ("pair",))


def test_changed_gameplay_hash_remains_a_repeat_failure():
    first, repeat = pair()
    repeat.update(
        attempted_actions=400, action_wait=390, executed_action_sequence_sha256="different"
    )
    assert set(repeat_difference(first, repeat)) == {"executed_action_sequence_sha256"}


def test_missing_field_is_not_downgraded_to_diagnostic():
    first, repeat = pair()
    repeat.pop("score")
    with pytest.raises(ValueError, match="schema"):
        check_repeat(first, repeat, ("pair",), [])


def test_failed_integrity_cannot_promote_favorable_performance():
    result = blocked_decision(
        {"status": "exploratory_pass", "selected_arm": "neutral"},
        [{"failure": True}],
        {"campaign": {"original_wall_pass": False}},
    )
    assert not result["automatic_continuation_allowed"]
    assert result["selected_arm"] is None and result["selected_replica"] is None
    assert not result["repeat_gate_pass"] and not result["original_campaign_wall_gate_pass"]


def test_portable_fingerprint_reproduces_windows_component_order(tmp_path):
    (tmp_path / "a").mkdir()
    files = [("a/z.py", b"x"), ("a.py", b"y"), ("B.py", b"z")]
    for name, data in files:
        (tmp_path / name).write_bytes(data)
    expected = hashlib.sha256()
    for name, data in files:
        expected.update(name.encode() + b"\0" + data + b"\0")
    assert fingerprint(tmp_path, (".",)) == expected.hexdigest()


def test_fingerprint_protects_executed_line_endings(tmp_path):
    file = tmp_path / "a.py"
    file.write_bytes(b"x\r\n")
    before = fingerprint(tmp_path, ("a.py",))
    file.write_bytes(b"x\n")
    assert fingerprint(tmp_path, ("a.py",)) != before


def test_payload_comparison_reports_optimizer_and_tensor_changes():
    import torch

    from training.verify_issue163_results import payload_differences

    first = {"online": torch.tensor([1.0]), "optimizer": {"step": 2}}
    second = {"online": torch.tensor([2.0]), "optimizer": {"step": 3}}
    assert payload_differences(first, second) == ["/online", "/optimizer/step"]
    second = {"online": torch.tensor([1.0]), "optimizer": {"step": 2}}
    assert payload_differences(first, second) == []
    second["online"] = torch.tensor([1.0], dtype=torch.float64)
    assert payload_differences(first, second) == ["/online"]
    assert payload_differences([1], (1,)) == ["/type"]
