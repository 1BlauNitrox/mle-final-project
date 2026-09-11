"""Campaign ordering, fail-closed evidence and selection contracts."""

from copy import deepcopy
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from training.analyze_issue124_campaign import eligibility, verify_resources
from training.run_issue124_campaign import (
    CONFIG,
    LIMITS,
    LocalMonitor,
    existing_authorization,
    require_current_approval,
    run_phases,
    sha256,
    validate_protocol,
    write_json,
)


def test_finalized_protocol_loads_all_six_plans():
    plans = validate_protocol(historical=True)
    assert len(plans) == 6
    assert all(plan.max_parallel_training == 1 for plan in plans.values())


@pytest.mark.parametrize("resume", [False, True])
def test_restart_cannot_overwrite_a_completed_result(tmp_path, resume):
    write_json(
        tmp_path / "authorization.json",
        {"reviewed_commit": "head", "config_sha256": sha256(CONFIG)},
    )
    write_json(tmp_path / "campaign-status.json", {"status": "completed"})
    write_json(tmp_path / "analysis/result.json", {"analysis_valid": True})
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError):
        existing_authorization(tmp_path, "head", resume)
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*.json")}


def test_monitor_counts_supervisor_usage_and_keeps_technical_cancel_resumable(
    tmp_path, monkeypatch
):
    from training import run_issue124_campaign as campaign

    cpu = [10.0]
    supervisor = SimpleNamespace(
        cpu_times=lambda: SimpleNamespace(user=cpu[0], system=0),
        memory_info=lambda: SimpleNamespace(rss=4096),
    )
    monkeypatch.setattr(campaign.psutil, "Process", lambda: supervisor)
    monitor = LocalMonitor(
        state_path=tmp_path / "resources.json",
        authorized_at="1970-01-01T00:00:00Z",
        limits=LIMITS,
        time_fn=lambda: 100,
    )
    cpu[0] = 12.0
    assert monitor._usage_locked() == (2.0, 4096)
    monitor.cancel("technical interruption")
    with pytest.raises(RuntimeError, match="technical interruption"):
        monitor.check()
    assert monitor._limit_reached is None


def test_four_training_arms_finish_before_any_serial_evaluation(tmp_path):
    plans = {cell: SimpleNamespace(plan_id=cell) for cell in [*"ABCD", "untrained", "frozen_task1"]}
    barrier, lock = Barrier(4), Lock()
    completed, evaluation = set(), []

    def executor(plan, **kwargs):
        if kwargs.get("training_only"):
            barrier.wait(timeout=10)
            with lock:
                completed.add(plan.plan_id)
            (tmp_path / plan.plan_id).mkdir()
        else:
            assert completed == set("ABCD")
            evaluation.append(plan.plan_id)
            assert kwargs["resume"] == (plan.plan_id in "ABCD")

    monitor = SimpleNamespace(check=lambda: None, cancel=lambda reason: None)
    run_phases(plans, tmp_path, False, monitor, executor)
    assert evaluation == list(plans)


def test_training_failure_cancels_siblings_and_prevents_evaluation(tmp_path):
    plans = {cell: SimpleNamespace(plan_id=cell) for cell in "ABCD"}
    cancelled = []

    def executor(plan, **kwargs):
        assert kwargs.get("training_only")
        if plan.plan_id == "B":
            raise RuntimeError("technical failure")

    monitor = SimpleNamespace(check=lambda: None, cancel=cancelled.append)
    with pytest.raises(RuntimeError, match="technical failure"):
        run_phases(plans, tmp_path, False, monitor, executor)
    assert cancelled


def contrasts():
    result = {
        name: {"mean_difference": 0.10, "bonferroni_98_75_lower": 0.001}
        for name in (
            "rehearsal_b_minus_a",
            "rehearsal_d_minus_c",
            "mask_c_minus_a",
            "mask_d_minus_b",
        )
    }
    for pair in ("b_minus_a", "c_minus_a", "d_minus_a", "d_minus_b", "d_minus_c"):
        result["guards_" + pair] = {"collection": {"ci95_lower": -0.049}}
    return result


def test_eligibility_requires_both_combined_effects_and_strict_guards():
    values = contrasts()
    assert all(eligibility(values).values())
    changed = deepcopy(values)
    changed["mask_d_minus_b"]["bonferroni_98_75_lower"] = 0
    assert not eligibility(changed)["D"]
    changed = deepcopy(values)
    changed["guards_d_minus_a"]["collection"]["ci95_lower"] = -0.05
    assert not eligibility(changed)["D"]
    changed = deepcopy(values)
    changed["rehearsal_b_minus_a"]["mean_difference"] = 0.099
    assert not eligibility(changed)["B"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("active_root_pids", [123]),
        ("limit_reached", "wall"),
        ("wall_seconds_elapsed", 36001),
        ("peak_memory_bytes", 9 * 1024**3),
        ("cpu_seconds_consumed", -1),
    ],
)
def test_invalid_resource_evidence_is_rejected(field, value):
    record = dict(
        limits=vars(LIMITS),
        active_root_pids=[],
        limit_reached=None,
        wall_seconds_elapsed=100,
        peak_memory_bytes=1024,
        cpu_seconds_consumed=100,
    )
    verify_resources(record)
    record[field] = value
    with pytest.raises(ValueError):
        verify_resources(record)


@pytest.mark.parametrize(
    "state,commit,author,valid",
    [
        ("APPROVED", "head", "peer", True),
        ("APPROVED", "old", "peer", False),
        ("APPROVED", "head", "owner", False),
        ("CHANGES_REQUESTED", "head", "peer", False),
        ("DISMISSED", "head", "peer", False),
    ],
)
def test_review_requires_non_author_approval_on_execution_commit(state, commit, author, valid):
    review = {
        "headRefOid": "head",
        "author": {"login": "owner"},
        "reviews": [{"state": state, "author": {"login": author}, "commit": {"oid": commit}}],
    }
    if valid:
        require_current_approval(review, "head")
        with pytest.raises(ValueError, match="differs"):
            require_current_approval(review, "different")
    else:
        with pytest.raises(ValueError, match="non-author approval"):
            require_current_approval(review, "head")


def test_outstanding_peer_change_request_blocks_other_approval():
    review = {
        "headRefOid": "head",
        "author": {"login": "owner"},
        "reviews": [
            {"state": "APPROVED", "author": {"login": "peer"}, "commit": {"oid": "head"}},
            {"state": "CHANGES_REQUESTED", "author": {"login": "peer2"}},
        ],
    }
    with pytest.raises(ValueError, match="non-author approval"):
        require_current_approval(review, "head")


def test_owner_exception_is_explicit_recorded_and_default_still_requires_review(monkeypatch):
    from training import run_issue124_campaign as campaign

    calls = []

    def reject(commit):
        calls.append(commit)
        raise ValueError("missing peer review")

    monkeypatch.setattr(campaign, "check_review", reject)
    with pytest.raises(ValueError, match="missing peer review"):
        campaign.review_authorization("head")
    record = campaign.review_authorization("head", owner_override=True)
    assert calls == ["head"]
    assert record["status"] == "owner_authorized_exception_no_peer_approval"
    assert record["authorized_by"] == "1BlauNitrox"
    assert "no PR approval or merge" in record["scope"]


def test_current_seed_audit_still_rejects_later_overlap():
    with pytest.raises(ValueError, match="Seed collision"):
        validate_protocol()
