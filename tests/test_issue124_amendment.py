"""Only the owner-authorized time/CPU extension may change on resume."""

from copy import deepcopy

import pytest

from scripts.resume_issue124_amendment import validate_amendment


def records():
    authorization = {
        "limits": {"cpu_seconds": 144000, "wall_seconds": 36000, "memory_bytes": 8589934592},
        "reviewed_commit": "original",
        "authorized_by": "1BlauNitrox",
    }
    amendment = {
        "original_limits": deepcopy(authorization["limits"]),
        "new_limits": {"cpu_seconds": 345600, "wall_seconds": 86400, "memory_bytes": 8589934592},
        "execution_commit": "original",
        "authorized_by": "1BlauNitrox",
        "workers": 4,
    }
    return amendment, authorization


def test_budget_extension_preserves_original_authorization():
    amendment, authorization = records()
    before = deepcopy(authorization)
    validate_amendment(amendment, authorization)
    assert authorization == before


@pytest.mark.parametrize(
    "field,value", [("memory_bytes", 17179869184), ("wall_seconds", 36000), ("cpu_seconds", 144000)]
)
def test_extension_cannot_change_memory_or_reset_ceilings(field, value):
    amendment, authorization = records()
    amendment["new_limits"][field] = value
    with pytest.raises(ValueError):
        validate_amendment(amendment, authorization)


@pytest.mark.parametrize(
    "field,value", [("execution_commit", "different"), ("workers", 8), ("authorized_by", "other")]
)
def test_extension_cannot_change_source_workers_or_owner(field, value):
    amendment, authorization = records()
    amendment[field] = value
    with pytest.raises(ValueError):
        validate_amendment(amendment, authorization)


def test_resume_normalizes_timestamp_without_resetting_consumed_cpu(tmp_path):
    import json

    from scripts.resume_issue124_amendment import resume_monitor_type
    from training.run_issue124_campaign import LIMITS, LocalMonitor

    path = tmp_path / "resources.json"
    first = LocalMonitor(
        state_path=path, authorized_at="2026-09-10T06:59:45.481673+00:00", limits=LIMITS
    )
    state = json.loads(path.read_text())
    state["cpu_seconds_consumed"] = 1234
    path.write_text(json.dumps(state))
    resumed = resume_monitor_type(LocalMonitor)(
        state_path=path, authorized_at="2026-09-10T06:59:45.481673+00:00", limits=LIMITS
    )
    assert resumed._completed_cpu_seconds == 1234
    assert resumed._authorized_epoch == first._authorized_epoch
