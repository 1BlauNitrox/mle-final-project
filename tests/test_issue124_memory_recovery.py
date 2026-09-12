from datetime import datetime, timezone

import pytest

from scripts.recover_issue124_memory import REASON, resumed_state


def state():
    return {"limit_reached": REASON, "active_root_pids": [],
            "authorized_at": "2026-09-10T06:59:45Z",
            "limits": {"wall_seconds": 86400, "cpu_seconds": 345600,
                       "memory_bytes": 8 * 1024**3},
            "cpu_seconds_consumed": 163584.828125,
            "peak_memory_bytes": 1350107136}


def test_recovery_preserves_accounting_and_original_clock():
    old = state()
    now = datetime(2026, 9, 10, 22, tzinfo=timezone.utc).timestamp()
    new = resumed_state(old, 5 * 1024**3, now)
    assert old["limit_reached"] == REASON
    assert new["limit_reached"] is None
    for key in ("authorized_at", "limits", "cpu_seconds_consumed", "peak_memory_bytes"):
        assert new[key] == old[key]
    assert new["wall_seconds_elapsed"] == 54015


@pytest.mark.parametrize("change", [
    {"limit_reached": "CPU budget exhausted"}, {"active_root_pids": [42]},
    {"cpu_seconds_consumed": 345600}, {"peak_memory_bytes": 9 * 1024**3},
])
def test_recovery_rejects_other_breaches_or_live_workers(change):
    with pytest.raises(ValueError):
        resumed_state({**state(), **change}, 5 * 1024**3,
                      datetime(2026, 9, 10, 22, tzinfo=timezone.utc).timestamp())


def test_recovery_rejects_low_memory_and_expired_wall_budget():
    now = datetime(2026, 9, 10, 22, tzinfo=timezone.utc).timestamp()
    with pytest.raises(ValueError):
        resumed_state(state(), 3 * 1024**3, now)
    with pytest.raises(ValueError):
        resumed_state(state(), 5 * 1024**3, now + 86400)
