import copy
from types import SimpleNamespace

import pytest

from scripts.resume_issue163 import GIB, amended_record, cumulative_monitor, resource_decision


def original(wall=36000):
    return {
        "authorized_at": "2026-09-12T22:31:32Z",
        "active_root_pids": [],
        "limit_reached": None,
        "cpu_seconds_consumed": 14000,
        "wall_seconds_elapsed": 19000,
        "peak_memory_bytes": 400000000,
        "limits": {"cpu_seconds": 43200, "wall_seconds": wall, "memory_bytes": 2 * GIB},
    }


def test_wall_only_retains_original_epoch_usage_and_bytes():
    before = original()
    saved = copy.deepcopy(before)
    new = amended_record(before, 36000)
    assert before == saved
    assert new["authorized_at"] == saved["authorized_at"]
    assert new["cpu_seconds_consumed"] == 14000
    assert new["limits"] == {**saved["limits"], "wall_seconds": 43200}


def test_outer_cpu_floor_cannot_reset_usage():
    new = amended_record(original(43200), 43200, 20000)
    assert new["cpu_seconds_consumed"] == 20000
    assert new["limits"]["wall_seconds"] == 50400
    with pytest.raises(ValueError, match="exhausted"):
        amended_record(original(), 36000, 43200)


@pytest.mark.parametrize("key,value", [("active_root_pids", [123]), ("limit_reached", "wall")])
def test_rejects_active_or_failed_records(key, value):
    record = original()
    record[key] = value
    with pytest.raises(ValueError, match="active roots"):
        amended_record(record, 36000)


def test_rejects_changed_cpu_limit():
    record = original()
    record["limits"]["cpu_seconds"] += 1
    with pytest.raises(ValueError, match="Unexpected"):
        amended_record(record, 36000)


def test_original_wall_gate_is_not_silently_passed():
    before = original()
    after = amended_record(before, 36000)
    after["wall_seconds_elapsed"] = 40000
    gates = resource_decision(before, after)
    assert gates["original_wall_gate_pass"] is False
    assert gates["approved_wall_gate_pass"] is True
    after["wall_seconds_elapsed"] = 43201
    with pytest.raises(ValueError, match="cap exceeded"):
        resource_decision(before, after)


def test_resource_decision_rejects_epoch_and_cpu_reset():
    before = original()
    after = amended_record(before, 36000)
    after["authorized_at"] = "later"
    with pytest.raises(ValueError, match="epoch"):
        resource_decision(before, after)
    after["authorized_at"] = before["authorized_at"]
    after["cpu_seconds_consumed"] = 1
    with pytest.raises(ValueError, match="reset"):
        resource_decision(before, after)


def test_cpu_retains_exited_descendants_and_distinguishes_pid_reuse():
    class Base:
        def __init__(self):
            self._active = {1: {"process": None, "cpu_seconds": 0}}

        def _tree(self, root):
            return self.processes

    def process(pid, birth, cpu):
        return SimpleNamespace(
            pid=pid,
            create_time=lambda: birth,
            cpu_times=lambda: SimpleNamespace(user=cpu, system=0),
        )

    monitor = cumulative_monitor(Base)()
    monitor.processes = [process(1, 10, 2), process(2, 20, 5)]
    monitor._sample_locked()
    monitor.processes = [process(1, 10, 3)]
    monitor._sample_locked()
    assert monitor._active[1]["cpu_seconds"] == 8
    monitor.processes = [process(1, 10, 4), process(2, 30, 6)]
    monitor._sample_locked()
    assert monitor._active[1]["cpu_seconds"] == 15
