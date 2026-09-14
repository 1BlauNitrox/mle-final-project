import json

import pytest

from scripts.task3_progress_retry import with_permission_retry


def test_retries_identical_write_and_preserves_audit(tmp_path):
    calls = []
    value = {"cpu_seconds": 10, "completed": ["reference"]}

    def write(path, payload):
        calls.append((path, payload))
        if len(calls) < 3:
            raise PermissionError("temporarily held by a reader")
        path.write_text(json.dumps(payload))

    target = tmp_path / "state.json"
    audit = tmp_path / "audit.jsonl"
    with_permission_retry(write, audit, delay=0)(target, value)
    assert len(calls) == 3
    assert all(payload is value and path == target for path, payload in calls)
    assert json.loads(target.read_text()) == value
    assert len(audit.read_text().splitlines()) == 2


def test_bounded_failure_does_not_hide_error(tmp_path):
    def write(path, value):
        raise PermissionError("permanent")

    audit = tmp_path / "audit.jsonl"
    with pytest.raises(PermissionError, match="permanent"):
        with_permission_retry(write, audit, retries=2, delay=0)(tmp_path / "state", {})
    assert len(audit.read_text().splitlines()) == 2


def test_other_errors_are_not_retried(tmp_path):
    def write(path, value):
        raise OSError("disk full")

    audit = tmp_path / "audit.jsonl"
    with pytest.raises(OSError, match="disk full"):
        with_permission_retry(write, audit, delay=0)(tmp_path / "state", {})
    assert not audit.exists()
