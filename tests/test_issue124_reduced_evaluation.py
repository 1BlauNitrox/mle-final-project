import hashlib

import pytest

from scripts.evaluate_issue124_reduced import evaluation_spec


def fixture(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"retained-final-checkpoint")
    raw = {"replicas": [{"id": f"r{i}"} for i in range(1, 6)],
           "training_stages": [{"rounds": 500}],
           "evaluation_suites": [{"world_seeds": [324001]}]}
    status = {"jobs": {f"train-r{i}-block-20": {
        "status": "completed", "artifact": {"path": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
        for i in range(1, 6)}}
    return raw, status


@pytest.mark.parametrize("cell,count", [("A", 5), ("B", 5), ("C", 5), ("D", 4)])
def test_exact_final_artifacts_and_registered_suites_retained(tmp_path, cell, count):
    raw, status = fixture(tmp_path)
    if cell == "D":
        status["jobs"]["train-r5-block-20"] = {"status": "pending"}
    plan = evaluation_spec(raw, cell, status, tmp_path)
    assert plan["training_stages"] == []
    assert len(plan["replicas"]) == count
    assert plan["evaluation_suites"] == raw["evaluation_suites"]
    assert raw["training_stages"] != []
    assert all(r["parent_artifact"] == str((tmp_path / "checkpoint.pt").resolve())
               for r in plan["replicas"])


def test_does_not_silently_drop_other_failed_replicas(tmp_path):
    raw, status = fixture(tmp_path)
    status["jobs"]["train-r2-block-20"]["status"] = "failed"
    with pytest.raises(ValueError, match="exact stage budget"):
        evaluation_spec(raw, "D", status, tmp_path)


def test_rejects_checkpoint_corruption(tmp_path):
    raw, status = fixture(tmp_path)
    (tmp_path / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        evaluation_spec(raw, "A", status, tmp_path)


def test_budget_extension_preserves_registration_and_rejects_other_changes():
    from scripts.evaluate_issue124_reduced import effective_limits

    protocol = {"runner_sha256": "old", "limits": {
        "cpu_seconds": 28800, "wall_seconds": 28800, "memory_bytes": 2 * 1024**3}}
    extension = {"original_runner_sha256": "old", "runner_sha256": "new",
                 "original_limits": dict(protocol["limits"]), "limits": {
                     "cpu_seconds": 115200, "wall_seconds": 86400,
                     "memory_bytes": 2 * 1024**3}}
    assert effective_limits(protocol, extension, "new") == extension["limits"]
    assert effective_limits(protocol, None, "old") == protocol["limits"]
    assert protocol["limits"]["wall_seconds"] == 28800
    with pytest.raises(ValueError):
        effective_limits(protocol, extension, "other")
    with pytest.raises(ValueError):
        effective_limits(protocol, {**extension, "limits": {
            **extension["limits"], "memory_bytes": 8 * 1024**3}}, "new")
