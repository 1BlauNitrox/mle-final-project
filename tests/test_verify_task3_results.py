"""Portable historical verification must retain identity and reject unsafe inputs."""

import hashlib
import io
import json
import tarfile

import pytest
import yaml

from training import verify_task3_results as verifier


def test_latency_round_lookup_survives_lexically_sorted_json():
    raw = {
        "by_round": {
            str(i): {"agents": {"learner": {"decision_times_ms": [i / 10]}}} for i in range(1, 13)
        }
    }
    raw = json.loads(json.dumps(raw, sort_keys=True))
    assert list(raw["by_round"])[1] == "10"
    indexed = verifier.analysis.index_rounds(raw)
    assert indexed[2]["agents"]["learner"]["decision_times_ms"] == [0.2]
    assert indexed[10]["agents"]["learner"]["decision_times_ms"] == [1.0]
    timestamped = {
        "by_round": {f"Round {key} (2026-09-11)": value for key, value in raw["by_round"].items()}
    }
    assert verifier.analysis.index_rounds(timestamped) == indexed
    raw["by_round"]["Round 1 (2026-09-11)"] = raw["by_round"]["1"]
    with pytest.raises(ValueError, match="Duplicate"):
        verifier.analysis.index_rounds(raw)


@pytest.mark.parametrize(
    "name,kind",
    [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)],
)
def test_archive_rejects_unsafe_members(tmp_path, name, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        archive.addfile(member)
    data.seek(0)
    with tarfile.open(fileobj=data) as archive, pytest.raises(ValueError):
        verifier.extract_files(archive, tmp_path)


def test_fingerprint_uses_posix_order_and_file_contents(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a/z.py").write_bytes(b"z")
    (tmp_path / "a/B.py").write_bytes(b"B")
    expected = hashlib.sha256(b"a/B.py\0B\0a/z.py\0z\0").hexdigest()
    assert verifier.fingerprint(tmp_path, ("a",)) == expected
    (tmp_path / "a/z.py").write_bytes(b"changed")
    assert verifier.fingerprint(tmp_path, ("a",)) != expected


def binding_fixture(root, monkeypatch):
    root.mkdir()
    binding = {
        "parent": {
            "path": "parent.pt",
            "source_commit": "cbd52be8392f5003a91c5600fda4efd544b48ec5",
        },
        "successor": {"path": "successor.pt"},
        "plans": {},
    }
    for key in ("parent", "successor"):
        record = binding[key]
        path = root / record["path"]
        path.write_bytes(key.encode())
        record.update(sha256=verifier.campaign.sha256(path), size_bytes=path.stat().st_size)
    monkeypatch.setattr(verifier, "PARENT", binding["parent"]["sha256"])
    for arm, key in (("candidate", "successor"), ("reference", "parent")):
        path = root / (arm + ".yaml")
        path.write_text(
            yaml.safe_dump(
                {
                    "replicas": [
                        {"id": "r1", "parent_artifact": "/server/binding/" + binding[key]["path"]}
                    ]
                }
            )
        )
        binding["plans"][arm] = {"path": path.name, "sha256": verifier.campaign.sha256(path)}
    (root / "binding.json").write_text(json.dumps(binding))


def test_relocation_preserves_original_evidence(tmp_path, monkeypatch):
    source, destination = tmp_path / "original", tmp_path / "relocated"
    binding_fixture(source, monkeypatch)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    fingerprints = verifier.rebind_copy(source, destination)
    assert set(fingerprints) == {"candidate", "reference"}
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    data = yaml.safe_load((destination / "candidate.yaml").read_text())
    assert data["replicas"][0]["parent_artifact"] == str(destination / "successor.pt")


@pytest.mark.parametrize("name", ["parent.pt", "successor.pt", "candidate.yaml"])
def test_relocation_rejects_changed_bytes(tmp_path, monkeypatch, name):
    source = tmp_path / "original"
    binding_fixture(source, monkeypatch)
    (source / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verifier.rebind_copy(source, tmp_path / "relocated")


def test_wrong_execution_revision_is_rejected_before_source_access(tmp_path):
    (tmp_path / "authorization.json").write_text(json.dumps({"identity": {"reviewed_commit": "x"}}))
    with (
        pytest.raises(ValueError, match="execution SHA"),
        verifier.historical_context(tmp_path, tmp_path),
    ):
        pytest.fail("Invalid evidence entered verification context")
