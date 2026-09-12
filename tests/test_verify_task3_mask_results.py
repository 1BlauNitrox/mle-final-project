"""Historical mask evidence must retain original identity and fail closed."""

import json
import tarfile

import pytest
import yaml

from training import verify_task3_mask_results as verifier


def fixture(root):
    root.mkdir()
    binding = {"artifacts": {}, "plans": {}}
    for arm in verifier.mask.ARMS:
        artifact = root / (arm + ".pt")
        artifact.write_bytes(arm.encode())
        binding["artifacts"][arm] = {
            "path": artifact.name,
            "sha256": verifier.mask.sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        plan = root / (arm + ".yaml")
        plan.write_text(
            yaml.safe_dump({"replicas": [{"parent_artifact": "/server/" + artifact.name}]})
        )
        binding["plans"][arm] = {"path": plan.name, "sha256": verifier.mask.sha256(plan)}
    (root / "binding.json").write_text(json.dumps(binding))


def test_relocation_only_changes_scratch_paths(tmp_path):
    source, target = tmp_path / "original", tmp_path / "copy"
    fixture(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    hashes = verifier.relocate(source, target)
    assert set(hashes) == set(verifier.mask.ARMS)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    for arm in verifier.mask.ARMS:
        plan = yaml.safe_load((target / (arm + ".yaml")).read_text())
        assert plan["replicas"][0]["parent_artifact"] == str(target / (arm + ".pt"))


@pytest.mark.parametrize("name", ["reference.pt", "control.pt", "masked.pt", "masked.yaml"])
def test_relocation_rejects_tampering(tmp_path, name):
    source = tmp_path / "original"
    fixture(source)
    (source / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verifier.relocate(source, tmp_path / "copy")


def test_execution_revision_checked_before_git_or_analysis(tmp_path):
    (tmp_path / "authorization.json").write_text(
        json.dumps({"identity": {"reviewed_commit": "bad"}})
    )
    with pytest.raises(ValueError, match="execution SHA"):
        verifier.verify(tmp_path, tmp_path, tmp_path / "output")


@pytest.mark.parametrize("damage", [None, "archive", "inventory", "member"])
def test_export_checks_outer_and_member_integrity(tmp_path, damage):
    data = tmp_path / "data"
    data.write_bytes(b"evidence")
    archive = tmp_path / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(data, arcname="campaign/data")
    manifest = {
        "sha256": verifier.mask.sha256(archive),
        "size_bytes": archive.stat().st_size,
        "files": {"campaign/data": {"sha256": verifier.mask.sha256(data), "size_bytes": 8}},
    }
    if damage == "archive":
        manifest["sha256"] = "wrong"
    elif damage == "inventory":
        manifest["files"] = {}
    elif damage == "member":
        manifest["files"]["campaign/data"]["sha256"] = "wrong"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    if damage:
        with pytest.raises(ValueError, match="mismatch"):
            verifier.verify_export(archive, path, tmp_path / "import")
    else:
        verifier.verify_export(archive, path, tmp_path / "import")
        assert (tmp_path / "import/campaign/data").read_bytes() == b"evidence"
        with pytest.raises(FileExistsError):
            verifier.verify_export(archive, path, tmp_path / "import")
