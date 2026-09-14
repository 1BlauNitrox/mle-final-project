import hashlib
import io
import json
import tarfile

import pytest

from scripts.package_issue163_evidence import record, verify


def archive_fixture(tmp_path, name="source/code.py", data=b"safe"):
    path = tmp_path / "evidence.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    manifest = {
        **record(path),
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}},
    }
    sidecar = tmp_path / "evidence.tar.gz.manifest.json"
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    return path, sidecar


def test_roundtrip_verifies_and_preserves_existing_destination(tmp_path):
    archive, _ = archive_fixture(tmp_path)
    destination = tmp_path / "extracted"
    assert verify(archive, destination)["verified_files"] == 1
    assert (destination / "source/code.py").read_bytes() == b"safe"
    with pytest.raises(FileExistsError):
        verify(archive, destination)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", r"a\escape"])
def test_rejects_unsafe_members(tmp_path, name):
    archive, _ = archive_fixture(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe"):
        verify(archive, tmp_path / "extracted")
    assert not (tmp_path / "escape").exists()


def test_detects_corrupted_archive_and_member_manifest(tmp_path):
    archive, sidecar = archive_fixture(tmp_path)
    manifest = json.loads(sidecar.read_text())
    manifest["files"]["source/code.py"]["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Member bytes differ"):
        verify(archive)
    with archive.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="Archive hash/size mismatch"):
        verify(archive)
