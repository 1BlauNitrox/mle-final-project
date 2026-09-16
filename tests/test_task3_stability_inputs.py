import hashlib
import io
import tarfile

import pytest

from scripts import fetch_task3_stability_inputs as fetch


def test_extract_verifies_both_inputs_and_preserves_existing(tmp_path, monkeypatch):
    files = {"initial.pt": b"initial", "reference.pt": b"reference"}
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(
        fetch, "INPUTS", {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    )
    monkeypatch.setattr(fetch, "ARCHIVE_SHA", fetch.digest(archive))
    output = tmp_path / "inputs"
    fetch.extract_inputs(archive, output)
    fetch.extract_inputs(archive, output)
    assert (output / "reference.pt").read_bytes() == b"reference"
    (output / "initial.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing"):
        fetch.extract_inputs(archive, output)
    assert (output / "initial.pt").read_bytes() == b"changed"
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="archive checksum"):
        fetch.extract_inputs(archive, tmp_path / "new")
    assert not (tmp_path / "new").exists()
