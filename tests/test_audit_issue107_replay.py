"""Checks for the read-only historical checkpoint audit."""

import numpy as np
import pytest

from scripts.audit_issue107_replay import audit, partition_digest


def test_audit_rejects_unregistered_archive_before_loading(tmp_path):
    archive = tmp_path / "wrong.tar.gz"
    archive.write_bytes(b"not registered")
    with pytest.raises(ValueError, match="registered"):
        audit(archive)


def test_partition_digest_covers_values_shape_and_type():
    original = {"states": np.zeros((2, 3), dtype=np.float32)}
    checksum = partition_digest(original)
    assert checksum != partition_digest({"states": np.ones((2, 3), dtype=np.float32)})
    assert checksum != partition_digest({"states": np.zeros((3, 2), dtype=np.float32)})
    assert checksum != partition_digest({"states": np.zeros((2, 3), dtype=np.int32)})
