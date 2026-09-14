"""Independent live workspaces and content-addressed, read-only retained inputs."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from pathlib import Path


class ImmutableSnapshots:
    """Copy bytes into a private object store, never link a live model into it."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self._lock = threading.Lock()

    def snapshot(self, source, destination):
        source, destination = Path(source).resolve(), Path(destination).absolute()
        if destination.exists():
            raise FileExistsError(f"Preserve retained snapshot: {destination}")
        if source == self.root or self.root in source.parents or source in self.root.parents:
            raise ValueError("Snapshot store and live source must be separate")
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            if temporary.exists():
                raise FileExistsError(f"Preserve incomplete snapshot: {temporary}")
            temporary.mkdir(parents=True)
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source)
                if {"logs", "__pycache__"} & set(relative.parts) or path.suffix == ".pyc":
                    continue
                if path.is_symlink():
                    raise ValueError("Refuse symlink in immutable input")
                if not path.is_file():
                    continue
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                obj = self.root / digest
                if obj.exists():
                    if obj.is_symlink() or obj.read_bytes() != data:
                        raise ValueError("Immutable snapshot object changed")
                else:
                    # Exclusive creation and fsync; failed writes remain evidence and fail closed.
                    with obj.open("xb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    obj.chmod(stat.S_IREAD)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(obj, target)
                if target.read_bytes() != data:
                    raise ValueError("Linked snapshot differs from input")
            temporary.replace(destination)


def probe_links(root):
    """Test support in a unique temporary directory before authorizing execution."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="snapshot-probe-", dir=root) as directory:
        source = Path(directory) / "source"
        target = Path(directory) / "target"
        source.write_bytes(b"probe")
        os.link(source, target)
        if not os.path.samefile(source, target):
            raise ValueError("Filesystem does not support hard links")
