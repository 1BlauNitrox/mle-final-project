"""Prospective #163 snapshot storage and bounded same-PC execution."""

import math
import shutil
from pathlib import Path

from training import issue150_storage as previous
from training.immutable_snapshots import ImmutableSnapshots, probe_links

GIB = 1024**3
STORE_BYTES = 2 * GIB
DIRECTORY_BYTES = 2 * GIB


def storage_budget(plans):
    original = previous.storage_budget(plans)
    records = original["record_bytes"]
    required = (
        3 * records
        + 3 * previous.WORKSPACE_BYTES
        + STORE_BYTES
        + DIRECTORY_BYTES
        + previous.FREE_RESERVE_BYTES
    )
    return {
        **original,
        "policy": "immutable_input_objects_v1",
        "retained_snapshot_bytes": STORE_BYTES,
        "directory_allowance_bytes": DIRECTORY_BYTES,
        "required_free_bytes": math.ceil(required / GIB) * GIB,
        "retries": "Full headroom required on resume; preserve every attempt.",
    }


def preflight(plans, output_root, source_root):
    report = storage_budget(plans)
    filesystem = previous.existing_directory(output_root)
    source_filesystem = previous.existing_directory(source_root)
    if shutil.disk_usage(filesystem).free < report["required_free_bytes"]:
        raise ValueError(f"Need {report['required_free_bytes'] / GIB:.0f} GiB free on {filesystem}")
    if shutil.disk_usage(source_filesystem).free < 6 * GIB:
        raise ValueError("Need 6 GiB free on source filesystem")
    probe_links(filesystem)
    for plan in plans.values():
        size = previous.tree_bytes(Path(source_root) / "agent_code" / plan.agent, snapshot=True)
        parent = max(
            (Path(r.parent_artifact).stat().st_size for r in plan.replicas if r.parent_artifact),
            default=0,
        )
        if size + parent > previous.SNAPSHOT_BYTES:
            raise ValueError("Agent/parent exceeds snapshot envelope")
    return {
        **report,
        "observed_free_bytes": shutil.disk_usage(filesystem).free,
        "output_filesystem": str(filesystem),
    }


class StorageMonitor(previous.StorageMonitor):
    snapshot_storage = "immutable_input_objects_v1"

    def __init__(self, **kwargs):
        self.snapshots = ImmutableSnapshots(Path(kwargs["state_path"]).parent / "input-objects")
        super().__init__(**kwargs)

    def snapshot_input(self, source, destination):
        self.snapshots.snapshot(source, destination)
        self.check()

    def check(self):
        # Retained input objects are bounded independently of live aliases and per-job outputs.
        with self._lock:
            if not self._limit_reached and previous.tree_bytes(self.snapshots.root) > STORE_BYTES:
                self._limit_reached = "Immutable input store exceeds 2 GiB envelope"
                self._sample_locked()
                self._persist(*self._usage_locked())
        super().check()
