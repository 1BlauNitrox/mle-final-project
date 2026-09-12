"""Conservative fresh-run storage sizing and fail-closed runtime guards for #150.

The envelopes are enforced operational limits, not extrapolations from compressed
archives. Exceeding one preserves an incomplete run; it never discards evidence.
Historical source 6f01448 and its external recovery adapters remain independent.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

from training.run_issue107_campaign import CampaignResourceMonitor

MIB = 1024**2
GIB = 1024**3
SNAPSHOT_BYTES = 8 * MIB
EVALUATION_BYTES = MIB
TRAINING_BYTES = 512 * MIB
WORKSPACE_BYTES = 2 * GIB
FREE_RESERVE_BYTES = 4 * GIB


def existing_directory(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"Storage path is not a directory: {path}")
    return path


def tree_bytes(path, *, snapshot=False):
    """Count logical bytes, conservatively charging hard links independently."""
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        relative = item.relative_to(path)
        if snapshot and ({"__pycache__", "logs"} & set(relative.parts) or item.suffix == ".pyc"):
            continue
        if item.is_symlink():
            raise ValueError(f"Symlink in campaign storage: {item}")
        if item.is_file():
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                # Atomic checkpoint replacement may remove its temporary file
                # between enumeration and stat; the replacement is sampled next.
                continue
    return total


def storage_budget(plans):
    jobs = [job for plan in plans.values() for job in plan.jobs]
    training = sum(job.kind == "training" for job in jobs)
    evaluation = sum(job.kind == "evaluation" for job in jobs)
    if training + evaluation != len(jobs):
        raise ValueError("Unknown job kind in storage matrix")
    # Each attempt retains its input-agent tree and may retain an agent_snapshot.
    # Live aliases, atomic replacements, final checkpoints and status writes get
    # a separate allowance. No credit is taken for hard-link deduplication.
    snapshots = len(jobs) * 2 * SNAPSHOT_BYTES
    records = training * TRAINING_BYTES + evaluation * EVALUATION_BYTES
    # Export omits per-attempt snapshots. Allow two full uncompressed copies of
    # records + workspaces for archive overhead, manifests and analysis output.
    export = 2 * (records + WORKSPACE_BYTES)
    required = snapshots + records + WORKSPACE_BYTES + export + FREE_RESERVE_BYTES
    return {
        "schema_version": 1,
        "training_jobs": training,
        "evaluation_jobs": evaluation,
        "snapshot_limit_bytes": SNAPSHOT_BYTES,
        "evaluation_output_limit_bytes": EVALUATION_BYTES,
        "training_output_limit_bytes": TRAINING_BYTES,
        "retained_snapshot_bytes": snapshots,
        "record_bytes": records,
        "workspace_bytes": WORKSPACE_BYTES,
        "export_analysis_bytes": export,
        "free_reserve_bytes": FREE_RESERVE_BYTES,
        "required_free_bytes": math.ceil(required / GIB) * GIB,
        "retries": "Recheck full headroom on resume; retained attempts are never removed.",
    }


def preflight(plans, output_root, source_root):
    """Check the output filesystem, not merely the launcher's working directory."""
    report = storage_budget(plans)
    output_device = existing_directory(output_root)
    source_device = existing_directory(source_root)
    free = shutil.disk_usage(output_device).free
    if free < report["required_free_bytes"]:
        raise ValueError(
            f"Issue 150 needs {report['required_free_bytes'] / GIB:.0f} GiB free on "
            f"{output_device}; found {free / GIB:.2f} GiB. Includes retained input "
            "snapshots, raw records, workspaces, export and reserve; 8 GiB is insufficient. "
            "Do not modify or restart the historical campaign with this source."
        )
    if shutil.disk_usage(source_device).free < WORKSPACE_BYTES + FREE_RESERVE_BYTES:
        raise ValueError("Need 6 GiB free on the source filesystem for live aliases and reserve")
    for plan in plans.values():
        size = tree_bytes(Path(source_root) / "agent_code" / plan.agent, snapshot=True)
        parent_size = max(
            (Path(r.parent_artifact).stat().st_size for r in plan.replicas if r.parent_artifact),
            default=0,
        )
        if size + parent_size > SNAPSHOT_BYTES:
            raise ValueError("Agent source/parent exceeds the 8 MiB snapshot envelope")
    return {**report, "output_filesystem": str(output_device), "observed_free_bytes": free}


class StorageMonitor(CampaignResourceMonitor):
    """Keep original CPU/wall/RAM enforcement and persist storage failures with it."""

    def __init__(self, *, source_root, **kwargs):
        self.storage_paths = (
            existing_directory(Path(kwargs["state_path"]).parent),
            existing_directory(source_root),
        )
        self.storage_jobs = {}
        state_path = Path(kwargs["state_path"])
        if state_path.is_file():
            stored = json.loads(state_path.read_text(encoding="utf-8"))["authorized_at"]
            supplied = kwargs["authorized_at"]
            left = datetime.fromisoformat(stored.replace("Z", "+00:00"))
            right = datetime.fromisoformat(supplied.replace("Z", "+00:00"))
            if left.utcoffset() is None or right.utcoffset() is None or left != right:
                raise ValueError("Campaign resource state belongs to another authorization")
            # Compare instants without rewriting authorization or resource data.
            # The original monitor still checks all limits and retained usage.
            kwargs["authorized_at"] = stored
        super().__init__(**kwargs)

    def begin_job(self, job, alias, attempt_root):
        with self._lock:
            self.storage_jobs[job.run_id] = (job.kind, Path(alias), Path(attempt_root))
        self.check()

    def end_job(self, job):
        with self._lock:
            self.storage_jobs.pop(job.run_id, None)

    def check(self):
        # The original monitor persists the breach and terminates its entire
        # registered process forest. Never reset a previous storage/CPU/wall stop.
        with self._lock:
            if not self._limit_reached:
                try:
                    for path in self.storage_paths:
                        if shutil.disk_usage(path).free < FREE_RESERVE_BYTES:
                            raise ValueError(f"Storage reserve below 4 GiB on {path}")
                    for kind, alias, attempt in self.storage_jobs.values():
                        if tree_bytes(alias, snapshot=True) > SNAPSHOT_BYTES:
                            raise ValueError("Agent workspace exceeds 8 MiB storage envelope")
                        for path in attempt.parent.glob(attempt.name + "-*-agent*"):
                            if tree_bytes(path, snapshot=True) > SNAPSHOT_BYTES:
                                raise ValueError("Input/failure snapshot exceeds storage envelope")
                        snapshot = attempt / "agent_snapshot"
                        if tree_bytes(snapshot, snapshot=True) > SNAPSHOT_BYTES:
                            raise ValueError("Metadata snapshot exceeds storage envelope")
                        output = tree_bytes(attempt) - tree_bytes(snapshot)
                        limit = TRAINING_BYTES if kind == "training" else EVALUATION_BYTES
                        if output > limit:
                            raise ValueError(f"{kind} output exceeds storage envelope: {output}")
                except (ValueError, OSError) as error:
                    self._limit_reached = f"Storage guard: {error}"
                    self._sample_locked()
                    self._persist(*self._usage_locked())
        super().check()
