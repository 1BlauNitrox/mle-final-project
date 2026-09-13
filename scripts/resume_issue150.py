"""Resume pinned Issue 150 with equivalent UTC spelling adapted only in memory.

Download outside the campaign checkout. Default: read-only preflight. Execution
retains the original monitor and all its limits, and logs this runtime adapter.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "6f014485a3026cc3707fa2cc3a379880dd0b74bd"
PROTOCOL = "69bff393fb2637d8be2d678cdec3d66aa7c8286e570b3e0b4c19c03b28fe8e27"
LIMITS = {"cpu_seconds": 86400, "wall_seconds": 36000, "memory_bytes": 8 * 1024**3}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def instant(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Authorization time must include a timezone")
    return parsed.astimezone(timezone.utc)


def monitor_adapter(original):
    """Delegate unchanged checks after proving both spellings mean the same instant."""

    def create(**kwargs):
        state = read(Path(kwargs["state_path"]))
        stored = state["authorized_at"]
        require(
            instant(stored) == instant(kwargs["authorized_at"]),
            "Campaign resource state belongs to another authorization instant",
        )
        return original(**dict(kwargs, authorized_at=stored))

    return create


def check_records(root, now):
    runs = root / "runs"
    require(not (runs / ".task3-campaign.lock").exists(), "Campaign lock exists")
    require(not (root / "run-completed").exists(), "Campaign already completed")
    auth = read(runs / "authorization.json")
    state = read(runs / "resources.json")
    identity = auth["identity"]
    require(identity["issue"] == 150, "Wrong campaign")
    require(identity["reviewed_commit"] == SOURCE, "Wrong campaign source")
    require(identity["protocol_sha256"] == PROTOCOL, "Wrong protocol")
    require(instant(auth["authorized_at"]) == instant(state["authorized_at"]), "Different instants")
    require(state["limits"] == LIMITS, "Changed resource limits")
    require(not state["active_root_pids"], "Recorded active workers")
    require(state["limit_reached"] is None, "Recorded resource limit reached")
    elapsed = (now - instant(auth["authorized_at"])).total_seconds()
    require(0 <= elapsed < LIMITS["wall_seconds"], "Original wall budget expired or future time")
    cpu = state["cpu_seconds_consumed"]
    require(math.isfinite(cpu) and 0 <= cpu < LIMITS["cpu_seconds"], "CPU budget exhausted")
    require(sha(root / "binding/binding.json") == identity["binding_sha256"], "Binding changed")
    hashes = {
        p.relative_to(root).as_posix(): sha(p)
        for p in (
            runs / "authorization.json",
            runs / "resources.json",
            root / "binding/binding.json",
        )
    }
    counts, training = {}, 0
    for status in sorted((runs / "plans").glob("*/status.json")):
        hashes[status.relative_to(root).as_posix()] = sha(status)
        counts[status.parent.name] = {}
        for job in read(status)["jobs"].values():
            tally = counts[status.parent.name]
            tally[job["status"]] = tally.get(job["status"], 0) + 1
            if job["kind"] == "training":
                training += 1
                require(job["status"] == "completed", "This adapter cannot resume training")
                artifact = job["artifact"]
                paths = [
                    status.parent / artifact["path"],
                    status.parent / "replicas" / job["replica"] / "agent/checkpoint.pt",
                ]
                for path in paths:
                    require(path.resolve().is_relative_to(runs.resolve()), "Checkpoint path escape")
                    value = sha(path)
                    require(value == artifact["sha256"], "Training checkpoint/workspace changed")
                    hashes[path.relative_to(root).as_posix()] = value
    require(training == 10, "Expected ten completed training jobs")
    require(
        set(counts) == {"issue150-" + arm for arm in ("reference", "control", "double")},
        "Unexpected plan set",
    )
    return auth, {
        "scope": "issue157_equivalent_timestamp_resume_adapter",
        "source": SOURCE,
        "helper_sha256": sha(Path(__file__)),
        "authorization_time": auth["authorized_at"],
        "resource_time": state["authorized_at"],
        "remaining_wall_seconds": LIMITS["wall_seconds"] - elapsed,
        "remaining_cpu_seconds": LIMITS["cpu_seconds"] - cpu,
        "retained_resources": state,
        "before_sha256": hashes,
        "jobs": counts,
    }


def run_campaign(root, auth):
    """Original CLI and validator still enforce complete identity, plans and limits."""
    from training import run_task3_campaign, task3_double_campaign

    original = run_task3_campaign.CampaignResourceMonitor
    old_argv = sys.argv
    identity = auth["identity"]
    run_task3_campaign.CampaignResourceMonitor = monitor_adapter(original)
    sys.argv = [
        "training.task3_double_campaign",
        "run",
        "--resume",
        "--authorize-compute",
        "--binding-dir",
        str(root / "binding"),
        "--output-root",
        str(root / "runs"),
        "--reviewed-commit",
        SOURCE,
        "--authorized-by",
        identity["authorized_by"],
        "--hardware-description",
        identity["hardware_description"],
        "--available-memory-gib",
        str(identity["available_memory_gib"]),
        "--allocation-hours",
        "12",
    ]
    try:
        task3_double_campaign.main()
    finally:
        run_task3_campaign.CampaignResourceMonitor = original
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    repo = root / "repo"

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()

    require(git("rev-parse", "HEAD") == SOURCE, "Keep original pinned checkout")
    require(not git("status", "--porcelain"), "Campaign checkout must be clean")
    require(not Path(__file__).resolve().is_relative_to(repo), "Download helper outside checkout")
    import shutil

    import psutil

    require(psutil.virtual_memory().available >= LIMITS["memory_bytes"], "Need 8 GiB free RAM")
    require(shutil.disk_usage(root).free >= LIMITS["memory_bytes"], "Need 8 GiB free disk")
    auth, report = check_records(root, datetime.now(timezone.utc))
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return
    require(
        os.environ.get("TASK150_AUTHORIZED") == "yes", "Explicit existing authorization required"
    )
    journal = root / ("resume-" + uuid.uuid4().hex + ".jsonl")
    with journal.open("x", encoding="utf-8") as log:

        def record(value):
            log.write(json.dumps(value) + "\n")
            log.flush()
            os.fsync(log.fileno())

        record(dict(report, phase="before"))
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        try:
            run_campaign(root, auth)
        except BaseException as error:
            record({"phase": "failed", "error": str(error)})
            raise
        require(
            sha(root / "runs/authorization.json")
            == report["before_sha256"]["runs/authorization.json"],
            "Authorization bytes changed",
        )
        with (root / "run-completed").open("x", encoding="utf-8") as marker:
            marker.write(SOURCE + "\n")
        record({"phase": "campaign_complete", "resources": read(root / "runs/resources.json")})
    # The original supervisor now performs its normal analysis, verification and export.
    environment = dict(os.environ, TASK150_PYTHON=sys.executable)
    subprocess.run(
        ["bash", str(repo / "scripts/run_issue150_server.sh"), str(root), SOURCE],
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
