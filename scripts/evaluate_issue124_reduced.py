"""Evaluate completed Issue 124 final checkpoints under a reduced-scope amendment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_spec(raw, cell, status, source):
    """Bind only exact-budget checkpoints; never silently discard another replica."""
    raw = dict(raw)
    raw["plan_id"] = "issue124-reduced-eval-" + cell.lower()
    raw["training_stages"] = []
    raw["max_parallel_training"] = 1
    replicas = []
    for replica in raw["replicas"]:
        replica = dict(replica)
        if cell == "D" and replica["id"] == "r5":
            continue
        job = status["jobs"][f"train-{replica['id']}-block-20"]
        if job["status"] != "completed":
            raise ValueError("Included replica has not completed its exact stage budget")
        artifact = source / job["artifact"]["path"]
        if digest(artifact) != job["artifact"]["sha256"]:
            raise ValueError("Final checkpoint checksum mismatch")
        replica["parent_artifact"] = str(artifact.resolve())
        replicas.append(replica)
    if len(replicas) != (4 if cell == "D" else 5):
        raise ValueError("Unexpected replica population")
    raw["replicas"] = replicas
    return raw


def effective_limits(protocol, extension, runner_digest):
    """Accept only the recorded owner extension; preserve original registration."""
    if extension is None:
        if protocol["runner_sha256"] != runner_digest:
            raise ValueError("Registered evaluation runner changed")
        return protocol["limits"]
    expected = {"cpu_seconds": 115200, "wall_seconds": 86400,
                "memory_bytes": 2 * 1024**3}
    if (extension["original_runner_sha256"] != protocol["runner_sha256"]
            or extension["runner_sha256"] != runner_digest
            or extension["original_limits"] != protocol["limits"]
            or extension["limits"] != expected):
        raise ValueError("Invalid owner evaluation extension")
    return expected

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    execution, source, output = (
        args.execution_root.resolve(), args.campaign_root.resolve(), args.output_root.resolve()
    )
    from scripts.resume_issue124_amendment import resume_monitor_type

    sys.path.insert(0, str(execution))
    from training import run_issue124_campaign as campaign
    from training.run_plan import execute_plan, load_plan

    if campaign.git("rev-parse", "HEAD") != "1ce18c8736b6a50773b60b95ec0f11dd4c901028":
        raise ValueError("Training execution source changed")
    campaign.validate_protocol()
    if (source / "campaign.lock").exists():
        raise ValueError("Training campaign lock exists")
    limits = campaign.CampaignLimits(cpu_seconds=28800, wall_seconds=28800,
                                    memory_bytes=2 * 1024**3)
    protocol_path = output / "evaluation-amendment.json"
    if not protocol_path.exists():
        output.mkdir(parents=True, exist_ok=False)
        plans = []
        for cell, original in campaign.PLANS.items():
            raw = yaml.safe_load(original.read_text())
            if cell in "ABCD":
                directory = source / "run-plans" / raw["plan_id"]
                status = json.loads((directory / "status.json").read_text())
                raw = evaluation_spec(raw, cell, status, directory)
            else:
                raw["plan_id"] = "issue124-reduced-eval-" + cell
                raw["training_stages"] = []
                for replica in raw["replicas"]:
                    replica["parent_artifact"] = str(
                        (original.parent / replica["parent_artifact"]).resolve())
            path = output / (cell + ".yaml")
            path.write_text(yaml.safe_dump(raw, sort_keys=False))
            plan = load_plan(path)
            if any(job.kind != "evaluation" for job in plan.jobs):
                raise ValueError("Training forbidden in reduced evaluation")
            plans.append({"cell": cell, "path": path.name, "sha256": digest(path),
                          "episodes": sum(job.rounds for job in plan.jobs),
                          "artifacts": {r.replica_id: r.parent_artifact_sha256
                                        for r in plan.replicas}})
        if sum(p["episodes"] for p in plans) != 4880:
            raise ValueError("Unexpected evaluation budget")
        campaign.write_json(protocol_path, {
            "issue": 124, "scope": "reduced_evaluation_after_technical_interruption",
            "owner_instruction": "Start the evaluation now", "limits": vars(limits),
            "training_episodes": 0, "evaluation_episodes": 4880,
            "excluded": "D/r5 incomplete; retain all four completed D replicas as exploratory",
            "claims": "A/B/C five-replica comparisons; no original full-matrix completion claim",
            "decision": "Original metrics, gates and four-test correction retained; D exploratory",
            "source": str(source), "source_resources_sha256": digest(source / "resources.json"),
            "runner_sha256": digest(__file__), "plans": plans,
        })
    protocol = campaign.read_json(protocol_path)
    extension_path = output / "runtime-extension.json"
    extension = campaign.read_json(extension_path) if extension_path.exists() else None
    limits = campaign.CampaignLimits(**effective_limits(protocol, extension, digest(__file__)))
    plans = []
    for record in protocol["plans"]:
        path = output / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("Registered evaluation plan changed")
        plan = load_plan(path)
        if any(j.kind != "evaluation" for j in plan.jobs):
            raise ValueError("Training forbidden")
        if {r.replica_id: r.parent_artifact_sha256 for r in plan.replicas} != record["artifacts"]:
            raise ValueError("Registered final checkpoint changed")
        plans.append(plan)
    if args.prepare_only:
        print(json.dumps(protocol, indent=2))
        return
    if psutil.virtual_memory().available < 2 * 1024**3:
        raise ValueError("Single-worker evaluation requires 2 GiB available RAM")
    # Exclusive lock prevents duplicate evaluation supervisors. Preserve stale locks for inspection.
    with (output / "evaluation.lock").open("x") as handle:
        handle.write(str(psutil.Process().pid))
    authorization = output / "authorization.json"
    if not authorization.exists():
        campaign.write_json(authorization, {
            "authorized_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "protocol_sha256": digest(protocol_path), "limits": vars(limits)})
    auth = campaign.read_json(authorization)
    monitor = resume_monitor_type(campaign.LocalMonitor)(
        state_path=output / "resources.json", authorized_at=auth["authorized_at"],
        limits=limits, campaign_metadata={"reduced_evaluation": protocol, "authorization": auth,
                                         "runtime_extension": extension})
    # Request wakefulness only for this supervisor's lifetime; do not change power settings.
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    stop = threading.Event()

    def watch():
        while not stop.wait(1):
            try:
                if psutil.virtual_memory().available < 1024**3:
                    monitor.abort("System free RAM fell below 1 GiB")
                monitor.check()
            except Exception as error:
                monitor.cancel(str(error))
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        campaign.write_json(output / "status.json", {"status": "running",
                                                    "pid": psutil.Process().pid})
        for plan in plans:
            monitor.check()
            execute_plan(plan, output_root=output / "run-plans",
                         resume=(output / "run-plans" / plan.plan_id).exists(),
                         process_monitor=monitor)
        monitor.check()
        campaign.write_json(output / "status.json", {
            "status": "evaluation_completed", "scientific_analysis": "pending",
            "episodes": 4880, "task2_complete": False})
    except BaseException as error:
        campaign.write_json(output / "status.json", {"status": "stopped_incomplete",
                                                    "error": str(error)})
        raise
    finally:
        stop.set()
        watcher.join(timeout=5)
        (output / "evaluation.lock").unlink(missing_ok=True)
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    # Keep the reviewed operational helpers available alongside immutable training code.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
