"""Bounded four-arm local campaign; train concurrently, then evaluate serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml

from training.run_experiment import REPOSITORY_ROOT
from training.run_issue107_campaign import (
    SOURCE_SHA256,
    TASK1_SHA256,
    CampaignLimits,
    CampaignResourceMonitor,
    _seed_values_from_path,
)
from training.run_plan import execute_plan, load_plan

ROOT = REPOSITORY_ROOT
CONFIG = ROOT / "experiments/2026-09-10-task2-rehearsal-mask/config.yaml"
PLANS = {cell.upper(): ROOT / f"training/run_plans/issue124-cell-{cell}.yaml" for cell in "abcd"}
PLANS.update(
    {
        name.replace("-", "_"): ROOT / f"training/run_plans/issue124-{name}.yaml"
        for name in ("untrained", "frozen-task1")
    }
)
LIMITS = CampaignLimits(cpu_seconds=40 * 3600, wall_seconds=10 * 3600, memory_bytes=8 * 1024**3)

ANALYSIS_SPEC = {
    "bootstrap_resamples": 10000,
    "confidence_interval_percent": 95,
    "efficacy_family_size": 4,
    "efficacy_interval_percent": 98.75,
    "rehearsal_coin_collection_improvement_min": 0.10,
    "mask_classic_collection_improvement_min": 0.10,
    "adjusted_lower_strictly_above": 0.0,
    "non_regression_collection_survival_margin": 0.05,
    "non_regression_ci_lower_strict": True,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_protocol():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "prospective_protocol" or config["issue"] != 124:
        raise ValueError("Issue #124 protocol is not finalized")
    if config.get("analysis") != ANALYSIS_SPEC:
        raise ValueError("Decision specification differs from the registered analyzer")
    if config["resources"] != {"workers": 4, "memory_gib": 8, "wall_hours": 10, "cpu_hours": 40}:
        raise ValueError("Registered resource limits changed")
    plans = {cell: load_plan(path) for cell, path in PLANS.items()}
    for cell, plan in plans.items():
        if sha256(PLANS[cell]) != config["plan_sha256"][cell]:
            raise ValueError(f"Registered plan bytes changed: {cell}")
        expected_parent = TASK1_SHA256 if cell == "frozen_task1" else SOURCE_SHA256
        if any(replica.parent_artifact_sha256 != expected_parent for replica in plan.replicas):
            raise ValueError(f"Parent artifact changed: {cell}")
        if plan.max_parallel_training != 1:
            raise ValueError("One worker per arm required by the global four-worker design")
    training = [job for plan in plans.values() for job in plan.jobs if job.kind == "training"]
    evaluation = [job for plan in plans.values() for job in plan.jobs if job.kind == "evaluation"]
    if len(training) != 400 or sum(job.rounds for job in training) != 200000:
        raise ValueError("Training matrix differs from registered budget")
    if len(evaluation) != 5120:
        raise ValueError("Evaluation matrix differs from registered budget")
    seeds = {seed for job in training + evaluation for seed in (job.world_seed, job.agent_seed)}
    for path in list((ROOT / "training/run_plans").glob("*.yaml")) + list(
        (ROOT / "experiments").glob("*/config.yaml")
    ):
        if path in PLANS.values() or path == CONFIG:
            continue
        if seeds & _seed_values_from_path(path):
            raise ValueError(f"Seed collision with {path}")
    return plans


class LocalMonitor(CampaignResourceMonitor):
    """Include supervisor usage and stop only this campaign's workers."""

    def __init__(self, **kwargs):
        self.cancelled = None
        self.supervisor = psutil.Process()
        times = self.supervisor.cpu_times()
        self.initial_cpu = times.user + times.system
        super().__init__(**kwargs)

    def _usage_locked(self):
        cpu, memory = super()._usage_locked()
        times = self.supervisor.cpu_times()
        memory += self.supervisor.memory_info().rss
        self._peak_memory_bytes = max(self._peak_memory_bytes, memory)
        return cpu + max(0, times.user + times.system - self.initial_cpu), memory

    def abort(self, reason):
        with self._lock:
            self._limit_reached = self._limit_reached or reason
        self.check()

    def cancel(self, reason):
        """Stop sibling arms on technical failure; preserve the original budget."""
        with self._lock:
            self.cancelled = reason
            processes = self._processes_locked()
        self._terminate(processes)

    def check(self):
        if self.cancelled:
            with self._lock:
                processes = self._processes_locked()
            self._terminate(processes)
            raise RuntimeError(self.cancelled)
        super().check()


def run_phases(plans, output, resume, monitor, executor=execute_plan):
    """No evaluation may begin until all four training arms have stopped."""

    def train(cell):
        plan = plans[cell]
        try:
            executor(
                plan,
                output_root=output,
                resume=resume and (output / plan.plan_id).exists(),
                training_only=True,
                process_monitor=monitor,
            )
        except BaseException:
            with suppress(Exception):
                monitor.cancel(f"Training arm {cell} failed; preserve all attempts")
            raise

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(train, cell) for cell in "ABCD"]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            monitor.cancel("Campaign interrupted; stop only its workers")
            raise
    for plan in plans.values():
        monitor.check()
        executor(
            plan,
            output_root=output,
            resume=(output / plan.plan_id).exists(),
            process_monitor=monitor,
        )


def existing_authorization(output, reviewed_commit, resume):
    """Reject incompatible restarts before changing any retained evidence."""
    path = output / "authorization.json"
    if not path.exists():
        if resume or any(
            (output / name).exists() for name in ("run-plans", "resources.json", "analysis")
        ):
            raise ValueError("No compatible campaign authorization to resume")
        return None
    authorization = read_json(path)
    if (
        not resume
        or authorization["reviewed_commit"] != reviewed_commit
        or authorization["config_sha256"] != sha256(CONFIG)
    ):
        raise ValueError("Cannot change an existing campaign authorization")
    status_path = output / "campaign-status.json"
    if status_path.exists() and read_json(status_path).get("status") == "completed":
        raise ValueError("Campaign is already complete; use the analysis command")
    if not (output / "resources.json").exists():
        raise ValueError("Resume requires resource accounting")
    resource = read_json(output / "resources.json")
    if resource.get("limits") != vars(LIMITS) or resource.get("limit_reached"):
        raise ValueError("Cannot resume changed or exhausted resource limits")
    if any(psutil.pid_exists(pid) for pid in resource["active_root_pids"]):
        raise ValueError("Previous worker PIDs still exist; inspect before resuming")
    return authorization


def execute(output, reviewed_commit, resume=False):
    plans = validate_protocol()
    if reviewed_commit != git("rev-parse", "HEAD") or len(reviewed_commit) != 40:
        raise ValueError("Execution requires the exact reviewed HEAD SHA")
    if git("status", "--porcelain"):
        raise ValueError("Execution requires a clean worktree")
    review = json.loads(
        subprocess.check_output(
            [
                "gh",
                "pr",
                "view",
                "131",
                "--repo",
                "1BlauNitrox/mle-final-project",
                "--json",
                "headRefOid,reviewDecision",
            ],
            text=True,
        )
    )
    if review.get("headRefOid") != reviewed_commit or review.get("reviewDecision") != "APPROVED":
        raise ValueError("PR #131 needs non-author approval on this exact commit")
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise ValueError("At least 4 GiB free RAM is required before starting four workers")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    authorization = existing_authorization(output, reviewed_commit, resume)
    lock = output / "campaign.lock"
    if lock.exists():
        previous = read_json(lock)
        try:
            alive = psutil.Process(previous["pid"]).create_time() == previous["created"]
        except psutil.NoSuchProcess:
            alive = False
        if alive or not resume:
            raise ValueError("Campaign already running or stale lock requires --resume")
        lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, handle)
    stop = threading.Event()
    watchdog = None
    try:
        authorization_path = output / "authorization.json"
        if authorization is None:
            authorization = {
                "issue": 124,
                "reviewed_commit": reviewed_commit,
                "authorized_by": "1BlauNitrox",
                "authorized_at": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256(CONFIG),
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "memory_total_bytes": psutil.virtual_memory().total,
                "limits": vars(LIMITS),
            }
            write_json(authorization_path, authorization)
        monitor = LocalMonitor(
            state_path=output / "resources.json",
            authorized_at=authorization["authorized_at"],
            limits=LIMITS,
            campaign_metadata=authorization,
        )

        def watch():
            while not stop.wait(1):
                try:
                    if psutil.virtual_memory().available < 1024**3:
                        monitor.abort("System free RAM fell below 1 GiB")
                    monitor.check()
                except Exception as error:
                    monitor.cancel(f"Resource watchdog stopped: {error}")
                    return

        watchdog = threading.Thread(target=watch, daemon=True)
        watchdog.start()
        write_json(
            output / "campaign-status.json",
            {"status": "running", "pid": os.getpid(), "reviewed_commit": reviewed_commit},
        )
        run_phases(plans, output / "run-plans", resume, monitor)
        monitor.check()
        from training.analyze_issue124_campaign import analyze

        result = analyze(output)
        monitor.check()
        stop.set()
        watchdog.join(timeout=5)
        manifest_path = output / "analysis/evidence-files.json"
        manifest = read_json(manifest_path)
        manifest["resources.json"] = {
            "sha256": sha256(output / "resources.json"),
            "size_bytes": (output / "resources.json").stat().st_size,
        }
        write_json(manifest_path, manifest)
        write_json(
            output / "campaign-status.json",
            {
                "status": "completed",
                "analysis_valid": result["analysis_valid"],
                "selection": result["selection"],
            },
        )
    except BaseException as error:
        result_path = output / "analysis/result.json"
        if result_path.exists():
            invalid = read_json(result_path)
            invalid.update(analysis_valid=False, selection=None, campaign_error=str(error))
            write_json(result_path, invalid)
        write_json(
            output / "campaign-status.json",
            {
                "status": "stopped_incomplete",
                "error": f"{type(error).__name__}: {error}",
                "scientific_selection": None,
            },
        )
        raise
    finally:
        stop.set()
        if watchdog is not None:
            watchdog.join(timeout=5)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize-compute", action="store_true")
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "training_outputs/issue124")
    args = parser.parse_args()
    if args.dry_run:
        plans = validate_protocol()
        print(
            json.dumps(
                {
                    "plans": list(plans),
                    "training_episodes": 200000,
                    "evaluation_episodes": 5120,
                    "limits": vars(LIMITS),
                    "training_started": False,
                }
            )
        )
        return
    if not args.authorize_compute or not args.reviewed_commit:
        parser.error("Execution requires --authorize-compute and --reviewed-commit")
    execute(args.output_root, args.reviewed_commit, args.resume)


if __name__ == "__main__":
    main()
