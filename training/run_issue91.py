"""Prepare and run the prospective Issue 91 two-arm discount-horizon experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml

from training.run_issue107_campaign import (
    CampaignLimits,
    CampaignResourceMonitor,
    _seed_values_from_path,
)
from training.run_plan import execute_plan, load_plan

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "agent_code/DagobertDuckDQNTask2/checkpoint.pt"
SOURCE_HASH = "4ad409472e7ca008dfcc82aa26017017c90259b65f9e593020d1b63921430f60"
CONFIG = ROOT / "experiments/2026-09-11-task2-discount-horizon/config.yaml"
LIMITS = CampaignLimits(cpu_seconds=48 * 3600, wall_seconds=24 * 3600, memory_bytes=4 * 1024**3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def prepare_parent(path, gamma):
    from agent_code.DagobertDuckDQNTask2.persistence import (
        load_training_checkpoint,
        save_checkpoint,
    )

    if sha(SOURCE) != SOURCE_HASH or gamma not in (0.9, 0.97) or path.exists():
        raise ValueError("Unexpected source, gamma or existing destination")
    loaded = load_training_checkpoint(SOURCE)
    if loaded.completed_episodes or len(loaded.replay_buffer) or loaded.learner.update_steps:
        raise ValueError("Initialization must be the fresh migration")
    config = replace(loaded.config, discount_factor=gamma, escape_continuation_features=True)
    loaded.learner.config = config
    loaded.learner.online_network.config = config
    loaded.learner.target_network.config = config
    save_checkpoint(
        learner=loaded.learner,
        replay_buffer=loaded.replay_buffer,
        action_rng=loaded.action_rng,
        epsilon=loaded.epsilon,
        completed_episodes=0,
        agent_seed=loaded.agent_seed,
        path=path,
    )


def plan_spec(cell, parent):
    frozen = cell == "frozen_task1"
    trained = cell in ("A", "B")
    replicas = (
        [
            {
                "id": f"r{i}",
                "world_seed": 9100000 + i,
                "agent_seed": 19100000 + i,
                "parent_artifact": str(parent),
            }
            for i in range(1, 6)
        ]
        if trained
        else [
            {
                "id": "reference",
                "world_seed": 9100099,
                "agent_seed": 19100099,
                "parent_artifact": str(parent),
            }
        ]
    )
    suites = []
    for scenario, offset in (("classic", 0), ("coin-heaven", 100), ("loot-crate", 200)):
        if frozen and scenario != "coin-heaven":
            continue
        for kind in ("primary", "repeat"):
            suites.append(
                {
                    "id": f"{scenario}-{kind}",
                    "population": "development",
                    "scenario": scenario,
                    "rounds": 1,
                    "opponents": [],
                    "world_seeds": list(range(29100001 + offset, 29100041 + offset)),
                    "agent_seeds": list(range(39100001 + offset, 39100041 + offset)),
                }
            )
    return {
        "schema_version": 1,
        "plan_id": f"issue91-{cell.lower()}",
        "agent": "DagobertDuckDQN" if frozen else "DagobertDuckDQNTask2",
        "artifact_path": "checkpoint.pt",
        "max_parallel_training": 1,
        "action_masking": "none",
        "escape_continuations": "off" if frozen else "on",
        "replay_treatment": "uniform",
        "reward_variant": "control",
        "replicas": replicas,
        "training_stages": [
            {"id": scenario, "scenario": scenario, "rounds": rounds, "opponents": []}
            for scenario, rounds in (("coin-heaven", 2000), ("loot-crate", 2000), ("classic", 6000))
        ]
        if trained
        else [],
        "evaluation_suites": suites,
    }


def prepare(root):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Preparation requires a new output directory")
    root.mkdir(parents=True)
    for cell, gamma in (("A", 0.9), ("B", 0.97)):
        prepare_parent(root / f"initial-{cell}.pt", gamma)
    records = {}
    for cell in ("A", "B", "untrained", "frozen_task1"):
        parent = (
            ROOT / "agent_code/DagobertDuckDQN/checkpoint.pt"
            if cell == "frozen_task1"
            else (root / f"initial-{'A' if cell == 'untrained' else cell}.pt")
        )
        path = root / f"{cell}.yaml"
        path.write_text(yaml.safe_dump(plan_spec(cell, parent), sort_keys=False), encoding="utf-8")
        plan = load_plan(path)
        records[cell] = {
            "path": path.name,
            "sha256": sha(path),
            "fingerprints": plan.fingerprints,
            "parents": {r.replica_id: r.parent_artifact_sha256 for r in plan.replicas},
        }
    manifest = {
        "issue": 91,
        "source_sha256": SOURCE_HASH,
        "config_sha256": sha(CONFIG),
        "prepared_commit": git("rev-parse", "HEAD"),
        "plans": records,
        "training_episodes": 100000,
        "evaluation_episodes": 2720,
        "gamma": {"A": 0.9, "B": 0.97},
        "limits": vars(LIMITS),
    }
    write(root / "protocol.json", manifest)
    validate(root)
    return manifest


def validate(root):
    root = Path(root)
    manifest = read(root / "protocol.json")
    if manifest["source_sha256"] != sha(SOURCE) or manifest["config_sha256"] != sha(CONFIG):
        raise ValueError("Registered source or configuration changed")
    plans = {}
    seeds = set()
    for cell, item in manifest["plans"].items():
        path = root / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError("Registered plan changed")
        plan = load_plan(path)
        if (
            plan.fingerprints != item["fingerprints"]
            or {r.replica_id: r.parent_artifact_sha256 for r in plan.replicas} != item["parents"]
        ):
            raise ValueError("Source, dependency or artifact fingerprint changed")
        plans[cell] = plan
        seeds |= {j.world_seed for j in plan.jobs} | {j.agent_seed for j in plan.jobs}
    for path in list((ROOT / "training/run_plans").glob("*.yaml")) + list(
        (ROOT / "experiments").rglob("*.yaml")
    ):
        if "2026-09-11-task2-discount-horizon" in path.parts:
            continue
        if seeds & _seed_values_from_path(path):
            raise ValueError(f"Historical/reserved seed collision: {path}")
    train = {j.world_seed for p in plans.values() for j in p.jobs if j.kind == "training"}
    evaluate = {j.world_seed for p in plans.values() for j in p.jobs if j.kind == "evaluation"}
    if train & evaluate:
        raise ValueError("Training/evaluation seeds overlap")
    return plans


class Monitor(CampaignResourceMonitor):
    def __init__(self, **kwargs):
        self.cancelled = None
        self.supervisor = psutil.Process()
        times = self.supervisor.cpu_times()
        self.cpu_origin = times.user + times.system
        super().__init__(**kwargs)

    def _usage_locked(self):
        cpu, memory = super()._usage_locked()
        times = self.supervisor.cpu_times()
        memory += self.supervisor.memory_info().rss
        self._peak_memory_bytes = max(self._peak_memory_bytes, memory)
        return cpu + max(0, times.user + times.system - self.cpu_origin), memory

    def cancel(self, reason):
        """Stop technical failures without turning them into resource exhaustion."""
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

    def abort(self, reason):
        with self._lock:
            self._limit_reached = self._limit_reached or reason
        self.check()


def execute(root, reviewed_commit, authorized_by, hardware, resume):
    root = Path(root).resolve()
    plans = validate(root)
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != reviewed_commit:
        raise ValueError("Execution requires the clean reviewed commit")
    if read(root / "protocol.json")["prepared_commit"] != reviewed_commit:
        raise ValueError("Prepare again from the reviewed commit into a new directory")
    if psutil.virtual_memory().available < 5 * 1024**3 or psutil.cpu_count() < 2:
        raise ValueError("Server requires >=5 GiB available RAM and >=2 logical CPUs")
    path = root / "authorization.json"
    if path.exists() and not resume:
        raise ValueError("Existing campaign requires --resume")
    if not path.exists():
        if resume:
            raise ValueError("No authorization to resume")
        write(
            path,
            {
                "reviewed_commit": reviewed_commit,
                "authorized_by": authorized_by,
                "hardware": hardware,
                "platform": platform.platform(),
                "authorized_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "protocol_sha256": sha(root / "protocol.json"),
                "limits": vars(LIMITS),
            },
        )
    auth = read(path)
    if auth["reviewed_commit"] != reviewed_commit or auth["protocol_sha256"] != sha(
        root / "protocol.json"
    ):
        raise ValueError("Authorization changed")
    with (root / "campaign.lock").open("x") as lock:
        lock.write(str(psutil.Process().pid))
    monitor = Monitor(
        state_path=root / "resources.json",
        authorized_at=auth["authorized_at"],
        limits=LIMITS,
        campaign_metadata=auth,
    )
    stop = threading.Event()

    def watch():
        while not stop.wait(1):
            try:
                if psutil.virtual_memory().available < 1024**3:
                    monitor.abort("System available memory below 1 GiB")
                monitor.check()
            except Exception:
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        write(root / "status.json", {"status": "training"})

        def train(cell):
            try:
                execute_plan(
                    plans[cell],
                    output_root=root / "run-plans",
                    training_only=True,
                    resume=(root / "run-plans" / plans[cell].plan_id).exists(),
                    process_monitor=monitor,
                )
            except BaseException:
                with suppress(Exception):
                    monitor.cancel(f"Training arm {cell} failed")
                raise

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(train, ("A", "B")))
        write(root / "status.json", {"status": "evaluation"})
        for plan in plans.values():
            monitor.check()
            execute_plan(
                plan,
                output_root=root / "run-plans",
                process_monitor=monitor,
                resume=(root / "run-plans" / plan.plan_id).exists(),
            )
        from training.analyze_issue91 import analyze

        monitor.check()
        result = analyze(root)
        monitor.check()
        stop.set()
        watcher.join(timeout=5)
        monitor.check()
        evidence = read(root / "analysis/evidence-files.json")
        evidence["resources.json"] = {
            "sha256": sha(root / "resources.json"),
            "size_bytes": (root / "resources.json").stat().st_size,
        }
        write(root / "analysis/evidence-files.json", evidence)
        write(root / "status.json", {"status": "completed", "decision": result["decision"]})
    except BaseException as error:
        result_path = root / "analysis/result.json"
        if result_path.exists():
            invalid = read(result_path)
            invalid.update(
                analysis_valid=False,
                selection=None,
                decision="invalid_execution",
                campaign_error=str(error),
            )
            write(result_path, invalid)
        write(root / "status.json", {"status": "stopped_incomplete", "error": str(error)})
        raise
    finally:
        stop.set()
        watcher.join(timeout=5)
        (root / "campaign.lock").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-compute", action="store_true")
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--authorized-by")
    parser.add_argument("--hardware-description")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.output_root), indent=2))
    elif args.dry_run:
        plans = validate(args.output_root)
        print(
            {
                "training": sum(
                    j.rounds for p in plans.values() for j in p.jobs if j.kind == "training"
                ),
                "evaluation": sum(
                    j.rounds for p in plans.values() for j in p.jobs if j.kind == "evaluation"
                ),
            }
        )
    else:
        if not (
            args.authorize_compute
            and args.reviewed_commit
            and args.authorized_by
            and args.hardware_description
        ):
            parser.error("Supply compute authorization, reviewed commit, authorizer and hardware")
        execute(
            args.output_root,
            args.reviewed_commit,
            args.authorized_by,
            args.hardware_description,
            args.resume,
        )


if __name__ == "__main__":
    main()
