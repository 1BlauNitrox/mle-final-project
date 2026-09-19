"""Issue 217: isolated paired curriculum workers, bounded overnight supervisor."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.curriculum_episode import play_episode  # noqa: E402
from scripts.curriculum_io import initialize, read, require, sha, write  # noqa: E402
from training.hunting_curriculum import UpdateTracker, training_setting  # noqa: E402

EXPERIMENT = "experiments/2026-09-19-hunting-curriculum"
CONFIG = ROOT / EXPERIMENT / "config.json"


def sources():
    paths = [CONFIG, CONFIG.with_name("seed-audit.json")]
    paths += [
        ROOT / "scripts" / name
        for name in (
            "run_hunting_curriculum.py",
            "curriculum_io.py",
            "curriculum_episode.py",
            "analyze_hunting_curriculum.py",
        )
    ]
    paths.append(ROOT / "training/hunting_curriculum.py")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def bound(root):
    binding, cfg = read(root / "binding.json"), read(root / "config.json")
    require(binding["tools"] == sources(), "Executed tools changed")
    require(sha(root / "config.json") == binding["config_sha256"], "Protocol changed")
    for name, digest in binding["inputs"].items():
        require(sha(root / name) == digest, f"Bound input changed: {name}")
    return cfg


def prepare(args):
    require(sys.version_info[:2] == (3, 13), "Use Python 3.13")
    cfg = read(CONFIG)
    require(not args.root.exists(), "New run root required; never overwrite evidence")
    require(sha(args.reference) == cfg["reference_sha256"], "Wrong parent checkpoint")
    require(read(CONFIG.with_name("seed-audit.json"))["passed"], "Seed audit required")
    if (ROOT / "manifest.json").exists():
        portable = read(ROOT / "manifest.json")
        for name, digest in sources().items():
            require(portable["files"][name]["sha256"] == digest, "Portable source changed")
        source_commit = portable["source_commit"]
    else:
        dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True)
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        require(
            not set(dirty.splitlines()).intersection(sources()) and set(sources()) <= set(tracked),
            "Commit technical sources before preparation",
        )
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    args.root.mkdir(parents=True)
    shutil.copy2(CONFIG, args.root / "config.json")
    shutil.copy2(args.reference, args.root / "reference.pt")
    archive = args.root / "runtime.tar"
    if args.runtime_archive:
        manifest = read(args.runtime_archive.with_suffix(".json"))
        require(
            manifest["runtime_commit"] == cfg["runtime_commit"]
            and sha(args.runtime_archive) == manifest["sha256"],
            "Wrong runtime archive",
        )
        shutil.copy2(args.runtime_archive, archive)
    else:
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", cfg["runtime_commit"]],
            cwd=ROOT,
            check=True,
        )
    with tarfile.open(archive) as stream:
        stream.extractall(args.root / "source", filter="data")
    sys.path.insert(0, str(args.root / "source"))
    (args.root / "initial").mkdir()
    records = {}
    for replica in cfg["devices"][args.device]["replicas"]:
        name = f"r{replica}"
        records[name] = initialize(
            args.root / "reference.pt",
            args.root / "initial" / f"{name}.pt",
            "preserve",
            cfg["learner_seeds"][replica - 1],
            cfg["learning_rate"],
        )
    write(args.root / "initialization.json", records)
    inputs = [
        args.root / "reference.pt",
        args.root / "initialization.json",
        *(args.root / "initial").glob("*.pt"),
    ]
    inputs += [
        p for p in (args.root / "source").rglob("*") if p.is_file() and "__pycache__" not in p.parts
    ]
    if args.prior_root:
        previous = args.prior_root.resolve()
        require(read(previous / "binding.json")["device"] == args.device, "Prior device mismatch")
        require(
            (previous / "STOP.json").exists() or (previous / "complete.json").exists(),
            "Prior attempt must be stopped",
        )
        owner = read(previous / "supervisor.json")
        if psutil.pid_exists(owner["pid"]):
            require(
                abs(psutil.Process(owner["pid"]).create_time() - owner["created"]) > 0.01,
                "Prior supervisor is still alive",
            )
        ledger = read(previous / "resources.json")
        inherited = args.root / "inherited-resources.json"
        write(
            inherited,
            {
                "cpu_seconds": ledger["cpu_seconds"] + 60.0,
                "first_start": ledger["first_start"],
                "source_resources_sha256": sha(previous / "resources.json"),
                "note": "Prior consumption plus60seconds conservative termination margin",
            },
        )
        inputs.append(inherited)
    write(
        args.root / "binding.json",
        {
            "tools": sources(),
            "device": args.device,
            "config_sha256": sha(CONFIG),
            "runtime_archive_sha256": sha(archive),
            "inputs": {p.relative_to(args.root).as_posix(): sha(p) for p in inputs},
            "python": sys.version,
            "source_commit": source_commit,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "pygame", "psutil")
            },
            "authorization": cfg["authorization"],
        },
    )


def epsilon(seed, index):
    return float(index % 5 == random.Random(seed + 168000000 + index // 5).randrange(5))


def compact(row):
    steps = row.pop("late_steps")
    eligible = loops = 0
    for i in range(23, len(steps)):
        window = steps[i - 23 : i + 1]
        if all(s["crates_left"] == 0 and not s["hazards"] and not s["progress"] for s in window):
            eligible += 1
            loops += len({tuple(s["position"]) for s in window}) <= 3
    row["loop_windows"] = {"eligible": eligible, "looping": loops}
    return row


def evaluate(root, cfg, checkpoint, directory, stage, *, latency=False):
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha(checkpoint)
    for suite, setting in cfg["evaluation"][stage].items():
        if (suite == "latency") != latency:
            continue
        path = directory / f"{suite}.json"
        if path.exists():
            require(read(path)["checkpoint_sha256"] == digest, "Stale evaluation")
            continue
        rows = []
        for i, seed in enumerate(range(setting["seeds"][0], setting["seeds"][1] + 1)):
            row = play_episode(
                root,
                checkpoint,
                world_seed=seed,
                opponents=setting["opponents"],
                training=False,
                epsilon=0,
                agent_seed=cfg["evaluation_agent_seed"],
                slot=i % (len(setting["opponents"]) + 1),
                scenario=setting["scenario"],
            )
            rows.append(compact(row))
        require(sha(checkpoint) == digest, "Evaluation mutated parameters")
        write(path, {"checkpoint_sha256": digest, "rows": rows})


def save_generation(directory, checkpoint, state):
    destination = directory / f"generation-{state['episodes']:06d}-{time.time_ns()}"
    destination.mkdir(parents=True)
    shutil.copy2(checkpoint, destination / "checkpoint.pt")
    write(destination / "state.json", state)
    record = {
        "generation": destination.name,
        "checkpoint_sha256": sha(destination / "checkpoint.pt"),
        "state_sha256": sha(destination / "state.json"),
    }
    write(directory / "resume.json", record)
    # Only two recovery generations are needed; named scientific snapshots are
    # separate and never pruned. Check every resolved deletion target first.
    generations = sorted(directory.glob("generation-*"), key=lambda p: p.name)
    for obsolete in generations[:-2]:
        target = obsolete.resolve()
        require(
            target.is_relative_to(directory.resolve())
            and target.name.startswith("generation-")
            and target != destination.resolve(),
            "Unsafe recovery cleanup target",
        )
        shutil.rmtree(target)
    return destination


def restore(directory, initial, initialization):
    if (directory / "resume.json").exists():
        record = read(directory / "resume.json")
        generation = directory / record["generation"]
        require(
            sha(generation / "checkpoint.pt") == record["checkpoint_sha256"]
            and sha(generation / "state.json") == record["state_sha256"],
            "Corrupt generation",
        )
        return generation / "checkpoint.pt", read(generation / "state.json")
    return initial, {
        "episodes": 0,
        "updates": 0,
        "rows": [],
        "tracker": {
            "transitions": 0,
            "generated": {},
            "sampled": {},
            "origins": ["parent"] * initialization["replay_size"],
        },
    }


def advance(root, cfg, replica, arm, check):
    directory = root / "pairs" / f"r{replica}" / arm
    directory.mkdir(parents=True, exist_ok=True)
    init = read(root / "initialization.json")[f"r{replica}"]
    previous, state = restore(directory, root / "initial" / f"r{replica}.pt", init)
    checkpoint = directory / "working.pt"
    shutil.copy2(previous, checkpoint)
    target = check.get("updates", cfg["updates"])
    while (
        state["updates"] < target if "updates" in check else state["episodes"] < check["episodes"]
    ):
        if state["episodes"] >= check.get("episode_stop", cfg["episodes"] + 1):
            break
        require(state["episodes"] < cfg["episodes"], "Episode ceiling before update endpoint")
        i = state["episodes"]
        setting = training_setting(arm, i)
        tracker = UpdateTracker(
            state["tracker"],
            capacity=10000,
            every=cfg["update_every"],
            maximum_updates=init["updates"] + target,
            origin=setting["kind"],
        )
        row = play_episode(
            root,
            checkpoint,
            world_seed=cfg["train_ranges"][replica - 1][0] + i,
            opponents=setting["opponents"],
            training=True,
            epsilon=epsilon(cfg["learner_seeds"][replica - 1], i),
            agent_seed=cfg["learner_seeds"][replica - 1],
            slot=i % 4,
            tracker=tracker,
            hunting_index=i if setting["kind"] == "hunting" else None,
        )
        require(row["completed_episodes"] == init["episodes"] + i + 1, "Episode counter drift")
        row["kind"] = setting["kind"]
        state["episodes"] += 1
        state["updates"] = row["optimizer_updates"] - init["updates"]
        state["tracker"] = tracker.snapshot()
        state["rows"].append(compact(row))
        if state["episodes"] % 10 == 0:
            save_generation(directory, checkpoint, state)
            write(
                directory / "progress.json",
                {
                    "episodes": state["episodes"],
                    "updates": state["updates"],
                    "check": check["name"],
                    "generated": state["tracker"]["generated"],
                    "sampled": state["tracker"]["sampled"],
                },
            )
    generation = save_generation(directory, checkpoint, state)
    snapshot = directory / "snapshots" / check["name"]
    snapshot.mkdir(parents=True, exist_ok=True)
    if (snapshot / "checkpoint.pt").exists():
        require(
            sha(snapshot / "checkpoint.pt") == sha(generation / "checkpoint.pt"),
            "Attempt to replace snapshot",
        )
    else:
        shutil.copy2(generation / "checkpoint.pt", snapshot / "checkpoint.pt")
        write(snapshot / "state.json", state)
    return snapshot


def pair(root, cfg, replica):
    from scripts.analyze_hunting_curriculum import safety_gate

    directory = root / "pairs" / f"r{replica}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "decision.json").exists():
        return
    reference = directory / "reference"
    evaluate(root, cfg, root / "reference.pt", reference / "pilot", "pilot")
    # Both initial copies share network weights and greedy action seed with the
    # frozen reference. Actual paired games verify this before any new update.
    initial = root / "initial" / f"r{replica}.pt"
    evaluate(root, cfg, initial, directory / "zero", "pilot")
    for suite in cfg["evaluation"]["pilot"]:
        left = read(reference / "pilot" / f"{suite}.json")["rows"]
        right = read(directory / "zero" / f"{suite}.json")["rows"]
        require(
            [r["actions_sha256"] for r in left] == [r["actions_sha256"] for r in right],
            "Zero-update greedy policy changed",
        )
    for check in cfg["checks"]:
        check_path = directory / f"gate-{check['name']}.json"
        if check_path.exists():
            require(read(check_path)["passed"], "Stopped pilot cannot be resumed")
            continue
        results = {}
        for arm in cfg["arms"]:
            snapshot = advance(root, cfg, replica, arm, check)
            stage = "final" if check["name"] == "final" else "pilot"
            evaluate(root, cfg, snapshot / "checkpoint.pt", snapshot / "evaluation", stage)
            if stage == "pilot":
                results[arm] = safety_gate(
                    snapshot / "evaluation", reference / "pilot", cfg["pilot"]
                )
                state = read(snapshot / "state.json")
                hunting = [r for r in state["rows"] if r["kind"] == "hunting"]
                if len(hunting) >= 25:
                    exposed = any(
                        r["initial_exposure"]["reachable_attack_position"] for r in hunting
                    )
                    results[arm]["gates"]["hunting_exposure"] = exposed
                    results[arm]["passed"] &= exposed
        passed = all(r["passed"] for r in results.values())
        write(check_path, {"passed": passed, "arms": results, "checkpoint": check})
        if not passed:
            write(
                directory / "decision.json",
                {"status": "stopped_safety", "check": check["name"], "submission_promotion": False},
            )
            return
    evaluate(root, cfg, root / "reference.pt", reference / "final", "final")
    write(
        directory / "decision.json", {"status": "training_complete", "submission_promotion": False}
    )


def latency(root, cfg):
    for replica in cfg["devices"][read(root / "binding.json")["device"]]["replicas"]:
        pair_root = root / "pairs" / f"r{replica}"
        decision = read(pair_root / "decision.json")
        if decision["status"] != "training_complete":
            continue
        evaluate(
            root, cfg, root / "reference.pt", pair_root / "reference/final", "final", latency=True
        )
        for arm in cfg["arms"]:
            snapshot = pair_root / arm / "snapshots/final"
            evaluate(
                root,
                cfg,
                snapshot / "checkpoint.pt",
                snapshot / "evaluation",
                "final",
                latency=True,
            )


def resource_breach(usage, limits, device, *, now, rss, available):
    if usage["cpu_seconds"] >= device["cpu_seconds"]:
        return "CPU budget exhausted"
    if usage["wall_seconds"] >= limits["wall_seconds"]:
        return "Elapsed budget exhausted"
    if now >= datetime.fromisoformat(limits["stop_utc"]).timestamp():
        return "Sunday absolute stop reached"
    if rss > device["rss_bytes"] or available < limits["minimum_available_bytes"]:
        return "Memory limit reached"
    return None


def process_usage(process, seen):
    """Account Windows venv launcher descendants by PID AND creation time."""
    rss = 0
    try:
        parent = psutil.Process(process.pid)
        family = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        family = []
    for member in family:
        try:
            key = f"{member.pid}:{member.create_time()}"
            seen[key] = max(seen.get(key, 0.0), sum(member.cpu_times()[:2]))
            rss += member.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss


def stop_tree(process):
    try:
        parent = psutil.Process(process.pid)
        family = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for member in reversed(family):
        with contextlib.suppress(psutil.NoSuchProcess):
            member.terminate()
    _, alive = psutil.wait_procs(family, timeout=5)
    for member in alive:
        member.kill()
    process.wait(timeout=10)


def initial_usage(root, cfg, device):
    if (root / "resources.json").exists():
        return read(root / "resources.json")
    previous = cfg["prior_usage"][device]
    value = {
        "cpu_seconds": previous["cpu_seconds"],
        "wall_seconds": 0.0,
        "first_start": previous.get("first_start", time.time()),
    }
    if (root / "inherited-resources.json").exists():
        inherited = read(root / "inherited-resources.json")
        value["cpu_seconds"] = max(value["cpu_seconds"], inherited["cpu_seconds"])
        value["first_start"] = min(value["first_start"], inherited["first_start"])
    return value


def supervise(root):
    cfg = bound(root)
    require(
        (root / "smoke.json").exists() and read(root / "smoke.json")["passed"],
        "Passing smoke required",
    )
    require(not (root / "STOP.json").exists(), "Stop is persistent; inspect evidence")
    lock = root / "supervisor.json"
    if lock.exists():
        old = read(lock)
        if psutil.pid_exists(old["pid"]):
            require(
                abs(psutil.Process(old["pid"]).create_time() - old["created"]) > 0.01,
                "Another supervisor owns this root",
            )
    owner = psutil.Process()
    write(lock, {"pid": os.getpid(), "created": owner.create_time()})
    device = cfg["devices"][read(root / "binding.json")["device"]]
    usage_path = root / "resources.json"
    usage = initial_usage(root, cfg, read(root / "binding.json")["device"])
    usage["wall_seconds"] = time.time() - usage["first_start"]
    base_cpu = usage["cpu_seconds"]
    self_start = sum(owner.cpu_times()[:2])
    measured, completed, running = {}, {}, {}
    logs = []
    attempts = str(time.time_ns())

    def launch(mode, replica=0):
        key = f"{attempts}-{mode}-{replica}"
        stream = (root / f"{key}.log").open("w", encoding="utf-8")
        logs.append(stream)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            mode,
            "--root",
            str(root),
            "--replica",
            str(replica),
            "--job-id",
            key,
        ]
        process = subprocess.Popen(
            cmd, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT
        )
        running[key] = process
        measured[key] = {}
        write(
            root / "workers.json",
            {
                name: {"pid": worker.pid, "created": psutil.Process(worker.pid).create_time()}
                for name, worker in running.items()
                if worker.poll() is None
            },
        )

    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        reason = resource_breach(
            usage,
            cfg["limits"],
            device,
            now=time.time(),
            rss=owner.memory_info().rss,
            available=psutil.virtual_memory().available,
        )
        require(not reason, reason or "")
        for replica in device["replicas"]:
            launch("_pair", replica)
        phase = "pairs"
        while running:
            rss = owner.memory_info().rss
            for key, process in list(running.items()):
                rss += process_usage(process, measured[key])
                code = process.poll()
                if code is not None:
                    final_usage = root / "worker-usage" / f"{key}.json"
                    if final_usage.exists():
                        final = read(final_usage)
                        identity = f"{final['pid']}:{final['created']}"
                        measured[key][identity] = max(
                            measured[key].get(identity, 0.0), final["cpu_seconds"]
                        )
                    completed[key] = code
                    del running[key]
                    require(code == 0, f"Worker failed: {key}; inspect its log")
            usage["cpu_seconds"] = (
                base_cpu
                + sum(sum(family.values()) for family in measured.values())
                + sum(owner.cpu_times()[:2])
                - self_start
            )
            usage["wall_seconds"] = time.time() - usage["first_start"]
            usage["peak_rss_bytes"] = max(usage.get("peak_rss_bytes", 0), rss)
            usage["stage"] = phase
            write(usage_path, usage)
            reason = resource_breach(
                usage,
                cfg["limits"],
                device,
                now=time.time(),
                rss=rss,
                available=psutil.virtual_memory().available,
            )
            require(not reason, reason or "")
            require(not (root / "STOP.request").exists(), "User stop requested")
            if not running and phase == "pairs":
                # All training processes have exited before serial latency begins.
                phase = "latency"
                launch("_latency")
            elif not running and phase == "latency":
                phase = "export"
                launch("_export")
            time.sleep(0.5)
        write(
            root / "complete.json",
            {
                "pipeline_complete": True,
                "submission_promotion": False,
                "decisions": [
                    read(root / "pairs" / f"r{r}" / "decision.json") for r in device["replicas"]
                ],
            },
        )
        exported = read(root / "export.json")
        write(
            root / "export.json",
            {
                **exported,
                "final_pipeline_resources": usage,
                "resource_note": "Archive ledger is the pre-export snapshot; "
                "this ledger includes export CPU and supervisor overhead",
            },
        )
    except BaseException as error:
        write(root / "STOP.json", {"reason": str(error), "time": time.time(), "usage": usage})
        raise
    finally:
        for process in running.values():
            if process.poll() is None:
                stop_tree(process)
        for stream in logs:
            stream.close()
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def export(root):
    names = [
        "config.json",
        "binding.json",
        "initialization.json",
        "resources.json",
        "smoke.json",
        "inherited-resources.json",
        "STOP.json",
    ]
    for pattern in (
        "pairs/*/decision.json",
        "pairs/*/gate-*.json",
        "pairs/*/reference/*/*.json",
        "pairs/*/zero/*.json",
        "pairs/*/*/snapshots/*/evaluation/*.json",
        "pairs/*/*/snapshots/*/state.json",
        "pairs/*/*/snapshots/*/checkpoint.pt",
    ):
        names += [p.relative_to(root).as_posix() for p in root.glob(pattern)]
    names += [p.relative_to(root).as_posix() for p in root.glob("worker-usage/*.json")]
    archive = root / f"issue217-results-{time.time_ns()}.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as stream:
        manifest = {}
        for name in sorted(set(names)):
            if (root / name).exists():
                content = (root / name).read_bytes()
                stream.writestr(name, content)
                manifest[name] = {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
        stream.writestr("manifest.json", json.dumps(manifest, indent=2))
    write(
        root / "export.json",
        {"path": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size},
    )


def smoke(root):
    cfg = bound(root)
    directory = root / "mechanics"
    require(not directory.exists(), "Preserve previous smoke")
    directory.mkdir()
    replica = cfg["devices"][read(root / "binding.json")["device"]]["replicas"][0]
    init = read(root / "initialization.json")[f"r{replica}"]
    records = []
    for arm in cfg["arms"]:
        checkpoint = directory / f"{arm}.pt"
        shutil.copy2(root / "initial" / f"r{replica}.pt", checkpoint)
        tracker = UpdateTracker(
            {
                "transitions": 0,
                "generated": {},
                "sampled": {},
                "origins": ["parent"] * init["replay_size"],
            },
            capacity=10000,
            every=cfg["update_every"],
            maximum_updates=init["updates"] + 2,
            origin="hunting" if arm == "curriculum" else "classic",
        )
        for i in range(3):
            row = play_episode(
                root,
                checkpoint,
                world_seed=cfg["smoke_seed"] + i,
                opponents=["peaceful_agent"] * 3,
                training=True,
                epsilon=0,
                agent_seed=cfg["learner_seeds"][replica - 1],
                tracker=tracker,
                hunting_index=1 if arm == "curriculum" else None,
            )
            if row["optimizer_updates"] > init["updates"]:
                break
        require(row["optimizer_updates"] > init["updates"], "Smoke requires real gradient updates")
        digest = sha(checkpoint)
        evaluated = play_episode(
            root,
            checkpoint,
            world_seed=cfg["smoke_seed"],
            opponents=["rule_based_agent"] * 3,
            training=False,
            epsilon=0,
            agent_seed=cfg["evaluation_agent_seed"],
        )
        require(sha(checkpoint) == digest, "Evaluation changed checkpoint")
        records.append({"arm": arm, "training": compact(row), "evaluation": compact(evaluated)})
    write(root / "smoke.json", {"passed": True, "scientific": False, "records": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["prepare", "smoke", "run", "export", "_pair", "_latency", "_export"]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=["pc", "laptop"], default="pc")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    args.root = args.root.resolve()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.job_id:

        def record_usage():
            owner = psutil.Process()
            write(
                args.root / "worker-usage" / f"{args.job_id}.json",
                {
                    "cpu_seconds": sum(owner.cpu_times()[:2]),
                    "pid": owner.pid,
                    "created": owner.create_time(),
                },
            )

        atexit.register(record_usage)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "smoke":
        smoke(args.root)
    elif args.mode == "run":
        supervise(args.root)
    elif args.mode in ("export", "_export"):
        export(args.root)
    else:
        cfg = bound(args.root)
        if args.mode == "_pair":
            require(
                args.replica
                in cfg["devices"][read(args.root / "binding.json")["device"]]["replicas"],
                "Replica not allocated to this device",
            )
            pair(args.root, cfg, args.replica)
        else:
            latency(args.root, cfg)


if __name__ == "__main__":
    main()
