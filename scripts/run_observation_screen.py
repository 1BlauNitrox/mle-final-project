"""Issue 211: immutable per-device triplets, durable resume and paired evaluation."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
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
CONFIG = ROOT / "experiments/2026-09-19-observation-screen/config.json"
AGENT = "DagobertDuckDQNObservation"
ARMS = ("control", "geometry", "memory")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def rows_write(path, rows):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(json.dumps(rows, separators=(",", ":")).encode())
        raw.flush()
        os.fsync(raw.fileno())


def rows_read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def code_hashes():
    files = list((ROOT / "agent_code" / AGENT).rglob("*.py"))
    files += [
        ROOT / "scripts" / name
        for name in (
            "run_observation_screen.py",
            "observation_episode.py",
            "observation_migration.py",
            "analyze_observation_screen.py",
            "audit_observation_seeds.py",
        )
    ]
    files += [CONFIG]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


def bound(root):
    cfg, binding = read(root / "config.json"), read(root / "binding.json")
    require(binding["code_hashes"] == code_hashes(), "Source/config changed after preparation")
    require(digest(root / "config.json") == binding["config_sha256"], "Bound protocol changed")
    require(digest(root / "reference.pt") == cfg["source"]["reference_sha256"], "Parent changed")
    for name, expected in binding["initial_hashes"].items():
        require(digest(root / "initial" / name) == expected, f"Initial checkpoint changed: {name}")
    for name, expected in binding["runtime_hashes"].items():
        require(digest(root / "source" / name) == expected, f"Runtime changed: {name}")
    return cfg, binding


def prepare(args):
    from scripts.observation_migration import migrate

    cfg = read(CONFIG)
    require(sys.version_info[:2] == (3, 13), "Use Python 3.13")
    require(not args.root.exists(), "Output root already exists; inspect/status/resume it")
    require(digest(args.reference) == cfg["source"]["reference_sha256"], "Wrong reference")
    require(
        digest(args.external_zip) == cfg["source"]["external_archive_sha256"], "Wrong opponent zip"
    )
    require(read(CONFIG.with_name("seed-audit.json"))["passed"], "Seed audit required")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True)
    require(
        not set(dirty.splitlines()).intersection(code_hashes()), "Commit technical changes first"
    )
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    require(set(code_hashes()) <= tracked, "Commit all technical source before preparation")
    args.root.mkdir(parents=True)
    shutil.copy2(args.reference, args.root / "reference.pt")
    shutil.copy2(CONFIG, args.root / "config.json")
    archive = args.root / "runtime.tar"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            f"--output={archive}",
            cfg["source"]["trained_runtime_commit"],
        ],
        cwd=ROOT,
        check=True,
    )
    source = args.root / "source"
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    shutil.copytree(
        ROOT / "agent_code" / AGENT,
        source / "agent_code" / AGENT,
        ignore=shutil.ignore_patterns("__pycache__", "*.pt", "README.md"),
    )
    opponent = args.root / "external"
    with zipfile.ZipFile(args.external_zip) as stream:
        for member in stream.infolist():
            target = (opponent / member.filename).resolve()
            require(target.is_relative_to(opponent.resolve()), "Unsafe zip member")
        stream.extractall(opponent)
    models = list(opponent.rglob("z_best-model.pt"))
    require(len(models) == 1, "Expected one external model")
    require(
        digest(models[0]) == cfg["source"]["external_checkpoint_sha256"], "External model mismatch"
    )
    shutil.copytree(models[0].parent, source / "agent_code/RUEHL_BASED_AGENT")
    seed = cfg["shared_training"]["learner_seeds"][args.replica - 1]
    initial = args.root / "initial"
    initial.mkdir()
    for arm in ARMS:
        migrate(args.reference, initial / f"{arm}.pt", arm, seed)
    shutil.copy2(initial / "control.pt", initial / "reference.pt")
    runtime_hashes = {
        p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*") if p.is_file()
    }
    write(
        args.root / "binding.json",
        {
            "replica": args.replica,
            "learner_seed": seed,
            "tool_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "config_sha256": digest(CONFIG),
            "code_hashes": code_hashes(),
            "runtime_hashes": runtime_hashes,
            "initial_hashes": {p.name: digest(p) for p in initial.glob("*.pt")},
            "prepared_at": datetime.now().astimezone().isoformat(),
            "python": sys.version,
            "host": os.environ.get("COMPUTERNAME"),
        },
    )
    print(f"Prepared replica {args.replica}: {args.root}")


def epsilon(seed, episode):
    selected = random.Random(seed + 168_000_000 + episode // 5).randrange(5)
    return float(episode % 5 == selected)


def checkpoint_for(root, artifact):
    if artifact == "reference":
        return root / "initial/reference.pt"
    state = read(root / "training" / artifact / "result.json")
    require(state["complete"], f"Incomplete model {artifact}")
    path = root / "training" / artifact / "final.pt"
    require(digest(path) == state["checkpoint_sha256"], "Final checkpoint changed")
    return path


def episode(root, checkpoint, cfg, binding, seed, *, arm=None, suite=None, index=0):
    from scripts.observation_episode import play_episode

    setting = cfg["shared_training"] if arm else cfg["evaluation_suites"][suite]
    return play_episode(
        root,
        checkpoint,
        world_seed=seed,
        opponents=setting["opponents"],
        training=arm is not None,
        epsilon=epsilon(
            cfg["shared_training"]["replica_world_seed_ranges_inclusive"][binding["replica"] - 1][
                0
            ],
            index,
        )
        if arm
        else 0.0,
        agent_seed=binding["learner_seed"],
        slot=index % (len(setting["opponents"]) + 1),
        scenario=setting.get("scenario", "classic"),
    )


def save_generation(directory, checkpoint, rows):
    """Publish pointer only after both immutable members are durable."""
    generation = directory / f"generation-{len(rows):06d}-{time.time_ns()}"
    generation.mkdir(parents=True)
    target = generation / "checkpoint.pt"
    shutil.copy2(checkpoint, target)
    with target.open("r+b") as stream:
        os.fsync(stream.fileno())
    rows_write(generation / "episodes.json.gz", rows)
    write(
        directory / "resume.json",
        {
            "generation": generation.name,
            "episodes": len(rows),
            "checkpoint_sha256": digest(target),
            "rows_sha256": digest(generation / "episodes.json.gz"),
        },
    )


def resume(directory, initial):
    checkpoint = directory / "working.pt"
    if not (directory / "resume.json").exists():
        shutil.copy2(initial, checkpoint)
        return checkpoint, []
    record = read(directory / "resume.json")
    generation = directory / record["generation"]
    require(
        digest(generation / "checkpoint.pt") == record["checkpoint_sha256"], "Corrupt resume model"
    )
    require(
        digest(generation / "episodes.json.gz") == record["rows_sha256"],
        "Corrupt resume observations",
    )
    rows = rows_read(generation / "episodes.json.gz")
    require(len(rows) == record["episodes"], "Resume counter mismatch")
    shutil.copy2(generation / "checkpoint.pt", checkpoint)
    return checkpoint, rows


def training_job(args):
    cfg, binding = bound(args.root)
    directory = args.root / "training" / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, rows = resume(directory, args.root / "initial" / f"{args.arm}.pt")
    seeds = cfg["shared_training"]["replica_world_seed_ranges_inclusive"][binding["replica"] - 1]
    budget = cfg["shared_training"]["episodes_per_arm_replica"]
    started, cpu = time.monotonic(), time.process_time()
    for index in range(len(rows), budget):
        if (
            args.root / "stop-request.json"
        ).exists() or datetime.now().astimezone() >= datetime.fromisoformat(
            cfg["resources"]["absolute_stop"]
        ):
            break
        row = episode(
            args.root, checkpoint, cfg, binding, seeds[0] + index, arm=args.arm, index=index
        )
        require(row["completed_episodes"] == index + 1, "Checkpoint and observations disagree")
        rows.append(row)
        if len(rows) % 25 == 0 or len(rows) == budget:
            save_generation(directory, checkpoint, rows)
            write(
                directory / "progress.json",
                {
                    "episodes": len(rows),
                    "budget": budget,
                    "updated_at": datetime.now().astimezone().isoformat(),
                    "cpu_seconds_attempt": time.process_time() - cpu,
                    "wall_seconds_attempt": time.monotonic() - started,
                },
            )
    if rows:
        save_generation(directory, checkpoint, rows)
    complete = len(rows) == budget
    if complete:
        shutil.copy2(checkpoint, directory / "final.pt")
    rows_write(directory / "episodes.json.gz", rows)
    write(
        directory / "result.json",
        {
            "complete": complete,
            "episodes": len(rows),
            "checkpoint_sha256": digest(checkpoint),
            "replica": binding["replica"],
            "arm": args.arm,
        },
    )
    return 0 if complete else 2


def smoke(args):
    cfg, binding = bound(args.root)
    output = args.root / "smoke"
    require(not output.exists(), "Preserve existing smoke output")
    output.mkdir()
    reports = []
    for index, arm in enumerate(ARMS):
        checkpoint = output / f"{arm}.pt"
        shutil.copy2(args.root / "initial" / f"{arm}.pt", checkpoint)
        start = time.monotonic()
        cpu = time.process_time()
        training_rows = []
        for step in range(cfg["smoke"]["maximum_episodes_per_arm"]):
            row = episode(
                args.root,
                checkpoint,
                cfg,
                binding,
                cfg["smoke"]["world_seeds"][index],
                arm=arm,
                index=4,
            )
            require(row["completed_episodes"] == step + 1, "Smoke resume counter mismatch")
            training_rows.append(row)
            if row["optimizer_updates"] > 0:
                break
        require(row["optimizer_updates"] > 0, "Smoke did not exercise gradient updates")
        before = digest(checkpoint)
        evaluation = episode(
            args.root,
            checkpoint,
            cfg,
            binding,
            cfg["smoke"]["world_seeds"][index],
            suite="primary-classic-rule-based",
            index=0,
        )
        require(digest(checkpoint) == before, "Smoke evaluation changed checkpoint")
        reports.append(
            {
                "arm": arm,
                "seconds": time.monotonic() - start,
                "cpu_seconds": time.process_time() - cpu,
                "rss_bytes": psutil.Process().memory_info().rss,
                "row": row,
                "training_rows": training_rows,
                "evaluation_row": evaluation,
            }
        )
    write(output / "report.json", {"passed": True, "runs": reports})
    print(json.dumps({"passed": True, "seconds": [r["seconds"] for r in reports]}))


def evaluation_job(args):
    cfg, binding = bound(args.root)
    checkpoint = checkpoint_for(args.root, args.artifact)
    before = digest(checkpoint)
    latency = args.suite == "latency"
    setting = cfg["latency"] if latency else cfg["evaluation_suites"][args.suite]
    target = args.root / ("latency" if latency else "evaluation") / args.artifact
    if not latency:
        target /= args.suite
    target.mkdir(parents=True, exist_ok=True)
    # Individual atomic rows permit recovery without discarding or duplicating worlds.
    rows = []
    first, last = setting["world_seed_range_inclusive"]
    for index, seed in enumerate(range(first, last + 1)):
        path = target / f"world-{seed}.json"
        if path.exists():
            row = read(path)
            require(
                row["checkpoint_sha256"] == before and row["world_seed"] == seed, "Stale eval row"
            )
        else:
            if (args.root / "stop-request.json").exists():
                return 2
            row = episode(
                args.root,
                checkpoint,
                cfg,
                binding,
                seed,
                suite="primary-classic-rule-based" if latency else args.suite,
                index=index,
            )
            row.update(artifact=args.artifact, suite=args.suite, checkpoint_sha256=before)
            write(path, row)
        rows.append(row)
    require(digest(checkpoint) == before, "Evaluation modified model")
    rows_write(target / "episodes.json.gz", rows)
    write(
        target / "result.json",
        {
            "complete": True,
            "episodes": len(rows),
            "checkpoint_sha256": before,
            "rows_sha256": digest(target / "episodes.json.gz"),
        },
    )
    return 0


@contextlib.contextmanager
def awake():
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def child_command(root, mode, **kwargs):
    command = [sys.executable, str(Path(__file__).resolve()), mode, "--root", str(root)]
    for key, value in kwargs.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def supervise(args, jobs, stage):
    cfg, _binding = bound(args.root)
    limits = cfg["resources"]
    active = {}
    logs = args.root / "logs"
    logs.mkdir(exist_ok=True)
    usage_path = args.root / f"{stage}-resources.json"
    usage = read(usage_path) if usage_path.exists() else {"cpu_seconds": 0.0, "wall_seconds": 0.0}
    other_usage = {"cpu_seconds": 0.0, "wall_seconds": 0.0}
    if stage in {"evaluate", "latency"}:
        other = "evaluate" if stage == "latency" else "latency"
        other_path = args.root / f"{other}-resources.json"
        if other_path.exists():
            other_usage = read(other_path)
    started, last_wall = time.monotonic(), time.monotonic()
    workers = 1 if stage == "latency" else args.workers
    require(1 <= workers <= 3, "Use 1..3 workers")
    cpu_limit = limits[
        "training_cpu_seconds_per_device"
        if stage == "train"
        else "evaluation_cpu_seconds_per_device"
    ]
    wall_limit = limits["training_wall_seconds" if stage == "train" else "evaluation_wall_seconds"]
    try:
        while jobs or active:
            memory = psutil.virtual_memory()
            rss = 0
            for item in active.values():
                try:
                    process = psutil.Process(item["process"].pid)
                    children = [process, *process.children(recursive=True)]
                    consumed = sum(p.cpu_times().user + p.cpu_times().system for p in children)
                    usage["cpu_seconds"] += max(0.0, consumed - item["cpu"])
                    item["cpu"] = consumed
                    rss += sum(p.memory_info().rss for p in children)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            now = time.monotonic()
            usage["wall_seconds"] += now - last_wall
            last_wall = now
            usage.update(
                updated_at=datetime.now().astimezone().isoformat(),
                rss_bytes=rss,
                available_bytes=memory.available,
                active=list(active),
                pending=[j[0] for j in jobs],
            )
            write(usage_path, usage)
            exceeded = usage["cpu_seconds"] + other_usage["cpu_seconds"] >= cpu_limit
            exceeded |= usage["wall_seconds"] + other_usage["wall_seconds"] >= wall_limit
            exceeded |= (
                memory.available < limits["minimum_available_ram_bytes"]
                or rss > limits["aggregate_workload_ram_bytes"]
            )
            if stage == "train":
                exceeded |= datetime.now().astimezone() >= datetime.fromisoformat(
                    limits["absolute_stop"]
                )
            if exceeded and not (args.root / "stop-request.json").exists():
                write(
                    args.root / "stop-request.json",
                    {"reason": "registered resource limit", "stage": stage, "usage": usage},
                )
            stopped = (args.root / "stop-request.json").exists()
            while jobs and len(active) < workers and not stopped:
                name, mode, options = jobs.pop(0)
                stream = (logs / f"{name}-{time.time_ns()}.log").open("w", encoding="utf-8")
                process = subprocess.Popen(
                    child_command(args.root, mode, **options),
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                active[name] = {"process": process, "stream": stream, "cpu": 0.0}
            for name, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["stream"].close()
                del active[name]
                if code:
                    write(
                        args.root / "stop-request.json",
                        {"reason": f"{name} exited {code}", "stage": stage},
                    )
            write(
                args.root / "status.json",
                {
                    "stage": stage,
                    "active": list(active),
                    "pending": [j[0] for j in jobs],
                    "elapsed_attempt": time.monotonic() - started,
                    "stopped": (args.root / "stop-request.json").exists(),
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
            )
            if stopped and not active:
                return False
            time.sleep(1)
    finally:
        # Only our children; never leave unsupervised jobs after an exception.
        for item in active.values():
            process = item["process"]
            with contextlib.suppress(psutil.NoSuchProcess):
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.terminate()
            process.terminate()
            process.wait()
            item["stream"].close()
    return not (args.root / "stop-request.json").exists()


def pipeline(args):
    cfg, binding = bound(args.root)
    require(read(args.root / "smoke/report.json")["passed"], "Passing smoke required")
    require((args.root / "authorization.json").exists(), "Explicit launch authorization required")
    with awake():
        jobs = [
            (arm, "_train", {"arm": arm})
            for arm in ARMS
            if not (args.root / "training" / arm / "result.json").exists()
            or not read(args.root / "training" / arm / "result.json")["complete"]
        ]
        if not supervise(args, jobs, "train"):
            return 2
        write(
            args.root / "training-summary.json", {"complete": True, "replica": binding["replica"]}
        )
        jobs = [
            (f"{arm}-{suite}", "_evaluate", {"artifact": arm, "suite": suite})
            for arm in (*ARMS, "reference")
            for suite in cfg["evaluation_suites"]
            if not (args.root / "evaluation" / arm / suite / "result.json").exists()
        ]
        if not supervise(args, jobs, "evaluate"):
            return 2
        write(args.root / "evaluation-summary.json", {"complete": True})
        jobs = [
            (f"{arm}-latency", "_evaluate", {"artifact": arm, "suite": "latency"})
            for arm in (*ARMS, "reference")
            if not (args.root / "latency" / arm / "result.json").exists()
        ]
        if not supervise(args, jobs, "latency"):
            return 2
        write(args.root / "latency-summary.json", {"complete": True})
        export(args.root)
        write(
            args.root / "status.json",
            {
                "stage": "complete",
                "replica": binding["replica"],
                "analysis_requires_all_three_replicas": True,
            },
        )
    return 0


def export(root):
    cfg, binding = bound(root)
    names = [
        "binding.json",
        "config.json",
        "training-summary.json",
        "evaluation-summary.json",
        "latency-summary.json",
        "authorization.json",
        "train-resources.json",
        "evaluate-resources.json",
        "latency-resources.json",
        "smoke/report.json",
    ]
    for arm in ARMS:
        names += [
            f"training/{arm}/{name}" for name in ("result.json", "episodes.json.gz", "final.pt")
        ]
    for arm in (*ARMS, "reference"):
        for suite in cfg["evaluation_suites"]:
            names += [
                f"evaluation/{arm}/{suite}/{name}" for name in ("result.json", "episodes.json.gz")
            ]
        names += [f"latency/{arm}/{name}" for name in ("result.json", "episodes.json.gz")]
    require(all((root / name).is_file() for name in names), "Cannot export incomplete triplet")
    manifest = {
        name: {"sha256": digest(root / name), "bytes": (root / name).stat().st_size}
        for name in names
    }
    write(root / "export-manifest.json", manifest)
    target = root / f"issue211-r{binding['replica']}-results.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for name in [*names, "export-manifest.json"]:
            stream.write(root / name, name)
    print(f"RESULT BUNDLE: {target}")


def launch(args):
    _cfg, binding = bound(args.root)
    require(args.authorize_compute, "Pass --authorize-compute for the registered triplet only")
    require(read(args.root / "smoke/report.json")["passed"], "Passing smoke required")
    require(
        not (args.root / "stop-request.json").exists(),
        "Inspect stop-request; do not blindly resume",
    )
    pid_path = args.root / "supervisor.json"
    if pid_path.exists():
        previous = read(pid_path)
        with contextlib.suppress(psutil.NoSuchProcess):
            process = psutil.Process(previous["pid"])
            require(
                abs(process.create_time() - previous["created"]) > 0.1, "Supervisor already running"
            )
    write(
        args.root / "authorization.json",
        {
            "authorized_by": "Julius",
            "replica": binding["replica"],
            "config_sha256": binding["config_sha256"],
            "at": datetime.now().astimezone().isoformat(),
        },
    )
    command = child_command(args.root, "_pipeline", workers=args.workers)
    flags = (
        (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
        if os.name == "nt"
        else 0
    )
    with (
        (args.root / "supervisor.stdout.log").open("a") as out,
        (args.root / "supervisor.stderr.log").open("a") as err,
    ):
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            creationflags=flags,
            close_fds=True,
            start_new_session=os.name != "nt",
        )
    write(
        pid_path,
        {
            "pid": process.pid,
            "created": psutil.Process(process.pid).create_time(),
            "command": command,
        },
    )
    print(f"Detached PID {process.pid}; status: {args.root / 'status.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "smoke",
            "launch",
            "status",
            "export",
            "_pipeline",
            "_train",
            "_evaluate",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--external-zip", type=Path)
    parser.add_argument("--replica", type=int, choices=(1, 2, 3))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--artifact", choices=(*ARMS, "reference"))
    parser.add_argument("--suite")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--authorize-compute", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    if args.mode == "prepare":
        require(
            all((args.reference, args.external_zip, args.replica)),
            "prepare requires reference, external zip and replica",
        )
        prepare(args)
    elif args.mode == "smoke":
        smoke(args)
    elif args.mode == "launch":
        launch(args)
    elif args.mode == "_pipeline":
        return pipeline(args)
    elif args.mode == "_train":
        return training_job(args)
    elif args.mode == "_evaluate":
        return evaluation_job(args)
    elif args.mode == "export":
        export(args.root)
    else:
        print(
            json.dumps(
                {
                    "status": read(args.root / "status.json")
                    if (args.root / "status.json").exists()
                    else "prepared",
                    "progress": {
                        arm: read(args.root / "training" / arm / "progress.json")
                        for arm in ARMS
                        if (args.root / "training" / arm / "progress.json").exists()
                    },
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
