"""Bounded, isolated replay/Adam continuation diagnostic for issue 213."""

from __future__ import annotations

import argparse
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
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "experiments/2026-09-19-continuation-stability/config.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources():
    paths = [CONFIG, CONFIG.with_name("seed-audit.json")]
    paths += [
        ROOT / "scripts" / name
        for name in ("run_stability_test.py", "stability_episode.py", "analyze_stability_test.py")
    ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def bound(root):
    binding = read(root / "binding.json")
    require(binding["tools"] == sources(), "Tool/config changed since prepare")
    require(sha(root / "config.json") == binding["config_sha256"], "Protocol changed")
    for key in ("runtime", "initial"):
        for name, checksum in binding[key].items():
            require(sha(root / name) == checksum, f"Bound input changed: {name}")
    return read(root / "config.json"), binding


def initialize(parent_path, destination, arm, seed, learning_rate):
    import numpy as np

    from agent_code.DagobertDuckDQNTask3.model import DQNLearner
    from agent_code.DagobertDuckDQNTask3.persistence import (
        load_training_checkpoint,
        save_checkpoint,
    )
    from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer

    parent = load_training_checkpoint(parent_path)
    require(parent.config.input_dim == 39, "Expected original 39-input checkpoint")
    require(arm in ("reset", "preserve"), "Unknown initialization arm")
    config = replace(parent.config, learning_rate=learning_rate)
    learner = DQNLearner(config=config, seed=seed)
    learner.online_network.load_state_dict(parent.learner.online_network.state_dict())
    learner.target_network.load_state_dict(parent.learner.target_network.state_dict())
    learner.update_steps = parent.learner.update_steps
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=seed)
    if arm == "preserve":
        learner.optimizer.load_state_dict(deepcopy(parent.learner.optimizer.state_dict()))
        for group in learner.optimizer.param_groups:
            group["lr"] = learning_rate
        state = parent.replay_buffer.state_dict()
        state["rng_state"] = deepcopy(replay.state_dict()["rng_state"])
        replay.load_state_dict(state)
    save_checkpoint(
        learner=learner,
        replay_buffer=replay,
        action_rng=np.random.default_rng(seed),
        epsilon=parent.epsilon,
        completed_episodes=parent.completed_episodes,
        agent_seed=seed,
        path=destination,
    )
    return {
        "episodes": parent.completed_episodes,
        "updates": parent.learner.update_steps,
        "replay_size": len(replay),
        "optimizer_entries": len(learner.optimizer.state),
    }


def prepare(args):
    cfg = read(CONFIG)
    require(sys.version_info[:2] == (3, 13), "Use Python 3.13")
    require(not args.root.exists(), "Use a new root; preserve existing evidence")
    require(sha(args.reference) == cfg["reference_sha256"], "Wrong reference")
    require(sha(args.external_zip) == cfg["external_sha256"], "Wrong opponent archive")
    require(read(CONFIG.with_name("seed-audit.json"))["passed"], "Seed audit required")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    require(not set(dirty.splitlines()).intersection(sources()), "Commit technical source first")
    require(set(sources()) <= set(tracked), "Untracked technical source")
    # Check availability before creating a partially prepared directory.
    subprocess.run(["git", "cat-file", "-e", cfg["runtime_commit"]], cwd=ROOT, check=True)
    args.root.mkdir(parents=True)
    shutil.copy2(args.reference, args.root / "reference.pt")
    shutil.copy2(CONFIG, args.root / "config.json")
    archive = args.root / "runtime.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", cfg["runtime_commit"]],
        cwd=ROOT,
        check=True,
    )
    with tarfile.open(archive) as stream:
        stream.extractall(args.root / "source", filter="data")
    external = args.root / "external"
    with zipfile.ZipFile(args.external_zip) as stream:
        for item in stream.infolist():
            require(
                (external / item.filename).resolve().is_relative_to(external.resolve()),
                "Unsafe zip member",
            )
        stream.extractall(external)
    models = list(external.rglob("z_best-model.pt"))
    require(
        len(models) == 1 and sha(models[0]) == cfg["external_model_sha256"], "Wrong external model"
    )
    shutil.copytree(models[0].parent, args.root / "source/agent_code/RUEHL_BASED_AGENT")
    sys.path.insert(0, str(args.root / "source"))
    initial = args.root / "initial"
    initial.mkdir()
    records = {}
    for replica, seed in enumerate(cfg["learner_seeds"], 1):
        for arm in cfg["arms"]:
            records[f"{arm}-r{replica}"] = initialize(
                args.root / "reference.pt",
                initial / f"{arm}-r{replica}.pt",
                arm,
                seed,
                cfg["learning_rate"],
            )
    write(args.root / "initialization.json", records)
    write(
        args.root / "binding.json",
        {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "tools": sources(),
            "config_sha256": sha(CONFIG),
            "runtime": {
                p.relative_to(args.root).as_posix(): sha(p)
                for p in (args.root / "source").rglob("*")
                if p.is_file()
            },
            "initial": {
                p.relative_to(args.root).as_posix(): sha(p)
                for p in [
                    args.root / "reference.pt",
                    args.root / "initialization.json",
                    *initial.glob("*.pt"),
                ]
            },
            "python": sys.version,
        },
    )


def epsilon(seed, index):
    return float(index % 5 == random.Random(seed + 168000000 + index // 5).randrange(5))


def play(root, checkpoint, cfg, replica, index, *, training=False, suite="classic", smoke=False):
    from scripts.stability_episode import play_episode

    setting = cfg["evaluation"][suite]
    seed = (
        cfg["smoke_seed"]
        if smoke
        else (
            cfg["train_ranges"][replica - 1][0] + index if training else setting["seeds"][0] + index
        )
    )
    opponents = cfg["training_opponents"] if training else setting["opponents"]
    row = play_episode(
        root,
        checkpoint,
        world_seed=seed,
        opponents=opponents,
        training=training,
        epsilon=epsilon(cfg["learner_seeds"][replica - 1], index) if training else 0.0,
        agent_seed=cfg["learner_seeds"][replica - 1],
        slot=index % (len(opponents) + 1),
        scenario="classic" if training else setting["scenario"],
    )
    # Correct the diagnostic, keeping the stale internal value explicitly distinct.
    metrics = row["native"].get("learning_metrics", {})
    if "epsilon" in metrics:
        metrics["checkpoint_internal_epsilon"] = metrics["epsilon"]
        metrics["epsilon"] = row["epsilon"]
    return row


def save_generation(directory, checkpoint, rows):
    generation = directory / f"generation-{len(rows):04d}-{time.time_ns()}"
    generation.mkdir()
    shutil.copy2(checkpoint, generation / "checkpoint.pt")
    with (generation / "checkpoint.pt").open("r+b") as stream:
        os.fsync(stream.fileno())
    write(generation / "rows.json", rows)
    write(
        directory / "resume.json",
        {
            "generation": generation.name,
            "episodes": len(rows),
            "checkpoint_sha256": sha(generation / "checkpoint.pt"),
            "rows_sha256": sha(generation / "rows.json"),
        },
    )


def train(root, cfg, artifact):
    arm, rep = artifact.rsplit("-r", 1)
    replica = int(rep)
    directory = root / "training" / artifact
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists() and read(directory / "result.json")["complete"]:
        return
    checkpoint, rows = directory / "working.pt", []
    initial = root / "initial" / f"{artifact}.pt"
    if (directory / "resume.json").exists():
        state = read(directory / "resume.json")
        generation = directory / state["generation"]
        require(
            sha(generation / "checkpoint.pt") == state["checkpoint_sha256"]
            and sha(generation / "rows.json") == state["rows_sha256"],
            "Corrupt generation",
        )
        rows = read(generation / "rows.json")
        require(len(rows) == state["episodes"], "Generation counter mismatch")
        initial = generation / "checkpoint.pt"
    shutil.copy2(initial, checkpoint)
    inherited = read(root / "initialization.json")[artifact]["episodes"]
    for index in range(len(rows), cfg["episodes"]):
        row = play(root, checkpoint, cfg, replica, index, training=True)
        require(row["completed_episodes"] == inherited + index + 1, "Training counter mismatch")
        rows.append(row)
        if len(rows) % 5 == 0 or len(rows) == cfg["episodes"]:
            save_generation(directory, checkpoint, rows)
            write(directory / "progress.json", {"episodes": len(rows), "budget": cfg["episodes"]})
    shutil.copy2(checkpoint, directory / "final.pt")
    with gzip.open(directory / "episodes.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(rows, stream)
    write(
        directory / "result.json",
        {
            "complete": True,
            "episodes": len(rows),
            "checkpoint_sha256": sha(directory / "final.pt"),
            "rows_sha256": sha(directory / "episodes.json.gz"),
        },
    )


def evaluate(root, cfg, artifact, suite):
    checkpoint = (
        root / "reference.pt"
        if artifact == "reference"
        else root / "training" / artifact / "final.pt"
    )
    checksum = sha(checkpoint)
    directory = root / "evaluation" / artifact / suite
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    setting = cfg["evaluation"][suite]
    for index, seed in enumerate(range(setting["seeds"][0], setting["seeds"][1] + 1)):
        path = directory / f"{seed}.json"
        if path.exists():
            row = read(path)
            require(
                row["world_seed"] == seed and row["checkpoint_sha256"] == checksum,
                "Stale evaluation",
            )
        else:
            row = play(root, checkpoint, cfg, 1, index, suite=suite)
            row["checkpoint_sha256"] = checksum
            write(path, row)
        rows.append(row)
    require(sha(checkpoint) == checksum, "Evaluation changed checkpoint")
    with gzip.open(directory / "episodes.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(rows, stream)
    write(
        directory / "result.json",
        {
            "complete": True,
            "checkpoint_sha256": checksum,
            "rows_sha256": sha(directory / "episodes.json.gz"),
        },
    )


def smoke(root):
    cfg, _ = bound(root)
    directory = root / "smoke"
    require(not directory.exists(), "Preserve existing smoke")
    directory.mkdir()
    report = {}
    for arm in cfg["arms"]:
        checkpoint = directory / f"{arm}.pt"
        shutil.copy2(root / "initial" / f"{arm}-r1.pt", checkpoint)
        rows = []
        for index in range(4):
            rows.append(play(root, checkpoint, cfg, 1, index, training=True, smoke=True))
            if sum(r["optimizer_updates_this_episode"] for r in rows) > 0:
                break
        require(
            sum(r["optimizer_updates_this_episode"] for r in rows) > 0, "Smoke needs actual updates"
        )
        checksum = sha(checkpoint)
        evaluation = play(root, checkpoint, cfg, 1, 0, smoke=True)
        require(sha(checkpoint) == checksum, "Smoke evaluation mutated checkpoint")
        report[arm] = {"training": rows, "evaluation": evaluation}
    write(directory / "report.json", {"passed": True, "scientific": False, "arms": report})


def export(root):
    from scripts.analyze_stability_test import analyze

    write(root / "analysis.json", analyze(root))
    names = [
        "config.json",
        "binding.json",
        "initialization.json",
        "analysis.json",
        "authorization.json",
        "resources.json",
        "smoke/report.json",
    ]
    for p in (root / "training").glob("*/result.json"):
        names += [
            q.relative_to(root).as_posix()
            for q in [p, p.with_name("final.pt"), p.with_name("episodes.json.gz")]
        ]
    for p in (root / "evaluation").glob("*/*/result.json"):
        names += [q.relative_to(root).as_posix() for q in [p, p.with_name("episodes.json.gz")]]
    manifest = {n: {"sha256": sha(root / n), "bytes": (root / n).stat().st_size} for n in names}
    archive = root / "issue213-results.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as stream:
        for name in names:
            stream.write(root / name, name)
        stream.writestr("manifest.json", json.dumps(manifest, indent=2))
    write(root / "export.json", {"sha256": sha(archive), "bytes": archive.stat().st_size})


def jobs(cfg):
    artifacts = [f"{arm}-r{replica}" for replica in range(1, 4) for arm in cfg["arms"]]
    return [("_train", artifact, "classic") for artifact in artifacts] + [
        ("_evaluate", artifact, suite)
        for artifact in ["reference", *artifacts]
        for suite in cfg["evaluation"]
    ]


def monitor(root):
    cfg, _ = bound(root)
    write(root / "supervisor.json", {"pid": os.getpid(), "created": psutil.Process().create_time()})
    require(read(root / "smoke/report.json")["passed"], "Passing smoke required")
    require((root / "authorization.json").exists(), "Explicit authorization required")
    require(not (root / "stop.json").exists(), "Inspect preserved stop record before any resume")
    usage_path = root / "resources.json"
    usage = read(usage_path) if usage_path.exists() else {"cpu_seconds": 0.0, "wall_seconds": 0.0}
    limits = cfg["limits"]
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for mode, artifact, suite in jobs(cfg):
            result = root / ("training" if mode == "_train" else "evaluation") / artifact
            if mode != "_train":
                result /= suite
            if (result / "result.json").exists() and read(result / "result.json")["complete"]:
                continue
            write(root / "status.json", {"stage": mode, "artifact": artifact, "suite": suite})
            with (root / f"{artifact}-{suite}-{mode}.log").open("a", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        mode,
                        "--root",
                        str(root),
                        "--artifact",
                        artifact,
                        "--suite",
                        suite,
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    cwd=ROOT,
                )
                last_time, seen = time.monotonic(), {}
                try:
                    while True:
                        alive = process.poll() is None
                        rss = 0
                        try:
                            parent = psutil.Process(process.pid)
                            for child in [parent, *parent.children(recursive=True)]:
                                ident = (child.pid, child.create_time())
                                cpu = sum(child.cpu_times()[:2])
                                usage["cpu_seconds"] += max(0, cpu - seen.get(ident, 0))
                                seen[ident] = cpu
                                rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                        now = time.monotonic()
                        usage["wall_seconds"] += now - last_time
                        last_time = now
                        write(usage_path, usage)
                        require(
                            usage["cpu_seconds"] < limits["cpu_seconds"]
                            and usage["wall_seconds"] < limits["wall_seconds"],
                            "Compute budget exhausted",
                        )
                        require(
                            rss <= limits["rss_bytes"]
                            and psutil.virtual_memory().available
                            >= limits["minimum_available_bytes"],
                            "RAM limit reached",
                        )
                        if not alive:
                            break
                        time.sleep(0.25)
                    require(process.returncode == 0, f"Worker failed: {artifact} {suite}")
                finally:
                    if process.poll() is None:
                        with __import__("contextlib").suppress(psutil.NoSuchProcess):
                            for child in psutil.Process(process.pid).children(recursive=True):
                                child.terminate()
                        process.terminate()
                        process.wait()
        write(root / "status.json", {"stage": "analysis_and_export"})
        export(root)
        write(root / "status.json", {"stage": "complete"})
    except Exception as exc:
        write(root / "stop.json", {"error": str(exc)})
        write(root / "status.json", {"stage": "stopped", "error": str(exc)})
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "prepare",
            "smoke",
            "launch",
            "status",
            "analyze",
            "_monitor",
            "_train",
            "_evaluate",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--external-zip", type=Path)
    parser.add_argument("--artifact")
    parser.add_argument("--suite", default="classic")
    parser.add_argument("--authorize-compute", action="store_true")
    parser.add_argument("--authorized-by")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "status":
        for name in ("status.json", "resources.json", "stop.json"):
            if (args.root / name).exists():
                print(name, json.dumps(read(args.root / name)))
    elif args.mode == "launch":
        cfg, _ = bound(args.root)
        require(
            args.authorize_compute and args.authorized_by,
            "Owner agreement and explicit compute authorization required",
        )
        require(read(args.root / "smoke/report.json")["passed"], "Smoke required")
        require(not (args.root / "stop.json").exists(), "Inspect stop record; no automatic retry")
        if (args.root / "supervisor.json").exists():
            require(args.resume, "Existing launch requires --resume")
            # Check command lines as well as PID identity: a crashed supervisor can
            # leave its Windows Python child alive. Never launch over that worker.
            for process in psutil.process_iter(["pid", "cmdline"]):
                if process.pid in {os.getpid(), psutil.Process().ppid()}:
                    continue
                command = process.info["cmdline"] or []
                require(
                    not (
                        str(args.root) in command
                        and any(mode in command for mode in ("_monitor", "_train", "_evaluate"))
                    ),
                    "Existing stability process still alive",
                )
            require(
                read(args.root / "authorization.json")["config_sha256"] == sha(CONFIG),
                "Resume authorization does not match protocol",
            )
        else:
            require(not args.resume, "No previous launch to resume")
            write(
                args.root / "authorization.json",
                {
                    "authorized_by": args.authorized_by,
                    "limits": cfg["limits"],
                    "config_sha256": sha(CONFIG),
                    "time": time.time(),
                },
            )
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )
        with (args.root / "supervisor.log").open("a") as log:
            p = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_monitor",
                    "--root",
                    str(args.root),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                start_new_session=os.name != "nt",
                cwd=ROOT,
            )
        write(
            args.root / "supervisor.json",
            {"pid": p.pid, "created": psutil.Process(p.pid).create_time()},
        )
        print("Detached supervisor", p.pid)
    elif args.mode == "analyze":
        from scripts.analyze_stability_test import analyze

        write(args.root / "analysis.json", analyze(args.root))
    else:
        cfg, _ = bound(args.root)
        if args.mode == "smoke":
            smoke(args.root)
        elif args.mode == "_monitor":
            monitor(args.root)
        elif args.mode in ("_train", "_evaluate"):
            require((args.root / "authorization.json").exists(), "No authorization")
            if args.mode == "_train":
                train(args.root, cfg, args.artifact)
            else:
                evaluate(args.root, cfg, args.artifact, args.suite)


if __name__ == "__main__":
    main()
