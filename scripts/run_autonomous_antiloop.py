"""Run the bounded Issue #225 frozen-policy anti-loop comparison."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
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
from scripts.curriculum_io import read, require, sha, write  # noqa: E402
from scripts.teacher_loop_episode import play_episode  # noqa: E402

EXPERIMENT = "experiments/2026-09-20-autonomous-antiloop-campaign"
CONFIG = ROOT / EXPERIMENT / "config.json"


def sources():
    paths = [CONFIG, CONFIG.with_name("seed-audit.json")]
    paths += [
        ROOT / "scripts" / name
        for name in (
            "run_autonomous_antiloop.py",
            "analyze_autonomous_antiloop.py",
            "audit_autonomous_antiloop_seeds.py",
            "curriculum_io.py",
            "teacher_loop_episode.py",
            "watch_hunting_curriculum.py",
        )
    ]
    paths.append(ROOT / "training/hunting_curriculum.py")
    paths.extend((ROOT / "agent_code/DagobertDuckDQNAntiLoop").rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def bound(root):
    binding, cfg = read(root / "binding.json"), read(root / "config.json")
    require(binding["tools"] == sources(), "Executed tools changed")
    require(sha(root / "config.json") == binding["config_sha256"], "Protocol changed")
    for name, digest in binding["inputs"].items():
        require(sha(root / name) == digest, f"Bound input changed: {name}")
    return cfg


def prepare(args):
    started = time.time()
    require(sys.version_info[:2] == (3, 13), "Use Python 3.13")
    cfg = read(CONFIG)
    require(not args.root.exists(), "New run root required; never overwrite evidence")
    require(sha(args.reference) == cfg["reference_sha256"], "Wrong frozen checkpoint")
    require(read(CONFIG.with_name("seed-audit.json"))["passed"], "Seed audit required")
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
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", cfg["runtime_commit"]],
        cwd=ROOT,
        check=True,
    )
    with tarfile.open(archive) as stream:
        stream.extractall(args.root / "source", filter="data")
    for relative in sources():
        source = ROOT / relative
        destination = args.root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    inputs = [args.root / "reference.pt"]
    inputs += [
        path
        for path in (args.root / "source").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    write(
        args.root / "binding.json",
        {
            "tools": sources(),
            "device": args.device,
            "config_sha256": sha(CONFIG),
            "runtime_archive_sha256": sha(archive),
            "inputs": {path.relative_to(args.root).as_posix(): sha(path) for path in inputs},
            "python": sys.version,
            "source_commit": source_commit,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "pygame", "psutil")
            },
            "authorization": cfg["authorization"],
        },
    )
    write(
        args.root / "resources.json",
        {
            "cpu_seconds": cfg["prior_usage"][args.device]["cpu_seconds"],
            "first_start": started,
            "wall_seconds": time.time() - started,
            "stage": "prepared",
            "peak_rss_bytes": psutil.Process().memory_info().rss,
        },
    )


def compact(row):
    steps = row.pop("late_steps")
    late_eligible = late_looping = broad_eligible = broad_looping = 0
    for index in range(23, len(steps)):
        window = steps[index - 23 : index + 1]
        signatures = {item["board_coins_sha256"] for item in window}
        scores = {item["score"] for item in window}
        opponents = {item["opponents_left"] for item in window}
        common = (
            all(not item["hazards"] and not item["progress"] for item in window)
            and len(signatures) == len(scores) == len(opponents) == 1
            and (
                window[-1]["crates_left"]
                or window[-1]["coins_visible"]
                or window[-1]["opponents_left"]
            )
        )
        if common:
            broad_eligible += 1
            broad_looping += len({tuple(item["position"]) for item in window}) <= 3
            if all(item["crates_left"] == 0 for item in window):
                late_eligible += 1
                late_looping += len({tuple(item["position"]) for item in window}) <= 3
    row["loop_windows"] = {"eligible": late_eligible, "looping": late_looping}
    row["broad_loop_windows"] = {"eligible": broad_eligible, "looping": broad_looping}
    return row


def evaluate(root, cfg, stage, arm):
    setting = cfg["evaluation"][stage]
    directory = root / "results" / stage / arm
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha(root / "reference.pt")
    for suite, specification in setting.items():
        path = directory / f"{suite}.json"
        if path.exists():
            require(read(path)["checkpoint_sha256"] == digest, "Stale evaluation")
            continue
        rows = []
        for index, seed in enumerate(
            range(specification["seeds"][0], specification["seeds"][1] + 1)
        ):
            row = play_episode(
                root,
                root / "reference.pt",
                world_seed=seed,
                opponents=specification["opponents"],
                training=False,
                epsilon=0,
                agent_seed=cfg["evaluation_agent_seed"],
                slot=index % (len(specification["opponents"]) + 1),
                scenario=specification["scenario"],
                guard_mode=cfg.get("arm_modes", {}).get(
                    arm, "on" if arm == "narrow_guard" else "off"
                ),
            )
            rows.append(compact(row))
        require(sha(root / "reference.pt") == digest, "Evaluation changed checkpoint")
        write(path, {"checkpoint_sha256": digest, "rows": rows})


def smoke(root):
    cfg = bound(root)
    directory = root / "mechanics"
    require(not directory.exists(), "Preserve previous smoke")
    directory.mkdir()
    digest = sha(root / "reference.pt")
    records = []
    for arm in cfg["arms"]:
        row = play_episode(
            root,
            root / "reference.pt",
            world_seed=cfg["smoke_seed"],
            opponents=["rule_based_agent"] * 3,
            training=False,
            epsilon=0,
            agent_seed=cfg["evaluation_agent_seed"],
            guard_mode=cfg.get("arm_modes", {}).get(arm, "on" if arm == "narrow_guard" else "off"),
        )
        records.append({"arm": arm, "row": compact(row)})
    require(sha(root / "reference.pt") == digest, "Smoke changed checkpoint")
    write(root / "smoke.json", {"passed": True, "scientific": False, "records": records})


def resource_breach(usage, cfg, device, now, rss, available):
    if usage["cpu_seconds"] >= device["cpu_seconds"]:
        return "CPU budget exhausted"
    if usage["wall_seconds"] >= cfg["limits"]["wall_seconds"]:
        return "Elapsed budget exhausted"
    if now >= datetime.fromisoformat(cfg["limits"]["pipeline_stop_utc"]).timestamp():
        return "Pipeline absolute stop reached"
    if rss > device["rss_bytes"] or available < cfg["limits"]["minimum_available_bytes"]:
        return "Memory limit reached"
    return None


def process_usage(process, seen):
    rss = 0
    try:
        family = [
            psutil.Process(process.pid),
            *psutil.Process(process.pid).children(recursive=True),
        ]
    except psutil.NoSuchProcess:
        family = []
    for member in family:
        try:
            identity = f"{member.pid}:{member.create_time()}"
            seen[identity] = max(seen.get(identity, 0.0), sum(member.cpu_times()[:2]))
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


def export(root):
    names = [
        "config.json",
        "binding.json",
        "resources.json",
        "smoke.json",
        "analysis.json",
        "complete.json",
        "STOP.json",
    ]
    names += [path.relative_to(root).as_posix() for path in root.glob("results/**/*.json")]
    names += [path.relative_to(root).as_posix() for path in root.glob("worker-usage/*.json")]
    archive = root / f"issue225-autonomous-antiloop-{time.time_ns()}.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as stream:
        manifest = {}
        for name in sorted(set(names)):
            path = root / name
            if path.exists():
                content = path.read_bytes()
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


def supervise(root):
    cfg = bound(root)
    analyze = importlib.import_module(
        cfg.get("analyzer", "scripts.analyze_autonomous_antiloop")
    ).analyze
    require(read(root / "smoke.json")["passed"], "Passing smoke required")
    require(not (root / "STOP.json").exists(), "Persistent stop exists")
    owner = psutil.Process()
    write(root / "supervisor.json", {"pid": owner.pid, "created": owner.create_time()})
    device = cfg["devices"][read(root / "binding.json")["device"]]
    usage = read(root / "resources.json")
    base_cpu = usage["cpu_seconds"]
    self_start = sum(owner.cpu_times()[:2])
    running, measured, logs = {}, {}, []

    def launch(mode, label, stage=None):
        log = (root / f"{label}.log").open("ab")
        logs.append(log)
        command = [sys.executable, __file__, mode, "--root", str(root), "--job-id", label]
        if stage:
            command += ["--stage", stage]
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        running[label] = child
        measured[label] = {}

    try:
        for stage in ("pilot_r1", "pilot_r2"):
            launch("_evaluate", stage, stage)
        phase = "pilot"
        while running:
            rss = owner.memory_info().rss
            for label, process in list(running.items()):
                rss += process_usage(process, measured[label])
                code = process.poll()
                if code is not None:
                    final_path = root / "worker-usage" / f"{label}.json"
                    if final_path.exists():
                        final = read(final_path)
                        identity = f"{final['pid']}:{final['created']}"
                        measured[label][identity] = max(
                            measured[label].get(identity, 0.0), final["cpu_seconds"]
                        )
                    del running[label]
                    require(code == 0, f"Worker failed: {label}; inspect its log")
            usage.update(
                cpu_seconds=base_cpu
                + sum(sum(group.values()) for group in measured.values())
                + sum(owner.cpu_times()[:2])
                - self_start,
                wall_seconds=time.time() - usage["first_start"],
                peak_rss_bytes=max(usage.get("peak_rss_bytes", 0), rss),
                stage=phase,
            )
            write(root / "resources.json", usage)
            reason = resource_breach(
                usage, cfg, device, time.time(), rss, psutil.virtual_memory().available
            )
            require(not reason, reason or "")
            require(not (root / "STOP.request").exists(), "User stop requested")
            if not running and phase == "pilot":
                analysis = analyze(root)
                write(root / "analysis.json", analysis)
                if (
                    analysis["eligible"]
                    and time.time()
                    < datetime.fromisoformat(cfg["limits"]["experiment_stop_utc"]).timestamp()
                ):
                    phase = "confirmation"
                    launch("_evaluate", "confirmation", "confirmation")
                else:
                    phase = "export"
                    break
            elif not running and phase == "confirmation":
                write(root / "analysis.json", analyze(root))
                phase = "export"
                break
            time.sleep(0.5)
        report = analyze(root)
        write(root / "analysis.json", report)
        write(
            root / "complete.json",
            {
                "pipeline_complete": True,
                "submission_promotion": False,
                "candidate_for_human_review": bool(report.get("confirmed_for_human_review")),
            },
        )
        export(root)
        usage["stage"] = "complete"
        usage["wall_seconds"] = time.time() - usage["first_start"]
        usage["cpu_seconds"] = (
            base_cpu
            + sum(sum(group.values()) for group in measured.values())
            + sum(owner.cpu_times()[:2])
            - self_start
        )
        write(root / "resources.json", usage)
        exported = read(root / "export.json")
        write(root / "export.json", {**exported, "final_pipeline_resources": usage})
    except BaseException as error:
        write(root / "STOP.json", {"reason": str(error), "time": time.time(), "usage": usage})
        raise
    finally:
        for process in running.values():
            if process.poll() is None:
                stop_tree(process)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "run", "_evaluate", "export"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="pc")
    parser.add_argument("--stage")
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
            process = psutil.Process()
            write(
                args.root / "worker-usage" / f"{args.job_id}.json",
                {
                    "cpu_seconds": sum(process.cpu_times()[:2]),
                    "pid": process.pid,
                    "created": process.create_time(),
                },
            )

        atexit.register(record_usage)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "smoke":
        smoke(args.root)
    elif args.mode == "run":
        supervise(args.root)
    elif args.mode == "_evaluate":
        cfg = bound(args.root)
        require(args.stage in cfg["evaluation"], "Unknown evaluation stage")
        require(
            time.time() < datetime.fromisoformat(cfg["limits"]["experiment_stop_utc"]).timestamp(),
            "Experiment stop reached",
        )
        for arm in cfg["arms"]:
            evaluate(args.root, cfg, args.stage, arm)
    else:
        export(args.root)


if __name__ == "__main__":
    main()
