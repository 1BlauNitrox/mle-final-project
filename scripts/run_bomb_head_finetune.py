"""Run the bounded, serial BOMB-output-only fine-tuning attempt for issue #225."""

from __future__ import annotations

import argparse
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

from scripts.curriculum_io import read, require, sha, write  # noqa: E402

EXPERIMENT = ROOT / "experiments/2026-09-21-bomb-head-finetune"
CONFIG = EXPERIMENT / "config.json"


def sources():
    paths = [CONFIG, CONFIG.with_name("seed-audit.json")]
    paths += [
        ROOT / "scripts" / name
        for name in (
            "run_bomb_head_finetune.py",
            "analyze_bomb_head_finetune.py",
            "audit_bomb_head_finetune_seeds.py",
            "audit_autonomous_antiloop_seeds.py",
            "curriculum_io.py",
            "teacher_loop_episode.py",
            "watch_hunting_curriculum.py",
        )
    ]
    paths.extend(
        [
            ROOT / "training/bomb_head_finetune.py",
            ROOT / "training/hunting_curriculum.py",
        ]
    )
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
    require(not args.root.exists(), "New run root required")
    require(sha(args.reference) == cfg["reference_sha256"], "Wrong parent checkpoint")
    require(read(CONFIG.with_name("seed-audit.json"))["passed"], "Passing seed audit required")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    require(
        not set(dirty.splitlines()).intersection(sources()) and set(sources()) <= set(tracked),
        "Commit technical sources before preparation",
    )
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
        destination = args.root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
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
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
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


class Meter:
    def __init__(self, root, cfg, stage):
        self.root, self.cfg, self.stage = root, cfg, stage
        self.process = psutil.Process()
        usage = read(root / "resources.json")
        self.base = usage["cpu_seconds"]
        self.started_cpu = sum(self.process.cpu_times()[:2])

    def heartbeat(self, stage=None, *, training=False):
        if stage:
            self.stage = stage
        family = [self.process, *self.process.children(recursive=True)]
        rss = 0
        cpu = 0.0
        for process in family:
            try:
                rss += process.memory_info().rss
                cpu += sum(process.cpu_times()[:2])
            except psutil.NoSuchProcess:
                pass
        usage = read(self.root / "resources.json")
        usage.update(
            cpu_seconds=self.base + cpu - self.started_cpu,
            wall_seconds=time.time() - usage["first_start"],
            stage=self.stage,
            peak_rss_bytes=max(usage.get("peak_rss_bytes", 0), rss),
        )
        write(self.root / "resources.json", usage)
        device = self.cfg["devices"][read(self.root / "binding.json")["device"]]
        now = time.time()
        reason = None
        if usage["cpu_seconds"] >= device["cpu_seconds"]:
            reason = "CPU budget exhausted"
        elif usage["wall_seconds"] >= self.cfg["limits"]["wall_seconds"]:
            reason = "Elapsed budget exhausted"
        elif now >= datetime.fromisoformat(self.cfg["limits"]["stop_utc"]).timestamp():
            reason = "Absolute deadline reached"
        elif (
            training
            and now >= datetime.fromisoformat(self.cfg["limits"]["training_stop_utc"]).timestamp()
        ):
            reason = "Training absolute deadline reached"
        elif (
            rss > device["rss_bytes"]
            or psutil.virtual_memory().available < self.cfg["limits"]["minimum_available_bytes"]
        ):
            reason = "Memory limit reached"
        require(not reason, reason or "")
        require(not (self.root / "STOP.request").exists(), "User stop requested")
        return usage


def compact(row):
    steps = row.pop("late_steps")
    broad_eligible = broad_looping = 0
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
    row["broad_loop_windows"] = {"eligible": broad_eligible, "looping": broad_looping}
    return row


def episode(root, checkpoint, **kwargs):
    from scripts.teacher_loop_episode import play_episode

    return play_episode(root, checkpoint, **kwargs)


def smoke(root):
    cfg = bound(root)
    require(not (root / "smoke.json").exists(), "Preserve previous smoke")
    row = episode(
        root,
        root / "reference.pt",
        world_seed=cfg["smoke_seed"],
        opponents=["rule_based_agent"] * 3,
        training=False,
        epsilon=0,
        agent_seed=cfg["evaluation_agent_seed"],
        guard_mode="broad_persistent",
        collect_head_training_trace=True,
        trace_stride=cfg["collection"]["trace_stride"],
    )
    require(row["head_training_trace"]["anchors"], "Smoke did not collect anchor states")
    write(root / "smoke.json", {"passed": True, "scientific": False, "row": compact(row)})


def exploration(agent_seed, index):
    chosen = random.Random(agent_seed + 168_000_000 + index // 5).randrange(5)
    return float(index % 5 == chosen)


def collection_rows(root):
    return [read(path) for path in sorted((root / "collection").glob("episode-*.json"))]


def positive_count(rows):
    return sum(
        int(bomb["kill_credits"])
        for row in rows
        for bomb in row["head_training_trace"]["bombs"]
        if int(bomb["self_kill_credits"]) == 0
    )


def collect(root, cfg, meter):
    directory = root / "collection"
    directory.mkdir(exist_ok=True)
    setting = cfg["collection"]
    existing = collection_rows(root)
    completed = len(existing)
    require(completed <= setting["maximum_episodes"], "Too many collection episodes")
    if (root / "collection-complete.json").exists():
        return read(root / "collection-complete.json")["passed"]
    while completed < setting["maximum_episodes"]:
        meter.heartbeat("collection", training=True)
        seed = setting["seed_range"][0] + completed
        require(seed <= setting["seed_range"][1], "Collection seed range exhausted")
        opponents = setting["lineups"][completed % len(setting["lineups"])]
        row = episode(
            root,
            root / "reference.pt",
            world_seed=seed,
            opponents=opponents,
            training=False,
            epsilon=exploration(cfg["evaluation_agent_seed"], completed),
            agent_seed=cfg["evaluation_agent_seed"],
            slot=completed % (len(opponents) + 1),
            scenario="classic",
            guard_mode="broad_persistent",
            collect_head_training_trace=True,
            trace_stride=setting["trace_stride"],
        )
        write(directory / f"episode-{completed:04d}.json", compact(row))
        completed += 1
        if completed % setting["block_episodes"] == 0:
            values = collection_rows(root)
            positives = positive_count(values)
            write(
                root / "collection-progress.json",
                {"episodes": completed, "empirical_positive_kill_credits": positives},
            )
            if (
                completed >= setting["minimum_episodes"]
                and positives >= setting["minimum_empirical_positive_bombs"]
            ):
                break
    values = collection_rows(root)
    positives = positive_count(values)
    result = {
        "passed": positives >= setting["minimum_empirical_positive_bombs"],
        "episodes": len(values),
        "empirical_positive_kill_credits": positives,
        "native_kills": sum(row["native"]["kills"] for row in values),
        "native_self_kills": sum(row["native"]["self_kills"] for row in values),
        "bomb_records": sum(len(row["head_training_trace"]["bombs"]) for row in values),
        "anchor_records": sum(len(row["head_training_trace"]["anchors"]) for row in values),
    }
    write(root / "collection-complete.json", result)
    return result["passed"]


def train(root, cfg, meter):
    if (root / "training.json").exists():
        return read(root / "training.json")
    meter.heartbeat("training", training=True)
    from training.bomb_head_finetune import build_dataset, install_online_state, train_snapshots

    dataset = build_dataset(
        collection_rows(root),
        input_dim=56,
        minimum_empirical_positives=cfg["collection"]["minimum_empirical_positive_bombs"],
    )
    setting = cfg["training"]
    snapshots, history = train_snapshots(
        root / "reference.pt",
        dataset,
        snapshot_steps=setting["snapshot_steps"],
        learning_rate=setting["learning_rate"],
        margin=setting["margin"],
        anchor_weight=setting["anchor_weight"],
        negative_weight=setting["negative_weight"],
        l2_weight=setting["l2_weight"],
        batch_size=setting["batch_size"],
        seed=setting["optimizer_seed"],
    )
    candidates = {}
    for step, state in snapshots.items():
        path = root / "candidates" / f"u{step}.pt"
        install_online_state(root / "reference.pt", path, state)
        candidates[f"u{step}"] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
    result = {
        "dataset": {
            "empirical_positive_count": dataset.empirical_positive_count,
            "trapped_teacher_count": dataset.trapped_teacher_count,
            "negative_count": dataset.negative_count,
            "unique_positive_count": len(dataset.positive),
            "unique_anchor_count": len(dataset.anchors),
        },
        "candidates": candidates,
        "loss_history": history,
        "evaluation_only_note": (
            "Online network contains the restricted learned head; inherited "
            "target/Adam/replay are retained for provenance and are not "
            "authorized for resume training."
        ),
    }
    write(root / "training.json", result)
    return result


def evaluate(root, cfg, meter, stage, arm, checkpoint):
    digest = sha(checkpoint)
    for suite, setting in cfg["evaluation"][stage].items():
        path = root / "results" / stage / arm / f"{suite}.json"
        if path.exists():
            require(read(path)["checkpoint_sha256"] == digest, "Stale evaluation checkpoint")
            continue
        values = []
        for index, seed in enumerate(range(setting["seeds"][0], setting["seeds"][1] + 1)):
            meter.heartbeat(f"{stage}:{arm}:{suite}")
            row = episode(
                root,
                checkpoint,
                world_seed=seed,
                opponents=setting["opponents"],
                training=False,
                epsilon=0,
                agent_seed=cfg["evaluation_agent_seed"],
                slot=index % (len(setting["opponents"]) + 1),
                scenario=setting["scenario"],
                guard_mode="broad_persistent",
            )
            values.append(compact(row))
        require(sha(checkpoint) == digest, "Evaluation changed checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, {"checkpoint_sha256": digest, "rows": values})


def export(root):
    names = [
        "config.json",
        "binding.json",
        "resources.json",
        "smoke.json",
        "collection-progress.json",
        "collection-complete.json",
        "training.json",
        "analysis.json",
        "complete.json",
        "STOP.json",
    ]
    names += [path.relative_to(root).as_posix() for path in root.glob("results/**/*.json")]
    names += [path.relative_to(root).as_posix() for path in root.glob("candidates/*.pt")]
    archive = root / f"issue225-bomb-head-{time.time_ns()}.zip"
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


def run(root):
    cfg = bound(root)
    require(read(root / "smoke.json")["passed"], "Passing smoke required")
    require(not (root / "STOP.json").exists(), "Persistent stop exists")
    owner = psutil.Process()
    write(root / "supervisor.json", {"pid": owner.pid, "created": owner.create_time()})
    meter = Meter(root, cfg, "starting")
    try:
        if not collect(root, cfg, meter):
            write(
                root / "analysis.json",
                {
                    "complete": True,
                    "eligible": [],
                    "selected": None,
                    "reason": "Insufficient empirical kill exposure",
                    "submission_promotion": False,
                },
            )
        else:
            training = train(root, cfg, meter)
            evaluate(root, cfg, meter, "pilot", "baseline", root / "reference.pt")
            for arm, record in training["candidates"].items():
                checkpoint = root / record["path"]
                require(sha(checkpoint) == record["sha256"], "Candidate changed")
                evaluate(root, cfg, meter, "pilot", arm, checkpoint)
            from scripts.analyze_bomb_head_finetune import analyze

            report = analyze(root)
            write(root / "analysis.json", report)
            if report["selected"]:
                selected = report["selected"]
                evaluate(root, cfg, meter, "confirmation", "baseline", root / "reference.pt")
                evaluate(
                    root,
                    cfg,
                    meter,
                    "confirmation",
                    selected,
                    root / training["candidates"][selected]["path"],
                )
                report = analyze(root)
                write(root / "analysis.json", report)
        report = read(root / "analysis.json")
        write(
            root / "complete.json",
            {
                "pipeline_complete": True,
                "submission_promotion": False,
                "candidate_for_human_review": bool(report.get("confirmed_for_human_review")),
                "selected": report.get("selected"),
            },
        )
        meter.heartbeat("export")
        export(root)
        final = meter.heartbeat("complete")
        exported = read(root / "export.json")
        write(root / "export.json", {**exported, "final_pipeline_resources": final})
    except BaseException as error:
        write(
            root / "STOP.json",
            {"reason": str(error), "time": time.time(), "usage": read(root / "resources.json")},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "run", "export"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="pc")
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
    if args.mode == "prepare":
        require(args.reference is not None, "prepare requires --reference")
        prepare(args)
    elif args.mode == "smoke":
        smoke(args.root)
    elif args.mode == "run":
        run(args.root)
    else:
        bound(args.root)
        export(args.root)


if __name__ == "__main__":
    main()
