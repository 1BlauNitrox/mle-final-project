"""Run the bounded, serial learned endgame-pursuit experiment for issue #225."""

from __future__ import annotations

import argparse
import hashlib
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

EXPERIMENT = ROOT / "experiments/2026-09-21-learned-endgame-pursuit"
CONFIG = EXPERIMENT / "config.json"


def sources():
    paths = [CONFIG, CONFIG.with_name("seed-audit.json")]
    paths += [
        ROOT / "scripts" / name
        for name in (
            "run_endgame_pursuit.py",
            "analyze_endgame_pursuit.py",
            "audit_endgame_pursuit_seeds.py",
            "curriculum_io.py",
            "teacher_loop_episode.py",
            "watch_hunting_curriculum.py",
        )
    ]
    paths.extend([ROOT / "training/endgame_pursuit.py", ROOT / "training/hunting_curriculum.py"])
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
        rss = cpu = 0.0
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
        now, reason = time.time(), None
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
    eligible = looping = 0
    for index in range(23, len(steps)):
        window = steps[index - 23 : index + 1]
        common = (
            all(not item["hazards"] and not item["progress"] for item in window)
            and len({item["board_coins_sha256"] for item in window})
            == len({item["score"] for item in window})
            == len({item["opponents_left"] for item in window})
            == 1
            and (
                window[-1]["crates_left"]
                or window[-1]["coins_visible"]
                or window[-1]["opponents_left"]
            )
        )
        if common:
            eligible += 1
            looping += len({tuple(item["position"]) for item in window}) <= 3
    row["broad_loop_windows"] = {"eligible": eligible, "looping": looping}
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
        hunting_index=0,
        collect_pursuit_training_trace=True,
    )
    write(
        root / "smoke.json",
        {
            "passed": bool(row["pursuit_training_trace"]),
            "scientific": False,
            "row": compact(row),
        },
    )


def collection_rows(root):
    return [read(path) for path in sorted((root / "collection").glob("episode-*.json"))]


def collection_counts(rows):
    examples = [example for row in rows for example in row["pursuit_training_trace"]]
    return len(examples), sum(example["teacher_action"] == "BOMB" for example in examples)


def collect(root, cfg, meter):
    setting = cfg["collection"]
    directory = root / "collection"
    directory.mkdir(exist_ok=True)
    if (root / "collection-complete.json").exists():
        return read(root / "collection-complete.json")["passed"]
    completed = len(collection_rows(root))
    while completed < setting["maximum_episodes"]:
        meter.heartbeat("collection", training=True)
        seed = setting["seed_range"][0] + completed
        require(seed <= setting["seed_range"][1], "Collection seed range exhausted")
        row = episode(
            root,
            root / "reference.pt",
            world_seed=seed,
            opponents=setting["lineups"][completed % len(setting["lineups"])],
            training=False,
            epsilon=0,
            agent_seed=cfg["evaluation_agent_seed"],
            slot=completed % 4,
            scenario="classic",
            guard_mode="broad_persistent",
            hunting_index=completed * 4,
            collect_pursuit_training_trace=True,
        )
        write(directory / f"episode-{completed:04d}.json", compact(row))
        completed += 1
        if completed % setting["block_episodes"] == 0:
            examples, bombs = collection_counts(collection_rows(root))
            write(
                root / "collection-progress.json",
                {"episodes": completed, "examples": examples, "bomb_examples": bombs},
            )
            if (
                completed >= setting["minimum_episodes"]
                and examples >= setting["minimum_examples"]
                and bombs >= setting["minimum_bomb_examples"]
            ):
                break
    rows = collection_rows(root)
    examples, bombs = collection_counts(rows)
    result = {
        "passed": examples >= setting["minimum_examples"]
        and bombs >= setting["minimum_bomb_examples"],
        "episodes": len(rows),
        "examples": examples,
        "bomb_examples": bombs,
    }
    write(root / "collection-complete.json", result)
    return result["passed"]


def train(root, cfg, meter):
    if (root / "training.json").exists():
        return read(root / "training.json")
    meter.heartbeat("training", training=True)
    from training.endgame_pursuit import (
        build_dataset,
        save_artifact,
        train_grouped_pursuit,
    )

    collection = cfg["collection"]
    dataset = build_dataset(
        collection_rows(root),
        minimum_examples=collection["minimum_examples"],
        minimum_bombs=collection["minimum_bomb_examples"],
    )
    setting = cfg["training"]
    parameters, cv = train_grouped_pursuit(
        dataset,
        folds=setting["folds"],
        epochs=setting["epochs"],
        learning_rate=setting["learning_rate"],
        l2_weight=setting["l2_weight"],
        seed=setting["optimizer_seed"],
        thresholds=setting["thresholds"],
        minimum_coverage=setting["minimum_oof_coverage"],
        minimum_accuracy=setting["minimum_oof_accuracy"],
        minimum_bomb_precision=setting["minimum_oof_bomb_precision"],
        minimum_bomb_recall=setting["minimum_oof_bomb_recall"],
    )
    result = {
        "classifier_accepted": parameters is not None,
        "cross_validation": cv,
        "candidates": {},
    }
    if parameters is not None:
        agent_dir = root / "source/agent_code/DagobertDuckDQNAntiLoop"
        for threshold in cv["accepted_thresholds"]:
            arm = f"t{int(threshold * 100):02d}"
            name = f"endgame-pursuit-{arm}.pt"
            path = root / "candidates" / name
            save_artifact(
                path,
                parameters,
                threshold=threshold,
                parent_checkpoint_sha256=cfg["reference_sha256"],
            )
            shutil.copy2(path, agent_dir / name)
            result["candidates"][arm] = {
                "path": path.relative_to(root).as_posix(),
                "file_name": name,
                "sha256": sha(path),
                "bytes": path.stat().st_size,
                "threshold": threshold,
            }
    write(root / "training.json", result)
    return result


def evaluate(root, cfg, meter, stage, arm, artifact=None):
    checkpoint = root / "reference.pt"
    for suite, setting in cfg["evaluation"][stage].items():
        path = root / "results" / stage / arm / f"{suite}.json"
        if path.exists():
            continue
        values = []
        for index, seed in enumerate(range(setting["seeds"][0], setting["seeds"][1] + 1)):
            meter.heartbeat(f"{stage}:{arm}:{suite}")
            values.append(
                compact(
                    episode(
                        root,
                        checkpoint,
                        world_seed=seed,
                        opponents=setting["opponents"],
                        training=False,
                        epsilon=0,
                        agent_seed=cfg["evaluation_agent_seed"],
                        slot=index % (len(setting["opponents"]) + 1),
                        scenario=setting["scenario"],
                        guard_mode="broad_learned_pursuit" if artifact else "broad_persistent",
                        pursuit_head_name=artifact["file_name"] if artifact else None,
                    )
                )
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        write(
            path,
            {
                "checkpoint_sha256": sha(checkpoint),
                "pursuit_head_sha256": artifact["sha256"] if artifact else None,
                "rows": values,
            },
        )


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
    names += [path.relative_to(root).as_posix() for path in root.glob("collection/*.json")]
    archive = root / f"issue225-endgame-pursuit-{time.time_ns()}.zip"
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
        collected = collect(root, cfg, meter)
        training = (
            train(root, cfg, meter)
            if collected
            else {
                "classifier_accepted": False,
                "reason": "Insufficient registered teacher examples",
                "candidates": {},
            }
        )
        if not collected:
            write(root / "training.json", training)
        if training["classifier_accepted"]:
            evaluate(root, cfg, meter, "pilot", "baseline")
            for arm, artifact in training["candidates"].items():
                require(sha(root / artifact["path"]) == artifact["sha256"], "Candidate changed")
                evaluate(root, cfg, meter, "pilot", arm, artifact)
        from scripts.analyze_endgame_pursuit import analyze

        report = analyze(root)
        write(root / "analysis.json", report)
        if report.get("selected"):
            selected = report["selected"]
            evaluate(root, cfg, meter, "confirmation", "baseline")
            evaluate(root, cfg, meter, "confirmation", selected, training["candidates"][selected])
            report = analyze(root)
            write(root / "analysis.json", report)
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
        write(
            root / "export.json", {**read(root / "export.json"), "final_pipeline_resources": final}
        )
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
