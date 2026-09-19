"""Prepare, smoke, and durably train the registered Issue #207 experiment."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/2026-09-19-five-step-dqn/config.json"
RUNTIME_COMMIT = "c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113"
EXPERIMENTAL_AGENT = ROOT / "agent_code/Bomb-omb-nstep"
GIB = 1024**3


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_rows(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(json.dumps(rows, separators=(",", ":")).encode())
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def config():
    value = read_json(CONFIG)
    require(value["status"] == "prospective" and value["issue"] == 207, "Wrong protocol")
    require(value["source"]["trained_runtime_commit"] == RUNTIME_COMMIT, "Wrong runtime")
    require(value["shared_training"]["total_episodes"] == 6000, "Wrong budget")
    return value


def inclusive(pair):
    return list(range(pair[0], pair[1] + 1))


def environment():
    import torch

    return {
        "hostname": os.environ.get("COMPUTERNAME"),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_memory_bytes": psutil.virtual_memory().total,
    }


def prepare(root: Path, reference: Path, external: Path):
    cfg = config()
    require(sys.version_info[:2] == (3, 13), "Python 3.13 is required")
    require(not root.exists(), f"Preserve and inspect existing root: {root}")
    require(sha256(reference) == cfg["source"]["reference_sha256"], "Wrong reference")
    require(
        sha256(external / "z_best-model.pt") == cfg["source"]["external_checkpoint_sha256"],
        "Wrong external checkpoint",
    )
    require(read_json(CONFIG.with_name("seed-audit.json"))["passed"], "Seed audit failed")
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "Commit technical source before preparation",
    )
    root.mkdir(parents=True)
    shutil.copy2(reference, root / "reference.pt")
    shutil.copy2(CONFIG, root / "config.json")
    archive = root / "runtime-source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", RUNTIME_COMMIT],
        cwd=ROOT,
        check=True,
    )
    source = root / "source"
    source.mkdir()
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter="data")
    runtime_agent = source / "agent_code/DagobertDuckDQNTask3"
    for name in ("model.py", "replay.py", "persistence.py"):
        shutil.copy2(EXPERIMENTAL_AGENT / name, runtime_agent / name)
    shutil.copytree(external, source / "agent_code/RUEHL_BASED_AGENT")
    source_manifest = {
        path.relative_to(source).as_posix(): sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    write_json(root / "source-manifest.json", source_manifest)
    for arm in cfg["arms"]:
        for replica in range(cfg["shared_training"]["replicas"]):
            destination = root / "initial" / f"{arm}-r{replica + 1}.pt"
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(reference, destination)
            require(sha256(destination) == cfg["source"]["reference_sha256"], "Initial changed")
    write_json(
        root / "binding.json",
        {
            "tool_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "config_sha256": sha256(CONFIG),
            "reference_sha256": sha256(root / "reference.pt"),
            "runtime_commit": RUNTIME_COMMIT,
            "source_manifest_sha256": hashlib.sha256(
                json.dumps(source_manifest, sort_keys=True).encode()
            ).hexdigest(),
            "prepared_at": datetime.now().astimezone().isoformat(),
            "environment": environment(),
        },
    )
    print(json.dumps({"prepared": str(root), "scientific_execution_authorized": False}, indent=2))


@contextlib.contextmanager
def release_agent_logs():
    before = {
        handler
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
        for handler in logger.handlers
    }
    try:
        yield
    finally:
        for logger in logging.Logger.manager.loggerDict.values():
            if not isinstance(logger, logging.Logger):
                continue
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler) and handler not in before:
                    logger.removeHandler(handler)
                    handler.close()


def seeded_event_wrapper(original, root_seed):
    """Give every non-learning AgentRunner an isolated legacy RNG stream."""

    def process_event(self, event_name, *event_args):
        if self.code_name == "DagobertDuckDQNTask3":
            return original(self, event_name, *event_args)
        if not hasattr(self, "_issue207_rng"):
            slot = int(hashlib.sha256(self.agent_name.encode()).hexdigest()[:8], 16)
            words = np.random.SeedSequence([root_seed, slot, 207]).generate_state(3)
            numpy_seed = int(words[0])
            python_seed = (int(words[1]) << 32) | int(words[2])
            self._issue207_seed = numpy_seed, python_seed
            self._issue207_rng = (
                np.random.RandomState(numpy_seed).get_state(),
                random.Random(python_seed).getstate(),
            )
        numpy_seed, python_seed = self._issue207_seed
        previous = np.random.get_state(), random.getstate()
        numpy_seed_function, python_seed_function = np.random.seed, random.seed
        np.random.set_state(self._issue207_rng[0])
        random.setstate(self._issue207_rng[1])
        try:
            if event_name == "setup":
                np.random.seed = lambda value=None: numpy_seed_function(
                    numpy_seed if value is None else value
                )
                random.seed = lambda value=None, version=2: python_seed_function(
                    python_seed if value is None else value, version=version
                )
            return original(self, event_name, *event_args)
        finally:
            self._issue207_rng = np.random.get_state(), random.getstate()
            np.random.seed, random.seed = numpy_seed_function, python_seed_function
            np.random.set_state(previous[0])
            random.setstate(previous[1])

    return process_event


def play_episode(
    root, checkpoint, *, world_seed, opponents, training, epsilon, n_step, scenario="classic"
):
    from scripts.hunting_n_step import n_step_training

    source = (root / "source").resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(source)
    from agent_code.DagobertDuckDQNTask3 import callbacks, train
    from agents import AgentRunner
    from environment import BombeRLeWorld, WorldArgs

    actions = []
    selector = callbacks.select_action

    def behavior(**kwargs):
        action = selector(**{**kwargs, "epsilon": epsilon})
        actions.append(action)
        return action

    log_dir = root / "framework-logs"
    log_dir.mkdir(exist_ok=True)
    args = WorldArgs(
        True,
        30,
        False,
        0,
        False,
        None,
        False,
        not training,
        str(log_dir),
        False,
        "issue207",
        world_seed,
        False,
        scenario,
    )
    rows = []
    env = {
        "BOMBERMAN_AGENT_SEED": str(config()["shared_training"]["checkpoint_internal_agent_seed"]),
        "BOMBERMAN_DQN_ACTION_MASKING": "framework_legal",
        "BOMBERMAN_DQN_ESCAPE_CONTINUATIONS": "on",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BOMBERMAN_COMPACT_LOGS": "1",
    }
    with (
        release_agent_logs(),
        patch.dict(os.environ, env),
        patch.object(callbacks, "CHECKPOINT_PATH", checkpoint),
        patch.object(train, "CHECKPOINT_PATH", checkpoint),
        patch.object(callbacks, "_evaluation_checkpoint_path", lambda: checkpoint),
        patch.object(callbacks, "select_action", behavior),
        patch.object(
            AgentRunner,
            "process_event",
            seeded_event_wrapper(AgentRunner.process_event, world_seed + 1_000_000),
        ),
        n_step_training(train, horizon=n_step, discount_factor=0.9),
    ):
        world = BombeRLeWorld(
            args,
            [("DagobertDuckDQNTask3", training), *[(name, False) for name in opponents]],
        )
        learner = world.agents[0]
        policy = learner.backend.runner.fake_self
        before_updates = policy.learner.update_steps if training else None
        world.new_round()
        while world.running:
            alive = not learner.dead
            world.do_step()
            if not alive:
                continue
            state = learner.last_game_state
            if state is not None:
                signature = hashlib.sha256(
                    json.dumps(
                        [state["field"].tolist(), sorted(state["coins"])],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                rows.append(
                    {
                        "step": int(state["step"]),
                        "position": [int(value) for value in state["self"][3]],
                        "crates_left": int((state["field"] == 1).sum()),
                        "coins_visible": len(state["coins"]),
                        "hazards": bool(state["bombs"] or np.any(state["explosion_map"])),
                        "progress": bool(
                            {"COIN_COLLECTED", "CRATE_DESTROYED", "KILLED_OPPONENT"}
                            & set(learner.events)
                        ),
                        "board_coins_sha256": signature,
                    }
                )
        world.end()
        native = world.round_statistics[world.round_id]["agents"][learner.name]
        return {
            "world_seed": world_seed,
            "epsilon": epsilon,
            "opponents": list(opponents),
            "native": native,
            "late_steps": rows,
            "actions_sha256": hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
            "completed_episodes": policy.completed_episodes,
            "optimizer_updates": policy.learner.update_steps if training else None,
            "optimizer_updates_this_episode": (
                policy.learner.update_steps - before_updates if training else None
            ),
        }


def episode_epsilon(replica_seed, episode):
    period = config()["shared_training"]["random_episode_period"]
    selected = random.Random(replica_seed + 168_000_000 + episode // period).randrange(period)
    return 1.0 if episode % period == selected else 0.0


def smoke(root):
    cfg = config()
    output = root / "smoke"
    require(not output.exists(), "Preserve existing smoke")
    output.mkdir()
    reports = []
    process = psutil.Process()
    for (arm, setting), world_seed in zip(
        cfg["arms"].items(), cfg["smoke"]["world_seeds"], strict=True
    ):
        checkpoint = output / f"{arm}.pt"
        shutil.copy2(root / "reference.pt", checkpoint)
        started = time.monotonic()
        row = play_episode(
            root,
            checkpoint,
            world_seed=world_seed,
            opponents=cfg["shared_training"]["opponents"],
            training=True,
            epsilon=1.0,
            n_step=setting["n_step"],
        )
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require("bootstrap_discounts" in payload["replay_state"], "Discounts not persisted")
        require(row["completed_episodes"] == 8001, "Smoke did not continue reference")
        require(
            row["optimizer_updates_this_episode"] == row["native"]["attempted_actions"],
            "One update per transition invariant failed",
        )
        reports.append(
            {
                "arm": arm,
                "n_step": setting["n_step"],
                "wall_seconds": time.monotonic() - started,
                "rss_bytes": process.memory_info().rss,
                "checkpoint_sha256": sha256(checkpoint),
                "row": row,
            }
        )
    write_json(output / "report.json", {"environment": environment(), "runs": reports})
    print(json.dumps({"smoke": "passed", "runs": reports}, indent=2))


def resume_state(directory):
    checkpoint = directory / "resume.pt"
    trail = directory / "episodes.json.gz"
    if not checkpoint.exists() and not trail.exists():
        return 0, []
    require(checkpoint.exists() and trail.exists(), "Incomplete durable resume pair")
    rows = read_rows(trail)
    return len(rows), rows


def training_job(root, arm, replica):
    cfg = config()
    setting = cfg["arms"][arm]
    output = root / "training" / f"{arm}-r{replica + 1}"
    durable = root / "resume" / f"{arm}-r{replica + 1}"
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    completed, rows = resume_state(durable)
    if completed:
        shutil.copy2(durable / "resume.pt", checkpoint)
    else:
        shutil.copy2(root / "initial" / f"{arm}-r{replica + 1}.pt", checkpoint)
    seeds = inclusive(cfg["shared_training"]["replica_world_seed_ranges_inclusive"][replica])
    require(len(seeds) == 1000, "Wrong replica budget")
    started, cpu = time.monotonic(), time.process_time()
    attempts = output / "attempts.jsonl"
    for index, world_seed in enumerate(seeds):
        if index < completed:
            continue
        if (
            root / "stop-request.json"
        ).exists() or datetime.now().astimezone() >= datetime.fromisoformat(
            cfg["resources"]["absolute_stop"]
        ):
            break
        row = play_episode(
            root,
            checkpoint,
            world_seed=world_seed,
            opponents=cfg["shared_training"]["opponents"],
            training=True,
            epsilon=episode_epsilon(seeds[0], index),
            n_step=setting["n_step"],
        )
        require(row["completed_episodes"] == 8000 + index + 1, "Episode counter mismatch")
        require(
            row["optimizer_updates_this_episode"] == row["native"]["attempted_actions"],
            "Update budget mismatch",
        )
        row["episode"] = index + 1
        rows.append(row)
        with attempts.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"episode": index + 1, "world_seed": world_seed}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        episode = index + 1
        if episode in cfg["shared_training"]["milestones"]:
            shutil.copy2(checkpoint, durable / f"milestone-{episode:06d}.pt")
        if episode % cfg["shared_training"]["durable_checkpoint_every"] == 0 or episode == 1000:
            write_rows(durable / "episodes.json.gz", rows)
            shutil.copy2(checkpoint, durable / "resume.pt.tmp")
            os.replace(durable / "resume.pt.tmp", durable / "resume.pt")
    complete = len(rows) == 1000
    write_rows(output / "episodes.json.gz", rows)
    result = {
        "arm": arm,
        "replica": replica + 1,
        "episodes": len(rows),
        "complete": complete,
        "checkpoint_sha256": sha256(checkpoint),
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time() - cpu,
        "ended_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if complete else 2


def process_rss(process):
    try:
        parent = psutil.Process(process.pid)
        return parent.memory_info().rss + sum(
            child.memory_info().rss for child in parent.children(recursive=True)
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 0


def train_supervisor(root, workers):
    cfg = config()
    require((root / "smoke/report.json").exists(), "Passing smoke required")
    require(1 <= workers <= cfg["resources"]["maximum_workers"], "Unregistered workers")
    jobs = [(arm, replica) for replica in range(3) for arm in cfg["arms"]]
    pending = []
    for arm, replica in jobs:
        result = root / "training" / f"{arm}-r{replica + 1}/result.json"
        if not result.exists() or not read_json(result).get("complete"):
            pending.append((arm, replica))
    active = {}
    logs = root / "supervisor-logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()
    while pending or active:
        now = datetime.now().astimezone()
        rss = sum(process_rss(process) for process in active)
        memory = psutil.virtual_memory()
        limit_hit = (
            rss > cfg["resources"]["aggregate_workload_ram_bytes"]
            or memory.available < cfg["resources"]["minimum_available_ram_bytes"]
            or time.monotonic() - started >= cfg["resources"]["training_wall_seconds"]
            or now >= datetime.fromisoformat(cfg["resources"]["absolute_stop"])
        )
        if limit_hit and not (root / "stop-request.json").exists():
            write_json(
                root / "stop-request.json",
                {
                    "time": now.isoformat(),
                    "workload_rss_bytes": rss,
                    "available_memory_bytes": memory.available,
                    "reason": "registered resource limit",
                },
            )
        while pending and len(active) < workers and not (root / "stop-request.json").exists():
            require(
                memory.available >= cfg["resources"]["minimum_available_ram_bytes"], "RAM floor"
            )
            arm, replica = pending.pop(0)
            name = f"{arm}-r{replica + 1}"
            stream = (logs / f"{name}.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_train-job",
                "--root",
                str(root),
                "--arm",
                arm,
                "--replica",
                str(replica),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            active[name] = (process, stream, command)
        for name, (process, stream, command) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del active[name]
            write_json(
                root / "training-state.json",
                {
                    "updated_at": datetime.now().astimezone().isoformat(),
                    "last_finished": name,
                    "exit_code": code,
                    "pending": [f"{arm}-r{replica + 1}" for arm, replica in pending],
                    "active": list(active),
                    "command": command,
                },
            )
            if code not in (0, 2):
                write_json(
                    root / "stop-request.json",
                    {
                        "time": datetime.now().astimezone().isoformat(),
                        "reason": f"worker {name} failed with {code}",
                    },
                )
        time.sleep(2)
    results = []
    for arm, replica in jobs:
        path = root / "training" / f"{arm}-r{replica + 1}/result.json"
        if path.exists():
            results.append(read_json(path))
    complete = len(results) == 6 and all(item["complete"] for item in results)
    write_json(
        root / "training-summary.json",
        {
            "complete": complete,
            "jobs": results,
            "ended_at": datetime.now().astimezone().isoformat(),
        },
    )
    return 0 if complete else 2


def artifact_paths(root):
    paths = {"reference": root / "reference.pt"}
    for replica in range(3):
        for arm in config()["arms"]:
            name = f"{arm}-r{replica + 1}"
            result = root / "training" / name / "result.json"
            require(
                result.exists() and read_json(result)["complete"], f"Incomplete artifact {name}"
            )
            paths[name] = root / "training" / name / "checkpoint.pt"
    return paths


def evaluation_job(root, artifact, suite_name, latency=False):
    cfg = config()
    checkpoint = artifact_paths(root)[artifact]
    before = sha256(checkpoint)
    if latency:
        setting = cfg["evaluation_suites"]["primary-classic-rule-based"]
        seeds = inclusive(cfg["latency"]["world_seed_range_inclusive"])
        output = root / "latency" / artifact
    else:
        setting = cfg["evaluation_suites"][suite_name]
        seeds = inclusive(setting["world_seed_range_inclusive"])
        output = root / "evaluation" / artifact / suite_name
    require(not output.exists(), f"Preserve existing evaluation output: {output}")
    output.mkdir(parents=True)
    rows = []
    started, cpu = time.monotonic(), time.process_time()
    for seed in seeds:
        row = play_episode(
            root,
            checkpoint,
            world_seed=seed,
            opponents=setting["opponents"],
            training=False,
            epsilon=0.0,
            n_step=1,
            scenario=setting["scenario"],
        )
        row.update({"artifact": artifact, "suite": suite_name})
        rows.append(row)
    require(sha256(checkpoint) == before, "Evaluation mutated checkpoint")
    write_rows(output / "episodes.json.gz", rows)
    result = {
        "artifact": artifact,
        "suite": suite_name,
        "stage": "latency" if latency else "evaluation",
        "episodes": len(rows),
        "checkpoint_sha256": before,
        "rows_sha256": sha256(output / "episodes.json.gz"),
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time() - cpu,
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2))


def evaluation_supervisor(root, workers):
    cfg = config()
    jobs = [
        (artifact, suite)
        for artifact in artifact_paths(root)
        for suite in cfg["evaluation_suites"]
        if not (root / "evaluation" / artifact / suite / "result.json").exists()
    ]
    active = {}
    logs = root / "evaluation-logs"
    logs.mkdir(exist_ok=True)
    while jobs or active:
        memory = psutil.virtual_memory()
        require(memory.available >= cfg["resources"]["minimum_available_ram_bytes"], "RAM floor")
        require(
            sum(process_rss(process) for process, _stream in active.values())
            <= cfg["resources"]["aggregate_workload_ram_bytes"],
            "Evaluation workload RAM ceiling",
        )
        while jobs and len(active) < workers:
            artifact, suite = jobs.pop(0)
            name = f"{artifact}--{suite}"
            stream = (logs / f"{name}.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_eval-job",
                "--root",
                str(root),
                "--artifact",
                artifact,
                "--suite",
                suite,
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            active[name] = (process, stream)
        for name, (process, stream) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del active[name]
            require(code == 0, f"Evaluation worker failed: {name}")
        time.sleep(2)
    write_json(
        root / "evaluation-summary.json",
        {"complete": True, "ended_at": datetime.now().astimezone().isoformat()},
    )


def latency(root):
    require((root / "evaluation-summary.json").exists(), "Evaluation must finish first")
    for artifact in artifact_paths(root):
        if (root / "latency" / artifact / "result.json").exists():
            continue
        evaluation_job(root, artifact, "primary-classic-rule-based", latency=True)
    write_json(
        root / "latency-summary.json",
        {"complete": True, "ended_at": datetime.now().astimezone().isoformat()},
    )


def status(root):
    progress = {}
    for replica in range(1, 4):
        for arm in config()["arms"]:
            name = f"{arm}-r{replica}"
            trail = root / "resume" / name / "episodes.json.gz"
            result = root / "training" / name / "result.json"
            progress[name] = {
                "durable_episodes": len(read_rows(trail)) if trail.exists() else 0,
                "result": read_json(result) if result.exists() else None,
            }
    report = {
        "time": datetime.now().astimezone().isoformat(),
        "available_memory_gib": psutil.virtual_memory().available / GIB,
        "stop_requested": (root / "stop-request.json").exists(),
        "progress": progress,
    }
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--reference", type=Path, required=True)
    prepare_parser.add_argument("--external", type=Path, required=True)
    for mode in ("smoke", "train", "evaluate", "latency", "status"):
        child = sub.add_parser(mode)
        child.add_argument("--root", type=Path, required=True)
        if mode in {"train", "evaluate"}:
            child.add_argument("--workers", type=int, default=3)
    job = sub.add_parser("_train-job")
    job.add_argument("--root", type=Path, required=True)
    job.add_argument("--arm", choices=("one-step", "five-step"), required=True)
    job.add_argument("--replica", type=int, choices=range(3), required=True)
    evaluation = sub.add_parser("_eval-job")
    evaluation.add_argument("--root", type=Path, required=True)
    evaluation.add_argument("--artifact", required=True)
    evaluation.add_argument("--suite", required=True)
    args = parser.parse_args()
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    if args.mode == "prepare":
        prepare(args.root, args.reference, args.external)
        return 0
    if args.mode == "smoke":
        smoke(args.root)
        return 0
    if args.mode == "train":
        return train_supervisor(args.root, args.workers)
    if args.mode == "evaluate":
        evaluation_supervisor(args.root, args.workers)
        return 0
    if args.mode == "latency":
        latency(args.root)
        return 0
    if args.mode == "status":
        status(args.root)
        return 0
    if args.mode == "_eval-job":
        evaluation_job(args.root, args.artifact, args.suite)
        return 0
    return training_job(args.root, args.arm, args.replica)


if __name__ == "__main__":
    raise SystemExit(main())
