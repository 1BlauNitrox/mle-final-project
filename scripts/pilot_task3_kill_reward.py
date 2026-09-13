"""Issue175 retention pilot, adapted from the tested #168 runner at dcff79f."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.task3_pilot_resources import (  # noqa: E402 - support direct CLI execution
    network_sha,
    sample_tree,
    sha,
    stop_owned,
)

INPUT = "99144d1688f66dcc6369d3efc02b2b9d5755cc8d21e6f2c1e1ffef466d0a7113"
SOURCE = "c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113"
CONFIG = ROOT / "experiments/2026-09-14-task3-elimination-reward/config.json"


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write(path, value):
    from scripts.task3_progress_retry import with_permission_retry

    audit = ROOT / "training_outputs/progress-write-retries.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    return with_permission_retry(_write, audit)(path, value)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def config():
    value = json.loads(CONFIG.read_text())
    if canonical_sha(value) != "facf56942df15b4fe86048d70df1d30a5046e71d0dcec8d313e8276740a08856":
        raise ValueError("Pilot registration changed")
    if (
        value["runtime_source"] != SOURCE
        or value["input_sha256"] != INPUT
        or value["training_episodes"] != 1000
        or value["evaluation_episodes"] != 3520
        or value["episodes_per_replica_arm"] != 100
        or value["replica_agent_seeds"] != [3975301, 3975302, 3975303, 3975304, 3975305]
        or value["initial_epsilon"] != 0.2
        or value["epsilon_decay"] != 1.0
    ):
        raise ValueError("Unregistered pilot configuration")
    return value


def episode_epsilon(arm, agent_seed, episode):
    if arm not in {"control", "treatment"} or not 0 <= episode < 100:
        raise ValueError("Unknown arm or episode")
    chosen = random.Random(agent_seed + 168000000 + episode // 5).randrange(5)
    return 1.0 if episode % 5 == chosen else 0.0


def zip_json(path, data):
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as file:
        file.write(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def read_json(path):
    return json.loads(path.read_text())


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "host": platform.node(),
        "processor": platform.processor(),
        "packages": {
            p: importlib.metadata.version(p) for p in ("numpy", "torch", "pygame", "psutil")
        },
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def code_hashes():
    names = [
        "scripts/pilot_task3_kill_reward.py",
        "scripts/task3_pilot_resources.py",
        "scripts/frozen_opponent_inputs.py",
        "scripts/task3_reward_configuration.py",
        "scripts/task3_progress_retry.py",
        "scripts/analyze_task3_kill_reward.py",
        "scripts/run_issue175_pilot.ps1",
        "scripts/audit_task3_kill_reward_seeds.py",
    ]
    return {
        name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in names
    }


def verify(root):
    binding = read_json(root / "binding.json")
    if canonical_sha(read_json(root / "config.json")) != canonical_sha(config()):
        raise ValueError("Prepared protocol copy changed")
    if binding["tool_hashes"] != code_hashes() or binding["config_sha256"] != canonical_sha(
        config()
    ):
        raise ValueError("Prepared tool/config source changed")
    if (
        sha(root / "reference.pt") != INPUT
        or sha(root / "initial.pt") != config()["initial_sha256"]
        or binding["initial_sha256"] != config()["initial_sha256"]
    ):
        raise ValueError("Prepared input changed")
    if any(
        sha(root / "source" / name) != digest
        for name, digest in read_json(root / "source-manifest.json").items()
    ):
        raise ValueError("Scientific runtime changed")
    for arm, record in read_json(root / "initialization-verification.json")["arms"].items():
        if sha(root / f"initial-{arm}.pt") != record["sha256"]:
            raise ValueError("Arm initialization changed")
    return binding


def prepare(root, parent, initial):
    cfg = config()
    if not read_json(CONFIG.with_name("seed-audit.json"))["passed"]:
        raise ValueError("Need passing registered seed audit")
    if sha(parent) != INPUT or sha(initial) != cfg["initial_sha256"]:
        raise ValueError("Wrong parent or fresh initialization")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit technical source before prepare")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent, root / "reference.pt")
    shutil.copyfile(initial, root / "initial.pt")
    shutil.copyfile(CONFIG, root / "config.json")
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={root / 'runtime-source.tar'}", SOURCE],
        cwd=ROOT,
        check=True,
    )
    (root / "source").mkdir()
    with tarfile.open(root / "runtime-source.tar") as archive:
        archive.extractall(root / "source", filter="data")
    write(
        root / "source-manifest.json",
        {
            p.relative_to(root / "source").as_posix(): sha(p)
            for p in (root / "source").rglob("*")
            if p.is_file()
        },
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_bind",
            "--root",
            str(root),
        ],
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    write(
        root / "binding.json",
        {
            "input_sha256": INPUT,
            "initial_sha256": sha(root / "initial.pt"),
            "runtime_source": SOURCE,
            "tool_source": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "tool_hashes": code_hashes(),
            "config_sha256": canonical_sha(cfg),
            "prepared_environment": environment(),
            "seed_audit_sha256": sha(CONFIG.with_name("seed-audit.json")),
        },
    )
    verify(root)
    print(
        json.dumps(
            {
                "prepared": str(root),
                "training_episodes": 1000,
                "evaluation_episodes": 3520,
                "scientific_execution_authorized": False,
            },
            indent=2,
        )
    )


def payload_equal(a, b):
    """Compare the entire restricted payload, including tensors and RNG/replay state."""
    import torch

    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(payload_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(payload_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def arm_payload(initial, arm):
    from copy import deepcopy

    result = deepcopy(initial)
    result["rewards"]["KILLED_OPPONENT"] = config()["kill_rewards"][arm]
    rate = config()["learning_rates"][arm]
    result["config"]["learning_rate"] = rate
    for group in result["learner_state"]["optimizer"]["param_groups"]:
        group["lr"] = rate
    return result


def bind(root):
    import torch

    torch.set_num_threads(1)
    initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    reference = torch.load(root / "reference.pt", map_location="cpu", weights_only=True)
    if (
        initial["completed_episodes"] != 0
        or initial["learner_state"]["update_steps"] != 0
        or initial["learner_state"]["optimizer"]["state"]
        or len(initial["replay_state"]["states"]) != 0
    ):
        raise ValueError("Need fresh initialization, empty optimizer and replay")
    expected = arm_payload(initial, "control")
    if not payload_equal(initial, expected):
        raise ValueError("Control LR does not match #168 initialization")
    for network in ("online_network", "target_network"):
        if not payload_equal(
            initial["learner_state"][network], reference["learner_state"][network]
        ):
            raise ValueError("Reference and initialization networks differ")
    report = {
        "initial_sha256": sha(root / "initial.pt"),
        "reference_sha256": sha(root / "reference.pt"),
        "only_changed_fields": ["rewards.KILLED_OPPONENT"],
        "training_restriction": config()["freeze_modes"],
        "identical_online_and_target_weights": True,
        "arms": {},
    }
    for arm in config()["arms"]:
        payload = arm_payload(initial, arm)
        path = root / f"initial-{arm}.pt"
        if arm == "control":
            shutil.copyfile(root / "initial.pt", path)
        else:
            torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not payload_equal(loaded, payload):
            raise ValueError("Arm initialization payload mismatch")
        report["arms"][arm] = {
            "sha256": sha(path),
            "learning_rate": payload["config"]["learning_rate"],
        }
    write(root / "initialization-verification.json", report)


def play(
    root,
    checkpoint,
    world_seed,
    agent_seed,
    scenario,
    opponents,
    training,
    epsilon,
    target,
    freeze=False,
    kill_reward=5.0,
):
    from unittest.mock import patch

    sys.path.insert(0, str(root / "source"))
    os.chdir(root / "source")
    from training.seeded_framework import configure_diagnostic_logging, wrap_process_event

    from agent_code.DagobertDuckDQNTask3 import callbacks, features, train
    from agent_code.DagobertDuckDQNTask3 import config as agent_config
    from agents import AgentRunner
    from environment import BombeRLeWorld, WorldArgs
    from scripts.task3_reward_configuration import configured_rewards

    configure_diagnostic_logging()
    actions = []
    selector = callbacks.select_action

    def behavior(**kwargs):
        action = selector(**{**kwargs, "epsilon": epsilon})
        actions.append(action)
        return action

    args = WorldArgs(
        True,
        30,
        False,
        0,
        False,
        None,
        False,
        not training,
        str(target),
        str(target / "native.json"),
        "issue175-retention-pilot",
        world_seed,
        False,
        scenario,
    )
    target.mkdir(parents=True, exist_ok=False)
    with (
        configured_rewards(agent_config.REWARDS, kill_reward),
        patch.dict(
            os.environ,
            {
                "BOMBERMAN_AGENT_SEED": str(agent_seed),
                "BOMBERMAN_DQN_ACTION_MASKING": "framework_legal",
                "BOMBERMAN_DQN_ESCAPE_CONTINUATIONS": "on",
            },
        ),
        patch.object(callbacks, "CHECKPOINT_PATH", checkpoint),
        patch.object(train, "CHECKPOINT_PATH", checkpoint),
        patch.object(callbacks, "_evaluation_checkpoint_path", lambda: checkpoint),
        patch.object(callbacks, "select_action", behavior),
        patch.object(
            AgentRunner,
            "process_event",
            wrap_process_event(
                AgentRunner.process_event,
                world_seed + 1000000,
                {name: i + 1 for i, name in enumerate(opponents)},
            ),
        ),
    ):
        world = BombeRLeWorld(
            args, [("DagobertDuckDQNTask3", training), *[(name, False) for name in opponents]]
        )
        learner = world.agents[0]
        policy = learner.backend.runner.fake_self
        before = network_sha(policy.policy_network)
        if training and any(
            g["lr"] != policy.config.learning_rate for g in policy.learner.optimizer.param_groups
        ):
            raise ValueError("Loaded optimizer/config learning rate mismatch")
        guard = None
        if freeze:
            if not training:
                raise ValueError("Freeze restriction is training-only")
            import torch

            from scripts.frozen_opponent_inputs import FrozenOpponentInputs

            anchor = torch.load(root / "initial.pt", weights_only=True, map_location="cpu")
            guard = FrozenOpponentInputs(policy.learner, anchor["learner_state"]["online_network"])
        steps, attacks = 0, 0
        world.new_round()
        while world.running:
            alive = not learner.dead
            world.do_step()
            if alive:
                steps += 1
                if training:
                    attacks += bool(features.state_to_features(learner.last_game_state)[30])
        world.end()
        if guard is not None:
            guard.validate()
        after = network_sha(policy.policy_network)
        if not training and before != after:
            raise ValueError("Evaluation mutated network")
        native = world.round_statistics[world.round_id]
        if native["agents"][learner.name]["survival_steps"] != steps:
            raise ValueError("Native/instrumented survival mismatch")
        return {
            "world_seed": world_seed,
            "agent_seed": agent_seed,
            "behavior_epsilon": epsilon,
            "training": training,
            "native": native,
            "attack_steps": attacks,
            "freeze_mode": "opponent_columns_only" if freeze else "none",
            "kill_reward": kill_reward,
            "greedy_actions_sha256": canonical_sha(actions),
            "online_before_sha256": before,
            "online_after_sha256": after,
            "completed_episodes": policy.completed_episodes,
            "optimizer_updates": policy.learner.update_steps if training else None,
        }


def train_job(root, output, arm, replica):
    cfg = config()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoint.pt"
    shutil.copyfile(root / f"initial-{arm}.pt", checkpoint)
    rows = []
    started, cpu = time.monotonic(), time.process_time()
    seed = cfg["replica_agent_seeds"][replica]
    for index, world_seed in enumerate(cfg["training_world_seeds"][replica]):
        row = play(
            root,
            checkpoint,
            world_seed,
            seed,
            "classic",
            ["peaceful_agent"],
            True,
            episode_epsilon(arm, seed, index),
            output / f"episode-{index + 1:02d}",
            freeze=True,
            kill_reward=cfg["kill_rewards"][arm],
        )
        if row["completed_episodes"] != index + 1:
            raise ValueError("Checkpoint episode counter mismatch")
        shutil.copyfile(checkpoint, output / f"checkpoint-{index + 1:02d}.pt")
        rows.append(row)
        zip_json(output / "episodes.json.gz", rows)
    write(
        output / "result.json",
        {
            "arm": arm,
            "replica": replica,
            "episodes": len(rows),
            "checkpoint_sha256": sha(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "optimizer_updates": rows[-1]["optimizer_updates"],
            "initial_online_sha256": rows[0]["online_before_sha256"],
            "final_online_sha256": rows[-1]["online_after_sha256"],
            "episodes_sha256": sha(output / "episodes.json.gz"),
            "environment": environment(),
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.monotonic() - started,
        },
    )


def artifacts(root):
    state = read_json(root / "training-state.json")
    expected = {f"{arm}-r{r}" for arm in config()["arms"] for r in range(1, 6)}
    if state["status"] != "completed" or set(state["completed"]) != expected:
        raise ValueError("Training incomplete")
    result = {"reference": root / "reference.pt"}
    for key, relative in state["completed"].items():
        directory = root / relative
        metadata = read_json(directory / "result.json")
        if (
            key != f"{metadata['arm']}-r{metadata['replica'] + 1}"
            or metadata["episodes"] != 100
            or sha(directory / "checkpoint.pt") != metadata["checkpoint_sha256"]
        ):
            raise ValueError("Training artifact mismatch")
        result[key] = directory / "checkpoint.pt"
    return result


def evaluate_job(root, output, artifact, suite):
    cfg = config()
    checkpoint = artifacts(root)[artifact]
    original_sha = sha(checkpoint)
    setting = cfg["evaluation_suites"][suite]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started, cpu = time.monotonic(), time.process_time()
    for repeat in range(2):
        for seed in setting["world_seeds"]:
            row = play(
                root,
                checkpoint,
                seed,
                seed + 1000000,
                setting["scenario"],
                setting["opponents"],
                False,
                0.0,
                output / f"{seed}-repeat{repeat}",
                kill_reward=cfg["kill_rewards"][
                    "treatment" if artifact.startswith("treatment") else "control"
                ],
            )
            rows.append({**row, "artifact": artifact, "suite": suite, "repeat": repeat})
            zip_json(output / "episodes.json.gz", rows)
    if sha(checkpoint) != original_sha:
        raise ValueError("Evaluation mutated checkpoint")
    write(
        output / "result.json",
        {
            "artifact": artifact,
            "suite": suite,
            "episodes": len(rows),
            "checkpoint_sha256": original_sha,
            "episodes_sha256": sha(output / "episodes.json.gz"),
            "environment": environment(),
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.monotonic() - started,
        },
    )


def supervise(root, stage, resume=False):
    import psutil

    verify(root)
    cfg = config()
    if os.environ.get("TASK175_PILOT_AUTHORIZED") != "yes":
        raise ValueError(
            "Explicit owner authorization required; "
            "set TASK175_PILOT_AUTHORIZED=yes only after approval"
        )
    if (
        psutil.virtual_memory().available < cfg["minimum_free_memory_bytes"]
        or shutil.disk_usage(root).free < cfg["minimum_free_disk_bytes"]
    ):
        raise ValueError("Need 3 GiB available RAM and 5 GiB free disk")
    stage_environment = {
        k: v for k, v in environment().items() if k not in {"omp_threads", "mkl_threads"}
    }
    path = root / f"{stage}-state.json"
    if path.exists():
        if not resume:
            raise ValueError("Existing stage; inspect state and use --resume explicitly")
        state = read_json(path)
        if state.get("environment") != stage_environment:
            raise ValueError("Stage environment changed; do not move a partial stage between hosts")
        if state["status"] == "completed":
            raise ValueError("Stage already complete")
        active = state.get("active")
        if active:
            try:
                process = psutil.Process(active["pid"])
                if process.create_time() == active["created"]:
                    raise ValueError("Owned worker is still active")
            except psutil.NoSuchProcess:
                pass
            raise ValueError(
                "Abrupt supervisor exit: resource accounting needs audit before resume"
            )
    else:
        state = {
            "status": "running",
            "environment": stage_environment,
            "completed": {},
            "attempts": [],
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
            "peak_memory_bytes": 0,
            "active": None,
        }
    if stage == "training":
        jobs = [
            (f"{arm}-r{replica + 1}", ["_train", "--arm", arm, "--replica", str(replica)])
            for replica in range(5)
            for arm in cfg["arms"]
        ]
    else:
        jobs = [
            (f"{artifact}-{suite}", ["_evaluate", "--artifact", artifact, "--suite", suite])
            for artifact in artifacts(root)
            for suite in cfg["evaluation_suites"]
        ]
    limits = cfg[f"{stage}_limits"]
    if (
        state["cpu_seconds"] >= limits["cpu_seconds"]
        or state["wall_seconds"] >= limits["wall_seconds"]
    ):
        raise ValueError("Existing allocation already exhausted")
    lock = root / f"{stage}.lock"
    if lock.exists():
        owner = read_json(lock)
        try:
            existing = psutil.Process(owner["pid"])
            if existing.create_time() == owner["created"]:
                raise ValueError("Another supervisor owns this stage")
        except psutil.NoSuchProcess:
            pass
        if not resume:
            raise ValueError("Stale lock requires explicit resume")
        lock.rename(root / f"{stage}-stale-lock-{time.time_ns()}.json")
    with lock.open("x") as file:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, file)
    cpu_base, wall_base = state["cpu_seconds"], state["wall_seconds"]
    started = time.monotonic()
    state["status"] = "running"
    try:
        for key, arguments in jobs:
            if key in state["completed"]:
                continue
            attempt = f"{stage}/{key}/attempt-{len(state['attempts']) + 1:03d}"
            target = root / attempt
            target.parent.mkdir(parents=True, exist_ok=True)
            record = {"job": key, "output": attempt, "status": "running"}
            state["attempts"].append(record)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *arguments,
                "--root",
                str(root),
                "--output",
                str(target),
            ]
            ledger, sampled = {}, 0.0
            with (target.parent / (target.name + ".log")).open("w") as log:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "BOMBERMAN_COMPACT_LOGS": "1",
                    },
                )
                process = psutil.Process(child.pid)
                state["active"] = {"pid": child.pid, "created": process.create_time()}
                write(path, state)
                try:
                    while child.poll() is None:
                        try:
                            sampled, memory = sample_tree(process, ledger)
                        except psutil.NoSuchProcess:
                            break
                        state["cpu_seconds"] = cpu_base + sampled
                        state["wall_seconds"] = wall_base + time.monotonic() - started
                        state["peak_memory_bytes"] = max(state["peak_memory_bytes"], memory)
                        write(path, state)
                        if (
                            state["cpu_seconds"] >= limits["cpu_seconds"]
                            or state["wall_seconds"] >= limits["wall_seconds"]
                            or memory > limits["memory_bytes"]
                        ):
                            raise RuntimeError("Registered allocation exhausted")
                        time.sleep(0.1)
                    if child.wait() != 0:
                        raise RuntimeError("Pilot worker failed: " + key)
                    result = read_json(target / "result.json")
                    cpu_base += max(sampled, result["cpu_seconds"])
                    record["status"] = "completed"
                    state["completed"][key] = attempt
                except BaseException:
                    if child.poll() is None:
                        stop_owned(child)
                    cpu_base += sampled
                    record["status"] = "failed"
                    raise
                finally:
                    state["active"] = None
                    state["cpu_seconds"] = cpu_base
                    state["wall_seconds"] = wall_base + time.monotonic() - started
                    write(path, state)
        state["status"] = "completed"
        verify(root)
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        write(path, state)
        lock.unlink()
    print(
        json.dumps(
            {"stage": stage, "status": state["status"], "completed_jobs": len(state["completed"])},
            indent=2,
        )
    )


def bundle(root, output, results=False):
    verify(root)
    artifacts(root)
    names = [
        "binding.json",
        "config.json",
        "source-manifest.json",
        "runtime-source.tar",
        "reference.pt",
        "initial.pt",
        "initial-control.pt",
        "initial-treatment.pt",
        "initialization-verification.json",
        "training-state.json",
    ]
    paths = [(root / name, name) for name in names]
    paths.extend((ROOT / name, "tools/" + name) for name in code_hashes())
    paths.append((CONFIG.with_name("seed-audit.json"), "seed-audit.json"))
    for path in (root / "training").rglob("*"):
        if path.is_file() and (
            (path.suffix == ".json" or path.name == "checkpoint.pt")
            or path.name.endswith(".json.gz")
        ):
            paths.append((path, path.relative_to(root).as_posix()))
    if results:
        from scripts.analyze_task3_kill_reward import analyze

        write(root / "analysis.json", analyze(root))
        paths.extend((root / name, name) for name in ("evaluation-state.json", "analysis.json"))
        for path in (root / "evaluation").rglob("*"):
            if path.is_file() and (path.suffix == ".json" or path.name.endswith(".json.gz")):
                paths.append((path, path.relative_to(root).as_posix()))
    with tarfile.open(output, "x:gz") as archive:
        for path, name in paths:
            archive.add(path, arcname=name, recursive=False)
    write(
        output.with_suffix(output.suffix + ".manifest.json"),
        {
            "sha256": sha(output),
            "size_bytes": output.stat().st_size,
            "files": len(paths),
            "tool_source": read_json(root / "binding.json")["tool_source"],
        },
    )
    print(
        json.dumps(
            {"bundle": str(output), "sha256": sha(output), "size_bytes": output.stat().st_size},
            indent=2,
        )
    )


def import_bundle(root, path, digest):
    if sha(path) != digest:
        raise ValueError("Transfer checksum mismatch")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(path) as archive:
        archive.extractall(root, filter="data")
    (root / "source").mkdir()
    with tarfile.open(root / "runtime-source.tar") as archive:
        archive.extractall(root / "source", filter="data")
    verify(root)
    artifacts(root)


def smoke_worker(root, output):
    verify(root)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "smoke-only.pt"
    shutil.copyfile(root / "initial-treatment.pt", checkpoint)
    training = []
    for episode in range(2):
        training.append(
            play(
                root,
                checkpoint,
                config()["smoke"]["world_seeds"][episode],
                config()["smoke"]["agent_seed"],
                "classic",
                ["peaceful_agent"],
                True,
                0.0,
                output / f"training-{episode}",
                freeze=True,
                kill_reward=config()["kill_rewards"]["treatment"],
            )
        )
    if training[-1]["completed_episodes"] != 2 or training[-1]["optimizer_updates"] <= 0:
        raise ValueError("Smoke did not exercise genuine checkpointed learning")
    evaluation = []
    for index, (suite, setting) in enumerate(config()["evaluation_suites"].items()):
        evaluation.append(
            play(
                root,
                checkpoint,
                config()["smoke"]["world_seeds"][index + 2],
                config()["smoke"]["agent_seed"],
                setting["scenario"],
                setting["opponents"],
                False,
                0.0,
                output / suite,
                kill_reward=config()["kill_rewards"]["treatment"],
            )
        )
    from scripts.analyze_task3_kill_reward import behavioral

    for index, (suite, setting) in enumerate(config()["evaluation_suites"].items()):
        if setting["opponents"]:
            continue
        reference = play(
            root,
            root / "reference.pt",
            config()["smoke"]["world_seeds"][index + 2],
            config()["smoke"]["agent_seed"],
            setting["scenario"],
            [],
            False,
            0.0,
            output / (suite + "-reference"),
        )
        if behavioral(reference["native"]) != behavioral(evaluation[index]["native"]):
            raise ValueError("Opponent-free smoke behavior changed")
        if reference["greedy_actions_sha256"] != evaluation[index]["greedy_actions_sha256"]:
            raise ValueError("Opponent-free smoke actions changed")
    verify(root)
    zip_json(
        output / "smoke-observations.json.gz", {"training": training, "evaluation": evaluation}
    )
    write(
        output / "result.json",
        {
            "scope": "mechanical_smoke_excluded_from_pilot",
            "training_episodes": 2,
            "evaluation_episodes": 7,
            "optimizer_updates": training[-1]["optimizer_updates"],
            "checkpoint_sha256": sha(checkpoint),
        },
    )


def smoke(root, output):
    import psutil

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_smoke",
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    log_path = output.with_suffix(".log")
    started = time.monotonic()
    ledger = {}
    cpu = peak = 0
    with log_path.open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "BOMBERMAN_COMPACT_LOGS": "1",
            },
        )
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                try:
                    cpu, memory = sample_tree(process, ledger)
                except psutil.NoSuchProcess:
                    break
                peak = max(peak, memory)
                if time.monotonic() - started > 120 or cpu > 120 or memory > 1073741824:
                    raise RuntimeError("Mechanical smoke allocation exhausted")
                time.sleep(0.1)
            if child.wait() != 0:
                raise RuntimeError("Mechanical smoke failed; inspect retained log")
        except BaseException:
            if child.poll() is None:
                stop_owned(child)
            raise
        finally:
            write(
                output.with_suffix(".resources.json"),
                {
                    "cpu_seconds": cpu,
                    "wall_seconds": time.monotonic() - started,
                    "peak_memory_bytes": peak,
                },
            )
    print(json.dumps(read_json(output / "result.json"), indent=2))


def chain(root):
    """Complete the authorized PC allocation; stop only owned children on deadline."""
    verify(root)
    path = root / "chain-state.json"
    if path.exists():
        raise ValueError("Existing chain: inspect retained state; no automatic retries")
    if os.environ.get("TASK175_PILOT_AUTHORIZED") != "yes":
        raise ValueError("Explicit authorization required")
    started = time.monotonic()
    state = {
        "status": "running",
        "stage": None,
        "started_unix": time.time(),
        "deadline_unix": time.time()
        + (config()["machine_allocation"]["overall_wall_seconds"] - 1800),
        "completed": [],
    }
    write(path, state)
    try:
        for mode in ("train", "evaluate", "results"):
            state["stage"] = mode
            write(path, state)
            command = [sys.executable, str(Path(__file__).resolve()), mode, "--root", str(root)]
            if mode == "results":
                command += ["--output", str(root / "issue175-pilot-evidence.tar.gz")]
            with (root / f"chain-{mode}.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
                state["active_pid"] = child.pid
                write(path, state)
                try:
                    while child.poll() is None:
                        if time.monotonic() - started >= (
                            config()["machine_allocation"]["overall_wall_seconds"] - 1800
                        ):
                            raise RuntimeError("Seven-hour chain allocation exhausted")
                        time.sleep(0.2)
                    if child.wait() != 0:
                        raise RuntimeError(
                            f"{mode} failed; inspect retained stage log and attempts"
                        )
                except BaseException:
                    if child.poll() is None:
                        stop_owned(child)
                    raise
            state["completed"].append(mode)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["active_pid"] = None
        state["wall_seconds"] = time.monotonic() - started
        write(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "chain",
            "prepare",
            "dry-run",
            "train",
            "evaluate",
            "bundle",
            "results",
            "analyze",
            "import",
            "smoke",
            "_smoke",
            "_bind",
            "_train",
            "_evaluate",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--arm")
    parser.add_argument("--replica", type=int)
    parser.add_argument("--artifact")
    parser.add_argument("--suite")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "chain":
        chain(root)
    elif args.mode == "prepare":
        prepare(root, args.parent, args.initial)
    elif args.mode == "dry-run":
        verify(root)
        print(
            json.dumps(
                {
                    "training_jobs": 10,
                    "training_episodes": 1000,
                    "evaluation_jobs": 44,
                    "evaluation_episodes": 3520,
                    "budgets": {k: v for k, v in config().items() if "limits" in k},
                },
                indent=2,
            )
        )
    elif args.mode in {"train", "evaluate"}:
        supervise(root, "training" if args.mode == "train" else "evaluation", args.resume)
    elif args.mode in {"bundle", "results"}:
        bundle(root, args.output.resolve(), args.mode == "results")
    elif args.mode == "analyze":
        from scripts.analyze_task3_kill_reward import analyze

        write(args.output, analyze(root))
    elif args.mode == "import":
        import_bundle(root, args.archive, args.sha256)
    elif args.mode == "smoke":
        smoke(root, args.output.resolve())
    elif args.mode == "_smoke":
        smoke_worker(root, args.output.resolve())
    elif args.mode == "_bind":
        bind(root)
    elif args.mode == "_train":
        train_job(root, args.output.resolve(), args.arm, args.replica)
    else:
        evaluate_job(root, args.output.resolve(), args.artifact, args.suite)


if __name__ == "__main__":
    main()
