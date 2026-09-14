"""Source-pinned, immutable-weight exploration exposure probe for issue #168."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "experiments/2026-09-13-task3-exploration-screen/config.json"
INPUT = "99144d1688f66dcc6369d3efc02b2b9d5755cc8d21e6f2c1e1ffef466d0a7113"
SOURCE = "c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113"


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate(config):
    if (
        config["runtime_source"] != SOURCE
        or config["input_sha256"] != INPUT
        or config["epsilons"] != [1.0, 0.2, 0.0]
        or config["world_seeds"] != list(range(1631101, 1631121))
        or config["agent_seed_offset"] != 1000000
        or config["training_episodes"] != 0
        or config["episodes"] != 60
        or config["scenario"] != "classic"
        or config["opponents"] != ["peaceful_agent"]
        or config["maximum_steps"] != 400
    ):
        raise ValueError("Unregistered source/input/matrix")
    if config["readiness_filter"] != {
        "mean_survival_ratio_min": 2.0,
        "strictly_longer_pairs_min": 14,
        "coin_difference_min": 0.0,
        "attack_episode_count_min": 2,
    }:
        raise ValueError("Unregistered readiness filter")
    if config["limits"] != {
        "wall_seconds": 900,
        "cpu_seconds": 900,
        "memory_bytes": 1073741824,
        "episode_seconds": 120,
        "minimum_free_memory_bytes": 3221225472,
    }:
        raise ValueError("Unregistered limits")


def decide(jobs):
    expected = {(seed, epsilon) for seed in range(1631101, 1631121) for epsilon in (1.0, 0.2, 0.0)}
    actual = {(j["world_seed"], j["epsilon"]) for j in jobs}
    if actual != expected or len(jobs) != len(expected):
        raise ValueError("Incomplete/duplicate matrix")
    if any(j["status"] != "completed" or not j["weights_unchanged"] for j in jobs):
        raise ValueError("Failed/mutable probe")
    groups = {
        eps: sorted([j for j in jobs if j["epsilon"] == eps], key=lambda j: j["world_seed"])
        for eps in (1.0, 0.2, 0.0)
    }
    control, low = groups[1.0], groups[0.2]
    survival = {eps: mean(j["survival_steps"] for j in rows) for eps, rows in groups.items()}
    longer = sum(
        b["survival_steps"] > a["survival_steps"] for a, b in zip(control, low, strict=True)
    )
    coin_diff = mean(j["coins"] for j in low) - mean(j["coins"] for j in control)
    attack_episodes = sum(j["attack_steps"] > 0 for j in low)
    gates = {
        "survival_ratio": survival[0.2] >= 2 * survival[1.0],
        "paired_survival": longer >= 14,
        "coins": coin_diff >= 0,
        "attack_exposure": attack_episodes >= 2,
    }
    return {
        "scope": "mechanism_only_not_task3_success",
        "phase_b_ready": all(gates.values()),
        "gates": gates,
        "mean_survival_steps": survival,
        "longer_pairs": longer,
        "mean_coin_difference": coin_diff,
        "low_epsilon_attack_episodes": attack_episodes,
        "selected_checkpoint": None,
        "long_training_authorized": False,
    }


def observe(features, state, events, action, legal_mask):
    if len(features) != 39:
        raise ValueError("Wrong feature schema")
    pos = state["self"][3]
    others = [o[3] for o in state["others"]]
    safe = bool(features[31])
    dropped = "BOMB_DROPPED" in events
    return {
        "step": int(state["step"]),
        "action": action,
        "events": list(events),
        "opponent_distance": min(
            (int(abs(pos[0] - p[0]) + abs(pos[1] - p[1])) for p in others), default=None
        ),
        "attack": bool(features[30]),
        "safe_attack": safe,
        "available_safe_attack": safe and bool(state["self"][2]) and bool(legal_mask[5]),
        "safe_cratefree_placement": bool(safe and features[19] == 0 and dropped),
        "wasteful_safe_attack_penalty": -0.5 if safe and features[19] == 0 and dropped else 0.0,
    }


def network_sha(network):
    result = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        result.update(name.encode() + tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def worker(source, checkpoint, seed, epsilon, output, phase="A"):
    # Runtime imports resolve only to the exact archived scientific source.
    sys.path.insert(0, str(source.resolve()))
    from unittest.mock import patch

    import psutil
    from training.seeded_framework import configure_diagnostic_logging, wrap_process_event

    from agent_code.DagobertDuckDQNTask3 import callbacks, features, legality
    from agents import AgentRunner
    from environment import BombeRLeWorld, WorldArgs

    allowed = range(1631101, 1631121) if phase == "A" else range(1682101, 1682121)
    if phase not in {"A", "C"} or seed not in allowed or epsilon not in (1.0, 0.2, 0.0):
        raise ValueError("Unregistered worker condition")
    if sha(checkpoint) != INPUT:
        raise ValueError("Wrong input checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    original_selector = callbacks.select_action

    def behavior_selector(**kwargs):
        return original_selector(**{**kwargs, "epsilon": epsilon})

    started, cpu_started = time.monotonic(), time.process_time()
    process = psutil.Process()
    configure_diagnostic_logging()
    args = WorldArgs(
        True,
        30,
        False,
        0,
        False,
        None,
        False,
        True,
        str(output),
        str(output / "framework_stats.json"),
        "issue168-probe",
        seed,
        False,
        "classic",
    )
    rows, peak = [], 0
    with (
        patch.dict(os.environ, {"BOMBERMAN_AGENT_SEED": str(seed + 1000000)}),
        patch.object(callbacks, "_evaluation_checkpoint_path", lambda: checkpoint),
        patch.object(callbacks, "select_action", behavior_selector),
        patch.object(
            AgentRunner,
            "process_event",
            wrap_process_event(AgentRunner.process_event, seed + 1000000, {"peaceful_agent": 1}),
        ),
    ):
        world = BombeRLeWorld(args, [("DagobertDuckDQNTask3", False), ("peaceful_agent", False)])
        learner = world.agents[0]
        policy = learner.backend.runner.fake_self
        if (
            policy.train
            or policy.completed_episodes != 0
            or not policy.config.action_masking
            or not policy.config.escape_continuation_features
            or policy.config.double_dqn
            or policy.config.neutral_safe_attack_bombs
        ):
            raise ValueError("Wrong parent/modes")
        before = network_sha(policy.policy_network)
        world.new_round()
        while world.running:
            peak = max(peak, process.memory_info().rss)
            if time.monotonic() - started > 120 or peak > 1073741824 or world.step >= 400:
                raise RuntimeError("Probe episode resource/step limit")
            alive = not learner.dead
            world.do_step()
            if alive:
                state = learner.last_game_state
                f = features.state_to_features(state, include_continuation_features=True)
                rows.append(
                    observe(
                        f,
                        state,
                        learner.events,
                        world.replay["actions"][learner.name][-1],
                        legality.framework_legal_action_mask(state),
                    )
                )
        world.end()
        unchanged = before == network_sha(policy.policy_network) and sha(checkpoint) == INPUT
        if not unchanged:
            raise ValueError("Probe changed weights")
    native = world.round_statistics[world.round_id]["agents"][learner.name]
    if native["survival_steps"] != len(rows):
        raise ValueError("Instrumented and native survival steps disagree")
    with gzip.GzipFile(filename=str(output / "observations.json.gz"), mode="wb", mtime=0) as file:
        file.write(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
    result = {
        "world_seed": seed,
        "agent_seed": seed + 1000000,
        "epsilon": epsilon,
        "status": "completed",
        "weights_unchanged": unchanged,
        "network_sha256": before,
        "survival_steps": len(rows),
        "coins": native["coins"],
        "crates": native["crates_destroyed"],
        "score": learner.score,
        "eliminations": native["kills"],
        "self_kills": native["self_kills"],
        "survived": not learner.dead,
        "invalid_actions": native["invalid"],
        "attack_steps": sum(r["attack"] for r in rows),
        "safe_attack_steps": sum(r["safe_attack"] for r in rows),
        "available_safe_attack_steps": sum(r["available_safe_attack"] for r in rows),
        "safe_cratefree_placements": sum(r["safe_cratefree_placement"] for r in rows),
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_memory_bytes": peak,
        "observations_sha256": sha(output / "observations.json.gz"),
        "framework_stats_sha256": sha(output / "framework_stats.json"),
    }
    write(output / "result.json", result)


def sample_tree(process, cpu_by_identity):
    import psutil

    memory = 0
    for p in [process, *process.children(recursive=True)]:
        try:
            t = p.cpu_times()
            key = (p.pid, p.create_time())
            cpu_by_identity[key] = max(cpu_by_identity.get(key, 0), t.user + t.system)
            memory += p.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return sum(cpu_by_identity.values()), memory


def stop_owned(child):
    import psutil

    try:
        parent = psutil.Process(child.pid)
        for p in parent.children(recursive=True):
            p.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    child.wait()


def prior_cost(previous, config_path=CONFIG):
    if previous is None:
        return 0.0, 0.0, None
    data = json.loads(previous.read_text(encoding="utf-8"))
    if data["status"] != "failed" or data["config_sha256"] != sha(config_path):
        raise ValueError("Previous attempt must be failed and share registration")
    return (
        data["cpu_seconds"],
        data["wall_seconds"],
        {"path": str(previous.resolve()), "sha256": sha(previous)},
    )


def run(checkpoint, output, previous=None, phase="A"):
    import psutil

    config_path = CONFIG if phase == "A" else CONFIG.with_name("phase-c-config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if phase == "A":
        validate(config)
        decision_function = decide
    elif phase == "C":
        from scripts.task3_episode_exploration import decide as decide_c
        from scripts.task3_episode_exploration import validate as validate_c

        validate_c(config)
        decision_function = decide_c
    else:
        raise ValueError("Unregistered phase")
    previous_cpu, previous_wall, previous_record = prior_cost(previous, config_path)
    if sha(checkpoint) != INPUT or checkpoint.stat().st_size != config["input_size_bytes"]:
        raise ValueError("Wrong input")
    if psutil.virtual_memory().available < config["limits"]["minimum_free_memory_bytes"]:
        raise ValueError("Need 3 GiB available system memory")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit technical source/config first; keep human draft outside checkout")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    bundle = output / "runtime-source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={bundle}", SOURCE], cwd=ROOT, check=True
    )
    with tarfile.open(bundle) as file:
        file.extractall(source, filter="data")
    source_hashes = {
        p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*") if p.is_file()
    }
    write(output / "source-manifest.json", source_hashes)
    write(output / "registration.json", config)
    report = {
        "jobs": [],
        "status": "running",
        "tool_sha256": sha(Path(__file__)),
        "tool_source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha(config_path),
        "phase": phase,
        "cpu_seconds": previous_cpu,
        "previous_attempt": previous_record,
        "peak_memory_bytes": 0,
    }
    started = time.monotonic()
    try:
        for seed in config["world_seeds"]:
            epsilons = (
                config["epsilons"]
                if phase == "A"
                else [0.2, 1.0 if seed in config["random_episode_world_seeds"] else 0.0]
            )
            for epsilon in epsilons:
                label = f"{seed}-epsilon{epsilon}"
                target = output / label
                environment = {
                    **os.environ,
                    "BOMBERMAN_COMPACT_LOGS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--phase",
                    phase,
                    "--source",
                    str(source),
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--seed",
                    str(seed),
                    "--epsilon",
                    str(epsilon),
                    "--output",
                    str(target),
                ]
                cpu_by_identity, cpu = {}, 0.0
                job_started = time.monotonic()
                with (output / (label + ".log")).open("w", encoding="utf-8") as log:
                    child = subprocess.Popen(
                        command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        process = psutil.Process(child.pid)
                        while child.poll() is None:
                            try:
                                cpu, memory = sample_tree(process, cpu_by_identity)
                            except psutil.NoSuchProcess:
                                break
                            report["peak_memory_bytes"] = max(report["peak_memory_bytes"], memory)
                            if (
                                time.monotonic() - started + previous_wall > 900
                                or time.monotonic() - job_started > 120
                                or report["cpu_seconds"] + cpu > 900
                                or memory > 1073741824
                            ):
                                raise RuntimeError("Probe allocation limit")
                            time.sleep(0.05)
                        if child.wait() != 0:
                            raise RuntimeError("Probe failed: " + label)
                    except BaseException:
                        if child.poll() is None:
                            stop_owned(child)
                        report["cpu_seconds"] += cpu
                        report["jobs"].append(
                            {
                                "world_seed": seed,
                                "epsilon": epsilon,
                                "status": "failed",
                                "output": label,
                            }
                        )
                        raise
                job = json.loads((target / "result.json").read_text(encoding="utf-8"))
                report["cpu_seconds"] += max(cpu, job["cpu_seconds"])
                report["jobs"].append({**job, "output": label})
                report["wall_seconds"] = time.monotonic() - started + previous_wall
                write(output / "status.json", report)
        if any(sha(source / name) != value for name, value in source_hashes.items()):
            raise ValueError("Runtime source changed")
        if sha(checkpoint) != INPUT:
            raise ValueError("Original checkpoint changed")
        report.update(status="completed", decision=decision_function(report["jobs"]))
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started + previous_wall
        write(output / "status.json", report)
    print(json.dumps(report["decision"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "worker"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--previous-status", type=Path)
    parser.add_argument("--phase", choices=["A", "C"], default="A")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epsilon", type=float)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.checkpoint, args.output, args.previous_status, args.phase)
    else:
        worker(args.source, args.checkpoint, args.seed, args.epsilon, args.output, args.phase)


if __name__ == "__main__":
    main()
