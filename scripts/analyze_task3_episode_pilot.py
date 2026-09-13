"""Selection-free phase-D learning-pilot analysis from retained native observations."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from statistics import mean

from scripts.pilot_task3_episode_exploration import artifacts, config, read_json, sha, verify


def metrics(row):
    agents = row["native"]["agents"]
    own = agents["DagobertDuckDQNTask3"]
    others = [value["score"] for key, value in agents.items() if key != "DagobertDuckDQNTask3"]
    denominator = own["initially_available_coins"]
    if denominator <= 0:
        raise ValueError("Missing collection denominator")
    return {
        "score": own["score"],
        "strict_win": int(bool(others) and own["score"] > max(others)),
        "collection_fraction": own["coins"] / denominator,
        "coins": own["coins"],
        "crates": own["crates_destroyed"],
        "eliminations": own["kills"],
        "survived": int(own["survived"]),
        "survival_steps": own["survival_steps"],
        "self_kills": own["self_kills"],
        "invalid_actions": own["invalid"],
        "bomb_actions": own["action_bomb"],
    }


def behavioral(value):
    if isinstance(value, dict):
        return {
            key: behavioral(item)
            for key, item in value.items()
            if not key.startswith("decision_time")
        }
    if isinstance(value, list):
        return [behavioral(item) for item in value]
    return value


def interval(matrix):
    rng = random.Random(1684000)
    draws = []
    for _ in range(10000):
        replicas = rng.choices(range(3), k=3)
        worlds = rng.choices(range(5), k=5)
        draws.append(mean(matrix[r][w] for r in replicas for w in worlds))
    draws.sort()
    return [draws[249], draws[9749]]


def verify_training_tensors(root):
    import torch

    torch.set_num_threads(1)
    initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)

    def digest(state):
        value = hashlib.sha256()
        for name, tensor in state.items():
            value.update(name.encode() + tensor.detach().cpu().numpy().tobytes())
        return value.hexdigest()

    for relative in read_json(root / "training-state.json")["completed"].values():
        directory = root / relative
        result = read_json(directory / "result.json")
        payload = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        if (
            payload["completed_episodes"] != 50
            or payload["agent_seed"] != config()["replica_agent_seeds"][result["replica"]]
            or payload["learner_state"]["update_steps"] != result["optimizer_updates"]
            or digest(payload["learner_state"]["online_network"]) != result["final_online_sha256"]
        ):
            raise ValueError("Checkpoint tensors/counters do not match execution evidence")
        for key in (
            "config",
            "actions",
            "rewards",
            "checkpoint_schema_version",
            "model_schema_version",
            "feature_schema_version",
        ):
            if payload[key] != initial[key]:
                raise ValueError("Checkpoint training controls changed")
    return digest(initial["learner_state"]["online_network"])


def analyze(root):
    verify(root)
    models = artifacts(root)
    initial_network_hash = verify_training_tensors(root)
    cfg = config()
    state = read_json(root / "evaluation-state.json")
    if state["status"] != "completed" or len(state["completed"]) != 28:
        raise ValueError("Incomplete evaluation")
    data, environments = {}, set()
    for relative in state["completed"].values():
        directory = root / relative
        result = read_json(directory / "result.json")
        if sha(directory / "episodes.json.gz") != result["episodes_sha256"]:
            raise ValueError("Evaluation observations changed")
        if sha(models[result["artifact"]]) != result["checkpoint_sha256"]:
            raise ValueError("Evaluation input changed")
        environments.add(json.dumps(result["environment"], sort_keys=True))
        with gzip.open(directory / "episodes.json.gz", "rt") as file:
            rows = json.load(file)
        for row in rows:
            key = (row["artifact"], row["suite"], row["world_seed"], row["repeat"])
            if key in data:
                raise ValueError("Duplicate evaluation episode")
            if (
                row["agent_seed"] != row["world_seed"] + 1000000
                or row["training"]
                or row["behavior_epsilon"] != 0
            ):
                raise ValueError("Wrong evaluation condition")
            if row["online_before_sha256"] != row["online_after_sha256"]:
                raise ValueError("Mutable evaluation")
            data[key] = row
    expected = {
        (artifact, suite, seed, repeat)
        for artifact in models
        for suite, setting in cfg["evaluation_suites"].items()
        for seed in setting["world_seeds"]
        for repeat in range(2)
    }
    if set(data) != expected or len(environments) != 1:
        raise ValueError("Incomplete matrix or mixed evaluation environment")
    repeat_ok = all(
        behavioral(data[(a, s, w, 0)]["native"]) == behavioral(data[(a, s, w, 1)]["native"])
        for a, s, w, r in expected
        if r == 0
    )
    summary, paired, retention = {}, {}, {}
    for suite, setting in cfg["evaluation_suites"].items():
        rows = {
            a: [metrics(data[(a, suite, w, 0)]) for w in setting["world_seeds"]] for a in models
        }
        summary[suite] = {
            a: {m: mean(row[m] for row in values) for m in values[0]} for a, values in rows.items()
        }
        paired[suite] = {}
        for comparison in ("stepwise", "reference"):
            paired[suite][comparison] = {}
            for metric in rows["reference"][0]:
                matrix = [
                    [
                        rows[f"episode_mixture-r{r}"][w][metric]
                        - rows["reference" if comparison == "reference" else f"stepwise-r{r}"][w][
                            metric
                        ]
                        for w in range(5)
                    ]
                    for r in range(1, 4)
                ]
                paired[suite][comparison][metric] = {
                    "mean_difference": mean(v for row in matrix for v in row),
                    "paired_crossed_bootstrap_95_percent": interval(matrix),
                }
        if suite != "classic-peaceful":
            effects = paired[suite]["reference"]
            retention[suite] = {
                "coins": effects["collection_fraction"]["mean_difference"] >= -0.05,
                "self_kills": effects["self_kills"]["mean_difference"] <= 0.05,
            }
    hunting = paired["classic-peaceful"]
    nonworse = sum(
        summary["classic-peaceful"][f"episode_mixture-r{r}"]["eliminations"]
        >= summary["classic-peaceful"]["reference"]["eliminations"]
        for r in range(1, 4)
    )
    training = read_json(root / "training-state.json")
    updated = True
    initial_hashes = set()
    training_summary = {}
    for key, relative in training["completed"].items():
        directory = root / relative
        result = read_json(directory / "result.json")
        initial_hashes.add(result["initial_online_sha256"])
        updated &= (
            result["optimizer_updates"] > 0
            and result["final_online_sha256"] != result["initial_online_sha256"]
        )
        if sha(directory / "episodes.json.gz") != result["episodes_sha256"]:
            raise ValueError("Training observations changed")
        with gzip.open(directory / "episodes.json.gz", "rt") as file:
            rows = json.load(file)
        replica = result["replica"]
        if [row["world_seed"] for row in rows] != cfg["training_world_seeds"][replica]:
            raise ValueError("Training seeds or counts differ")
        training_summary[key] = {
            "optimizer_updates": result["optimizer_updates"],
            "mean_survival_steps": mean(metrics(row)["survival_steps"] for row in rows),
            "attack_episodes": sum(row["attack_steps"] > 0 for row in rows),
            "eliminations": sum(metrics(row)["eliminations"] for row in rows),
        }
    if initial_hashes != {initial_network_hash}:
        raise ValueError("Unequal initialization")
    times = [
        t
        for row in data.values()
        for t in row["native"]["agents"]["DagobertDuckDQNTask3"]["decision_times_ms"]
    ]
    times.sort()
    import numpy as np

    latency = {"p95_ms": float(np.percentile(times, 95)), "max_ms": max(times)}
    gates = {
        "genuine_updates": bool(updated),
        "behavioral_repeats": repeat_ok,
        "hunting_effect": hunting["stepwise"]["eliminations"]["mean_difference"] >= 0.1,
        "hunting_parent_retention": hunting["reference"]["eliminations"]["mean_difference"] >= 0,
        "nonworse_replicas": nonworse >= 2,
        "earlier_task_retention": all(all(values.values()) for values in retention.values()),
        "invalid_actions": sum(metrics(row)["invalid_actions"] for row in data.values()) == 0,
        "latency": latency["p95_ms"] < 50 and latency["max_ms"] < 100,
    }
    return {
        "pilot_screen_passed": all(gates.values()),
        "gates": gates,
        "retention_suites": retention,
        "summary": summary,
        "paired_mixture_minus": paired,
        "training_summary": training_summary,
        "latency": latency,
        "nonworse_replicas": nonworse,
        "selected_checkpoint": None,
        "task2_complete": False,
        "original_cumulative_gates": "not_established_by_this_small_pilot",
        "next_long_experiment_authorized": False,
    }
