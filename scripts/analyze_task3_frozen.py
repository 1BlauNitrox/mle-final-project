"""Selection-free retention-pilot analysis; adapted from #168 at dcff79f."""

from __future__ import annotations

import gzip
import hashlib
import json
from statistics import mean

from scripts.pilot_task3_frozen import (
    arm_payload,
    artifacts,
    config,
    episode_epsilon,
    read_json,
    sha,
    verify,
)


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
    import numpy as np

    values = np.asarray(matrix, dtype=float)
    if values.shape != (5, 40) or not np.isfinite(values).all():
        raise ValueError("Expected five paired replicas and forty shared worlds")
    rng = np.random.default_rng(config()["uncertainty"]["seed"])
    replicas = rng.integers(0, 5, size=(10000, 5))
    worlds = rng.integers(0, 40, size=(10000, 40))
    draws = values[replicas[:, :, None], worlds[:, None, :]].mean(axis=(1, 2))
    return np.quantile(draws, [0.025, 0.975]).tolist()


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
            payload["completed_episodes"] != 100
            or payload["agent_seed"] != config()["replica_agent_seeds"][result["replica"]]
            or payload["learner_state"]["update_steps"] != result["optimizer_updates"]
            or digest(payload["learner_state"]["online_network"]) != result["final_online_sha256"]
        ):
            raise ValueError("Checkpoint tensors/counters do not match execution evidence")
        expected_initial = arm_payload(initial, result["arm"])
        if result["arm"] == "treatment":
            from scripts.frozen_opponent_inputs import validate_frozen

            validate_frozen(
                payload["learner_state"]["online_network"],
                initial["learner_state"]["online_network"],
            )

        if any(
            g["lr"] != expected_initial["config"]["learning_rate"]
            for g in payload["learner_state"]["optimizer"]["param_groups"]
        ):
            raise ValueError("Optimizer learning rate differs from registration")
        for key in (
            "config",
            "actions",
            "rewards",
            "checkpoint_schema_version",
            "model_schema_version",
            "feature_schema_version",
        ):
            if payload[key] != expected_initial[key]:
                raise ValueError("Checkpoint training controls changed")
    return digest(initial["learner_state"]["online_network"])


def analyze(root):
    verify(root)
    models = artifacts(root)
    initial_network_hash = verify_training_tensors(root)
    cfg = config()
    state = read_json(root / "evaluation-state.json")
    if state["status"] != "completed" or len(state["completed"]) != 44:
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
        for comparison in ("control", "reference"):
            paired[suite][comparison] = {}
            for metric in rows["reference"][0]:
                matrix = [
                    [
                        rows[f"treatment-r{r}"][w][metric]
                        - rows["reference" if comparison == "reference" else f"control-r{r}"][w][
                            metric
                        ]
                        for w in range(40)
                    ]
                    for r in range(1, 6)
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
        summary["classic-peaceful"][f"treatment-r{r}"]["eliminations"]
        >= summary["classic-peaceful"]["reference"]["eliminations"]
        for r in range(1, 6)
    )
    training = read_json(root / "training-state.json")
    updated = True
    initial_hashes = set()
    training_summary = {}
    training_environments = set()
    for key, relative in training["completed"].items():
        directory = root / relative
        result = read_json(directory / "result.json")
        initial_hashes.add(result["initial_online_sha256"])
        training_environments.add(json.dumps(result["environment"], sort_keys=True))
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
        for index, row in enumerate(rows):
            expected_seed = cfg["replica_agent_seeds"][replica]
            if (
                row["agent_seed"] != expected_seed
                or not row["training"]
                or row["completed_episodes"] != index + 1
                or row["behavior_epsilon"] != episode_epsilon(result["arm"], expected_seed, index)
                or row["freeze_mode"] != cfg["freeze_modes"][result["arm"]]
            ):
                raise ValueError("Executed exploration schedule differs from registration")

        training_summary[key] = {
            "optimizer_updates": result["optimizer_updates"],
            "mean_survival_steps": mean(metrics(row)["survival_steps"] for row in rows),
            "attack_episodes": sum(row["attack_steps"] > 0 for row in rows),
            "eliminations": sum(metrics(row)["eliminations"] for row in rows),
        }
    if len(training_environments) != 1 or training_environments != environments:
        raise ValueError("Training arms were split across environments")
    if initial_hashes != {initial_network_hash}:
        raise ValueError("Unequal initialization")
    times = [
        t
        for row in data.values()
        for t in row["native"]["agents"]["DagobertDuckDQNTask3"]["decision_times_ms"]
    ]
    times.sort()
    import numpy as np

    latency = {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": max(times),
    }
    resources = {}
    for stage in ("training", "evaluation"):
        record = read_json(root / (stage + "-state.json"))
        limits = cfg[stage + "_limits"]
        resources[stage] = {
            key: record[key] for key in ("cpu_seconds", "wall_seconds", "peak_memory_bytes")
        }
        resources[stage]["passed"] = (
            record["cpu_seconds"] <= limits["cpu_seconds"]
            and record["wall_seconds"] <= limits["wall_seconds"]
            and record["peak_memory_bytes"] <= limits["memory_bytes"]
        )
    gates = {
        "resources": all(r["passed"] for r in resources.values()),
        "genuine_updates": bool(updated),
        "behavioral_repeats": repeat_ok,
        "hunting_improvement": hunting["control"]["eliminations"]["mean_difference"] >= 0.1,
        "hunting_improvement_ci": hunting["control"]["eliminations"][
            "paired_crossed_bootstrap_95_percent"
        ][0]
        > 0,
        "hunting_self_kills_control": hunting["control"]["self_kills"]["mean_difference"] <= 0.05,
        "hunting_self_kills_parent": hunting["reference"]["self_kills"]["mean_difference"] <= 0.05,
        "frozen_online_weights": True,
        "opponent_free_invariance": all(
            behavioral(data[(f"treatment-r{r}", s, w, 0)]["native"])
            == behavioral(data[("reference", s, w, 0)]["native"])
            and data[(f"treatment-r{r}", s, w, 0)]["greedy_actions_sha256"]
            == data[("reference", s, w, 0)]["greedy_actions_sha256"]
            for s, setting in cfg["evaluation_suites"].items()
            if not setting["opponents"]
            for w in setting["world_seeds"]
            for r in range(1, 6)
        ),
        "hunting_parent_retention": hunting["reference"]["eliminations"]["mean_difference"] >= 0,
        "nonworse_replicas": nonworse >= 4,
        "earlier_task_retention": all(all(values.values()) for values in retention.values()),
        "invalid_actions": sum(metrics(row)["invalid_actions"] for row in data.values()) == 0,
        "latency": latency["p95_ms"] < 50 and latency["max_ms"] < 100,
    }
    return {
        "pilot_screen_passed": all(gates.values()),
        "gates": gates,
        "retention_suites": retention,
        "summary": summary,
        "paired_treatment_minus": paired,
        "training_summary": training_summary,
        "latency": latency,
        "resources": resources,
        "invalid_actions_by_artifact": {
            a: sum(metrics(row)["invalid_actions"] for key, row in data.items() if key[0] == a)
            for a in models
        },
        "nonworse_replicas": nonworse,
        "selected_checkpoint": None,
        "task2_complete": False,
        "original_cumulative_gates": "not_established_by_this_small_pilot",
        "next_long_experiment_authorized": False,
    }
