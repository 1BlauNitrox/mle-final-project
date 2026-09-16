"""Registered analysis for the Task 3 approach pilots.

Every criterion below is fixed before execution in the profile's `config.json`.
The analyzer implements all of them, including the ones a favourable result
would make convenient to drop, and it reports the registered selection outcome
rather than letting a reader pick a replica after seeing the table.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from statistics import mean

from scripts.pilot_task4_competition import (
    arm_of,
    arm_payload,
    artifact_names,
    artifacts,
    config,
    episode_epsilon,
    read_json,
    sha,
    verify,
)

AGENT = "DagobertDuckDQNTask3"


def require_versioned_registration(cfg):
    """Allow the corrected rule only for a protocol registered as version 2."""
    if cfg.get("promotion_rule_version") != 2:
        raise ValueError("Corrected promotion rule requires promotion_rule_version=2")
    return cfg


def config_v2():
    return require_versioned_registration(config())


def blocking_gates(integrity):
    """Integrity verdicts that can veto promotion.

    A gate reported as null is not applicable to the programme that ran - an arm
    registered to move the inherited weights cannot also be required to have left
    them frozen - and a gate that cannot apply must not be able to veto.
    """
    return [value for value in integrity.values() if value is not None]


def legality_not_worse(invalid_by_artifact):
    """Did training make legality worse than leaving the agent untrained?

    The original gate summed invalid actions over every artifact, the unchanged
    reference included, and required zero. The reference emits its own, so the
    gate could only ever veto. Each trained artifact is compared against the
    reference instead, with no tolerance.
    """
    reference = invalid_by_artifact["reference"]
    return all(
        count <= reference
        for artifact, count in invalid_by_artifact.items()
        if artifact != "reference"
    )


def median_replica_by_score(suite_summary, arm, replicas):
    """The median-final replica by hunting score; the best-looking one is not selectable."""
    ordered = sorted(
        range(replicas), key=lambda r: (suite_summary[f"{arm}-r{r + 1}"]["score"], r)
    )
    return ordered[len(ordered) // 2]


def metrics(row):
    agents = row["native"]["agents"]
    own = agents[AGENT]
    others = [value["score"] for key, value in agents.items() if key != AGENT]
    denominator = own["initially_available_coins"]
    if denominator <= 0:
        raise ValueError("Missing collection denominator")
    steps = own["survival_steps"]
    present = row["opponent_present_steps"]
    return {
        "score": own["score"],
        "strict_win": int(bool(others) and own["score"] > max(others)),
        "collection_fraction": own["coins"] / denominator,
        "coins": own["coins"],
        "crates": own["crates_destroyed"],
        "eliminations": own["kills"],
        "survived": int(own["survived"]),
        "survival_steps": steps,
        "self_kills": own["self_kills"],
        "invalid_actions": own["invalid"],
        "bomb_actions": own["action_bomb"],
        # Mechanism endpoints. `attack_opportunities_per_100_steps` is the
        # registered primary for the shaping profile: it is the quantity the
        # potential is meant to move, and it is dense enough to estimate at
        # this budget. Distance is descriptive support for the same mechanism.
        "attack_opportunities_per_100_steps": 100.0 * row["attack_steps"] / max(steps, 1),
        "opponent_present_fraction": present / max(steps, 1),
        "mean_opponent_distance": (
            row["opponent_distance_sum"] / present if present else float("nan")
        ),
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


def interval(matrix, *, replicas, worlds, percent, seed, resamples):
    """Hierarchical paired bootstrap over replicas and shared worlds."""
    import numpy as np

    values = np.asarray(matrix, dtype=float)
    if values.shape != (replicas, worlds):
        raise ValueError("Expected the registered paired replica and world counts")
    if not np.isfinite(values).all():
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    replica_draws = rng.integers(0, replicas, size=(resamples, replicas))
    world_draws = rng.integers(0, worlds, size=(resamples, worlds))
    draws = values[replica_draws[:, :, None], world_draws[:, None, :]].mean(axis=(1, 2))
    tail = (100.0 - percent) / 200.0
    return np.quantile(draws, [tail, 1.0 - tail]).tolist()


def verify_training_tensors(root):
    import torch

    from scripts.frozen_opponent_inputs import validate_frozen

    torch.set_num_threads(1)
    cfg = config_v2()
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
            payload["completed_episodes"] != cfg["episodes_per_replica_arm"]
            or payload["agent_seed"] != cfg["replica_agent_seeds"][result["replica"]]
            or payload["learner_state"]["update_steps"] != result["optimizer_updates"]
            or digest(payload["learner_state"]["online_network"]) != result["final_online_sha256"]
        ):
            raise ValueError("Checkpoint tensors/counters do not match execution evidence")
        if cfg["arm_settings"][result["arm"]]["trainable_scope"] == "opponent_columns":
            validate_frozen(
                payload["learner_state"]["online_network"],
                initial["learner_state"]["online_network"],
            )
        expected_initial = arm_payload(initial, result["arm"])
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


def _load_stage(root, stage, models, cfg, *, expected_jobs):
    state = read_json(root / f"{stage}-state.json")
    if state["status"] != "completed" or len(state["completed"]) != expected_jobs:
        raise ValueError(f"Incomplete {stage} stage")
    data, environments = {}, set()
    for relative in state["completed"].values():
        directory = root / relative
        result = read_json(directory / "result.json")
        if sha(directory / "episodes.json.gz") != result["episodes_sha256"]:
            raise ValueError("Observations changed after execution")
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
                or row["potential_scale"] != 0.0
                or row["update_every"] != 1
                or row["kill_reward"] != cfg["arm_settings"][arm_of(row["artifact"])]["kill_reward"]
            ):
                raise ValueError("Wrong evaluation condition")
            if row["online_before_sha256"] != row["online_after_sha256"]:
                raise ValueError("Mutable evaluation")
            data[key] = row
    if len(environments) != 1:
        raise ValueError("Mixed evaluation environment")
    return data, environments


def analyze(root):
    verify(root)
    cfg = config_v2()
    models = artifacts(root)
    initial_network_hash = verify_training_tensors(root)
    arms = cfg["arms"]
    treatments = [arm for arm in arms if arm != "control"]
    replicas = cfg["replicas"]
    uncertainty = cfg["uncertainty"]
    # The promotion rule is conjunctive, so its individual tests need no
    # adjustment (an intersection-union test holds its level). The multiplicity
    # that does inflate the error rate is the number of treatment arms that each
    # get a chance to be promoted, so the family size is that arm count.
    percent = 100.0 - (100.0 - uncertainty["interval_percent"]) / max(len(treatments), 1)

    data, environments = _load_stage(
        root,
        "evaluation",
        models,
        cfg,
        expected_jobs=len(models) * len(cfg["evaluation_suites"]),
    )
    expected = {
        (artifact, suite, seed, repeat)
        for artifact in models
        for suite, setting in cfg["evaluation_suites"].items()
        for seed in setting["world_seeds"]
        for repeat in range(cfg["evaluation_repeats"])
    }
    if set(data) != expected:
        raise ValueError("Incomplete evaluation matrix")
    repeat_ok = all(
        behavioral(data[(a, s, w, 0)]["native"]) == behavioral(data[(a, s, w, r)]["native"])
        for a, s, w, r in expected
        if r
    )

    hunting_suite = cfg["hunting_suite"]
    summary, paired, retention = {}, {}, {}
    for suite, setting in cfg["evaluation_suites"].items():
        seeds = setting["world_seeds"]
        rows = {a: [metrics(data[(a, suite, w, 0)]) for w in seeds] for a in models}
        opponent_free = not setting["opponents"]
        summary[suite] = {
            a: {
                m: None
                if opponent_free and m == "mean_opponent_distance"
                else mean(row[m] for row in values)
                for m in values[0]
            }
            for a, values in rows.items()
        }
        paired[suite] = {}
        for arm in arms:
            paired[suite][arm] = {}
            for baseline in ("control", "reference"):
                if arm == baseline:
                    continue
                paired[suite][arm][baseline] = {}
                for metric in rows["reference"][0]:
                    if opponent_free and metric in {
                        "mean_opponent_distance",
                        "attack_opportunities_per_100_steps",
                        "opponent_present_fraction",
                    }:
                        continue
                    matrix = [
                        [
                            rows[f"{arm}-r{r + 1}"][w][metric]
                            - rows[
                                "reference" if baseline == "reference" else f"control-r{r + 1}"
                            ][w][metric]
                            for w in range(len(seeds))
                        ]
                        for r in range(replicas)
                    ]
                    paired[suite][arm][baseline][metric] = {
                        "mean_difference": mean(v for row in matrix for v in row),
                        "paired_crossed_bootstrap": interval(
                            matrix,
                            replicas=replicas,
                            worlds=len(seeds),
                            percent=percent,
                            seed=uncertainty["seed"],
                            resamples=uncertainty["resamples"],
                        ),
                    }
        if suite != hunting_suite:
            retention[suite] = {
                arm: {
                    "coins": paired[suite][arm]["reference"]["collection_fraction"][
                        "mean_difference"
                    ]
                    >= -cfg["retention_margins"]["collection_fraction"],
                    "self_kills": paired[suite][arm]["reference"]["self_kills"]["mean_difference"]
                    <= cfg["retention_margins"]["self_kills"],
                }
                for arm in arms
            }

    hunting = paired[hunting_suite]
    nonworse = {
        arm: sum(
            summary[hunting_suite][f"{arm}-r{r + 1}"]["eliminations"]
            >= summary[hunting_suite]["reference"]["eliminations"]
            for r in range(replicas)
        )
        for arm in arms
    }

    training = read_json(root / "training-state.json")
    updated, initial_hashes, training_summary, training_environments = True, set(), {}, set()
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
        setting = cfg["arm_settings"][result["arm"]]
        for index, row in enumerate(rows):
            expected_seed = cfg["replica_agent_seeds"][replica]
            if (
                row["agent_seed"] != expected_seed
                or not row["training"]
                or row["completed_episodes"] != index + 1
                or row["behavior_epsilon"] != episode_epsilon(result["arm"], expected_seed, index)
                or row["trainable_scope"] != setting["trainable_scope"]
                or row["opponents"] != setting["training_opponents"]
                or row["kill_reward"] != setting["kill_reward"]
                or row["potential_scale"] != setting["potential_scale"]
                or row["update_every"] != setting["update_every"]
                or row.get("proximity_gate") != setting.get("proximity_gate")
                or row.get("l2_opponent", 0.0) != setting.get("l2_opponent", 0.0)
            ):
                raise ValueError("Executed schedule differs from registration")
        steps = sum(metrics(row)["survival_steps"] for row in rows)
        training_summary[key] = {
            "optimizer_updates": result["optimizer_updates"],
            "mean_survival_steps": mean(metrics(row)["survival_steps"] for row in rows),
            "attack_episodes": sum(row["attack_steps"] > 0 for row in rows),
            "attack_opportunities_per_100_steps": 100.0
            * sum(row["attack_steps"] for row in rows)
            / max(steps, 1),
            "eliminations": sum(metrics(row)["eliminations"] for row in rows),
            # Non-zero only under a proximity gate; it records how much
            # experience the gate withheld from replay, which is the quantity
            # the gated comparison is actually manipulating.
            "gated_out_transitions": sum(row.get("gated_out_transitions", 0) for row in rows),
        }
    if len(training_environments) != 1 or training_environments != environments:
        raise ValueError("Arms were split across environments")
    if initial_hashes != {initial_network_hash}:
        raise ValueError("Unequal initialization")

    latency_data, _ = _load_stage(
        root, "latency", models, cfg, expected_jobs=len(models)
    )
    times = sorted(
        t
        for row in latency_data.values()
        for t in row["native"]["agents"][AGENT]["decision_times_ms"]
    )
    contended = sorted(
        t for row in data.values() for t in row["native"]["agents"][AGENT]["decision_times_ms"]
    )
    import numpy as np

    def distribution(values):
        return {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(max(values)),
            "decisions": len(values),
        }

    latency = distribution(times)
    resources = {}
    for stage in ("training", "evaluation", "latency"):
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

    # Full fine-tuning is expected to change opponent-free behaviour, so the two
    # invariance gates are only asserted when every arm froze inheritance. They
    # are reported either way so a reader can see which regime applied.
    frozen_programme = all(
        cfg["arm_settings"][arm]["trainable_scope"] == "opponent_columns" for arm in arms
    )
    invalid_by_artifact = {
        a: sum(metrics(row)["invalid_actions"] for key, row in data.items() if key[0] == a)
        for a in artifact_names(cfg)
    }
    integrity = {
        "resources": all(r["passed"] for r in resources.values()),
        "genuine_updates": bool(updated),
        "behavioral_repeats": repeat_ok,
        # Null means "not applicable to this programme" and does not block promotion;
        # an arm registered to move the inherited weights cannot also be required to
        # have left them frozen.
        "frozen_online_weights": frozen_programme or None,
        "opponent_free_invariance": (not frozen_programme)
        or all(
            behavioral(data[(f"{arm}-r{r + 1}", s, w, 0)]["native"])
            == behavioral(data[("reference", s, w, 0)]["native"])
            and data[(f"{arm}-r{r + 1}", s, w, 0)]["greedy_actions_sha256"]
            == data[("reference", s, w, 0)]["greedy_actions_sha256"]
            for s, setting in cfg["evaluation_suites"].items()
            if not setting["opponents"]
            for w in setting["world_seeds"]
            for arm in arms
            for r in range(replicas)
        ),
        "invalid_actions": legality_not_worse(invalid_by_artifact),
        "latency": latency["p95_ms"] < cfg["decision_time_limits"]["p95_ms"]
        and latency["max_ms"] < cfg["decision_time_limits"]["max_ms"],
    }

    thresholds = cfg["decision_thresholds"]
    efficacy, diagnostics, promotable = {}, {}, []
    for arm in treatments:
        versus_control = hunting[arm]["control"]
        versus_reference = hunting[arm]["reference"]
        arm_retention = all(
            retention[suite][arm][key]
            for suite in retention
            for key in ("coins", "self_kills")
        )
        # Re-specified: the registered primary endpoint is score, because tournament
        # placing is decided by score. The +0.10 elimination thresholds were carried
        # over from the Task 3 hunting programme, where eliminations was the endpoint;
        # here they demanded roughly tripling the reference's elimination rate before
        # any score gain could be promoted. They are retained as reported diagnostics
        # and eliminations keeps a non-worse guard, so an arm cannot buy score by
        # abandoning opponents entirely.
        checks = {
            "score_versus_reference_ci": versus_reference["score"]["paired_crossed_bootstrap"][0]
            > 0,
            "eliminations_not_worse_than_reference": versus_reference["eliminations"][
                "mean_difference"
            ]
            >= -thresholds["self_kills"],
            "self_kills_versus_control": versus_control["self_kills"]["mean_difference"]
            <= thresholds["self_kills"],
            "self_kills_versus_reference": versus_reference["self_kills"]["mean_difference"]
            <= thresholds["self_kills"],
            "nonworse_replicas": nonworse[arm] >= thresholds["nonworse_replicas"],
            "earlier_task_retention": arm_retention,
        }
        diagnostics[arm] = {
            "exposure_increase": versus_control["attack_opportunities_per_100_steps"][
                "mean_difference"
            ]
            > 0,
            "exposure_increase_ci": versus_control["attack_opportunities_per_100_steps"][
                "paired_crossed_bootstrap"
            ][0]
            > 0,
            "hunting_improvement": versus_control["eliminations"]["mean_difference"]
            >= thresholds["eliminations"],
            "hunting_improvement_ci": versus_control["eliminations"][
                "paired_crossed_bootstrap"
            ][0]
            > 0,
            "hunting_versus_reference": versus_reference["eliminations"]["mean_difference"]
            >= thresholds["eliminations"],
            "hunting_versus_reference_ci": versus_reference["eliminations"][
                "paired_crossed_bootstrap"
            ][0]
            > 0,
        }
        efficacy[arm] = checks
        if all(blocking_gates(integrity)) and all(checks.values()):
            promotable.append(arm)

    # Registered selection, as the configuration states it: "the largest mean score
    # gain over the unchanged incumbent is chosen, and within it the median-final
    # replica by that score". The implementation ranked by eliminations instead,
    # which contradicted the registration it was supposed to execute. The
    # best-looking replica remains deliberately not selectable.
    selection = {"promotable_arms": promotable, "selected_arm": None, "selected_checkpoint": None}
    if promotable:
        chosen = max(
            promotable,
            key=lambda arm: (
                hunting[arm]["reference"]["score"]["mean_difference"],
                hunting[arm]["reference"]["eliminations"]["mean_difference"],
                -arms.index(arm),
            ),
        )
        median_replica = median_replica_by_score(summary[hunting_suite], chosen, replicas)
        selection["selected_arm"] = chosen
        selection["selected_checkpoint"] = {
            "artifact": f"{chosen}-r{median_replica + 1}",
            "rule": "median_final_replica_by_hunting_score",
            "sha256": sha(models[f"{chosen}-r{median_replica + 1}"]),
        }

    return {
        "profile": cfg["profile"],
        "registered_screen_passed": bool(promotable),
        "integrity_gates": integrity,
        "efficacy_gates": efficacy,
        "reported_diagnostics_not_gating": diagnostics,
        "retention_suites": retention,
        "selection": selection,
        "interval_percent": percent,
        "bonferroni_family_size": max(len(treatments), 1),
        "summary": summary,
        "paired_differences": paired,
        "training_summary": training_summary,
        "latency_serial_stage": latency,
        "latency_contended_evaluation_stage_not_gated": distribution(contended),
        "resources": resources,
        "invalid_actions_by_artifact": invalid_by_artifact,
        "nonworse_replicas": nonworse,
        "task2_complete": False,
        "original_cumulative_gates": "not_established_by_this_exploratory_pilot",
        "next_long_experiment_authorized": False,
    }
