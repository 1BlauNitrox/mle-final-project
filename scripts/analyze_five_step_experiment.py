"""Apply the prospectively registered Issue #207 decision rule."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def native_metric(row, metric):
    native = row["native"]
    if metric == "collection_fraction":
        denominator = native["initially_available_coins"]
        return native["coins"] / denominator if denominator else 0.0
    return float(native[metric])


def hierarchical_interval(matrix, *, resamples, seed):
    values = np.asarray(matrix, dtype=float)
    require(values.ndim == 2 and values.shape[0] == 3, "Expected three paired replicas")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for index in range(resamples):
        replicas = rng.integers(0, values.shape[0], size=values.shape[0])
        worlds = rng.integers(0, values.shape[1], size=values.shape[1])
        draws[index] = values[np.ix_(replicas, worlds)].mean()
    return {
        "estimate": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "replica_means": [float(value) for value in values.mean(axis=1)],
    }


def is_loop_window(part, maximum_period=4):
    if len(part) != 24:
        return False
    if any(
        row["crates_left"] != 0 or row["coins_visible"] != 0 or row["hazards"] or row["progress"]
        for row in part
    ):
        return False
    if len({row["board_coins_sha256"] for row in part}) != 1:
        return False
    positions = [tuple(row["position"]) for row in part]
    return any(
        all(positions[index] == positions[index % period] for index in range(len(positions)))
        for period in range(1, maximum_period + 1)
    )


def loop_summary(rows):
    denominator = 0
    total = 0
    windows = 0
    for episode in rows:
        steps = episode["late_steps"]
        total += len(steps)
        denominator += sum(row["crates_left"] == 0 and row["coins_visible"] == 0 for row in steps)
        windows += sum(is_loop_window(steps[end - 24 : end]) for end in range(24, len(steps) + 1))
    return {
        "late_state_steps": denominator,
        "observed_steps": total,
        "late_state_time_fraction": denominator / total if total else 0.0,
        "looping_windows": windows,
        "late_loop_rate": windows / denominator if denominator else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    require(read_json(args.root / "training-summary.json")["complete"], "Training incomplete")
    require(read_json(args.root / "evaluation-summary.json")["complete"], "Evaluation incomplete")
    require(read_json(args.root / "latency-summary.json")["complete"], "Latency incomplete")
    artifacts = ["reference"] + [
        f"{arm}-r{replica}" for arm in cfg["arms"] for replica in range(1, 4)
    ]
    suites = list(cfg["evaluation_suites"])
    data = {}
    for artifact in artifacts:
        for suite in suites:
            path = args.root / "evaluation" / artifact / suite / "episodes.json.gz"
            require(path.exists(), f"Missing evaluation: {artifact}/{suite}")
            data[artifact, suite] = read_rows(path)

    uncertainty = cfg["uncertainty"]

    def contrast(treatment, baseline, suite, metric):
        matrix = []
        for replica in range(1, 4):
            left_name = f"{treatment}-r{replica}" if treatment != "reference" else "reference"
            right_name = f"{baseline}-r{replica}" if baseline != "reference" else "reference"
            left, right = data[left_name, suite], data[right_name, suite]
            require(
                [row["world_seed"] for row in left] == [row["world_seed"] for row in right],
                "World pairing mismatch",
            )
            matrix.append(
                [
                    native_metric(a, metric) - native_metric(b, metric)
                    for a, b in zip(left, right, strict=True)
                ]
            )
        return hierarchical_interval(
            matrix, resamples=uncertainty["resamples"], seed=uncertainty["seed"]
        )

    primary = "primary-classic-rule-based"
    metrics = ("kills", "score", "survived", "self_kills", "invalid", "coins")
    five_vs_one = {metric: contrast("five-step", "one-step", primary, metric) for metric in metrics}
    five_vs_reference = {
        metric: contrast("five-step", "reference", primary, metric) for metric in metrics
    }
    retention = {
        "five_vs_one": {
            "coin_heaven_collection_fraction": contrast(
                "five-step", "one-step", "coin-heaven", "collection_fraction"
            ),
            "classic_peaceful_survived": contrast(
                "five-step", "one-step", "classic-peaceful", "survived"
            ),
            "classic_peaceful_self_kills": contrast(
                "five-step", "one-step", "classic-peaceful", "self_kills"
            ),
            "loot_crate_coins": contrast("five-step", "one-step", "loot-crate", "coins"),
            "loot_crate_self_kills": contrast("five-step", "one-step", "loot-crate", "self_kills"),
        },
        "five_vs_reference": {
            "coin_heaven_collection_fraction": contrast(
                "five-step", "reference", "coin-heaven", "collection_fraction"
            ),
            "classic_peaceful_survived": contrast(
                "five-step", "reference", "classic-peaceful", "survived"
            ),
            "classic_peaceful_self_kills": contrast(
                "five-step", "reference", "classic-peaceful", "self_kills"
            ),
            "loot_crate_coins": contrast("five-step", "reference", "loot-crate", "coins"),
            "loot_crate_self_kills": contrast("five-step", "reference", "loot-crate", "self_kills"),
        },
    }
    loops = {}
    for artifact in artifacts:
        all_rows = [row for suite in suites for row in data[artifact, suite]]
        loops[artifact] = loop_summary(all_rows)
    arm_loops = {}
    for arm in cfg["arms"]:
        summaries = [loops[f"{arm}-r{replica}"] for replica in range(1, 4)]
        arm_loops[arm] = {
            "late_state_steps": sum(item["late_state_steps"] for item in summaries),
            "observed_steps": sum(item["observed_steps"] for item in summaries),
            "looping_windows": sum(item["looping_windows"] for item in summaries),
        }
        item = arm_loops[arm]
        item["late_state_time_fraction"] = item["late_state_steps"] / item["observed_steps"]
        item["late_loop_rate"] = (
            item["looping_windows"] / item["late_state_steps"] if item["late_state_steps"] else 0.0
        )
    loop_differences = {
        key: arm_loops["five-step"][key] - arm_loops["one-step"][key]
        for key in ("late_state_time_fraction", "late_loop_rate")
    }

    latency = {}
    for artifact in artifacts:
        rows = read_rows(args.root / "latency" / artifact / "episodes.json.gz")
        times = [value for row in rows for value in row["native"]["decision_times_ms"]]
        latency[artifact] = {
            "decisions": len(times),
            "p95_ms": float(np.percentile(times, 95)),
            "maximum_ms": float(np.max(times)),
            "passes": (
                np.percentile(times, 95) <= cfg["latency"]["p95_limit_ms"]
                and np.max(times) <= cfg["latency"]["maximum_limit_ms"]
            ),
        }
    threshold = cfg["decision_rule"]["five_step_minus_one_step"]
    treatment_latency = all(latency[f"five-step-r{replica}"]["passes"] for replica in range(1, 4))
    retention_one = retention["five_vs_one"]
    claim_gates = {
        "kills": (
            five_vs_one["kills"]["estimate"] >= threshold["kills_estimate_minimum"]
            and five_vs_one["kills"]["ci95"][0] > threshold["kills_interval_lower_strictly_above"]
            and sum(value >= 0 for value in five_vs_one["kills"]["replica_means"])
            >= threshold["replica_pairs_nonnegative_minimum"]
        ),
        "score": (
            five_vs_one["score"]["estimate"] >= threshold["score_estimate_minimum"]
            and five_vs_one["score"]["ci95"][0] > threshold["score_interval_lower_strictly_above"]
        ),
        "survival": five_vs_one["survived"]["estimate"] >= threshold["survival_minimum"],
        "self_kills": five_vs_one["self_kills"]["estimate"] <= threshold["self_kills_maximum"],
        "invalid": five_vs_one["invalid"]["estimate"] <= threshold["invalid_actions_maximum"],
        "coin_heaven": retention_one["coin_heaven_collection_fraction"]["estimate"]
        >= threshold["coin_heaven_collection_fraction_minimum"],
        "classic_peaceful": (
            retention_one["classic_peaceful_survived"]["estimate"]
            >= threshold["classic_peaceful_survival_minimum"]
            and retention_one["classic_peaceful_self_kills"]["estimate"]
            <= threshold["classic_peaceful_self_kills_maximum"]
        ),
        "loot_crate": (
            retention_one["loot_crate_coins"]["estimate"] >= threshold["loot_crate_coins_minimum"]
            and retention_one["loot_crate_self_kills"]["estimate"]
            <= threshold["loot_crate_self_kills_maximum"]
        ),
        "late_loop_rate": loop_differences["late_loop_rate"] <= threshold["late_loop_rate_maximum"],
        "late_state_time": loop_differences["late_state_time_fraction"]
        <= threshold["late_state_time_fraction_maximum"],
        "latency": treatment_latency,
    }
    claim_success = all(claim_gates.values())
    promotion_threshold = cfg["decision_rule"]["promotion_versus_fallback"]
    retention_reference = retention["five_vs_reference"]
    fallback_retention = (
        retention_reference["coin_heaven_collection_fraction"]["estimate"] >= -0.05
        and retention_reference["classic_peaceful_survived"]["estimate"] >= -0.05
        and retention_reference["classic_peaceful_self_kills"]["estimate"] <= 0.05
        and retention_reference["loot_crate_coins"]["estimate"] >= -0.25
        and retention_reference["loot_crate_self_kills"]["estimate"] <= 0.05
    )
    promotion_gates = {
        "claim_success": claim_success,
        "score": (
            five_vs_reference["score"]["estimate"] >= promotion_threshold["score_estimate_minimum"]
            and five_vs_reference["score"]["ci95"][0]
            > promotion_threshold["score_interval_lower_strictly_above"]
        ),
        "kills": (
            five_vs_reference["kills"]["estimate"] >= promotion_threshold["kills_estimate_minimum"]
            and five_vs_reference["kills"]["ci95"][0]
            > promotion_threshold["kills_interval_lower_strictly_above"]
        ),
        "survival": five_vs_reference["survived"]["estimate"]
        >= promotion_threshold["survival_minimum"],
        "self_kills": five_vs_reference["self_kills"]["estimate"]
        <= promotion_threshold["self_kills_maximum"],
        "retention": fallback_retention,
        "latency": treatment_latency,
    }
    promotable = all(promotion_gates.values())
    selected = "reference"
    if promotable:
        candidates = []
        for replica in range(1, 4):
            rows = data[f"five-step-r{replica}", primary]
            candidates.append(
                (
                    float(np.mean([native_metric(row, "score") for row in rows])),
                    float(np.mean([native_metric(row, "kills") for row in rows])),
                    -replica,
                    f"five-step-r{replica}",
                )
            )
        selected = sorted(candidates)[1][3]
    result = {
        "issue": 207,
        "complete": True,
        "five_step_minus_one_step": five_vs_one,
        "five_step_minus_reference": five_vs_reference,
        "retention": retention,
        "looping_by_artifact": loops,
        "looping_by_arm": arm_loops,
        "loop_differences": loop_differences,
        "latency": latency,
        "claim_gates": claim_gates,
        "claim_success": claim_success,
        "promotion_gates": promotion_gates,
        "promotable": promotable,
        "selected_artifact": selected,
        "language": "registered decision rule, not a significance test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    print(f"FIVE-STEP CLAIM: {'PASS' if claim_success else 'FAIL'}")
    print(f"PROMOTION: {'PASS' if promotable else 'FAIL'}; selected={selected}")
    for metric in ("kills", "score", "survived", "self_kills", "invalid"):
        item = five_vs_one[metric]
        print(f"{metric}: {item['estimate']:.4f} [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
