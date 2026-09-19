"""Paired three-replica screen; validity, promise and promotion are separate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

# Replica 3 ran from the same commit on Windows. Git checked these six text files
# out with CRLF, while the laptops used LF. Each pair below was verified by
# normalizing the preserved Windows bytes to LF; no content difference remained.
# Accept only those exact pairs so a real source change still blocks pooling.
LINE_ENDING_HASH_PAIRS = {
    "agent_code/DagobertDuckDQNObservation/callbacks.py": frozenset(
        {
            "7d9ed6f23883237897b0282484ed45813aa6c8617c1700ef73ebd27c19fa3f26",
            "af652c666177c77375886fa2e1c343ce40afd9d7b8aa6bc1c0bb50637478f93b",
        }
    ),
    "agent_code/DagobertDuckDQNObservation/config.py": frozenset(
        {
            "e6a457109970355f3c04cf9e0878f865ecb8df32706f397a5ea8ca27a767add7",
            "5aadf05e16200e7335ff2ac398f17686f6ee5a2b8bd5348ef8d9bc89f415eeb1",
        }
    ),
    "agent_code/DagobertDuckDQNObservation/features/assemble.py": frozenset(
        {
            "742ed3b6fffbf7b488c11dfa9f06a6088853a2317eb204b22a777a49ac205432",
            "4cde8817c7d81c6052e89ad2dd2ad87ab9503d373feacdd609757d38d8aba7ec",
        }
    ),
    "agent_code/DagobertDuckDQNObservation/train.py": frozenset(
        {
            "2547456e3fbbaa1788adaff641b4c6eec16f09d2e8b68d34d9c95d66b3990f7d",
            "5dd0362b75cac2483ff33632e5b412fc079db53ed63f865da755e0514cf8c5e2",
        }
    ),
    "scripts/audit_observation_seeds.py": frozenset(
        {
            "3b330c825ceec00fd26e0db584a33ec14cf4898d98ac5a43563b1c79ca7b0e0b",
            "b23df545e9f83bd38382f14d625240afdcb937737fc9bf46be1e029ec0106250",
        }
    ),
    "scripts/observation_episode.py": frozenset(
        {
            "17c3d52f9cfd9ab5b4402ba8b34cbd84f0519614ae2903066c20a8337d3514ba",
            "d1c77e2ac6ec1fbdf85512491948e286c0bce58d6a84a0f7d5e4674d614ef3ef",
        }
    ),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric(row, name):
    native = row["native"]
    if name == "collection_fraction":
        return (
            native["coins"] / native["initially_available_coins"]
            if native["initially_available_coins"]
            else 0.0
        )
    return float(native[name])


def interval(matrix, *, resamples, seed):
    values = np.asarray(matrix, dtype=float)
    require(values.ndim == 2 and values.shape[0] == 3, "Need all three paired replicas")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        replicas = rng.integers(0, 3, size=3)
        worlds = rng.integers(0, values.shape[1], size=values.shape[1])
        draws.append(values[np.ix_(replicas, worlds)].mean())
    return {
        "estimate": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "replica_means": values.mean(axis=1).tolist(),
    }


def loops(rows):
    eligible = cyclic = late = observed = 0
    for row in rows:
        steps = row["late_steps"]
        observed += len(steps)
        late += sum(s["crates_left"] == 0 and s["coins_visible"] == 0 for s in steps)
        for end in range(24, len(steps) + 1):
            window = steps[end - 24 : end]
            if any(
                s["crates_left"] or s["coins_visible"] or s["hazards"] or s["progress"]
                for s in window
            ):
                continue
            if len({s["board_coins_sha256"] for s in window}) != 1:
                continue
            eligible += 1
            positions = [s["position"] for s in window]
            cyclic += any(
                all(positions[i] == positions[i % p] for i in range(24)) for p in range(1, 5)
            )
    return {
        "eligible_windows": eligible,
        "looping_windows": cyclic,
        "rate": cyclic / eligible if eligible else None,
        "late_steps": late,
        "observed_steps": observed,
    }


def classify(gates, efficacy, clear_harm):
    if clear_harm:
        return "reject_for_this_deadline"
    if all(gates.values()) and efficacy:
        return "eligible_for_confirmation"
    if efficacy:
        return "promising_but_unresolved"
    return "inconclusive"


def equivalent_code_hashes(bindings):
    maps = [binding["code_hashes"] for binding in bindings]
    if any(set(item) != set(maps[0]) for item in maps[1:]):
        return False
    for path in maps[0]:
        values = frozenset(item[path] for item in maps)
        if len(values) > 1 and values != LINE_ENDING_HASH_PAIRS.get(path):
            return False
    return True


def analyze(roots):
    require(len(roots) == 3, "Three replica bundles required; no partial pooled conclusion")
    roots = sorted(roots, key=lambda p: read(p / "binding.json")["replica"])
    bindings = [read(root / "binding.json") for root in roots]
    require([b["replica"] for b in bindings] == [1, 2, 3], "Duplicate or missing replica")
    require(equivalent_code_hashes(bindings), "Different execution sources")
    require(
        all(b["config_sha256"] == bindings[0]["config_sha256"] for b in bindings),
        "Different protocols",
    )
    cfg = read(roots[0] / "config.json")
    data = {}
    for root, binding in zip(roots, bindings, strict=True):
        require(
            hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
            == binding["config_sha256"],
            "Changed protocol",
        )
        for stage in ("training", "evaluation", "latency"):
            require(read(root / f"{stage}-summary.json")["complete"], f"Incomplete {stage}")
        for arm in (*cfg["arms"], "reference"):
            if arm != "reference":
                record = read(root / "training" / arm / "result.json")
                require(
                    record["complete"] and record["episodes"] == 1000, "Incomplete training budget"
                )
                model_hash = record["checkpoint_sha256"]
            else:
                model_hash = binding["initial_hashes"]["reference.pt"]
            for suite, setting in {**cfg["evaluation_suites"], "latency": cfg["latency"]}.items():
                directory = root / ("latency" if suite == "latency" else "evaluation") / arm
                if suite != "latency":
                    directory /= suite
                rowfile = directory / "episodes.json.gz"
                result = read(directory / "result.json")
                require(result["checkpoint_sha256"] == model_hash, "Wrong evaluated checkpoint")
                require(
                    hashlib.sha256(rowfile.read_bytes()).hexdigest() == result["rows_sha256"],
                    "Corrupt rows",
                )
                with gzip.open(rowfile, "rt", encoding="utf-8") as stream:
                    rows = json.load(stream)
                first, last = setting["world_seed_range_inclusive"]
                require(
                    [r["world_seed"] for r in rows] == list(range(first, last + 1)),
                    "Missing/duplicate worlds",
                )
                opponents = (
                    cfg["evaluation_suites"]["primary-classic-rule-based"]["opponents"]
                    if suite == "latency"
                    else setting["opponents"]
                )
                require(
                    all(
                        r["slot"] == i % (len(opponents) + 1) and r["opponents"] == opponents
                        for i, r in enumerate(rows)
                    ),
                    "Unpaired slots/opponents",
                )
                data[binding["replica"], arm, suite] = rows

    def contrast(arm, baseline, suite, name):
        values = [
            [
                metric(a, name) - metric(b, name)
                for a, b in zip(data[r, arm, suite], data[r, baseline, suite], strict=True)
            ]
            for r in (1, 2, 3)
        ]
        return interval(
            values, resamples=cfg["uncertainty"]["resamples"], seed=cfg["uncertainty"]["seed"]
        )

    primary = "primary-classic-rule-based"
    output = {
        "issue": 211,
        "valid_complete_screen": True,
        "automatic_promotion": False,
        "selected_submission_artifact": None,
        "arms": {},
        "config_sha256": bindings[0]["config_sha256"],
        "source_equivalence": {
            "status": "verified_exact_lf_crlf_pairs",
            "files": sorted(
                path
                for path in bindings[0]["code_hashes"]
                if len({binding["code_hashes"][path] for binding in bindings}) > 1
            ),
        },
    }
    loop_data = {
        arm: loops([row for r in (1, 2, 3) for row in data[r, arm, primary]])
        for arm in (*cfg["arms"], "reference")
    }
    output["loops"] = loop_data
    thresholds = cfg["decision_rule"]
    for arm in ("geometry", "memory"):
        comparisons, gates = {}, {}
        clear_harm = False
        for baseline in ("control", "reference"):
            estimates = {
                name: contrast(arm, baseline, primary, name)
                for name in (
                    "score",
                    "kills",
                    "survived",
                    "self_kills",
                    "invalid",
                    "coins",
                    "crates_destroyed",
                )
            }
            retention = {
                name: contrast(arm, baseline, suite, name)
                for suite, name in [("coin-heaven", "collection_fraction"), ("loot-crate", "coins")]
            }
            peaceful_survival = contrast(arm, baseline, "classic-peaceful", "survived")
            peaceful_self = contrast(arm, baseline, "classic-peaceful", "self_kills")
            crate_self = contrast(arm, baseline, "loot-crate", "self_kills")
            comparisons[baseline] = {
                "primary": estimates,
                "retention": retention,
                "peaceful_survival": peaceful_survival,
                "peaceful_self_kills": peaceful_self,
                "crate_self_kills": crate_self,
                "mixed": {
                    m: contrast(arm, baseline, "mixed", m)
                    for m in ("score", "kills", "survived", "self_kills")
                },
            }
            checks = {
                "score": estimates["score"]["estimate"] >= thresholds["score_minimum"],
                "score_interval": estimates["score"]["ci95"][0]
                > thresholds["score_lower_bound_minimum"],
                "survival": estimates["survived"]["estimate"] >= thresholds["survival_minimum"],
                "self_kills": estimates["self_kills"]["estimate"]
                <= thresholds["self_kills_maximum"],
                "invalid": estimates["invalid"]["estimate"] <= thresholds["invalid_maximum"],
                "collection": retention["collection_fraction"]["estimate"]
                >= thresholds["collection_fraction_minimum"],
                "crates_coins": retention["coins"]["estimate"]
                >= thresholds["loot_crate_coins_minimum"],
                "peaceful_survival": peaceful_survival["estimate"]
                >= thresholds["survival_minimum"],
                "peaceful_self": peaceful_self["estimate"] <= thresholds["self_kills_maximum"],
                "crate_self": crate_self["estimate"] <= thresholds["self_kills_maximum"],
            }
            if baseline == "reference":
                checks["kills"] = estimates["kills"]["estimate"] >= 0.0
            gates.update({f"{baseline}:{key}": value for key, value in checks.items()})
            clear_harm |= (
                estimates["score"]["ci95"][1] < -0.3 or estimates["self_kills"]["ci95"][0] > 0.05
            )
            clear_harm |= peaceful_survival["ci95"][1] < -0.05 or crate_self["ci95"][0] > 0.05
            clear_harm |= estimates["survived"]["ci95"][1] < thresholds["survival_minimum"]
            clear_harm |= estimates["invalid"]["ci95"][0] > thresholds["invalid_maximum"]
            clear_harm |= (
                retention["collection_fraction"]["ci95"][1]
                < thresholds["collection_fraction_minimum"]
            )
            clear_harm |= retention["coins"]["ci95"][1] < thresholds["loot_crate_coins_minimum"]
            clear_harm |= peaceful_self["ci95"][0] > thresholds["self_kills_maximum"]
        latency = {}
        for replica in (1, 2, 3):
            times = [
                t
                for row in data[replica, arm, "latency"]
                for t in row["native"]["decision_times_ms"]
            ]
            require(bool(times), "Missing latency observations")
            latency[replica] = {"p95_ms": float(np.percentile(times, 95)), "max_ms": max(times)}
        gates["latency"] = all(
            item["p95_ms"] <= cfg["latency"]["p95_limit_ms"]
            and item["max_ms"] <= cfg["latency"]["maximum_limit_ms"]
            for item in latency.values()
        )
        if arm == "geometry":
            efficacy = (
                comparisons["control"]["primary"]["kills"]["estimate"]
                >= thresholds["geometry_kills_minimum"]
            )
        else:
            baseline_rate, treatment_rate = loop_data["control"]["rate"], loop_data[arm]["rate"]
            efficacy = (
                baseline_rate is not None
                and baseline_rate > 0
                and treatment_rate is not None
                and treatment_rate <= thresholds["memory_loop_ratio_maximum"] * baseline_rate
            )
        screen_gates = {
            key: value
            for key, value in gates.items()
            if thresholds["score_interval_is_screen_gate"] or not key.endswith(":score_interval")
        }
        output["arms"][arm] = {
            "comparisons": comparisons,
            "gates": screen_gates,
            "diagnostic_gates": gates,
            "warnings": [
                key for key, value in gates.items() if not value and key.endswith(":score_interval")
            ],
            "efficacy_signal": bool(efficacy),
            "classification": classify(screen_gates, efficacy, clear_harm),
            "latency": latency,
        }
    output["next_confirmation_arm"] = next(
        (
            arm
            for arm in ("geometry", "memory")
            if output["arms"][arm]["classification"] == "eligible_for_confirmation"
        ),
        None,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.roots)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: row["classification"] for arm, row in result["arms"].items()}))


if __name__ == "__main__":
    main()
