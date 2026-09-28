"""Prospective three-replica stability gates; no submission promotion."""

import gzip
import json

import numpy as np

from scripts.run_stability_test import read, require, sha


def metric(row, key):
    native = row["native"]
    if key == "collection_fraction":
        return native["coins"] / native["initially_available_coins"]
    return float(native[key])


def interval(matrix, seed, samples):
    values = np.asarray(matrix)
    require(values.ndim == 2 and values.shape[0] == 3, "Three paired replicas required")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        replicas = rng.integers(0, 3, 3)
        worlds = rng.integers(0, values.shape[1], values.shape[1])
        draws.append(values[np.ix_(replicas, worlds)].mean())
    return {
        "difference": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "replica_differences": values.mean(axis=1).tolist(),
    }


def analyze(root):
    # Exported evidence can be analyzed without reconstructing an executable runtime.
    cfg, binding = read(root / "config.json"), read(root / "binding.json")
    require(sha(root / "config.json") == binding["config_sha256"], "Changed protocol")
    require(cfg["issue"] == 213, "Wrong experiment")
    artifacts = [f"{arm}-r{r}" for arm in cfg["arms"] for r in range(1, 4)]
    model_hashes = {"reference": cfg["reference_sha256"]}
    training = {}
    for artifact in artifacts:
        directory = root / "training" / artifact
        result = read(directory / "result.json")
        require(result["complete"] and result["episodes"] == cfg["episodes"], "Incomplete training")
        require(sha(directory / "final.pt") == result["checkpoint_sha256"], "Wrong final artifact")
        require(
            sha(directory / "episodes.json.gz") == result["rows_sha256"], "Corrupt training rows"
        )
        with gzip.open(directory / "episodes.json.gz", "rt", encoding="utf-8") as stream:
            rows = json.load(stream)
        replica = int(artifact.rsplit("-r", 1)[1])
        first, last = cfg["train_ranges"][replica - 1]
        require(
            [r["world_seed"] for r in rows] == list(range(first, last + 1)), "Wrong training seeds"
        )
        require(
            all(
                r["opponents"] == cfg["training_opponents"] and r["slot"] == i % 4
                for i, r in enumerate(rows)
            ),
            "Wrong training opponents/slots",
        )
        from scripts.run_stability_test import epsilon

        require(
            all(
                r["epsilon"] == epsilon(cfg["learner_seeds"][replica - 1], i)
                for i, r in enumerate(rows)
            ),
            "Wrong exploration",
        )
        model_hashes[artifact] = result["checkpoint_sha256"]
        training[artifact] = {
            "updates": sum(r["optimizer_updates_this_episode"] for r in rows),
            "episodes": len(rows),
        }
    data, summary, times = {}, {}, []
    for artifact in ["reference", *artifacts]:
        summary[artifact] = {}
        for suite, setting in cfg["evaluation"].items():
            directory = root / "evaluation" / artifact / suite
            result = read(directory / "result.json")
            require(
                result["complete"] and result["checkpoint_sha256"] == model_hashes[artifact],
                "Wrong evaluated artifact",
            )
            require(
                sha(directory / "episodes.json.gz") == result["rows_sha256"],
                "Corrupt evaluation rows",
            )
            with gzip.open(directory / "episodes.json.gz", "rt", encoding="utf-8") as stream:
                rows = json.load(stream)
            first, last = setting["seeds"]
            require(
                [r["world_seed"] for r in rows] == list(range(first, last + 1)),
                "Wrong evaluation worlds",
            )
            require(
                all(
                    r["opponents"] == setting["opponents"]
                    and r["slot"] == i % (len(setting["opponents"]) + 1)
                    and r["epsilon"] == 0
                    and r["checkpoint_sha256"] == model_hashes[artifact]
                    for i, r in enumerate(rows)
                ),
                "Unpaired evaluation",
            )
            data[artifact, suite] = rows
            summary[artifact][suite] = {
                m: float(np.mean([metric(r, m) for r in rows]))
                for m in [
                    "score",
                    "kills",
                    "self_kills",
                    "survived",
                    "coins",
                    "invalid",
                    "collection_fraction",
                ]
            }
            if suite == "latency":
                times.extend(t for row in rows for t in row["native"]["decision_times_ms"])

    def contrast(baseline, suite, name):
        matrix = [
            [
                metric(a, name) - metric(b, name)
                for a, b in zip(
                    data[f"preserve-r{r}", suite],
                    data["reference" if baseline == "reference" else f"reset-r{r}", suite],
                    strict=True,
                )
            ]
            for r in range(1, 4)
        ]
        return interval(matrix, cfg["bootstrap_seed"], cfg["bootstrap_resamples"])

    contrasts = {
        b: {
            s: {
                m: contrast(b, s, m)
                for m in [
                    "score",
                    "kills",
                    "self_kills",
                    "survived",
                    "coins",
                    "collection_fraction",
                ]
            }
            for s in cfg["evaluation"]
            if s != "latency"
        }
        for b in ["reference", "reset"]
    }
    gates = cfg["gates"]
    ref, reset = contrasts["reference"], contrasts["reset"]
    checks = {
        "score_reference": ref["classic"]["score"]["difference"] >= gates["score_vs_reference"],
        "self_kills_reference": ref["classic"]["self_kills"]["difference"]
        <= gates["self_kills_increase"],
        "survival_reference": ref["classic"]["survived"]["difference"] >= -gates["survival_loss"],
        "coin_retention": ref["coins"]["collection_fraction"]["difference"]
        >= -gates["coin_fraction_loss"],
        "crate_retention": ref["crates"]["coins"]["difference"] >= -gates["crate_coins_loss"],
        "crate_self_kills": ref["crates"]["self_kills"]["difference"]
        <= gates["crate_self_kills_increase"],
        "score_reset": reset["classic"]["score"]["difference"] >= gates["score_vs_reset"],
        "replicas_improve": sum(v > 0 for v in reset["classic"]["score"]["replica_differences"])
        >= gates["improving_replicas"],
        "latency": float(np.quantile(times, 0.95)) < gates["latency_p95_ms"]
        and max(times) < gates["latency_max_ms"],
    }
    return {
        "complete": True,
        "issue": 213,
        "config_sha256": binding["config_sha256"],
        "eligible_for_further_stability_testing": all(checks.values()),
        "submission_promotion": False,
        "gates": checks,
        "contrasts": contrasts,
        "summary": summary,
        "training": training,
        "latency": {"p95_ms": float(np.quantile(times, 0.95)), "max_ms": max(times)},
        "limitation": (
            "Replay and Adam are one bundled factor. Equal episodes need not mean equal updates. "
            "Changed LR/runtime versus #211 prevent attribution of between-experiment differences."
        ),
    }
