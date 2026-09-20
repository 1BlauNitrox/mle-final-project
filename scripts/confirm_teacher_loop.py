"""Independent paired-world confirmation; no automatic submission promotion."""

import numpy as np

from scripts.analyze_hunting_curriculum import difference, latency_gate, rows
from scripts.analyze_teacher_loop import loop_rate
from scripts.curriculum_io import require


def confirmation(candidate, reference, cfg):
    report = {"passed": False, "gates": {}, "effects": {}, "submission_promotion": False}
    limits = cfg["screen"]
    for suite, setting in cfg["evaluation"]["confirmation"].items():
        for directory in (candidate, reference):
            observations = rows(directory, suite)
            require(
                [r["world_seed"] for r in observations]
                == list(range(setting["seeds"][0], setting["seeds"][1] + 1)),
                "Confirmation seeds/count changed",
            )
            require(
                all(
                    r["scenario"] == setting["scenario"]
                    and r["opponents"] == setting["opponents"]
                    and r["epsilon"] == 0
                    and r["slot"] == i % (len(setting["opponents"]) + 1)
                    for i, r in enumerate(observations)
                ),
                "Confirmation conditions changed",
            )
        if suite == "latency":
            result = latency_gate(candidate, limits)
            report["effects"][suite] = result
            report["gates"][suite] = result["passed"]
            continue
        thresholds = {
            "score": (0, "min"),
            "kills": (0, "min"),
            "survived": (-limits["survival_loss"], "min"),
            "self_kills": (limits["self_kills_increase"], "max"),
            "invalid": (limits["invalid_actions"]["pooled_increase_per_game"], "max"),
        }
        if not setting["opponents"]:
            thresholds = {
                "collection_fraction": (-limits["collection_loss"], "min"),
                "self_kills": (limits["self_kills_increase"], "max"),
            }
            report["gates"][suite + ".zero_invalid"] = all(
                r["native"]["invalid"] == 0 for r in rows(candidate, suite)
            )
        for metric, (threshold, direction) in thresholds.items():
            values = difference(candidate, reference, suite, metric)
            rng = np.random.default_rng(cfg["bootstrap_seed"])
            draws = [
                float(values[rng.integers(0, len(values), len(values))].mean())
                for _ in range(cfg["bootstrap_resamples"])
            ]
            value = float(values.mean())
            key = f"{suite}.{metric}"
            report["effects"][key] = {
                "difference": value,
                "paired_world_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
            report["gates"][key] = (
                value >= threshold - 1e-12 if direction == "min" else value <= threshold + 1e-12
            )
        if suite == "classic":
            a, b = loop_rate(rows(candidate, suite)), loop_rate(rows(reference, suite))
            report["effects"]["loops"] = {"candidate": a, "reference": b}
            report["gates"]["loops"] = bool(
                a["eligible"]
                and b["eligible"]
                and b["rate"] > 0
                and a["rate"] <= limits["loop_ratio"] * b["rate"]
            )
    report["passed"] = all(report["gates"].values())
    return report
