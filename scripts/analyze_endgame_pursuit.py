"""Analyze the prospectively registered learned endgame-pursuit experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.curriculum_io import read, require


def metric(row, name):
    native = row["native"]
    if name == "collection_fraction":
        require(native["initially_available_coins"] > 0, "Undefined collection fraction")
        return native["coins"] / native["initially_available_coins"]
    return float(native[name])


def rows(root, stage, arm, suite):
    return read(Path(root) / "results" / stage / arm / f"{suite}.json")["rows"]


def loop_rate(values):
    eligible = sum(row["broad_loop_windows"]["eligible"] for row in values)
    looping = sum(row["broad_loop_windows"]["looping"] for row in values)
    return {
        "eligible": eligible,
        "looping": looping,
        "rate": looping / eligible if eligible else None,
    }


def diagnostics(root, cfg, stage, arm):
    result = {}
    for suite in cfg["evaluation"][stage]:
        values = rows(root, stage, arm, suite)
        result[suite] = {
            "games": len(values),
            "score": float(np.mean([metric(row, "score") for row in values])),
            "kills": float(np.mean([metric(row, "kills") for row in values])),
            "survival": float(np.mean([metric(row, "survived") for row in values])),
            "self_kills": float(np.mean([metric(row, "self_kills") for row in values])),
            "invalid": float(np.mean([metric(row, "invalid") for row in values])),
            "collection_fraction": (
                float(np.mean([metric(row, "collection_fraction") for row in values]))
                if suite in {"coins", "crates"}
                else None
            ),
            "loops": loop_rate(values),
            "pursuit": {
                key: sum(row["endgame_pursuit_guard"].get(key, 0) for row in values)
                for key in (
                    "eligible",
                    "overrides",
                    "rejected_confidence",
                    "rejected_bomb_safety",
                    "rejected_bomb_override",
                    "bomb_overrides",
                    "rejected_movement_override",
                    "rejected_bomb_rank",
                    "rejected_bomb_limit",
                    "rejected_target_mobility",
                    "escape_checks",
                    "escape_redirects",
                    "escape_rejected_no_candidate",
                )
            },
        }
    return result


def gate(root, cfg, stage, candidate, *, latency=False):
    screen = cfg["screen"]
    multiplayer = screen["multiplayer_suites"]

    def contrast(name, suites):
        output = []
        for suite in suites:
            left, right = rows(root, stage, candidate, suite), rows(root, stage, "baseline", suite)
            require(len(left) == len(right), "Unpaired evaluation")
            output.extend(
                metric(a, name) - metric(b, name) for a, b in zip(left, right, strict=True)
            )
        return np.asarray(output, dtype=float)

    effects = {
        name: float(contrast(name, multiplayer).mean())
        for name in ("score", "kills", "survived", "self_kills", "invalid")
    }
    gates = {
        "kills": effects["kills"] >= screen["kills_gain"] - 1e-12,
        "score": effects["score"] >= screen["score_gain"] - 1e-12,
        "survival": effects["survived"] >= -screen["survival_loss"] - 1e-12,
        "self_kills": effects["self_kills"] <= screen["self_kills_increase"] + 1e-12,
        "invalid_pooled": effects["invalid"]
        <= screen["invalid_actions"]["pooled_increase_per_game"] + 1e-12,
    }
    for suite in multiplayer:
        value = float(contrast("invalid", [suite]).mean())
        effects[f"{suite}.invalid"] = value
        gates[f"{suite}.invalid"] = (
            value <= screen["invalid_actions"]["block_increase_per_game"] + 1e-12
        )

    candidate_values = [row for suite in multiplayer for row in rows(root, stage, candidate, suite)]
    baseline_values = [row for suite in multiplayer for row in rows(root, stage, "baseline", suite)]
    candidate_loop, baseline_loop = loop_rate(candidate_values), loop_rate(baseline_values)
    effects["loops"] = {"candidate": candidate_loop, "baseline": baseline_loop}
    gates["loops"] = bool(
        candidate_loop["eligible"]
        and baseline_loop["eligible"]
        and candidate_loop["rate"] is not None
        and baseline_loop["rate"] is not None
        and candidate_loop["rate"]
        <= max(screen["loop_absolute_rate"], baseline_loop["rate"] * screen["loop_ratio"])
    )
    overrides = sum(row["endgame_pursuit_guard"]["overrides"] for row in candidate_values)
    effects["pursuit_overrides"] = overrides
    gates["pursuit_exposure"] = overrides >= screen["minimum_pursuit_overrides"]

    for suite in ("coins", "crates"):
        collection = float(contrast("collection_fraction", [suite]).mean())
        self_kills = float(contrast("self_kills", [suite]).mean())
        invalid_total = sum(metric(row, "invalid") for row in rows(root, stage, candidate, suite))
        effects[f"{suite}.collection"] = collection
        effects[f"{suite}.self_kills"] = self_kills
        effects[f"{suite}.invalid_total"] = invalid_total
        gates[f"{suite}.collection"] = collection >= -screen["collection_loss"] - 1e-12
        gates[f"{suite}.self_kills"] = self_kills <= screen["self_kills_increase"] + 1e-12
        gates[f"{suite}.invalid"] = invalid_total <= screen["invalid_actions"]["solo_maximum_total"]

    if latency:
        times = [
            value
            for row in rows(root, stage, candidate, "latency")
            for value in row["native"]["decision_times_ms"]
        ]
        require(times, "No latency decisions")
        effects["latency"] = {
            "p95_ms": float(np.quantile(times, 0.95)),
            "max_ms": float(max(times)),
        }
        gates["latency"] = (
            effects["latency"]["p95_ms"] < screen["latency_p95_ms"]
            and effects["latency"]["max_ms"] < screen["latency_max_ms"]
        )
    return {"passed": all(gates.values()), "gates": gates, "effects": effects}


def analyze(root):
    root = Path(root)
    cfg = read(root / "config.json")
    training = read(root / "training.json")
    candidates = list(training.get("candidates", {}))
    report = {"complete": False, "eligible": [], "selected": None, "submission_promotion": False}
    if not training.get("classifier_accepted"):
        return {
            **report,
            "complete": True,
            "reason": "Grouped OOF pursuit gate failed",
            "training": training,
        }
    required = [
        root / "results" / "pilot" / arm / f"{suite}.json"
        for arm in ["baseline", *candidates]
        for suite in cfg["evaluation"]["pilot"]
    ]
    if not all(path.exists() for path in required):
        return report
    report["complete"] = True
    report["training"] = training
    report["diagnostics"] = {
        arm: diagnostics(root, cfg, "pilot", arm) for arm in ["baseline", *candidates]
    }
    report["pilot"] = {arm: gate(root, cfg, "pilot", arm) for arm in candidates}
    report["eligible"] = [arm for arm in candidates if report["pilot"][arm]["passed"]]
    if report["eligible"]:
        report["selected"] = max(
            report["eligible"],
            key=lambda arm: (
                report["pilot"][arm]["effects"]["kills"],
                report["pilot"][arm]["effects"]["score"],
                training["candidates"][arm]["threshold"],
            ),
        )
    if report["selected"] and (root / "results" / "confirmation").exists():
        report["confirmation"] = gate(root, cfg, "confirmation", report["selected"], latency=True)
        report["confirmation_diagnostics"] = {
            arm: diagnostics(root, cfg, "confirmation", arm)
            for arm in ("baseline", report["selected"])
        }
        report["confirmed_for_human_review"] = bool(report["confirmation"]["passed"])
    return report
