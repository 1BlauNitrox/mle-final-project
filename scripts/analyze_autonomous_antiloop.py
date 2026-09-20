"""Analyze the prospectively registered frozen-policy anti-loop comparison."""

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


def stage_rows(root, stage, arm, suite):
    return read(root / "results" / stage / arm / f"{suite}.json")["rows"]


def validate_stage(root, cfg, stage):
    setting = cfg["evaluation"][stage]
    for arm in cfg["arms"]:
        for suite, specification in setting.items():
            path = root / "results" / stage / arm / f"{suite}.json"
            require(path.exists(), f"Missing {stage}/{arm}/{suite}")
            observations = read(path)["rows"]
            seeds = list(range(specification["seeds"][0], specification["seeds"][1] + 1))
            require([row["world_seed"] for row in observations] == seeds, "Wrong seed set")
            require(
                all(
                    row["slot"] == index % (len(specification["opponents"]) + 1)
                    and row["opponents"] == specification["opponents"]
                    and row["scenario"] == specification["scenario"]
                    and row["epsilon"] == 0
                    for index, row in enumerate(observations)
                ),
                "Evaluation conditions changed",
            )
    for suite in setting:
        left = stage_rows(root, stage, "control", suite)
        right = stage_rows(root, stage, "narrow_guard", suite)
        require(
            [(r["world_seed"], r["slot"], r["opponents"]) for r in left]
            == [(r["world_seed"], r["slot"], r["opponents"]) for r in right],
            "Unpaired evaluation",
        )


def loop_rate(observations):
    eligible = sum(row["loop_windows"]["eligible"] for row in observations)
    looping = sum(row["loop_windows"]["looping"] for row in observations)
    return {
        "eligible": eligible,
        "looping": looping,
        "rate": looping / eligible if eligible else None,
    }


def contrast(root, stage, suites, name):
    values = []
    for suite in suites:
        candidate = stage_rows(root, stage, "narrow_guard", suite)
        control = stage_rows(root, stage, "control", suite)
        values.extend(
            metric(a, name) - metric(b, name) for a, b in zip(candidate, control, strict=True)
        )
    return np.asarray(values, dtype=float)


def interval(values, cfg, offset):
    values = np.asarray(values, dtype=float)
    require(values.size > 0, "Empty contrast")
    rng = np.random.default_rng(cfg["bootstrap_seed"] + offset)
    draws = np.empty(cfg["bootstrap_resamples"])
    for index in range(len(draws)):
        draws[index] = rng.choice(values, len(values), replace=True).mean()
    return {
        "difference": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "games": int(values.size),
    }


def diagnostics(root, cfg, stage, arm):
    result = {}
    for suite in cfg["evaluation"][stage]:
        observations = stage_rows(root, stage, arm, suite)
        result[suite] = {
            "games": len(observations),
            "score_mean": float(np.mean([metric(row, "score") for row in observations])),
            "kills_mean": float(np.mean([metric(row, "kills") for row in observations])),
            "survival_rate": float(np.mean([metric(row, "survived") for row in observations])),
            "self_kills_mean": float(np.mean([metric(row, "self_kills") for row in observations])),
            "invalid_mean": float(np.mean([metric(row, "invalid") for row in observations])),
            "loops": loop_rate(observations),
            "guard": {
                key: sum(row["narrow_loop_guard"][key] for row in observations)
                for key in ("eligible", "overrides", "rejected_contested", "rejected_no_candidate")
            },
            "guarded_games": sum(row["narrow_loop_guard"]["overrides"] > 0 for row in observations),
        }
    return result


def gate_stage(root, cfg, stages, *, latency=False):
    limits = cfg["screen"]
    multi = limits["multiplayer_suites"]
    def all_stage_rows(stage, arm, suites):
        return [row for suite in suites for row in stage_rows(root, stage, arm, suite)]
    gates, effects, loop_reports = {}, {}, {}
    offsets = 0
    for name in ("score", "kills", "survived", "self_kills", "invalid"):
        values = np.concatenate([contrast(root, stage, multi, name) for stage in stages])
        effects[name] = interval(values, cfg, offsets)
        offsets += 1
    gates["score"] = effects["score"]["difference"] >= limits["score_gain"] - 1e-12
    gates["kills"] = effects["kills"]["difference"] >= limits["kills_gain"] - 1e-12
    gates["survival"] = effects["survived"]["difference"] >= -limits["survival_loss"] - 1e-12
    gates["self_kills"] = (
        effects["self_kills"]["difference"] <= limits["self_kills_increase"] + 1e-12
    )
    gates["invalid_pooled"] = effects["invalid"]["difference"] <= (
        limits["invalid_actions"]["pooled_increase_per_game"] + 1e-12
    )
    invalid_blocks = []
    guarded_games = 0
    for stage in stages:
        invalid = float(contrast(root, stage, multi, "invalid").mean())
        invalid_blocks.append(invalid)
        gates[f"{stage}.invalid"] = invalid <= (
            limits["invalid_actions"]["block_increase_per_game"] + 1e-12
        )
        candidate = loop_rate(all_stage_rows(stage, "narrow_guard", limits["loop_suites"]))
        control = loop_rate(all_stage_rows(stage, "control", limits["loop_suites"]))
        loop_reports[stage] = {"candidate": candidate, "control": control}
        gates[f"{stage}.loops"] = bool(
            candidate["eligible"]
            and control["eligible"]
            and control["rate"]
            and candidate["rate"] <= limits["loop_ratio"] * control["rate"]
        )
        guarded_games += sum(
            row["narrow_loop_guard"]["overrides"] > 0
            for row in all_stage_rows(stage, "narrow_guard", limits["loop_suites"])
        )
    gates["guard_exposure"] = guarded_games >= limits["minimum_guarded_games"]
    effects["invalid_block_differences"] = invalid_blocks
    effects["guarded_games"] = guarded_games
    for suite in ("coins", "crates"):
        values = np.concatenate(
            [contrast(root, stage, [suite], "collection_fraction") for stage in stages]
        )
        effects[f"{suite}.collection_fraction"] = interval(values, cfg, offsets)
        offsets += 1
        gates[f"{suite}.retention"] = values.mean() >= -limits["collection_loss"] - 1e-12
        self_kills = np.concatenate(
            [contrast(root, stage, [suite], "self_kills") for stage in stages]
        )
        effects[f"{suite}.self_kills"] = interval(self_kills, cfg, offsets)
        offsets += 1
        gates[f"{suite}.self_kills"] = self_kills.mean() <= limits["self_kills_increase"] + 1e-12
        candidate_invalid = sum(
            metric(row, "invalid")
            for stage in stages
            for row in stage_rows(root, stage, "narrow_guard", suite)
        )
        effects[f"{suite}.candidate_invalid_total"] = candidate_invalid
        gates[f"{suite}.zero_invalid"] = (
            candidate_invalid <= limits["invalid_actions"]["solo_maximum_total"]
        )
    if latency:
        observations = stage_rows(root, stages[0], "narrow_guard", "latency")
        times = [value for row in observations for value in row["native"]["decision_times_ms"]]
        require(times, "No latency observations")
        effects["latency"] = {"p95_ms": float(np.quantile(times, 0.95)), "max_ms": max(times)}
        gates["latency"] = (
            effects["latency"]["p95_ms"] < limits["latency_p95_ms"]
            and effects["latency"]["max_ms"] < limits["latency_max_ms"]
        )
    return {
        "passed": all(bool(value) for value in gates.values()),
        "gates": {key: bool(value) for key, value in gates.items()},
        "effects": effects,
        "loops": loop_reports,
    }


def analyze(root):
    root = Path(root)
    cfg = read(root / "config.json")
    pilot_stages = ("pilot_r1", "pilot_r2")
    report = {"complete": False, "eligible": False, "submission_promotion": False}
    if not all(
        (root / "results" / stage / arm).exists() for stage in pilot_stages for arm in cfg["arms"]
    ):
        return report
    for stage in pilot_stages:
        validate_stage(root, cfg, stage)
    report.update(
        complete=True,
        pilot=gate_stage(root, cfg, pilot_stages),
        diagnostics={
            stage: {arm: diagnostics(root, cfg, stage, arm) for arm in cfg["arms"]}
            for stage in pilot_stages
        },
    )
    report["eligible"] = report["pilot"]["passed"]
    confirmation = root / "results" / "confirmation"
    if confirmation.exists():
        validate_stage(root, cfg, "confirmation")
        report["confirmation"] = gate_stage(root, cfg, ("confirmation",), latency=True)
        report["confirmation_diagnostics"] = {
            arm: diagnostics(root, cfg, "confirmation", arm) for arm in cfg["arms"]
        }
        report["confirmed_for_human_review"] = bool(
            report["eligible"] and report["confirmation"]["passed"]
        )
    return report
