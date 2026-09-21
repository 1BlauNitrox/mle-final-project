"""Export and evaluate the final tabular freeze confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "training_outputs/run-plans/issue230-final-tabular-freeze-confirmation"
)
EXPERIMENT = (
    ROOT
    / "experiments/2026-09-21-final-tabular-freeze-"
    "DerKleineKonkurrenzvernichter"
)
MODEL = ROOT / "agent_code/DerKleineKonkurrenzvernichter/model.npz"
EVIDENCE = EXPERIMENT / "evidence.csv"
SUMMARY = EXPERIMENT / "summary.csv"
RESULT = EXPERIMENT / "result.json"

EXPECTED_SUITES = (
    "competitive",
    "mixed",
    "peaceful",
    "classic",
    "coin-heaven",
    "loot-crate",
)
REPEAT_FIELDS = (
    "score",
    "coins_collected",
    "self_kills",
    "opponents_eliminated",
    "survived",
    "termination_reason",
    "executed_action_sequence_sha256",
    "first_place",
    "tied_first",
)


def _number(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value else 0.0


def _integer(row: dict[str, str], name: str) -> int:
    return int(_number(row, name))


def _boolean(row: dict[str, str], name: str) -> bool:
    return row.get(name, "").lower() == "true"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_rows(run_root: Path = RUN_ROOT) -> list[dict[str, Any]]:
    """Collect the single observed-agent row from every completed job."""
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError("Freeze confirmation run plan is not complete")

    rows: list[dict[str, Any]] = []
    for job_name, job in sorted(status["jobs"].items()):
        if job["status"] != "completed" or len(job["attempts"]) != 1:
            raise ValueError(f"Job is not singly completed: {job_name}")
        attempt = run_root / job["attempts"][0]["output"]
        metadata = json.loads((attempt / "metadata.json").read_text(encoding="utf-8"))
        with (attempt / "episodes.csv").open(newline="", encoding="utf-8") as handle:
            episodes = list(csv.DictReader(handle))
        observed = metadata["observed_agent"]
        observed_rows = [row for row in episodes if row["agent"] == observed]
        if len(observed_rows) != 1:
            raise ValueError(f"Expected one observed-agent episode for {job_name}")
        episode = observed_rows[0]
        stage = metadata["run_plan"]["stage_or_suite"]
        phase = "repeat" if stage.endswith("-repeat") else "primary"
        suite = stage.removesuffix("-primary").removesuffix("-repeat")
        available = _integer(episode, "initially_available_coins")
        collected = _integer(episode, "coins_collected")
        rows.append(
            {
                "suite": suite,
                "phase": phase,
                "world_seed": metadata["world_seed"],
                "agent_seed": metadata["agent_seed"],
                "score": _number(episode, "score"),
                "coins_collected": collected,
                "initially_available_coins": available,
                "collection_fraction": collected / available if available else 0.0,
                "self_kills": _integer(episode, "self_kills"),
                "opponents_eliminated": _integer(episode, "opponents_eliminated"),
                "survived": _boolean(episode, "survived"),
                "termination_reason": episode["termination_reason"],
                "executed_action_sequence_sha256": episode[
                    "executed_action_sequence_sha256"
                ],
                "first_place": _boolean(episode, "first_place"),
                "tied_first": _boolean(episode, "tied_first"),
                "decision_time_p95_ms": _number(episode, "decision_time_p95_ms"),
                "decision_time_max_ms": _number(episode, "decision_time_max_ms"),
            }
        )
    if len(rows) != 240:
        raise ValueError(f"Expected 240 rows, found {len(rows)}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate primary confirmation rows by suite."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "primary":
            groups[row["suite"]].append(row)
    if set(groups) != set(EXPECTED_SUITES):
        raise ValueError("Confirmation suites are incomplete")

    output = []
    for suite in EXPECTED_SUITES:
        suite_rows = groups[suite]
        if len(suite_rows) != 20:
            raise ValueError(f"Expected 20 primary rows for {suite}")
        output.append(
            {
                "suite": suite,
                "rounds": len(suite_rows),
                "mean_score": fmean(row["score"] for row in suite_rows),
                "collection_fraction": fmean(
                    row["collection_fraction"] for row in suite_rows
                ),
                "self_kill_rate": fmean(row["self_kills"] for row in suite_rows),
                "eliminations_per_round": fmean(
                    row["opponents_eliminated"] for row in suite_rows
                ),
                "survival_rate": fmean(row["survived"] for row in suite_rows),
                "first_place_rate": fmean(row["first_place"] for row in suite_rows),
                "tied_first_rate": fmean(row["tied_first"] for row in suite_rows),
                "decision_time_p95_ms_max": max(
                    row["decision_time_p95_ms"] for row in suite_rows
                ),
                "decision_time_max_ms_max": max(
                    row["decision_time_max_ms"] for row in suite_rows
                ),
            }
        )
    return output


def repeat_mismatches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return fixed-seed primary/repeat disagreements."""
    primary = {
        (row["suite"], row["world_seed"], row["agent_seed"]): row
        for row in rows
        if row["phase"] == "primary"
    }
    mismatches = []
    for row in rows:
        if row["phase"] != "repeat":
            continue
        key = (row["suite"], row["world_seed"], row["agent_seed"])
        original = primary.get(key)
        if original is None:
            raise ValueError(f"Repeat has no primary row: {key}")
        changed = [field for field in REPEAT_FIELDS if row[field] != original[field]]
        if changed:
            mismatches.append({"key": list(key), "fields": changed})
    if len(primary) != 120:
        raise ValueError("Expected 120 primary rows")
    return mismatches


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the freeze criteria fixed before the run."""
    summaries = summarize(rows)
    by_suite = {row["suite"]: row for row in summaries}
    mismatches = repeat_mismatches(rows)
    aggregate_self_kill = fmean(
        row["self_kill_rate"] for row in summaries
    )
    max_p95 = max(row["decision_time_p95_ms_max"] for row in summaries)
    max_latency = max(row["decision_time_max_ms_max"] for row in summaries)
    gates = {
        "classic_collection_at_least_0_15": (
            by_suite["classic"]["collection_fraction"] >= 0.15
        ),
        "coin_heaven_collection_at_least_0_90": (
            by_suite["coin-heaven"]["collection_fraction"] >= 0.90
        ),
        "loot_crate_collection_at_least_0_20": (
            by_suite["loot-crate"]["collection_fraction"] >= 0.20
        ),
        "peaceful_eliminations_at_least_0_10": (
            by_suite["peaceful"]["eliminations_per_round"] >= 0.10
        ),
        "aggregate_self_kill_at_most_0_20": aggregate_self_kill <= 0.20,
        "p95_latency_at_most_50_ms": max_p95 <= 50.0,
        "max_latency_at_most_100_ms": max_latency <= 100.0,
        "exact_repeats": not mismatches,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "issue": 230,
        "plan_id": "issue230-final-tabular-freeze-confirmation",
        "candidate": "installed_task3_incumbent",
        "model": {
            "path": "agent_code/DerKleineKonkurrenzvernichter/model.npz",
            "sha256": _sha256(MODEL),
            "size_bytes": MODEL.stat().st_size,
        },
        "primary_rows": 120,
        "repeat_rows": 120,
        "repeat_mismatch_count": len(mismatches),
        "repeat_mismatches": mismatches,
        "aggregate_self_kill_rate": aggregate_self_kill,
        "maximum_suite_p95_ms": max_p95,
        "maximum_decision_time_ms": max_latency,
        "summaries": summaries,
        "gates": gates,
        "passed": passed,
        "decision": (
            "freeze_installed_task3_incumbent"
            if passed
            else "do_not_freeze_confirmation_failed"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Write the evidence and print the decision."""
    rows = collect_rows()
    result = analyze(rows)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    _write_csv(EVIDENCE, rows)
    _write_csv(SUMMARY, result["summaries"])
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"], indent=2, sort_keys=True))
    print(f"Overall pass: {result['passed']}")


if __name__ == "__main__":
    main()
