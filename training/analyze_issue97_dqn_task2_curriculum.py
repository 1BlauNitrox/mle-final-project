"""Validate and summarize the preregistered Issue #97 run-plan outputs.

Compares arm A (direct classic-only training, freshly run for this issue)
against arm B (Issue #86's retained unmasked arm, staged curriculum,
reused as-is and not retrained here -- see
experiments/2026-09-06-dqn-task2-curriculum/README.md). The two arms live
under different plan roots: arm A under this repository's own
training_outputs, arm B wherever #86's raw archive was extracted (its
result.json/summary.csv are committed, but the raw per-episode evidence
needed for a *new* paired comparison is the external archive named in that
experiment's README, not something this script can locate on its own).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from training.aggregate import _parse_csv_row, read_episodes_csv
from training.paired_bootstrap import paired_bootstrap
from training.run_experiment import REPOSITORY_ROOT

DIRECT_PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
DIRECT_PLAN_ID = "issue97-dqn-task2-direct-classic-unmasked"
STAGED_PLAN_ID = "issue86-dqn-task2-unmasked"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "training_outputs" / "issue97-analysis"
PRIMARY_SUITES = {
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
}
STAGED_ARM_MISSING_NOTE = (
    "The staged arm (Issue #86's retained unmasked artifact) was not available "
    "locally when this ran: its raw per-episode evidence lives in an external "
    "server archive (SHA-256 841f01f86719a28d7a9d10d69685f6293c94e281b0dd39379d0"
    "9947a4c180c1f, see experiments/2026-09-06-dqn-task2-legal-action-masking/"
    "README.md), not in this checkout. This result therefore only describes the "
    "direct arm; the registered direct-vs-staged comparison is still open. "
    "Re-run this script with --staged-plan-root pointing at that archive's "
    "extracted training_outputs/run-plans directory to complete it."
)


def analyze(
    direct_plan_root: Path = DIRECT_PLAN_ROOT,
    staged_plan_root: Path | None = None,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Compute the direct arm's own metrics, and the staged comparison if available.

    The staged arm (Issue #86's retained unmasked artifact) lives in an
    external raw archive not present in this checkout as of this run; see
    STAGED_ARM_MISSING_NOTE below. When it is missing, this still produces a
    complete direct-arm summary/result rather than failing outright, with
    the comparison fields explicitly null and `staged_arm_available: false`
    so nothing downstream mistakes an absent comparison for a null result.
    """
    direct_plan_root = Path(direct_plan_root).resolve()
    staged_plan_root = Path(staged_plan_root).resolve() if staged_plan_root else direct_plan_root
    rows: list[dict[str, Any]] = []
    deterministic_repeats = True

    arms = {
        "direct": (direct_plan_root, DIRECT_PLAN_ID),
        "staged": (staged_plan_root, STAGED_PLAN_ID),
    }
    staged_arm_available = (staged_plan_root / STAGED_PLAN_ID / "status.json").is_file()
    active_arms = arms if staged_arm_available else {"direct": arms["direct"]}

    for arm, (plan_root, plan_id) in active_arms.items():
        directory = plan_root / plan_id
        status_path = directory / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(
                f"No completed run-plan output found for the '{arm}' arm at {directory}. "
                "Run issue97-dqn-task2-direct-classic-unmasked.yaml first."
            )
        status = _read_json(status_path)
        resolved = _read_json(directory / "resolved_plan.json")
        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")

        for replica in (item["replica_id"] for item in resolved["replicas"]):
            for suite_id, scenario in PRIMARY_SUITES.items():
                primary = _suite_rows(directory, status, resolved, replica, suite_id)
                repeat = _suite_rows(
                    directory,
                    status,
                    resolved,
                    replica,
                    suite_id.replace("-primary", "-repeat"),
                )
                for primary_row, repeat_row in zip(primary, repeat, strict=True):
                    if primary_row["world_seed"] != repeat_row["world_seed"]:
                        raise ValueError("Primary and repeat world seeds differ")
                    if primary_row.get("executed_action_sequence_sha256") != repeat_row.get(
                        "executed_action_sequence_sha256"
                    ):
                        deterministic_repeats = False
                    available = primary_row.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError("Evaluation row has no available-coin count")
                    rows.append(
                        {
                            **primary_row,
                            "arm": arm,
                            "scenario": scenario,
                            "replica": replica,
                            "collection_fraction": primary_row["coins_collected"] / available,
                        }
                    )

    return _compute_and_write_result(rows, deterministic_repeats, staged_arm_available, output)


def rows_from_evidence(evidence_root: Path) -> tuple[list[dict[str, Any]], bool, bool]:
    """Rebuild the same analysis-ready rows from committed evidence CSVs.

    Reads only `evidence_root/<plan_id>/evaluation-episodes.csv` (written by
    `training/export_evidence.py`) for whichever arms have a directory
    present there -- this is what makes the committed result independently
    reproducible without the raw output tree or an external archive: a
    reviewer with only this repository checkout can run this against the
    committed evidence and get byte-identical numbers for the direct arm
    (and the staged arm too, if its evidence is ever exported alongside it).
    """
    evidence_root = Path(evidence_root).resolve()
    rows: list[dict[str, Any]] = []
    deterministic_repeats = True

    arm_plan_ids = {"direct": DIRECT_PLAN_ID, "staged": STAGED_PLAN_ID}
    staged_arm_available = (evidence_root / STAGED_PLAN_ID / "manifest.json").is_file()
    active = arm_plan_ids if staged_arm_available else {"direct": arm_plan_ids["direct"]}

    for arm, plan_id in active.items():
        by_key: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
        with (evidence_root / plan_id / "evaluation-episodes.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row_number, raw_row in enumerate(csv.DictReader(handle), start=2):
                row = _parse_csv_row(raw_row, row_number)
                row["world_seed"] = int(raw_row["world_seed"])
                row["agent_seed"] = int(raw_row["agent_seed"])
                by_key[(row["replica"], row["stage_or_suite"])][row["world_seed"]] = row

        replicas = {
            row["replica"] for rows_by_seed in by_key.values() for row in rows_by_seed.values()
        }
        for replica in replicas:
            for suite_id, scenario in PRIMARY_SUITES.items():
                primary = by_key[(replica, suite_id)]
                repeat = by_key[(replica, suite_id.replace("-primary", "-repeat"))]
                for world_seed in sorted(primary):
                    primary_row, repeat_row = primary[world_seed], repeat[world_seed]
                    if primary_row.get("executed_action_sequence_sha256") != repeat_row.get(
                        "executed_action_sequence_sha256"
                    ):
                        deterministic_repeats = False
                    available = primary_row.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError("Evaluation row has no available-coin count")
                    rows.append(
                        {
                            **primary_row,
                            "arm": arm,
                            "scenario": scenario,
                            "collection_fraction": primary_row["coins_collected"] / available,
                        }
                    )
    return rows, deterministic_repeats, staged_arm_available


def verify_from_evidence(
    evidence_root: Path, expected_result_path: Path
) -> tuple[dict[str, Any], bool]:
    """Recompute the result from evidence and compare it to the committed one."""
    rows, deterministic_repeats, staged_arm_available = rows_from_evidence(evidence_root)
    recomputed = _compute_and_write_result(
        rows, deterministic_repeats, staged_arm_available, output=None
    )
    expected = _read_json(Path(expected_result_path))
    return recomputed, recomputed == expected


def _compute_and_write_result(
    rows: list[dict[str, Any]],
    deterministic_repeats: bool,
    staged_arm_available: bool,
    output: Path | None,
) -> dict[str, Any]:
    summaries = _summaries(rows)
    timing = [
        row
        for row in rows
        if row.get("decision_time_p95_ms") is not None
        and row.get("decision_time_max_ms") is not None
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "issue": 97,
        "staged_arm_available": staged_arm_available,
        "deterministic_repeats": deterministic_repeats,
        "primary_evaluation_episodes": len(rows),
        "paired_collection_fraction_staged_minus_direct": (
            _paired_comparisons(rows, "collection_fraction") if staged_arm_available else None
        ),
        "paired_survival_rate_staged_minus_direct": (
            _paired_comparisons(rows, "survived") if staged_arm_available else None
        ),
        "timing_within_limits": bool(timing)
        and all(
            row["decision_time_p95_ms"] < 50 and row["decision_time_max_ms"] < 100
            for row in timing
        ),
    }
    if not staged_arm_available:
        result["note"] = STAGED_ARM_MISSING_NOTE
    if output is not None:
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        _write_summary(output / "summary.csv", summaries)
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def _suite_rows(
    directory: Path,
    status: dict[str, Any],
    resolved: dict[str, Any],
    replica: str,
    suite: str,
) -> list[dict[str, Any]]:
    expected = {
        job["run_id"]: job
        for job in resolved["jobs"]
        if job["kind"] == "evaluation"
        and job["replica"] == replica
        and job["stage_or_suite"] == suite
    }
    actual = {
        run_id: job
        for run_id, job in status["jobs"].items()
        if run_id in expected
    }
    if set(actual) != set(expected):
        raise ValueError(f"Evaluation matrix mismatch for {replica}/{suite}")

    rows: list[dict[str, Any]] = []
    for run_id in sorted(actual):
        job = actual[run_id]
        attempts = job.get("attempts", [])
        if job.get("status") != "completed" or not attempts:
            raise ValueError(f"Evaluation job is not complete: {run_id}")
        run_directory = directory / attempts[-1]["output"]
        episode_rows = read_episodes_csv(run_directory / "episodes.csv")
        if len(episode_rows) != 1:
            raise ValueError(f"Expected exactly one episode row: {run_id}")
        metadata = _read_json(run_directory / "metadata.json")
        registered = expected[run_id]
        for field in ("world_seed", "agent_seed", "scenario", "rounds", "opponents"):
            if metadata.get(field) != registered[field]:
                raise ValueError(f"Metadata mismatch for {run_id}: {field}")
        if metadata.get("mode") != "evaluation":
            raise ValueError(f"Job is not evaluation mode: {run_id}")
        row = dict(episode_rows[0])
        row["world_seed"] = metadata["world_seed"]
        rows.append(row)
    return rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["arm"])].append(row)
    result: list[dict[str, Any]] = []
    for scenario in PRIMARY_SUITES.values():
        for arm in ("direct", "staged"):
            group = groups[(scenario, arm)]
            if not group:
                continue
            episodes = len(group)
            attempted = sum(row["attempted_actions"] for row in group)
            coins = sum(row["coins_collected"] for row in group)
            survival_steps = sum(row["survival_steps"] for row in group)
            decision_medians = [
                row["decision_time_median_ms"]
                for row in group
                if row["decision_time_median_ms"] is not None
            ]
            decision_p95 = [
                row["decision_time_p95_ms"]
                for row in group
                if row["decision_time_p95_ms"] is not None
            ]
            decision_max = [
                row["decision_time_max_ms"]
                for row in group
                if row["decision_time_max_ms"] is not None
            ]
            result.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "episodes": episodes,
                    "mean_collection_fraction": fmean(row["collection_fraction"] for row in group),
                    "survival_rate": sum(row["survived"] for row in group) / episodes,
                    "self_kill_rate": sum(row["self_kills"] for row in group) / episodes,
                    "invalid_action_rate": sum(row["invalid_actions"] for row in group) / attempted,
                    "mean_coins": coins / episodes,
                    "mean_crates_destroyed": fmean(row["crates_destroyed"] for row in group),
                    "mean_survival_steps": survival_steps / episodes,
                    "steps_per_coin": survival_steps / coins if coins else None,
                    "action_up": sum(row["action_up"] for row in group),
                    "action_right": sum(row["action_right"] for row in group),
                    "action_down": sum(row["action_down"] for row in group),
                    "action_left": sum(row["action_left"] for row in group),
                    "action_wait": sum(row["action_wait"] for row in group),
                    "action_bomb": sum(row["action_bomb"] for row in group),
                    "action_unknown": sum(row["action_unknown"] for row in group),
                    "decision_time_median_ms": median(decision_medians)
                    if decision_medians
                    else None,
                    "decision_time_p95_ms": max(decision_p95) if decision_p95 else None,
                    "decision_time_max_ms": max(decision_max) if decision_max else None,
                }
            )
    return result


def _paired_comparisons(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, dict[str, float | int]]:
    comparisons: dict[str, dict[str, float | int]] = {}
    for scenario in PRIMARY_SUITES.values():
        arms: dict[str, dict[str, dict[int, float]]] = {"direct": {}, "staged": {}}
        for row in rows:
            if row["scenario"] != scenario:
                continue
            arms[row["arm"]].setdefault(row["replica"], {})[row["world_seed"]] = float(
                row[metric]
            )
        comparisons[scenario] = paired_bootstrap(
            arms["staged"], arms["direct"], resampler_seed=97
        ).as_dict()
    return comparisons


def _write_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-plan-root", type=Path, default=DIRECT_PLAN_ROOT)
    parser.add_argument("--staged-plan-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-from-evidence",
        type=Path,
        default=None,
        metavar="EVIDENCE_ROOT",
        help=(
            "Instead of reading the raw job tree, recompute the result from "
            "committed evidence CSVs under EVIDENCE_ROOT (see "
            "training/export_evidence.py) and compare it to --output's "
            "committed result.json. Exits non-zero on any mismatch."
        ),
    )
    args = parser.parse_args()
    if args.verify_from_evidence is not None:
        recomputed, matches = verify_from_evidence(
            args.verify_from_evidence, args.output / "result.json"
        )
        print(json.dumps(recomputed, indent=2, sort_keys=True))
        print("MATCHES committed result.json" if matches else "MISMATCH vs committed result.json")
        return 0 if matches else 1
    print(
        json.dumps(
            analyze(args.direct_plan_root, args.staged_plan_root, args.output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
