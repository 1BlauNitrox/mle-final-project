"""Apply the registered decision rule of the survival-guard factorial.

Reads one games.jsonl per cell, pairs every guarded game with the control game
on the same checkpoint and the same world, and reports a hierarchical paired
bootstrap that resamples the checkpoints first and then the worlds, because the
checkpoint level is what dominates this kind of contrast.

The untrained reference is played for context and never enters the decision.
Only the primary contrast decides; the rest is description.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRATION = REPO_ROOT / "experiments/2026-09-18-survival-guard-factorial/config.json"
CONTROL = "control"
REFERENCE = "reference"
METRICS = ("score", "self_kills", "survived", "kills", "coins", "collection_fraction", "invalid")


def load(folder: Path, expected: str) -> list[dict]:
    log = folder / "games.jsonl"
    if not log.is_file():
        raise SystemExit(f"missing {log}")
    games = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    labels = {game.get("variant", "") for game in games}
    if labels != {expected}:
        raise SystemExit(f"{log} holds variant {sorted(labels)}, expected [{expected!r}]")
    return games


def table(games: list[dict], metric: str) -> dict[tuple[str, int], float]:
    values = {}
    for game in games:
        value = game[metric]
        if value is not None:
            values[(game["artifact"], game["world_seed"])] = float(value)
    return values


def paired(
    cell: list[dict], control: list[dict], metric: str, checkpoints: list[str], worlds: list[int]
) -> np.ndarray | None:
    """(checkpoint, world) matrix of cell minus control, NaN where either is missing.

    Cells resume from games.jsonl, so an interrupted run leaves whole worlds
    unplayed. Rows and columns with nothing in them are dropped, so a partial
    run is analysed on exactly the pairs it actually has.
    """
    left, right = table(cell, metric), table(control, metric)
    matrix = np.full((len(checkpoints), len(worlds)), np.nan)
    for i, artifact in enumerate(checkpoints):
        for j, world in enumerate(worlds):
            key = (artifact, world)
            if key in left and key in right:
                matrix[i, j] = left[key] - right[key]
    if np.isnan(matrix).all():
        return None
    matrix = matrix[~np.isnan(matrix).all(axis=1)][:, ~np.isnan(matrix).all(axis=0)]
    return matrix


def bootstrap(matrix: np.ndarray, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows, cols = matrix.shape
    means = np.full(resamples, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for index in range(resamples):
            sample = matrix[np.ix_(rng.integers(0, rows, rows), rng.integers(0, cols, cols))]
            means[index] = np.nanmean(sample)
        usable = means[~np.isnan(means)]
        low, high = np.percentile(usable, [2.5, 97.5])
        mean = float(np.nanmean(matrix))
    return {
        "checkpoints": rows,
        "worlds": cols,
        "pairs": int(np.count_nonzero(~np.isnan(matrix))),
        "resamples_used": int(usable.size),
        "mean": mean,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def verdict(score: dict, self_kills: dict, thresholds: dict) -> tuple[bool, list[str]]:
    checks = [
        (
            self_kills["ci_high"] < thresholds["self_kills_upper_bound_below"],
            f"self-kill interval upper bound {self_kills['ci_high']:+.3f} below "
            f"{thresholds['self_kills_upper_bound_below']:+.2f}",
        ),
        (
            score["ci_low"] > thresholds["score_lower_bound_above"],
            f"score interval lower bound {score['ci_low']:+.3f} above "
            f"{thresholds['score_lower_bound_above']:+.2f}",
        ),
        (
            score["mean"] >= thresholds["score_point_estimate_at_least"],
            f"score point estimate {score['mean']:+.3f} at least "
            f"{thresholds['score_point_estimate_at_least']:+.2f}",
        ),
    ]
    lines = [f"  {'PASS' if ok else 'FAIL'}  {text}" for ok, text in checks]
    return all(ok for ok, _ in checks), lines


def report(
    name: str,
    cell: list[dict],
    control: list[dict],
    checkpoints: list[str],
    worlds: list[int],
    resamples: int,
    seed: int,
    metrics=METRICS,
) -> dict:
    print(f"\n=== {name} minus control ===")
    results = {}
    for metric in metrics:
        matrix = paired(cell, control, metric, checkpoints, worlds)
        if matrix is None:
            print(f"  {metric:<22} no paired games")
            continue
        stat = results[metric] = bootstrap(matrix, resamples, seed)
        star = "" if stat["ci_low"] <= 0 <= stat["ci_high"] else "  *"
        print(
            f"  {metric:<22} {stat['mean']:+.4f}  [{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}]"
            f"  ({stat['pairs']} pairs){star}"
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Directory holding one folder per cell"
    )
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--folder-prefix", default="survival-guard-")
    args = parser.parse_args()

    cfg = json.loads(args.registration.read_text(encoding="utf-8"))
    rule = cfg["decision_rule"]
    resamples = rule["interval"]["resamples"]
    seed = rule["interval"]["seed"]
    names = [cell["name"] for cell in cfg["design"]["cells"]]
    worlds = list(cfg["suite"]["world_seeds"])

    cells = {}
    for name in names:
        folder = args.root / f"{args.folder_prefix}{name}"
        if folder.is_dir():
            cells[name] = load(folder, name)
        else:
            print(f"cell {name}: not run ({folder} missing)")
    if CONTROL not in cells:
        raise SystemExit("the control cell is missing; nothing can be paired")

    control = cells[CONTROL]
    judged = sorted({g["artifact"] for g in control if g["artifact"] != REFERENCE})
    print(
        f"worlds: {len(worlds)}  judged checkpoints: {len(judged)}  "
        f"cells: {sorted(cells)}  resamples: {resamples}"
    )
    for name, games in cells.items():
        print(f"  {name:<8} {len(games)} games, {len({g['artifact'] for g in games})} checkpoints")

    primary = rule["primary_contrast"].split()[0]
    summary = {}
    for name, games in cells.items():
        if name == CONTROL:
            continue
        summary[name] = report(name, games, control, judged, worlds, resamples, seed)
        print(f"  --- {name}, the untrained reference for context only ---")
        report(
            f"{name} (reference)",
            games,
            control,
            [REFERENCE],
            worlds,
            resamples,
            seed,
            metrics=("score", "self_kills", "survived"),
        )
        for episode in sorted({int(a.split("@")[1]) for a in judged}):
            subset = [a for a in judged if a.endswith(f"@{episode}")]
            stat = paired(games, control, "self_kills", subset, worlds)
            score = paired(games, control, "score", subset, worlds)
            if stat is None or score is None:
                continue
            s, k = bootstrap(score, resamples, seed), bootstrap(stat, resamples, seed)
            print(
                f"  episode {episode}: score {s['mean']:+.3f} "
                f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
                f"   self-kills {k['mean']:+.3f} [{k['ci_low']:+.3f}, {k['ci_high']:+.3f}]"
            )

    print(f"\n================ registered decision: {primary} minus control ================")
    if primary not in summary:
        print(f"the primary cell {primary!r} has no games yet, so the rule cannot be applied")
        return 1
    passed, lines = verdict(
        summary[primary]["score"], summary[primary]["self_kills"], rule["thresholds"]
    )
    print("\n".join(lines))
    print("VERDICT: SHIP THE GUARD" if passed else "VERDICT: DO NOT SHIP")
    if not passed:
        print(rule["otherwise"])
        for name in summary:
            if name == primary:
                continue
            other, _ = verdict(
                summary[name]["score"], summary[name]["self_kills"], rule["thresholds"]
            )
            if other:
                print(f"NOTE: cell {name!r} would pass the same thresholds. {rule['multiplicity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
