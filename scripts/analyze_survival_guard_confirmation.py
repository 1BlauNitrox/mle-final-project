"""Apply the registered decision rule of the survival-guard confirmation.

This one is deliberately not a significance test. The registration says so and
says why: with a handful of checkpoints the interval would fail to clear zero
about two times in three even if the effect is real, so the interval is reported
and the decision rests on point estimates fixed in advance.

Pairing, bootstrap and the exclusion of the untrained reference are identical to
the factorial, and reused from it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so it runs as a script as well as a module

from scripts.analyze_survival_guard_factorial import bootstrap, load, paired  # noqa: E402

REGISTRATION = REPO_ROOT / "experiments/2026-09-18-survival-guard-confirmation/config.json"
CONTROL = "control"
TREATMENT = "both"
REFERENCE = "reference"
METRICS = ("score", "self_kills", "survived", "kills", "coins", "collection_fraction", "invalid")


def decide(score: dict, self_kills: dict) -> tuple[bool, list[str]]:
    """The three conditions of the registration, in its own order."""
    checks = [
        (score["mean"] >= 0.0, f"score point estimate {score['mean']:+.4f} is at least 0"),
        (
            score["ci_low"] > -0.15,
            f"score interval lower bound {score['ci_low']:+.4f} is above -0.15",
        ),
        (
            self_kills["mean"] <= 0.0,
            f"self-kill point estimate {self_kills['mean']:+.4f} is at most 0",
        ),
    ]
    return all(ok for ok, _ in checks), [
        f"  {'PASS' if ok else 'FAIL'}  {text}" for ok, text in checks
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Directory holding one folder per cell"
    )
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--folder-prefix", default="confirmation-")
    args = parser.parse_args()

    cfg = json.loads(args.registration.read_text(encoding="utf-8"))
    rule = cfg["decision_rule"]
    resamples, seed = rule["interval"]["resamples"], rule["interval"]["seed"]
    worlds = list(cfg["suite"]["world_seeds"])

    cells = {
        name: load(args.root / f"{args.folder_prefix}{name}", name) for name in (CONTROL, TREATMENT)
    }
    judged = sorted({g["artifact"] for g in cells[CONTROL] if g["artifact"] != REFERENCE})
    print(
        f"worlds {len(worlds)} | judged checkpoints {len(judged)} | "
        f"games {', '.join(f'{n} {len(g)}' for n, g in cells.items())} | resamples {resamples}"
    )

    results = {}
    print(f"\n=== {TREATMENT} minus {CONTROL}, the registered contrast ===")
    for metric in METRICS:
        matrix = paired(cells[TREATMENT], cells[CONTROL], metric, judged, worlds)
        if matrix is None:
            print(f"  {metric:<22} no paired games")
            continue
        stat = results[metric] = bootstrap(matrix, resamples, seed)
        clears = "" if stat["ci_low"] <= 0 <= stat["ci_high"] else "  (clears zero)"
        print(
            f"  {metric:<22} {stat['mean']:+.4f}  [{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}]"
            f"  ({stat['pairs']} pairs){clears}"
        )

    print("\nreported, not gating:")
    for episode in sorted({int(a.split("@")[1]) for a in judged}):
        subset = [a for a in judged if a.endswith(f"@{episode}")]
        score = bootstrap(
            paired(cells[TREATMENT], cells[CONTROL], "score", subset, worlds), resamples, seed
        )
        kills = bootstrap(
            paired(cells[TREATMENT], cells[CONTROL], "self_kills", subset, worlds), resamples, seed
        )
        print(
            f"  episode {episode}: score {score['mean']:+.3f} "
            f"[{score['ci_low']:+.3f}, {score['ci_high']:+.3f}]"
            f"   self-kills {kills['mean']:+.3f} "
            f"[{kills['ci_low']:+.3f}, {kills['ci_high']:+.3f}]"
        )
    per_checkpoint = np.nanmean(
        paired(cells[TREATMENT], cells[CONTROL], "score", judged, worlds), axis=1
    )
    print(f"  checkpoints improved on score: {(per_checkpoint > 0).sum()} of {len(per_checkpoint)}")
    reference = paired(cells[TREATMENT], cells[CONTROL], "score", [REFERENCE], worlds)
    if reference is not None:
        stat = bootstrap(reference, resamples, seed)
        print(
            f"  untrained reference, context only: score {stat['mean']:+.3f} "
            f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}]"
        )

    print("\n================ registered decision ================")
    passed, lines = decide(results["score"], results["self_kills"])
    print("\n".join(lines))
    print(
        "VERDICT: SHIP THE GUARD WITH THE SELECTED CHECKPOINT" if passed else "VERDICT: DO NOT SHIP"
    )
    print(rule["what_we_will_not_claim"] if passed else rule["otherwise"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
