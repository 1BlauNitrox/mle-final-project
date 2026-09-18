"""Apply the registered decision rule of the attack-rule test.

Primary contrast is `attack` minus `control`: does taking the attack the policy
declines produce kills, and does it pay for itself in score? The third cell,
`attack-selective`, fires only when the opponent cannot walk out of the blast, so
the second pair of contrasts asks which version of the rule to ship and what
holding fire costs.

Like the guard confirmation this is a decision rule, not a significance test. The
registration says so and says why; the interval is reported and does not gate.

Pairing, bootstrap and the exclusion of the untrained reference are reused from
the factorial analyzer.
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

REGISTRATION = REPO_ROOT / "experiments/2026-09-18-attack-rule/config.json"
CONTROL, ATTACK, SELECTIVE = "control", "attack", "attack-selective"
REFERENCE = "reference"
METRICS = ("score", "kills", "self_kills", "survived", "coins", "collection_fraction", "invalid")


def decide(score: dict, kills: dict) -> tuple[bool, list[str]]:
    """The three conditions of the registration, in its own order."""
    checks = [
        (kills["mean"] > 0.0,
         f"kill point estimate {kills['mean']:+.4f} is above 0"),
        (score["mean"] >= 0.0,
         f"score point estimate {score['mean']:+.4f} is at least 0"),
        (score["ci_low"] > -0.15,
         f"score interval lower bound {score['ci_low']:+.4f} is above -0.15"),
    ]
    return all(ok for ok, _ in checks), [
        f"  {'PASS' if ok else 'FAIL'}  {text}" for ok, text in checks
    ]


def contrast(cells, left, right, judged, worlds, resamples, seed, title):
    print(f"\n=== {left} minus {right} ===  {title}")
    results = {}
    for metric in METRICS:
        matrix = paired(cells[left], cells[right], metric, judged, worlds)
        if matrix is None:
            print(f"  {metric:<22} no paired games")
            continue
        stat = results[metric] = bootstrap(matrix, resamples, seed)
        clears = "" if stat["ci_low"] <= 0 <= stat["ci_high"] else "  (clears zero)"
        print(f"  {metric:<22} {stat['mean']:+.4f}  [{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}]"
              f"  ({stat['pairs']} pairs){clears}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory holding one folder per cell")
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--folder-prefix", default="attack-")
    args = parser.parse_args()

    cfg = json.loads(args.registration.read_text(encoding="utf-8"))
    rule = cfg["decision_rule"]
    resamples, seed = rule["interval"]["resamples"], rule["interval"]["seed"]
    worlds = list(cfg["suite"]["world_seeds"])

    cells = {}
    for name in (CONTROL, ATTACK, SELECTIVE):
        folder = args.root / f"{args.folder_prefix}{name}"
        if folder.is_dir():
            cells[name] = load(folder, name)
        else:
            print(f"cell {name}: not run ({folder} missing)")
    if CONTROL not in cells or ATTACK not in cells:
        raise SystemExit("the control and attack cells are both required")

    judged = sorted({g["artifact"] for g in cells[CONTROL] if g["artifact"] != REFERENCE})
    print(f"worlds {len(worlds)} | judged checkpoints {len(judged)} | "
          f"games {', '.join(f'{n} {len(g)}' for n, g in cells.items())} | resamples {resamples}")

    primary = contrast(cells, ATTACK, CONTROL, judged, worlds, resamples, seed,
                       "the registered contrast")

    print("\nreported, not gating:")
    for episode in sorted({int(a.split('@')[1]) for a in judged}):
        subset = [a for a in judged if a.endswith(f"@{episode}")]
        score = bootstrap(paired(cells[ATTACK], cells[CONTROL], "score", subset, worlds), resamples, seed)
        kills = bootstrap(paired(cells[ATTACK], cells[CONTROL], "kills", subset, worlds), resamples, seed)
        print(f"  episode {episode}: score {score['mean']:+.3f} [{score['ci_low']:+.3f}, {score['ci_high']:+.3f}]"
              f"   kills {kills['mean']:+.3f} [{kills['ci_low']:+.3f}, {kills['ci_high']:+.3f}]")
    for metric in ("score", "kills"):
        per_checkpoint = np.nanmean(paired(cells[ATTACK], cells[CONTROL], metric, judged, worlds), axis=1)
        print(f"  checkpoints improved on {metric}: {(per_checkpoint > 0).sum()} of {len(per_checkpoint)}")
    reference = paired(cells[ATTACK], cells[CONTROL], "score", [REFERENCE], worlds)
    if reference is not None:
        stat = bootstrap(reference, resamples, seed)
        print(f"  untrained reference, context only: score {stat['mean']:+.3f} "
              f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}]")

    print("\n================ registered decision ================")
    passed, lines = decide(primary["score"], primary["kills"])
    print("\n".join(lines))
    print("VERDICT: SHIP THE ATTACK RULE" if passed else "VERDICT: DO NOT SHIP")
    print(rule["what_we_will_not_claim"] if passed else rule["otherwise"])

    if SELECTIVE in cells:
        selective = contrast(cells, SELECTIVE, CONTROL, judged, worlds, resamples, seed,
                             "the variant that only fires on a trapped opponent")
        contrast(cells, SELECTIVE, ATTACK, judged, worlds, resamples, seed,
                 "what holding fire costs")
        selective_passed, lines = decide(selective["score"], selective["kills"])
        print("\n================ the selective variant ================")
        print("\n".join(lines))
        if passed and selective_passed:
            better = "selective" if selective["score"]["mean"] > primary["score"]["mean"] else "permissive"
            print(f"WHICH VARIANT: both pass; ship the {better} one on the larger score point "
                  f"estimate ({selective['score']['mean']:+.4f} selective against "
                  f"{primary['score']['mean']:+.4f} permissive). That choice rests on point "
                  f"estimates, not on a demonstrated difference between them.")
        elif selective_passed:
            print("WHICH VARIANT: only the selective variant passes; it ships.")
        elif passed:
            print("WHICH VARIANT: only the permissive rule passes; it ships.")
        else:
            print("WHICH VARIANT: neither passes; Bomb-omb ships without an attack rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
