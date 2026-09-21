# Final tabular submission freeze confirmation

> status: completed

## Metadata

- Issue: #230
- Agent: `DerKleineKonkurrenzvernichter`
- Registered source commit: `e1c2814a604c5ba4cbc498fe415ed559112b7b35`
- Executed revision: `34b0b600c44579860d2373f5a6428afcef81c8b4`
- Run plan: `issue230-final-tabular-freeze-confirmation`
- Date: 2026-09-21
- Training performed: none

## Objective and selection rule

This work selects and confirms a final tabular submission checkpoint without
new training, tuning, or post-hoc seed selection. A treatment is eligible only
if it passed all of its registered adoption and retention gates. Eligible
candidates are ranked by strict first-place rate, competitive score, survival,
opponent eliminations, self-kills, Classic collection, and stable identifier.

The shared-target, five-step Q-learning, Double Q-learning, kill-reward
redistribution, and Task 4 competitive-baseline experiments all failed their
registered adoption rules. Their best-looking replicas are therefore excluded.
The unchanged installed Task 3 incumbent is the only candidate admitted to the
confirmation.

## Confirmation protocol

The incumbent was evaluated without training on 20 fixed seeds in each of six
suites: three rule-based opponents, a mixed opponent line-up, a peaceful
opponent, solo Classic, Coin Heaven, and Loot Crate. The same 120 cases were
then repeated exactly. The run completed all 240 jobs without failure.

The prospective retention gates required Classic collection of at least 0.15,
Coin Heaven collection of at least 0.90, Loot Crate collection of at least
0.20, peaceful-opponent eliminations of at least 0.10, equal-suite aggregate
self-kills no greater than 0.20, p95 decision latency no greater than 50 ms,
maximum latency no greater than 100 ms, and exact repeats.

## Results

| Suite | Score | Collection | Self-kill | Eliminations | Survival | Strict first |
|---|---:|---:|---:|---:|---:|---:|
| Competitive | 1.45 | 0.133 | 0.40 | 0.05 | 0.10 | 0.00 |
| Mixed | 1.65 | 0.156 | 0.55 | 0.05 | 0.15 | 0.00 |
| Peaceful | 2.65 | 0.211 | 0.10 | 0.15 | 0.90 | 0.00 |
| Solo Classic | 2.20 | 0.244 | 0.05 | 0.00 | 0.95 | 0.00 |
| Coin Heaven | 50.00 | 1.000 | 0.00 | 0.00 | 1.00 | 0.00 |
| Loot Crate | 11.80 | 0.236 | 0.35 | 0.00 | 0.65 | 0.00 |

Collection retention, peaceful elimination, and latency passed. The equal-suite
aggregate self-kill rate was 0.2417, above the registered 0.20 ceiling. Forty of
120 primary/repeat pairs differed, so exact repeatability also failed. The
largest suite p95 decision time was 1.84 ms and the largest decision was 12.17
ms.

The evaluated model has SHA-256
`9ef02537efd75b70cbfdd5f973d391ed4924bec91735e07a51b866f3beeee031`
and size 1,493 bytes.

## Interpretation

The incumbent retains the earlier collection capabilities and shows some
peaceful-opponent elimination behavior, but it is unsafe in opponent-heavy play
and does not produce exact fixed-seed repeats under the registered protocol. It
also achieved no strict first places in the 20-game competitive confirmation.
These failures are directly relevant to a final submission freeze and cannot be
overridden by choosing an attractive failed replica from another experiment.

## Decision

The prospective confirmation selected no checkpoint because the safety and
repeatability gates failed. For time reasons we still chose to freeze `r5`
from Issue #228, which ranked first under that experiment's prospective
lexicographic ordering. This is a submission-packaging decision, not a scientific
promotion and not evidence that Task 4 passed.

The frozen model contains 7,909 learned states after 10,000 training episodes.
Its SHA-256 is
`945b2cf0176b4ed57922aa6e12347f1ebbc19d675021dadb38eed6449075eb2d`
and its size is 197,177 bytes. Training is disabled. The manifest preserves the
failed gates and identifies the human deadline override explicitly.

## Reproduction

Run the confirmation with:

    python -m training.run_plan training/run_plans/issue230-final-tabular-freeze-confirmation.yaml

Regenerate evidence and figures with:

    python -m training.analyze_issue230_freeze
    python -m training.plot_issue230_freeze

The compact evidence is in `evidence.csv`, `summary.csv`, and `result.json`.
