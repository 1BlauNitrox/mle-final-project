# Issue #97 staged curriculum for Task 2 DQN

> Status: **Partially completed.** Arm A (direct) ran to completion on
> 2026-09-07; arm B (staged, reused from #86) could not be included because
> its raw per-episode evidence is not available on the machine that ran arm
> A. See "Result and decision" for what this does and does not show, and
> "Known gaps" for what completes the registered comparison.

## Hypothesis and single factor

Holding action masking fixed at `none` (Issue #86 rejected `framework_legal`
masking; the retained arm is unmasked), does the staged curriculum
(`coin-heaven` → `loot-crate` → `classic`) improve `classic` coin collection
and self-survival, and preserve Task 1 retention better, than training
directly on `classic` alone for the same total episode budget?

This isolates the curriculum schedule. It does **not** isolate scenario-mix
from exploration state at entry: the direct-classic arm enters `classic` near
its initial epsilon, while the staged arm enters it after 4,000 prior
episodes, close to the exploration floor under the current
`epsilon_decay=0.9997` schedule. Report the measured epsilon at first entry
into `classic` for both arms alongside the result; do not attribute an
observed effect to the scenario mix alone.

## Resolved from Issue #86

Issue #86 rejected `framework_legal` masking (collection-fraction and
survival non-regression gates failed in all three scenarios) and retained the
unmasked arm. That decision settles both open questions this experiment was
waiting on:

1. **Which direct-classic plan to run.** Only the unmasked plan is prepared
   here now; the masked variant was deleted, since #86's outcome means it
   would never have been executed.
2. **Which artifact the staged arm reuses.** `staged_curriculum_arm_source`
   below now names #86's unmasked arm explicitly, not a placeholder
   "winning arm."

`action_masking: none` is the default `run_plan.py` already applies when the
field is absent, so — unlike the masked variant would have — this plan's
result does not depend on whether PR #95 has merged. Running it after #95
merges is still preferred for clean provenance (the fingerprint then reflects
a `main` that actually recognizes the field), but it is no longer a
correctness requirement.

## Proposed comparison

- **A (direct):** `training/run_plans/issue97-dqn-task2-direct-classic-unmasked.yaml`
  — 10,000 episodes of `classic` only, 5 replicas, no `coin-heaven`/`loot-crate`
  stages. Same starting artifact, replica seeds (`51001`-`51005` /
  `61001`-`61005`), and evaluation seed pairs as #86, so replicas pair
  directly against #86's unmasked arm.
- **B (staged):** #86's unmasked arm, reused as-is — not retrained by this
  experiment.

Both arms start from the same reviewed, corrected migration artifact as #86
(`checkpoint-issue85-zero-suffix.pt`). The two arms are matched on total
episode budget (10,000 each); this is the reason arm A's training budget is
not enlarged even though the team has otherwise allowed this round of
experiments to run larger — doing so here would invalidate the "same total
episode budget" comparison the hypothesis depends on.

## Decision rule

Not yet fixed. Per the owner: exact numeric criteria (minimum improvement,
non-inferiority margins, replicas-agreeing threshold) are proposed and
accepted by the team before the result is interpreted, following the
#46/#86 pattern — this file does not set them. This does not block execution:
#86 followed the same order (registered, then run, then the decision applied
to the completed data).

## Execution

```bash
python -m training.run_plan training/run_plans/issue97-dqn-task2-direct-classic-unmasked.yaml --dry-run
```

Confirm the dry-run matrix (5 replicas × [1 training stage + 6 evaluation
suites × 10 seed pairs] = 305 jobs, 50,000 training episodes total) before
removing `--dry-run`. Then:

```bash
tmux new -s issue97
python -m training.run_plan training/run_plans/issue97-dqn-task2-direct-classic-unmasked.yaml 2>&1 | tee logs/issue97-direct-classic.log
```

This plan was drafted for at most two training workers, 8 GiB RAM, and a
24 CPU-hour / 15 wall-hour ceiling — half of #86's, since only one arm (not
two) is freshly trained here. Executed here with `max_parallel_training: 1`
instead (committed as actually run): this plan ran concurrently with all
three of Issue #103's reward-shaping arms on one machine, so bounding each
plan to one worker kept total concurrent training processes at four rather
than eight. Detach with `Ctrl-b d`; resume an interrupted run with
`--resume`. Do not alter the plan, source tree, or artifact between a failed
run and its resume.

## Result and decision

Arm A (direct) ran on the owner's own machine on 2026-09-07 (concurrently
with Issue #103's three arms) and completed in full: 305/305 jobs, 0
failures, in about 6 hours. Deterministic repeats matched exactly (identical
`executed_action_sequence_sha256` between every primary/repeat pair; only
decision-time measurements differed). Compact evidence:
`result.json`, `summary.csv` (via
`training/analyze_issue97_dqn_task2_curriculum.py`, which produces both
even without arm B, clearly marked; see "Known gaps"). Raw per-episode data
(~1.5 GiB: `training_outputs/run-plans/issue97-dqn-task2-direct-classic-unmasked/`)
is retained on the owner's machine and not committed.

**No comparison against the staged arm was computed.** Arm B is Issue #86's
retained unmasked artifact, and its raw per-episode evidence lives in an
external server archive (SHA-256
`841f01f86719a28d7a9d10d69685f6293c94e281b0dd39379d09947a4c180c1f`, see
`experiments/2026-09-06-dqn-task2-legal-action-masking/README.md`) that was
not present on the machine that ran arm A. The direct-vs-staged question
this issue was registered to answer is therefore **still open** — see
"Known gaps".

What arm A's own numbers show, descriptively (no registered decision rule
applies to a single arm; not compared against control or staged):

| Scenario | Collection fraction | Survival rate | Self-kill rate | Invalid-action rate |
| --- | --- | --- | --- | --- |
| `classic` | 0.042 | 0.36 | 0.64 | 0.171 |
| `coin-heaven` | 0.077 | 0.62 | 0.38 | 0.285 |
| `loot-crate` | 0.024 | 0.42 | 0.58 | 0.229 |

These are broadly the same order of magnitude as Issue #103's `control` arm
(same reward values, same direct-classic protocol, different replica seeds:
`51001`-`51005` here vs. `91001`-`91005` there) — e.g. `classic` survival
0.36 here vs. 0.48 there. The two are not formally paired (different seeds),
so this is a rough consistency check, not evidence either result is "more
correct"; the spread between them is a reasonable indication of how much
across-seed variance to expect at n=5 replicas.

## Known gaps

- **The registered direct-vs-staged comparison is not computed.** Completing
  it needs #86's raw archive (see above) extracted somewhere reachable, then:
  ```bash
  python -m training.analyze_issue97_dqn_task2_curriculum \
    --direct-plan-root training_outputs/run-plans \
    --staged-plan-root <path to the extracted archive's training_outputs/run-plans> \
    --output training_outputs/issue97-analysis
  ```
  The script already supports this (`--staged-plan-root`); it only produced
  a direct-arm-only result here because that archive was not available.
  No decision rule was ever fixed for this comparison either (see below),
  so completing it produces descriptive paired statistics, not an
  adopt/reject call, until the team sets one.
- **No decision rule was fixed before running**, per the owner's standing
  preference to set numeric criteria themselves — this was true when arm A
  was registered and remains true now that it has run.
