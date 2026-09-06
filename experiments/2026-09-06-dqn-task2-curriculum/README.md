# Issue #97 staged curriculum for Task 2 DQN

> Status: **Registered — ready to execute.** Issue #86 reached its decision
> (masking rejected, unmasked control retained); see "Resolved from Issue #86"
> below for what that fixed and what it did not change.

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

This plan uses at most two training workers, 8 GiB RAM, and a 24 CPU-hour /
15 wall-hour ceiling — half of #86's, since only one arm (not two) is freshly
trained here. Detach with `Ctrl-b d`; resume an interrupted run with
`--resume`. Do not alter the plan, source tree, or artifact between a failed
run and its resume.
