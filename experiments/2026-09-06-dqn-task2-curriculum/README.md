# Issue #97 staged curriculum for Task 2 DQN

> Status: **Backlog — blocked on Issue #86.** Neither run plan referenced here
> may be executed yet. See "Blocking dependencies" below before doing
> anything else with this directory.

## Hypothesis and single factor

Holding the winning action-masking mode from #86 fixed, does the staged
curriculum (`coin-heaven` → `loot-crate` → `classic`) improve `classic` coin
collection and self-survival, and preserve Task 1 retention better, than
training directly on `classic` alone for the same total episode budget?

This isolates the curriculum schedule. It does **not** isolate scenario-mix
from exploration state at entry: the direct-classic arm enters `classic` near
its initial epsilon, while the staged arm enters it after 4,000 prior
episodes, close to the exploration floor under the current
`epsilon_decay=0.9997` schedule. Report the measured epsilon at first entry
into `classic` for both arms alongside the result; do not attribute an
observed effect to the scenario mix alone.

## Blocking dependencies

1. **Issue #86 (legal-action masking) must reach an adopt/reject decision
   first.** That decision selects which one of the two `direct_classic_plans`
   below is actually run — the other is discarded, not executed. It also
   determines whether the "staged" comparison arm is #86's masked or
   unmasked winning artifact.
2. **`run_plan.py` on `main` does not yet recognize `action_masking`.** It is
   added by PR #95 (Issue #86). Confirmed on this branch: both
   `issue97-dqn-task2-direct-classic-*.yaml` plans load and dry-run without
   error on current `main`, because unrecognized top-level YAML keys are
   silently ignored — **not** because masking is applied. Running the
   `*-masked` plan before PR #95 merges would silently train unmasked while
   labeled masked. Rebase this branch onto `main` after #95 merges and
   re-run both `--dry-run` checks before trusting either plan.
3. **The staged curriculum arm is not retrained by this experiment.** It
   reuses #86's winning arm's already-trained artifact and evaluation results
   directly. Only the direct-classic arm needs a new training run.

## Proposed comparison

- **A (direct):** `training/run_plans/issue97-dqn-task2-direct-classic-{unmasked,masked}.yaml`
  — 10,000 episodes of `classic` only, 5 replicas, no `coin-heaven`/`loot-crate`
  stages. Same starting artifact, replica seeds (`51001`-`51005` /
  `61001`-`61005`), and evaluation seed pairs as #86, so replicas pair
  directly against #86's staged arm.
- **B (staged):** #86's winning arm, reused as-is.

Both plans here start from the same reviewed, corrected migration artifact as
#86 (`checkpoint-issue85-zero-suffix.pt`).

## Decision rule

Not yet fixed. Per the owner: exact numeric criteria (minimum improvement,
non-inferiority margins, replicas-agreeing threshold) are proposed and
accepted by the team before training starts, following the #46/#86 pattern —
this file does not set them.

## Execution (once unblocked)

```bash
python -m training.run_plan training/run_plans/issue97-dqn-task2-direct-classic-<mode>.yaml --dry-run
```

Confirm the dry-run matrix (5 replicas × 10,000 training episodes + 1,200
evaluation episodes = 305 jobs, 50,000 training episodes total) before
removing `--dry-run`. Do not execute both `<mode>` plans — only the one
matching #86's decision.
