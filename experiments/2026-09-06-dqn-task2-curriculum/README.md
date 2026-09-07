# Issue #97 staged curriculum for Task 2 DQN

> Status: **Incomplete — does not satisfy #97's acceptance criteria; this
> issue stays open.** Arm A (direct) ran to completion on 2026-09-07, but
> two of #97's five specific acceptance criteria are unmet: the matched
> direct-vs-staged comparison was not computed (arm B's raw evidence is
> unavailable), and the numeric decision criteria were not accepted *before*
> arm A's training run, as #97 requires. See "Result" for what arm A's own
> numbers show and "Known gaps" / "Process note" for what is unmet and why.
> This PR documents arm A as a descriptive record, not a completion of #97 —
> see [Waffelmanufaktur's review](https://github.com/1BlauNitrox/mle-final-project/pull/99#pullrequestreview-5130878281).

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
even without arm B, clearly marked; see "Known gaps").

## Evidence: committed, not the ~1.5 GiB raw output tree

Arm A's full raw output tree (every job's attempt directory, including a
full agent-source snapshot per job) runs to ~1.5 GiB, overwhelmingly
redundant source snapshots and per-round framework dumps rather than
evidence. The team decided not to retain that centrally. What actually
matters for verifying the claimed result -- every episode row the analyzer
reads -- is committed instead, at `evidence/issue97-dqn-task2-direct-classic-unmasked/`
(`training/export_evidence.py`): `evaluation-episodes.csv`,
`training-episodes.csv.gz`, and `manifest.json` (plan fingerprints,
per-job provenance, evidence-file checksums), ~6.1 MiB total.

Independently checkable without the raw tree or any archive:

```bash
python -m training.analyze_issue97_dqn_task2_curriculum \
  --verify-from-evidence experiments/2026-09-06-dqn-task2-curriculum/evidence \
  --output experiments/2026-09-06-dqn-task2-curriculum
# -> MATCHES committed result.json
```

Confirmed to print exactly that against this experiment's own committed
evidence. If arm B's evidence is ever exported into the same evidence root
(as `issue86-dqn-task2-unmasked/`), this same command picks it up
automatically and verifies the completed direct-vs-staged comparison too.

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

**Epsilon at first entry into `classic`** (acceptance criterion 3, required
disclosure, not a free parameter):

- **Arm A (direct):** exactly the configured `initial_epsilon` = **1.0**.
  Arm A trains on `classic` from its very first episode, so this is the
  recorded `epsilon` column's value on episode 1 of every replica
  (confirmed directly from the raw training data, not assumed).
- **Arm B (staged):** **≈0.301**. This is not read from arm B's raw data
  (unavailable, see above) but computed analytically from the documented,
  deterministic schedule: `epsilon_decay=0.9997` applied for exactly 4,000
  episodes (2,000 `coin-heaven` + 2,000 `loot-crate`) starting from
  `initial_epsilon=1.0`, i.e. `1.0 * 0.9997**4000 ≈ 0.3011`, still well
  above the `minimum_epsilon=0.1` floor. This value depends only on
  already-registered hyperparameters and the schedule's episode counts, not
  on anything specific to a given training run, so it did not need the
  missing archive to compute.

As #97 itself says: this ~0.70 gap in starting exploration is a structural
difference between the arms, not something either arm's result should be
attributed to without a further isolated experiment.

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

## Process note: acceptance criterion 4 was not met before arm A ran

#97's own acceptance criteria require "exact numeric decision criteria...
accepted by the team **before training starts**." Arm A's 50,000-episode run
was started, and has now completed, without that happening — an ordering
violation, not a documentation gap, and not something this PR can fix after
the fact. Concretely, this means:

- No decision rule is applied to arm A's numbers anywhere in this document;
  the table above is descriptive only, exactly because there is no
  legitimate rule to apply retroactively.
- Whoever sets #97's decision rule going forward should treat arm A's
  already-visible numbers as a source of bias: a rule chosen with knowledge
  of this specific result is no longer the pre-registered, blind criterion
  #97 called for. The methodologically clean options are (a) set the rule
  now while explicitly disregarding arm A's numbers and accept the
  resulting rule is not provably blind, or (b) treat arm A as informational
  only and retrain a fresh direct arm once a rule is fixed first. Neither
  is this PR's call to make.
- This PR does not close #97 (`Refs #97`, not `Closes #97`) precisely
  because of this and the missing staged comparison above — see
  [Waffelmanufaktur's review](https://github.com/1BlauNitrox/mle-final-project/pull/99#pullrequestreview-5130878281).
