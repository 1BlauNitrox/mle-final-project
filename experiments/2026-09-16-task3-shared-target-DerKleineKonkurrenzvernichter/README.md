# Shared crate-and-opponent target representation

> status: completed-failed

## Metadata

- Issue: #204
- Agent: `DerKleineKonkurrenzvernichter`
- Date: 2026-09-16
- Parent SHA-256: `93470f92082597b1848f2fa65265c7288dbb1b50ac90370e5747e085e775ea3e`
- Control plan: `training/run_plans/issue204-task3-control.yaml`
- Candidate plan: `training/run_plans/issue204-task3-shared-target.yaml`

## Hypothesis

The candidate may transfer the frozen Task 2 crate-target prior to opponent
hunting more effectively by representing crates and opponents through the same
target direction, target kind, and bomb-status positions. This could apply,
because they are handeled the same, e.g. they should both be bombed.

## Protocol

Five matched replicas per arm train for 10,000 Classic episodes against one
`peaceful_agent`. Control uses `compact_opponent`; candidate uses
`compact_shared_target`. Both arms use the same corrected, checksum-pinned Task
2 prior, rewards, hyperparameters, stages, seeds, opponents, and evaluation
matrix. Evaluation covers peaceful and coin-collector opponents plus solo
Classic, Coin Heaven, and Loot Crate, with exact repeats.

The Issue #186 callback loaded but did not pass Task 2 Q-values for
`task2_prior`. That defect is corrected equally for both Issue #204 arms, so the
control is a corrected reproduction rather than the historical #186 execution.

## Registered decision

The candidate must improve mean peaceful eliminations, improve at least four
matched replicas, retain a paired-bootstrap lower bound of at least `-0.02`,
stay within the three collection margins and the `+0.03` self-kill margin, and
pass repeatability and latency gates. The full numerical rule is fixed in
`config.yaml` before execution.

## Results

Both run plans completed all 1,005 jobs without a failed job. Across the five
matched replicas, both representations eliminated 0.14 peaceful opponents per
episode on average. The registered candidate-minus-control difference was
therefore 0.00, with a 95% paired-bootstrap interval of [-0.17, 0.16]. The
candidate did not improve at least four replicas.

The candidate retained Classic and Loot Crate collection within the registered
margins, but Coin Heaven collection fell from 0.9612 to 0.9030 (difference
-0.0582). Its mean self-kill rates were higher in all five suites; averaged
equally across the suites, the increase was 0.054 and exceeded the +0.03 gate.

The abstraction did make learning denser. Mean final Q-table size decreased
from 2,226.2 to 1,465.2 states, mean visits per state rose from 243.5 to 372.0,
and the singleton-state fraction fell from 0.267 to 0.130. All latency limits
passed. Exact repeats failed in 225 of 1,000 primary/repeat pairs, across both
treatments, so the registered determinism gate also failed.

## Interpretation

The shared representation created fewer states and more reuse, but opponent
hunting did not improve. Crates and opponents sometimes need different actions,
so combining them probably merges states that should stay separate. This fits
the unchanged eliminations and the worse safety and Coin Heaven results.

Both variants used the corrected Task 2 prior, so the comparison is fair within
this experiment but is not an exact repeat of Issue #186. Repeatability failed
for both variants and should be investigated separately. It does not change the
decision because hunting did not improve and the safety criteria also failed.

## Decision

Reject `compact_shared_target` and keep `compact_opponent`. Fewer states and
more visits per state are not useful when different targets need different
behavior.
