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

The shared representation substantially reduces state fragmentation, but the
extra reuse did not translate into better opponent hunting. Combining crate and
opponent targets appears to introduce harmful aliasing: states that require
different pursuit and bomb-risk behaviour receive the same value estimates.
That interpretation is consistent with unchanged peaceful eliminations and the
simultaneous safety and Coin Heaven regressions.

Because both arms used the corrected Task 2 prior, the comparison isolates the
representation change within this experiment. It should not be compared as an
exact reproduction of the historical Issue #186 numbers. The repeatability
failure affects both treatments and should be investigated separately, but it
does not rescue the candidate: its primary paired hunting estimate is neutral
and its registered retention and safety gates independently fail.

## Decision

Reject `compact_shared_target` as the Task 3 successor. Keep
`compact_opponent` as the current representation. The compact candidate may be
useful as evidence that fewer states and more visits per state are not
sufficient when the abstraction merges behaviourally different targets.
