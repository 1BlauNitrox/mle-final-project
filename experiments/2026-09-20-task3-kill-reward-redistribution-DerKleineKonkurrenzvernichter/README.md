# Delayed opponent-kill reward redistribution

> status: completed-failed

## Metadata

- Issue: #224
- Agent: `DerKleineKonkurrenzvernichter`
- Date: 2026-09-20
- Control plan: `training/run_plans/issue224-task3-native-kill.yaml`
- Candidate plan: `training/run_plans/issue224-task3-causal-kill.yaml`
- Literature: Arjona-Medina et al. (2019), *RUDDER: Return Decomposition for
  Delayed Rewards*

## Hypothesis

The native `KILLED_OPPONENT: +5.0` reward occurs several transitions after the
bomb-placement action that caused it. Moving its discounted contribution to
that causal transition may improve credit assignment and peaceful-opponent
elimination without changing the return or degrading retained Task 2 skills.

## Protocol

Five matched replicas per arm train for 10,000 Classic episodes against one
`peaceful_agent`. Both arms use single-table Q-learning, `compact_opponent`, the
same Task 2 prior, hyperparameters, exploration, opponents, seeds, budgets, and
evaluation matrix. The control retains the native delayed reward. The candidate
holds each own-bomb transition until its bomb resolves; a kill after `d`
transitions adds `5.0 * 0.9^d` to the bomb transition and removes the native
reward from the explosion transition. A bomb without an opponent kill receives
no such reward.

Evaluation covers peaceful and coin-collector opponents plus opponent-free
Classic, Coin Heaven, and Loot Crate, with exact repeat suites. This is a
deterministic, domain-specific redistribution inspired by RUDDER, not an
implementation of RUDDER's learned return decomposition.

## Registered decision

The candidate must improve mean peaceful eliminations, improve at least four
matched replicas, retain a paired-bootstrap lower bound of at least `-0.02`,
stay within the three collection margins and the `+0.03` aggregate self-kill
margin, and pass exact-repeat and latency gates. The complete numerical rule is
fixed in `config.yaml` before execution.

## Results

Both run plans completed all 1,005 jobs without a failed job. Across five
matched replicas, the native control eliminated 0.15 peaceful opponents per
episode and the causal candidate eliminated 0.13. The registered
candidate-minus-control difference was therefore -0.02, with a 95% paired
bootstrap interval of [-0.14, 0.09]. Only one replica improved, one regressed,
and three tied, so the primary, four-replica, and confidence-bound gates failed.

All collection-retention gates passed. Candidate-minus-control collection was
+0.0189 in opponent-free Classic, 0.0000 in Coin Heaven, and +0.0550 in Loot
Crate. Averaged equally across the five suites, self-kill rate decreased by
0.004; this passed the registered +0.03 margin. The candidate improved
coin-collector-opponent self-kill rate from 0.36 to 0.28, but peaceful self-kill
rate rose from 0.03 to 0.05.

Decision latency passed comfortably: the worst per-replica/suite p95 was 1.93
ms for control and 2.59 ms for candidate, while maxima were 10.26 ms and 27.61
ms. Exact repeats failed in 222 of 1,000 primary/repeat pairs (115 control, 107
candidate), so the determinism gate failed in both arms.

Training-state density changed little: candidate mean Q-table size was 2,233.2
versus 2,264.0 for control, with 246.4 versus 237.3 mean visits per state and
singleton fractions of 0.262 versus 0.264.

## Interpretation

Moving the discounted kill contribution to the bomb-placement transition did
not improve the registered peaceful-opponent objective. The paired estimate is
slightly negative and its uncertainty includes both harm and modest benefit;
the per-replica evidence also lacks the required consistency. This result does
not support replacing the native delayed reward.

The candidate retained earlier-task collection and aggregate safety, and its
coin-collector safety result is potentially useful exploratory evidence.
However, those secondary observations cannot override the failed prospective
hunting gates and should not be treated as confirmation of a safety benefit.

This experiment implements deterministic causal redistribution only. It does
not test RUDDER's learned return-decomposition model, so the negative result
applies to this domain-specific approximation rather than to full RUDDER. The
repeatability failure affects both treatments and remains a separate shared
limitation, but it does not rescue the candidate because the independent
primary and replica gates also fail.

## Decision

Reject `causal_bomb` reward timing and keep the native delayed
`KILLED_OPPONENT: +5.0` reward as the current Task 3 configuration.
