# Time-aware escape-distance potential shaping

> Status: completed_mixed

## Metadata

- Issue: #135
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-11
- Registration base: `34c6a67`
- Implementation commit: `3bc9411`
- Experiment commit: `579d7878a3aaa318558b4b46560fa84e427cb6e4`
- Framework revision: `34c6a67`
- Approval: intermediate peer approval explicitly waived by the owner for this run

## Research question

Does time-aware escape-distance potential shaping reduce `classic` self-kills
relative to the compact agent without shaping?

## Hypothesis

Rewarding measurable progress along a complete time-aware route to persistent
safety will propagate safety information through intermediate decisions and
reduce self-kills without materially reducing collection or state reuse.

## Treatments

- Control: compact state with no potential shaping.
- Candidate: identical compact state with `escape_distance` shaping.

Both arms use fresh zero-initialized Q-tables. The shaping mode is the only
treatment difference.

## Registered potential

The shortest route to persistent safety reuses the agent's deterministic,
time-aware breadth-first search. It respects walls, crates, current occupancy,
bomb occupancy, blast propagation, bomb timers, active explosions and arrival
times. Persistent safety is a reachable tile outside every predicted lethal
interval.

Potential values are:

- already safe: `0`;
- safety one action away: `-1`;
- safety two actions away: `-2`;
- safety at least three actions away: `-3`;
- no complete safe route within the registered search horizon: `-5`.

The transition reward is `F(s,s') = 0.9 * Phi(s') - Phi(s)`. The absorbing
terminal state has potential zero. The calculation only changes the learning
reward: it does not mask, prescribe or execute actions.

## Controlled variables

Both arms retain the compact five-feature representation, one-step tabular
Q-learning, learning rate `0.05`, discount factor `0.9`, epsilon schedule
`1.0 * 0.99` with minimum `0.1`, zero initialization, no action masking, no
useful-bomb bonus, identical curriculum, paired seeds and final-checkpoint
selection.

## Protocol

Train five replicas per treatment for 2,000 `coin-heaven`, 2,000 `loot-crate`
and 6,000 `classic` episodes. Evaluate each final checkpoint on 40 paired fresh
development seeds in all three scenarios, followed by identical deterministic
repeats. Confirmation and final-test seeds remain unused.

## Decision rule

Accept the candidate only if:

- aggregate `classic` self-kill rate is below control;
- the paired 95% bootstrap CI upper bound is below zero;
- at least four of five replicas have lower `classic` self-kill rates;
- `classic` collection decreases by no more than `0.05`;
- candidate/control mean visits per state is at least `0.90`;
- all deterministic repeats match;
- decision-time p95 is below `50 ms`;
- maximum decision time is below `100 ms`.

## Registered run plans

```bash
python -m training.run_plan \
  training/run_plans/issue135-compact-task2-control.yaml \
  --dry-run

python -m training.run_plan \
  training/run_plans/issue135-compact-task2-escape-distance.yaml \
  --dry-run
```

The owner explicitly authorized immediate execution without the intermediate
peer-approval pause normally required by the experiment workflow. This waiver
changes only the approval gate, not the registered treatments, seeds, budgets,
metrics or decision criteria.

## Results

Both plans completed all 1,215 jobs. The retained evidence contains 1,200
primary and 1,200 deterministic-repeat evaluation episodes across both
treatments.

### Safety

| Scenario | No shaping | Escape-distance shaping |
| --- | ---: | ---: |
| `classic` | 0.055 | 0.015 |
| `coin-heaven` | 0.000 | 0.000 |
| `loot-crate` | 0.195 | 0.120 |

The paired candidate-minus-control `classic` self-kill difference was
`-0.040`, with a 95% nested-bootstrap interval of `[-0.085, 0.000]`. Four of
five replicas improved (`r1`, `r2`, `r4`, `r5`); `r3` remained at zero. The
mean and replica-count safety gates passed, but the strict CI gate failed
because its upper bound was exactly zero rather than below zero.

### Collection

| Scenario | No shaping | Escape-distance shaping |
| --- | ---: | ---: |
| `classic` | 0.132 | 0.048 |
| `coin-heaven` | 0.977 | 1.000 |
| `loot-crate` | 0.172 | 0.118 |

The `classic` collection difference was `-0.0839`, with a 95% interval of
`[-0.1328, -0.0167]`. This exceeds the registered maximum loss of `0.05`, so
the performance guard failed. Coin Heaven performance was retained, while
Loot Crate collection also decreased.

### Learning efficiency

| Diagnostic | No shaping | Escape-distance shaping |
| --- | ---: | ---: |
| Mean materialized states | 1,075.2 | 1,067.2 |
| Mean visits per state | 657.7 | 830.8 |
| Mean singleton-state fraction | 0.0608 | 0.0627 |
| Evaluation unseen-state rate | 0.0074% | 0.0051% |

The mean-visits ratio was `1.263`, comfortably above the registered `0.90`
minimum. State reuse was therefore preserved.

### Reproducibility and latency

All primary and repeated outcomes matched deterministically. Decision-time
p95 passed with a worst primary value of approximately `1.70 ms`. The strict
maximum-time gate failed because one candidate Loot Crate decision reached
`197.68 ms`; other aggregate latency results remained well below the limit.

Generated evidence:

- [`summary.csv`](summary.csv);
- [`result.json`](result.json);
- [`figures/performance_and_safety.png`](figures/performance_and_safety.png);
- [`figures/learning_efficiency.png`](figures/learning_efficiency.png).

## Decision

Reject `escape_distance` shaping as the new default because three registered
gates failed: the safety CI upper bound, Classic collection retention and
maximum decision time.

The treatment nevertheless produced the strongest safety signal observed in
this experiment sequence: aggregate Classic self-kills fell by 72.7%, four
replicas improved, and Loot Crate self-kills also decreased. The accompanying
collection loss suggests that the `-5` no-route potential or the coarse
distance buckets make the policy too conservative. A follow-up should preserve
the time-aware progress signal while reducing its magnitude or separating bomb
placement risk from post-placement escape progress.

## AI assistance

AI assistance was used for implementation, test design, experiment
registration, execution monitoring, analysis scaffolding, plotting and result
interpretation. All changes were validated locally with Ruff, pytest and the
registered deterministic experiment protocol. The owner explicitly authorized
execution without the normal intermediate approval pause.
