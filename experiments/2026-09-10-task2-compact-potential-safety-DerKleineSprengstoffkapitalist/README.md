# Compact Task 2 potential-based safety shaping

> Status: completed_rejected

## Metadata

- Issue: #133
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-10
- Registration base: `124e5fa`
- Experiment commit: `6ae0de76ae544eeae94f8ca94f32b54eabe3e8b8`
- Framework revision: `b31c6c16cab62f47794f2600a8c3b1b677cb05d1`
- Approval: verbally approved by `1BlauNitrox` for commit `d8b8148`

## Research question

Can potential-based safety shaping reduce the compact Task 2 agent's
self-kill rate without materially reducing coin collection or state reuse?

## Hypothesis

Safety potential shaping will make transitions out of danger more attractive
and transitions into danger less attractive. This should reduce self-kills
while preserving the compact representation's collection performance.

## Treatments

- Control: compact state representation without potential shaping.
- Candidate: compact state representation with `compact_safety` potential
  shaping.

Both treatments use zero-initialized Q-tables and differ only in the potential
shaping mode.

## Potential function

The potential is:

- `0` when the agent is safe;
- `-1` when the agent is in danger and at least one safe direction exists;
- `-2` when the agent is in danger and no safe direction exists.

The additional learning reward is:

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

with `gamma = 0.9`. For terminal transitions, `Phi(s') = 0`.

The shaping reward modifies only the Q-learning update. It does not filter
actions, prescribe an action, or replace the learned policy.

## Controlled variables

The following remain identical between treatments:

- compact five-feature state representation;
- one-step tabular Q-learning;
- learning rate `0.05`;
- discount factor `0.9`;
- epsilon schedule `1.0`, multiplied by `0.99` per episode to a minimum of
  `0.1`;
- zero Q-value initialization;
- no action masking;
- no useful-bomb bonus;
- training curriculum and episode budget;
- training and evaluation seeds;
- scenarios and opponents;
- final-checkpoint evaluation.

## Training protocol

Train five independent replicas per treatment:

1. 2,000 `coin-heaven` episodes;
2. 2,000 `loot-crate` episodes;
3. 6,000 `classic` episodes.

Only the checkpoint after exactly 10,000 episodes is evaluated.

## Evaluation protocol

Evaluate every replica on 40 paired development seeds in:

- `classic`;
- `coin-heaven`;
- `loot-crate`.

Repeat every evaluation with identical seeds to verify determinism.
Confirmation seeds remain unused.

## Primary metric

Candidate-minus-control self-kill rate on `classic`, paired by replica and
world seed.

## Secondary metrics

- `classic` coin collection fraction;
- `coin-heaven` coin collection fraction;
- `loot-crate` coin collection fraction;
- mean visits per materialized Q-table state;
- Q-table size;
- evaluation unseen-state rate;
- decision-time p95 and maximum.

## Decision rule

Accept the candidate only if:

- its aggregate `classic` self-kill rate is below control;
- at least four of five replicas have a lower self-kill rate;
- its `classic` collection fraction is no more than `0.05` below control;
- its mean-visits-per-state ratio relative to control is at least `0.90`;
- all deterministic repeats match;
- decision-time p95 is below `50 ms`;
- maximum decision time is below `100 ms`.

## Compute budget

- maximum training episodes: 100,000;
- planned evaluation episodes: 2,400;
- maximum evaluation episodes: 2,480;
- maximum CPU time: 36 CPU-hours;
- paid resources: none.

## Registered run plans

```bash
python -m training.run_plan \
  training/run_plans/issue133-compact-task2-control.yaml \
  --dry-run

python -m training.run_plan \
  training/run_plans/issue133-compact-task2-potential-safety.yaml \
  --dry-run
```

No training may begin until the protocol commit and both dry runs have been
reviewed and personally approved by another team member.

## Results

Both run plans completed all 1,215 jobs. The analysis contains 1,200 primary
and 1,200 deterministic-repeat evaluation episodes across both treatments.

### Safety

| Scenario | No shaping | Safety shaping |
| --- | ---: | ---: |
| `classic` | 0.040 | 0.050 |
| `coin-heaven` | 0.000 | 0.010 |
| `loot-crate` | 0.135 | 0.160 |

On `classic`, the candidate-minus-control self-kill difference was `0.010`,
with a paired 95% bootstrap interval of `[-0.040, 0.065]`. Only replica `r3`
had a lower self-kill rate. Replicas `r2` and `r5` were unchanged, while
replicas `r1` and `r4` were worse. Both registered safety criteria failed.

### Collection performance

| Scenario | No shaping | Safety shaping |
| --- | ---: | ---: |
| `classic` | 0.123 | 0.166 |
| `coin-heaven` | 1.000 | 0.990 |
| `loot-crate` | 0.224 | 0.179 |

The paired candidate-minus-control `classic` collection difference was
`0.043`, with a 95% bootstrap interval of `[-0.003, 0.093]`. The registered
maximum collection-loss criterion passed.

### Learning efficiency

| Diagnostic | No shaping | Safety shaping |
| --- | ---: | ---: |
| Mean materialized states | 1,082.4 | 1,088.8 |
| Mean visits per state | 665.9 | 701.7 |
| Mean singleton-state fraction | 0.0582 | 0.0582 |
| Mean model size | 40.4 KiB | 40.5 KiB |

The candidate/control mean-visits-per-state ratio was approximately `1.054`,
above the registered minimum of `0.90`.

All deterministic repeats matched. All registered decision-time limits passed.

Generated evidence:

- [`summary.csv`](summary.csv);
- [`result.json`](result.json);
- [`figures/performance_and_safety.png`](figures/performance_and_safety.png);
- [`figures/learning_efficiency.png`](figures/learning_efficiency.png).

## Decision

Reject `compact_safety` potential shaping as the default. It preserved the
compact representation's learning efficiency and did not reduce `classic`
collection, but it failed to reduce self-kills and slightly increased the
aggregate self-kill rate.

The result suggests that this coarse state-only potential does not distinguish
enough between genuinely safe escape progress and locally safe but ultimately
trapped movement. A future experiment should improve the safety information or
the potential definition rather than merely increasing the shaping magnitude.