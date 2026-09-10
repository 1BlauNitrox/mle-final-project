# Time-aware escape-distance potential shaping

> Status: registered_execution_authorized

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

Pending execution.

## Decision

Pending evaluation.
