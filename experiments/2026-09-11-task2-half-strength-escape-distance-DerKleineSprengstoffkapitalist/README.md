# Half-strength escape-distance potential shaping

> Status: registered and ready to run

## Metadata

- Issue: #139
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-11
- Registration base: `974fa7c`
- Implementation commit: `b46a54984d25cd4e15096eebbfa876614257590b`
- Experiment commit: pending registration commit
- Approval: intermediate peer approval explicitly waived by the owner

## Research question and hypothesis

Does halving the time-aware escape-distance potential recover Classic
collection performance relative to the full-strength treatment while retaining
a lower self-kill rate than the Issue #135 no-shaping reference?

Issue #135 reduced Classic self-kills from `0.055` to `0.015`, but also reduced
collection from `0.132` to `0.048`. The hypothesis is that half strength keeps
the useful safety signal without dominating task progress as strongly.

## Treatments

- Control: `escape_distance` with potentials `0, -1, -2, -3, -5`.
- Candidate: `escape_distance_half` with potentials
  `0, -0.5, -1, -1.5, -2.5`.

Both use `F(s,s') = 0.9 * Phi(s') - Phi(s)`. The scale is the only treatment
difference. It changes learning rewards, not evaluation-time action rules.

## Controlled protocol

Both arms use compact state, zero-initialized one-step Q-learning, learning
rate `0.05`, discount factor `0.9`, epsilon `1.0 * 0.99` down to `0.1`, no
action mask, no useful-bomb bonus, identical paired seeds and final-checkpoint
selection. Five replicas each train for 2,000 Coin Heaven, 2,000 Loot Crate
and 6,000 Classic episodes. Each final model is evaluated on 40 fresh paired
development seeds per scenario and repeated deterministically.

## Decision rule

Accept half strength only if all registered criteria pass:

- mean Classic collection exceeds full strength;
- the paired 95% bootstrap CI lower bound for that difference is above zero;
- candidate Classic self-kill rate remains below the Issue #135 no-shaping
  reference of `0.055`;
- at least four of five candidate replicas remain below `0.055`;
- Coin Heaven collection decreases by no more than `0.05`;
- candidate/control mean visits per state is at least `0.90`;
- deterministic repeats match;
- decision-time p95 is below `50 ms` and maximum below `100 ms`.

The historical no-shaping value is only a preregistered safety guard. The new
scientific paired contrast is full strength versus half strength.

## Execution

```bash
python -m training.run_plan \
  training/run_plans/issue139-compact-task2-full-strength.yaml

python -m training.run_plan \
  training/run_plans/issue139-compact-task2-half-strength.yaml
```

The owner explicitly authorized immediate execution without the intermediate
approval pause. This waiver does not alter treatments, seeds, budgets, metrics
or criteria.

## Results

Pending execution.

## Interpretation and decision

Pending execution.

## AI assistance

OpenAI Codex assisted with implementation, tests, protocol registration,
execution, analysis, plotting and documentation. AI output is not experimental
evidence; all claims must be computed from retained outputs and reviewed by the
owner and a non-author reviewer.
