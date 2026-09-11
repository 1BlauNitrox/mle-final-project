# Full no-route penalty with half-strength escape progress

> Status: registered and ready to run

## Metadata

- Issue: #142
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-11
- Registration base: `08ca233`
- Implementation commit: `77819f57a4ee0d413cbce462187f7eb2e38632fa`
- Experiment commit: pending
- Approval: intermediate peer approval explicitly waived by the owner

## Research question and hypothesis

Does retaining the full `-5` potential only for states without a complete safe
route reduce Classic self-kills relative to uniform half strength without
losing more than `0.05` Classic collection?

Issue #139 showed that uniform half strength restored collection but failed its
safety guard. The hypothesis is that the no-route bucket, rather than ordinary
distance progress, carries the useful catastrophic-risk signal.

## Treatments

- Control: half-strength potentials `0, -0.5, -1, -1.5, -2.5`.
- Candidate: identical progress potentials, but `-5` for no complete route.

Both use `F(s,s') = 0.9 * Phi(s') - Phi(s)`. The no-route value is the only
treatment difference and changes learning rewards only.

## Controlled protocol

Both arms use compact state, zero-initialized one-step Q-learning, learning
rate `0.05`, discount factor `0.9`, epsilon `1.0 * 0.99` down to `0.1`, no
action mask, no useful-bomb bonus and final-checkpoint selection. Five paired
replicas each train for 2,000 Coin Heaven, 2,000 Loot Crate and 6,000 Classic
episodes. Each final model is evaluated on 40 fresh paired development seeds
per scenario and repeated deterministically.

## Decision rule

Accept the candidate only if all registered criteria pass:

- mean Classic self-kill rate is below control;
- its paired 95% bootstrap CI upper bound is below zero;
- at least four of five paired replicas have fewer Classic self-kills;
- Classic and Coin Heaven collection each decrease by no more than `0.05`;
- candidate/control mean visits per state is at least `0.90`;
- deterministic repeats match;
- decision-time p95 is below `50 ms` and maximum below `100 ms`.

## Execution

```bash
python -m training.run_plan \
  training/run_plans/issue142-compact-task2-uniform-half.yaml
python -m training.run_plan \
  training/run_plans/issue142-compact-task2-full-no-route.yaml
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
