# Safety-constrained exploration for compact Task 2

> Status: registered and ready to run

## Metadata

- Issue: #153
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-12
- Registration base: `08ca233`
- Implementation commit: `4b6b3cf7f22888dd10b19b3aca36da8990709123`
- Experiment commit: `7c4f11191d76efdb33f5e80a5e686dd4e7907544`
- Approval: intermediate peer approval explicitly waived by the owner

## Research question and hypothesis

Does excluding a clearly unsafe `BOMB` action only from random epsilon-greedy
exploration reduce final Classic self-kills without reducing collection?

The hypothesis is that fewer avoidable lethal exploration transitions improve
the learned policy. The treatment does not constrain greedy action selection,
Bellman targets, or evaluation.

## Treatments

- Control: `compact_decision` with standard epsilon-greedy exploration.
- Candidate: the same configuration, except `BOMB` is removed from the random
  exploration candidates when `bomb_status == UNSAFE`.

Both arms use no action masking and no potential shaping. The candidate does
not force an action or encode a rule-based escape policy.

## Controlled protocol

Both arms use zero-initialized one-step tabular Q-learning, learning rate
`0.05`, discount factor `0.9`, epsilon `1.0 * 0.99` down to `0.1`, unchanged
rewards, no action masking, no potential shaping, no useful-bomb reward, and
final-checkpoint selection. Five paired replicas train for 2,000 Coin Heaven,
2,000 Loot Crate, and 6,000 Classic episodes. Each final model is evaluated on
40 fresh paired development seeds in all three scenarios and repeated exactly.

## Decision rule

Accept the candidate only if all ten registered criteria pass:

- candidate Classic self-kill rate is at most `0.055` and below control;
- the paired 95% bootstrap interval upper bound for candidate minus control
  Classic self-kills is below zero;
- at least four of five replicas lower Classic self-kills;
- Classic collection decreases by no more than `0.01`;
- Coin Heaven collection decreases by no more than `0.05`;
- candidate/control mean visits per state is at least `0.90`;
- all deterministic repeats match;
- decision-time p95 is below `50 ms` and maximum below `100 ms`.

## Execution authorization

The owner explicitly instructed Codex to implement and execute this experiment
without the intermediate approval pause. This prospective waiver does not alter
the treatments, seeds, budget, metrics, or decision criteria.

## Results

Pending execution.

## Interpretation and decision

Pending execution.

## AI assistance

OpenAI Codex assisted with implementation, tests, registration, execution,
analysis, plotting, and documentation. AI output is not experimental evidence;
reported values must come from retained framework outputs and receive human
review before merge.
