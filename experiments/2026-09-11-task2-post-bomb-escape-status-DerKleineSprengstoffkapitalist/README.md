# Compact post-bomb escape-status experiment

> Status: registered and ready to run

## Metadata

- Issue: #144
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-11
- Registration base: `08ca233`
- Implementation commit: `c849077e9a2782f284029edca1e3cd0705c202c1`
- Experiment commit: pending registration commit
- Approval: intermediate peer approval explicitly waived by the owner

## Research question and hypothesis

Does appending a categorical post-bomb escape status to the compact state
reduce Classic self-kills while retaining or improving hidden-coin collection?

The hypothesis is that distinguishing complete escape, temporary local safety
and no route resolves an information ambiguity that reward shaping could not
address without suppressing collection.

## Treatments

- Control: the existing five-value `compact_decision` representation.
- Candidate: the same five values plus `escape_after_bomb_status` with values
  `NOT_APPLICABLE`, `COMPLETE_ROUTE`, `TEMPORARY_SAFETY_ONLY`, and `NO_ROUTE`.

The additional value describes geometry only. It neither chooses nor masks an
action. Both arms use no potential shaping.

## Controlled protocol

Both arms use zero-initialized one-step tabular Q-learning, learning rate
`0.05`, discount factor `0.9`, epsilon `1.0 * 0.99` down to `0.1`, no action
mask, no potential shaping, no useful-bomb reward and final-checkpoint
selection. Five paired replicas train for 2,000 Coin Heaven, 2,000 Loot Crate
and 6,000 Classic episodes. Each final model is evaluated on 40 fresh paired
development seeds in all three scenarios and repeated deterministically.

## Decision rule

Accept the candidate only if all ten registered criteria pass:

- mean Classic collection is above control;
- the paired 95% bootstrap interval lower bound is above zero;
- at least four of five replicas improve Classic collection;
- candidate Classic self-kill rate is at most `0.055` and no higher than control;
- Coin Heaven collection decreases by no more than `0.05`;
- candidate/control mean visits per state is at least `0.90`;
- all deterministic repeats match;
- decision-time p95 is below `50 ms` and maximum below `100 ms`.

## Execution authorization

The owner explicitly instructed Codex to implement and execute the experiment
without the intermediate approval pause. This waiver is recorded before
training and does not alter the treatments, seeds, budget, metrics or criteria.

## Results

Pending execution.

## Interpretation and decision

Pending execution.

## AI assistance

OpenAI Codex assisted with implementation, tests, registration, execution,
analysis, plotting and documentation. AI output is not experimental evidence;
all reported values must be derived from retained framework outputs and
reviewed by the owner and a non-author reviewer.
