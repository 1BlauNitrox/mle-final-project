# Tabular Task 2 legal-action masking

> Status: registered

## Metadata

- Issue: #110
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-07
- Experiment commit: `6ceebe3fe6c75a378512cb7c7ebd022ce302df1c`
- Framework revision: `0f55c1d`

## Research question

Does consistent framework-legal action masking improve the Task 2 tabular
agent compared with an otherwise identical unmasked control?

## Hypothesis

Applying the same framework-legal mask during exploration, greedy action
selection and Bellman bootstrapping reduces invalid actions and improves
hidden-coin collection on `classic`.

## Experimental variable

The only changed factor is action masking:

- control: `action_masking: none`;
- candidate: `action_masking: framework_legal`.

The mask excludes only actions rejected by the framework. It does not exclude
dangerous but executable actions.

## Controlled variables

Both treatments use identical features, rewards, hyperparameters, parent
artifact, training curriculum, training budget, seeds, scenarios, opponents,
checkpoint selection and evaluation suites.

## Success criteria

The candidate passes if:

- mean classic collection fraction exceeds the control;
- the paired 95% confidence-interval lower bound is above zero;
- at least four of five replicas improve;
- invalid-action rate is below `0.01`;
- Task 1 retention CI remains above `-0.05`;
- no bombs are selected on `coin-heaven`;
- self-kill rate increases by no more than `0.02`;
- p95 decision time is below `50 ms`;
- maximum decision time is below `100 ms`;
- deterministic repeats match.

Confirmation seeds remain unused.

## Execution

Training begins only after the experiment definition has been reviewed.