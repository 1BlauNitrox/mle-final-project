# Tabular Task 2 immediate useful-bomb reward

> Status: prospectively_registered

## Metadata

- Issue: #114
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-07
- Registration base: `933a8fe`
- Experiment commit: `eb215172b16146772dd543ae90813f852ad041e4`
- Framework revision: `0f55c1d`
- Execution authorization: user-directed; no separate pre-training peer approval

The requester explicitly authorized immediate execution without waiting for the
usual pre-training peer approval. This is a documented process exception, not a
claim that independent review occurred.

## Research question

Does an immediate reward for placing a bomb that can destroy at least one crate
improve Task 2 credit assignment and hidden-coin collection?

## Hypothesis

Adding `USEFUL_BOMB_PLACED: +1.0` will increase useful bomb placement, crate
destruction, and `classic` coin collection without materially increasing
self-kills.

## Baseline and treatment

- Control: unchanged unmasked Issue #102 configuration with a useful-bomb
  reward of `0.0`.
- Candidate: the same configuration with a useful-bomb reward of `+1.0`.

Both arms use the same untrained Task 2 artifact, SHA-256
`8f2e618bfb38d690b565be1d3034f153d120887a36d90a61f8adcc1a765c1bbb`.
No action masking is used.

## Controlled variables

Features, all other rewards, hyperparameters, parent artifact, curriculum,
training budget, seeds, scenarios, opponents, checkpoint selection, and
evaluation suites are identical between arms.

## Training protocol

Train five replicas per arm using the Issue #102 seeds. Each replica receives
2,000 `coin-heaven`, 2,000 `loot-crate`, and 6,000 `classic` episodes. Only the
final checkpoint after 10,000 episodes is evaluated.

## Evaluation protocol

Evaluate each replica on 40 paired development seeds in `classic`,
`coin-heaven`, and `loot-crate`, followed by an identical deterministic repeat.
Evaluate the frozen Task 1 artifact on the paired `coin-heaven` seeds.
Confirmation seeds remain unused.

## Metrics and decision rule

Primary metric: candidate-minus-control hidden-coin collection fraction on
`classic`.

The candidate passes only if:

- the primary mean difference is above zero;
- the paired 95% CI lower bound is above zero;
- at least four of five replicas improve;
- mean useful-bomb placements per training episode increase;
- mean crates destroyed per `classic` evaluation episode increase;
- aggregate `classic` self-kill rate is no more than control plus `0.02`;
- Task 1 retention CI is above `-0.05`;
- deterministic repeats and the registered runtime limits pass.

## Execution

```bash
python -m training.run_plan training/run_plans/issue114-tabular-task1-frozen.yaml --dry-run
python -m training.run_plan training/run_plans/issue114-tabular-task2-control.yaml --dry-run
python -m training.run_plan training/run_plans/issue114-tabular-task2-useful-bomb-reward.yaml --dry-run
```

No result was inspected before this protocol and its decision rule were written.
