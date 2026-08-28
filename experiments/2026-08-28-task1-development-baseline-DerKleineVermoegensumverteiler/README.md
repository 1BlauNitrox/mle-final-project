# Task 1 Q-learning development baseline

## Metadata
- Issue: [#36](https://github.com/1BlauNitrox/mle-final-project/issues/36)
- Coordinating optimization issue:
  [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Commit:
- Agent implementation commit before merge: `6471fd27f2b8c673e2f38c099a9943227d85cbe2`
- PR-32 merge commit: `59f454c3cd84f03832b7c43f5b1b4adf053001d0`
- Framework revision: `0f55c1d`
- Agent: `DerKleineVermoegensumverteiler`
- Date: 2026-08-28
- Owner: LiliWestermann

## Hypothesis
Five training runs will provide a good development baseline for task 1 with the agent 
`DerKleineVermoegensumverteiler` unchanged after PR #32 implementation.

We expect at least 0.80 aggregated mean coin collection fraction -> 4 out of 5 
models reach 0.75 and all invalif action rates are below 0.01.

A negative or mixed result remains valid and will be retained.

## Setup
The complete machine-readable configuration is stored in `config.yaml`.

- Scenario: `coin-heaven`
- Opponents: none
- Training runs: 5
- Training episodes per run: 10,000
- Evaluation episodes: 40 per model
- Checkpoint selection: final checkpoint
- Training disabled during evaluation

## Independent Variables
Source of variation is the training run.

Each uses:
- a distinct training world seed
- a distinct agent seed
- a fresh model initialization

No design variable is deliberatly changed.

## Controlled Variables
The complete Agent implementation including Hyperparameters and Reward configuration,
aswell as the training and evaluation conditions remain unchanged.

## Metrics and success criterion
The metrics and thresholds are preregistered in Issue #36 and reproduced
in `config.yaml`. They must not be changed after the first scientific
training run begins.

## Results

## Interpretation

## Decision and follow-up