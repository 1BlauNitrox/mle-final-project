# Task 1 Penalty for movement in general

## Metadata
- Issue: [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Experiment run commit: auto
- Agent implementation commit before merge: `6471fd27f2b8c673e2f38c099a9943227d85cbe2`
- PR-32 merge commit: `59f454c3cd84f03832b7c43f5b1b4adf053001d0`
- Framework revision: `0f55c1d`
- Agent: `DerKleineVermoegensumverteiler`
- Date: 2026-09-02
- Owner: LiliWestermann

## Research question
Can the agent get faster in collecting all coins? So are the steps required to
collect each coin getting smaller if we change the rewards?

## Hypothesis
If we add a penalty for movement in general, the agent will take as few steps
as possible and the steps per coin will therefore decrease.

## Baseline
The immediate baseline is
`2026-09-01-task1-MovementCoinReward-DerKleineVermoegensumverteiler`.
Its aggregate steps-per-coin value is `3.598` and its aggregate coin collection
fraction is `0.9196`.

## Setup
The complete machine-readable configuration is stored in `config.yaml`.

- Scenario: `coin-heaven`
- Opponents: none
- Training runs: 5
- Training episodes per run: 10,000
- Evaluation episodes: 40 per model
- Checkpoint selection: final checkpoint
- Training disabled during evaluation

## Independent variable
The independent variable is the reward function. We added the four Rewards 
MOVED_LEFT: -0.1, MOVED_RIGHT: -0.1, MOVED_UP: -0.1 and MOVED_DOWN: -0.1.

## Controlled variables
Compared with the baseline experiment, the following variables remain unchanged:

- agent features and Q-learning algorithm
- learning rate and discount factor
- epsilon schedule
- scenario and number of available coins
- number of training runs
- training episodes per run
- evaluation seeds
- number of evaluation episodes
- deterministic evaluation policy
- checkpoint selection
- compute and runtime settings

Only the reward function is changed.

## Metrics and success criterion
The metrics and thresholds are registered in `config.yaml`. They must not be changed 
after the first scientific training run begins.

## Results


## Interpretation


## Decision and follow-up

