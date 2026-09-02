# Task 1 Stability improvement through slower epsilon decay

## Metadata
- Issue: [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Experiment run commit: auto
- Agent implementation commit: `6046e822bbf4e97dd9ff9dfe0f11b28afea81f2d`
- Agent implementation commit before merge: `6471fd27f2b8c673e2f38c099a9943227d85cbe2`
- PR-32 merge commit: `59f454c3cd84f03832b7c43f5b1b4adf053001d0`
- Framework revision: `0f55c1d`
- Agent: `DerKleineVermoegensumverteiler`
- Date: 2026-09-02
- Owner: LiliWestermann

## Research question
Can we get more stable training?

## Hypothesis
If we change the epsilon decay from 0.99 to 0.9995 the epsilon reaches its 
minimal value after approximatly 4600 episodes and not already after approximatly
229 episodes. Therefore the agents explores longer and gets less stuck early in
bad strategies.

## Baseline
The immediate baseline is
`2026-09-01-task1-MovementCoinReward-DerKleineVermoegensumverteiler`.

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
The independent variable is the epsilon decay. We change it from the initial
0.99 to 0.9995.

## Controlled variables
Compared with the baseline experiment, the following variables remain unchanged:

- agent features and Q-learning algorithm
- learning rate and discount factor
- rewards
- scenario and number of available coins
- number of training runs
- training episodes per run
- evaluation seeds
- number of evaluation episodes
- deterministic evaluation policy
- checkpoint selection
- compute and runtime settings

Only the epsilon decay is changed.

## Metrics and success criterion
The metrics and thresholds are registered in `config.yaml`. They must not be changed 
after the first scientific training run begins.

## Results


## Interpretation


## Decision and follow-up
