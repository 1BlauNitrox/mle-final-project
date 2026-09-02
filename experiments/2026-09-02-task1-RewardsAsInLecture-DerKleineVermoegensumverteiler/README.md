# Task 1 Rewards according to formula from lecture

## Metadata
- Issue: [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Experiment run commit: auto
- Agent implementation commit: `401feb66b772f902a5bd7702bb8549b7d56a2f0d`
- Agent implementation commit before merge: `6471fd27f2b8c673e2f38c099a9943227d85cbe2`
- PR-32 merge commit: `59f454c3cd84f03832b7c43f5b1b4adf053001d0`
- Framework revision: `0f55c1d`
- Agent: `DerKleineVermoegensumverteiler`
- Date: 2026-09-02
- Owner: LiliWestermann

## Research question
Does changeing the rewards according to the lecture improve the coin
collection.

## Hypothesis
If we use the formula F(s,s') = γΨ(s') - Ψ(s) instead of fixed rewards
MOVED_TOWARDS_COIN and MOVED_AWAY_FROM_COIN we will preseve the tasks
optimal policy while still providing dense feedback during training.

Ψ is the negative Manhatten distance to the nearest remaining coin,
scaled by `0.1`and the terminal potential is `0.0`.

We expect the agent to preserve the strong aggregate performance of the
immediate lower-learning-rate baseline. The experiment will test whether this
policy-preserving shaping formulation also reduces variation between
independently trained models compared with the fixed `+0.1/-0.1` directional
rewards.

## Baseline
The immediate baseline is
`2026-09-01-task1-LowerLearningRate-DerKleineVermoegensumverteiler`.

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
The independent variables are the rewards. We add reward shaping instead of
the fixed rewards "MOVED_TOWARDS_COIN" and "MOVED_AWAY_FROM_COIN".

## Controlled variables
Compared with the baseline experiment, the following variables remain unchanged:

- agent features and Q-learning algorithm
- epsilon decay
- learning rate
- scenario and number of available coins
- number of training runs
- training episodes per run
- evaluation seeds
- number of evaluation episodes
- deterministic evaluation policy
- checkpoint selection
- compute and runtime settings

Only the rewards are changed.

## Metrics and success criterion
The metrics and thresholds are registered in `config.yaml`. They must not be changed
after the first scientific training run begins.

## Results

## Interpretation

## Decision and follow-up

