# Task 1 Reward for movement in regard to coins

## Metadata
- Issue: [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Experiment run commit: `bf6f24abb9fa09a909ea5c27a42db394fc970bc9`
- Agent implementation commit before merge: `6471fd27f2b8c673e2f38c099a9943227d85cbe2`
- PR-32 merge commit: `59f454c3cd84f03832b7c43f5b1b4adf053001d0`
- Framework revision: `0f55c1d`
- Agent: `DerKleineVermoegensumverteiler`
- Date: 2026-09-01
- Owner: LiliWestermann

## Research question
Can the agent get faster in collecting all coins? So are the steps required to
collect each coin getting smaller if we change the rewards?

## Hypothesis
If we add a reward for movement in the direction of a coin the agent will prefer
to collect coins that are closer together and therefore won't have to walk 
across the whole board to collect the last coins, because he left them in 
otherwise deserted corners.

We expect a smaller steps_per_coin value than baseline.

## Baseline
see 2026-08-28-task1-development-baseline-DerKleineVermoegensumverteiler

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
The independent variable is the reward function. We added the two Rewards 
MOVED_TOWARDS_COIN: 0.1 and MOVED_AWAY_FROM_COIN: -0.1.

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

All five training runs completed successfully. Each final model was evaluated
deterministically on the 40 registered development seeds, resulting in 200
evaluation episodes.

### Aggregate evaluation results

| Model | Mean coins | Collection fraction | Full-clear rate | Steps per coin | Invalid-action rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run 1 | 47.225 | 0.9445 | 0.925 | 3.245 | 0.000816 |
| Run 2 | 50.000 | 1.0000 | 1.000 | 2.647 | 0.000000 |
| Run 3 | 50.000 | 1.0000 | 1.000 | 2.678 | 0.000747 |
| Run 4 | 32.675 | 0.6535 | 0.450 | 8.464 | 0.000452 |
| Run 5 | 50.000 | 1.0000 | 1.000 | 2.625 | 0.000381 |
| **Aggregate** | **45.980** | **0.9196** | **0.875** | **3.598** | **0.000484** |

### Success criteria

| Criterion | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Completed models | 5 | 5 | Pass |
| Evaluation episodes | 200 | 200 | Pass |
| Aggregate collection fraction | >= 0.80 | 0.9196 | Pass |
| Models with collection fraction >= 0.75 | >= 4 | 4 | Pass |
| Invalid-action rate | < 0.01 | 0.000484 | Pass |
| Bomb actions | 0 | 0 | Pass |
| Maximum decision time | < 100 ms | 1.307 ms | Pass |
| Individual steps per coin | < 2.6 | 2.625–8.464 | Fail |
| Aggregate steps per coin | < 4 | 3.598 | Pass |

## Interpretation

The Hypothesis is supported on the aggregate level. With the change we have
3.598 steps per coin, whereas the baseline had 4.031. And the coin collection
fraction increased from 0.8995 to 0.9196. 

The full-clear rate also increases from 81.5% to 87.5%. Three model collected
every coin over all evaluation episodes, but its not fully stable: run 3
collected only 65.35% an took 8.464 steps per coin.

The aggregate collection fraction criterion was met and as requested four out of 
five models met the collection fraction above 0.75.

The individual steps per coin threshold was not met, we hoped for values 
beneath 2.6, but no run reached that goal. For this criterion in overall best run
is still run 03 from the baseline experiment with 2.6.

## Decision and follow-up

The changes are a improvement because the aggregate collection efficiency and 
full-clear rate improved relative to the baseline.

Because the goal for individual steps per coin was not met we could try 
another reward instead, maybe a penalty for every movement, so that the agent chooses
to make as few as possible.

The variation between training runs also remains a concern. We should do an
follow up experiment, which investigates training stability.