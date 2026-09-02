# Task 1 Penalty for movement in general

## Metadata
- Issue: [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Experiment run commit: `971c5aca516ed2a09476091c6bea8a44c3ddad4e`
- Agent implementation commit: `cb87843d7087c6a1cb1893778256f1781be71856`
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
All five training runs completed successfully. Each model was trained for
10,000 episodes and evaluated deterministically on the 40 registered
development seeds. This resulted in a total of 200 evaluation episodes.

| Model | Training seed | Agent seed | Mean coins | Collection fraction | Full-clear rate | Steps per coin | Invalid-action rate | WAIT actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 1 | 11011 | 21011 | 48.800 | 0.9760 | 0.975 | 3.004 | 0.000682 | 3 |
| Run 2 | 11012 | 21012 | 16.025 | 0.3205 | 0.025 | 24.565 | 0.547758 | 745 |
| Run 3 | 11013 | 21013 | 45.400 | 0.9080 | 0.825 | 3.964 | 0.000139 | 4 |
| Run 4 | 11014 | 21014 | 41.575 | 0.8315 | 0.500 | 6.347 | 0.000095 | 0 |
| Run 5 | 11015 | 21015 | 11.450 | 0.2290 | 0.025 | 34.358 | 0.000318 | 14,265 |
| **Aggregate** | — | — | **32.650** | **0.6530** | **0.470** | **8.438** | **0.156739** | **15,017** |

Across all 200 evaluation episodes, the five models collected 6,530 of the
10,000 available coins. A complete clear was achieved in 94 episodes, resulting
in an aggregate full-clear rate of 47.0%. Three episodes ended without
collecting a coin.

The models performed 55,098 attempted actions, including 8,636 invalid actions.
This corresponds to an aggregate invalid-action rate of 15.674%. No bomb
actions were selected.

### Success criteria
| Criterion | Required | Observed | Result |
| --- | ---: | ---: | :---: |
| Completed models | 5 | 5 | Pass |
| Completed evaluation episodes | 200 | 200 | Pass |
| Aggregate mean coin collection fraction | >= 0.90 | 0.6530 | **Fail** |
| Models reaching individual collection fraction | At least 4 models >= 0.75 | 3 models | **Fail** |
| Aggregate invalid-action rate | < 0.01 | 0.156739 | **Fail** |
| Bomb action count | 0 | 0 | Pass |
| Deterministic evaluation | Required | Enabled | Pass |
| Model unchanged during evaluation | Required | Verified | Pass |
| Mean episode p95 decision time | < 50 ms | 0.0305 ms | Pass |
| Maximum decision time | < 100 ms | 2.465 ms | Pass |
| Individual steps per coin | < 2.6 | Best model: 3.004 | **Fail** |
| Aggregate steps per coin | < 3.598 | 8.438 | **Fail** |

### Comparison with the immediate baseline
| Metric | Coin-distance baseline | Movement-penalty experiment | Difference |
| --- | ---: | ---: | ---: |
| Mean coins | 45.980 | 32.650 | -13.330 |
| Collection fraction | 0.9196 | 0.6530 | -0.2666 |
| Full-clear rate | 0.875 | 0.470 | -0.405 |
| Steps per coin | 3.598 | 8.438 | +4.839 |
| Coins per 100 steps | 27.790 | 11.852 | -15.938 |
| Invalid-action rate | 0.000484 | 0.156739 | +0.156255 |
| WAIT actions | 0 | 15,017 | +15,017 |

Compared with the immediate baseline, the movement penalty reduced the
collection fraction by 26.66 percentage points and increased the number of
steps per coin by approximately 134.5%.

## Interpretation
The hypothesis was not supported. Adding the general movement penalty increased
the steps per coin from 3.598 to 8.438 and the aggregated coin collection
fraction decreased form 0.9196 to 0.653.

We see that the runs vary even more, as the use of th WAIT action was extremely
high in run 5 and run 2 produced an invalid-action rate od 0.5478. This
indicates, that the new reward encourages undesirable behaviour rather than
reduce unnecessary movement as hoped.

An explanation could be, that the reward does not distinguish between necessary
and unnecessary movent, but penalizes each movement the same.

The registered collection-fraction, invalid-action-rate, individual-model and
steps-per-coin success criterie were not met.

## Decision and follow-up
This reward change is rejected and we continue with the previous coin-distance
rewards.

With that as the new baseline we now test if we can get stability improvements.
