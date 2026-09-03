# Task 1 Lower learningrate

## Metadata
- Issue: [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Experiment run commit: `5af393d4c2c5b46d04751b270f4d865aec41ccaf`
- Agent implementation commit: `0df4eb1b01d1dd6cef5c4111c42544468db1fc28`
- Agent implementation commit before merge: `6471fd27f2b8c673e2f38c099a9943227d85cbe2`
- PR-32 merge commit: `59f454c3cd84f03832b7c43f5b1b4adf053001d0`
- Framework revision: `0f55c1d`
- Agent: `DerKleineVermoegensumverteiler`
- Date: 2026-09-02
- Owner: LiliWestermann

## Research question
Can we get more stable training?

## Hypothesis
If we change the learning rate from 0.1 to 0.05 singular random events won't
change the Q-values as much which can reduce the differences between the
independent training runs.

## Baseline
The immediate baseline is
`2026-09-01-task1-MovementCoinReward-DerKleineVermoegensumverteiler`.

Its between-model standard deviation was `0.1507`, its worst-model collection
fraction was `0.6535`, and its aggregate collection fraction was `0.9196`.

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
The independent variable is the learning rate. We change it from the initial
0.1 to 0.05.

## Controlled variables
Compared with the baseline experiment, the following variables remain unchanged:

- agent features and Q-learning algorithm
- epsilon decay
- rewards
- scenario and number of available coins
- number of training runs
- training episodes per run
- evaluation seeds
- number of evaluation episodes
- deterministic evaluation policy
- checkpoint selection
- compute and runtime settings

Only the learning rate is changed.

## Metrics and success criterion
The metrics and thresholds are registered in `config.yaml`. They must not be changed
after the first scientific training run begins.

## Results
All five training runs completed successfully. Each model was trained for
10,000 episodes and evaluated deterministically on the 40 registered
development seeds. This resulted in 200 evaluation episodes.

| Model | Training seed | Agent seed | Mean coins | Collection fraction | Full-clear rate | Steps per coin | Invalid-action rate | WAIT actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 1 | 11006 | 21006 | 50.000 | 1.0000 | 1.000 | 2.607 | 0.000384 | 4 |
| Run 2 | 11007 | 21007 | 45.300 | 0.9060 | 0.650 | 4.976 | 0.000665 | 0 |
| Run 3 | 11008 | 21008 | 50.000 | 1.0000 | 1.000 | 2.614 | 0.000000 | 0 |
| Run 4 | 11009 | 21009 | 50.000 | 1.0000 | 1.000 | 2.625 | 0.000191 | 3 |
| Run 5 | 11010 | 21010 | 50.000 | 1.0000 | 1.000 | 2.631 | 0.001140 | 0 |
| **Aggregate** | — | — | **49.060** | **0.9812** | **0.930** | **3.054** | **0.000501** | **7** |

Across all 200 evaluation episodes, the models collected 9,812 of the 10,000
available coins. A complete clear was achieved in 186 episodes, resulting in
an aggregate full-clear rate of 93.0%. No evaluation episode ended without
collecting a coin.

### Stability comparison with the immediate baseline
| Stability metric | Immediate baseline | Lower learning rate | Result |
| --- | ---: | ---: | :---: |
| Between-model SD collection fraction | 0.1507 | 0.0420 | Better |
| Minimum model collection fraction | 0.6535 | 0.9060 | Better |
| Maximum model collection fraction | 1.0000 | 1.0000 | Equal |
| Collection-fraction range | 0.3465 | 0.0940 | Better |
| Between-model SD steps per coin | 2.547 | 1.054 | Better |

### Aggregate comparison with the immediate baseline
| Metric | Immediate baseline | Lower learning rate | Difference |
| --- | ---: | ---: | ---: |
| Mean coins | 45.980 | 49.060 | +3.080 |
| Collection fraction | 0.9196 | 0.9812 | +0.0616 |
| Full-clear rate | 0.875 | 0.930 | +0.055 |
| Steps per coin | 3.598 | 3.054 | -0.544 |
| Coins per 100 steps | 27.790 | 32.743 | +4.953 |
| Invalid-action rate | 0.000484 | 0.000501 | +0.000017 |

The between-model standard deviation of the collection fraction decreased by
approximately 72.1%. The aggregate collection fraction increased by 6.16
percentage points, while steps per coin decreased by approximately 15.1%.

### Success criteria
| Criterion | Required | Observed | Result |
| --- | ---: | ---: | :---: |
| Completed models | 5 | 5 | Pass |
| Completed evaluation episodes | 200 | 200 | Pass |
| Aggregate collection fraction | >= 0.80 | 0.9812 | Pass |
| Models reaching collection fraction >= 0.75 | 5 | 5 | Pass |
| Aggregate invalid-action rate | < 0.01 | 0.000501 | Pass |
| Bomb action count | 0 | 0 | Pass |
| Deterministic evaluation | Required | Enabled | Pass |
| Models unchanged during evaluation | Required | Verified | Pass |
| Mean episode p95 decision time | < 50 ms | 0.0298 ms | Pass |
| Maximum decision time | < 100 ms | 0.780 ms | Pass |
| Individual steps per coin | < 2.7 | Best: 2.607; Run 2: 4.976 | **Fail** |
| Aggregate steps per coin | < 3.599 | 3.054 | Pass |

## Interpretation
The hypothesis was supported. Reducing the learning rate from `0.1` to `0.05`
substantially reduced the variation between the five paired training runs.

The between-model standard deviation of the mean collection fraction decreased
from `0.1507` to `0.0420`, and the collection-fraction range decreased from
`0.3465` to `0.0940`. The worst-model collection fraction increased from
`0.6535` to `0.9060`. These results show that the lower learning rate produced
more consistent policies across the registered training seeds.

Aggregate performance also improved. The coin collection fraction increased
from `0.9196` to `0.9812`, while the full-clear rate increased from `87.5%` to
`93.0%`. At the same time, steps per coin decreased from `3.598` to `3.054`,
so that the improvement in stability did not come at the cost of
collection efficiency.

A likely explanation is that the lower learning rate reduces the influence of
individual random transitions on the Q-values. Updates are less volatile, which
allows the models to converge to more similar policies across different
training trajectories.

Run 2 remained weaker than the other four models. Its collection fraction was
still high at `0.9060`, but it required `4.976` steps per coin. Therefore,
training stability improved substantially but is not perfect.

All registered aggregate performance and integrity criteria were met. The
individual steps-per-coin criterion was not met because Run 2 exceeded the
threshold of `2.7`. Overall, the lower learning rate of `0.05` should be
preferred over the previous value of `0.1`.

## Decision and follow-up
The lower learning rate of 0.05 is accepted and will become the new baseline,
because it substantially reduced between-model variation while also improving
aggregate coin collection, full clear rate and steps per coin.
