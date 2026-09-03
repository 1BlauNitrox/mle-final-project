# Task 1 Rewards according to formula from lecture

## Metadata
- Issue: [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Experiment run commit: `fa23b903ca3e6ad4df809739d1ae61ffaa8681ac`
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
`2026-09-02-task1-LowerLearningRate-DerKleineVermoegensumverteiler`.

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
All five registered training runs completed successfully. Each model was
trained for 10,000 episodes and evaluated deterministically on the 40
registered development seeds. This resulted in 200 evaluation episodes.

| Model | Training seed | Agent seed | Mean coins | Collection fraction | Full-clear rate | Steps per coin | Invalid-action rate | WAIT actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 1 | 11006 | 21006 | 50.000 | 1.0000 | 1.000 | 2.669 | 0.000562 | 1 |
| Run 2 | 11007 | 21007 | 48.425 | 0.9685 | 0.875 | 3.364 | 0.000153 | 0 |
| Run 3 | 11008 | 21008 | 10.300 | 0.2060 | 0.000 | 38.835 | 0.000375 | 14,703 |
| Run 4 | 11009 | 21009 | 50.000 | 1.0000 | 1.000 | 2.614 | 0.000191 | 0 |
| Run 5 | 11010 | 21010 | 48.175 | 0.9635 | 0.875 | 3.429 | 0.000303 | 0 |
| **Aggregate** | — | — | **41.380** | **0.8276** | **0.750** | **4.796** | **0.000328** | **14,704** |

Across all 200 evaluation episodes, the models collected 8,276 of the 10,000
available coins. A complete clear was achieved in 150 episodes, resulting in
an aggregate full-clear rate of 75.0%. Three evaluation episodes ended without
collecting a coin.

Four models achieved collection fractions above 0.96. However, Run 3 only
achieved a collection fraction of 0.2060 and selected WAIT 14,703 times.
This single unstable training result strongly reduced aggregate performance
and increased between-model variation.

### Comparison with the immediate baseline
| Metric | Lower-learning-rate baseline | Potential-based shaping | Difference |
| --- | ---: | ---: | ---: |
| Mean coins | 49.060 | 41.380 | -7.680 |
| Collection fraction | 0.9812 | 0.8276 | -0.1536 |
| Full-clear rate | 0.930 | 0.750 | -0.180 |
| Steps per coin | 3.054 | 4.796 | +1.742 |
| Coins per 100 steps | 32.743 | 20.852 | -11.891 |
| Invalid-action rate | 0.000501 | 0.000328 | -0.000173 |

### Stability comparison
| Stability metric | Lower-learning-rate baseline | Potential-based shaping | Result |
| --- | ---: | ---: | :---: |
| Between-model SD collection fraction | 0.0420 | 0.3318 | Worse |
| Minimum model collection fraction | 0.9060 | 0.2060 | Worse |
| Maximum model collection fraction | 1.0000 | 1.0000 | Equal |
| Collection-fraction range | 0.0940 | 0.7940 | Worse |

### Success criteria
| Criterion | Required | Observed | Result |
| --- | ---: | ---: | :---: |
| Completed models | 5 | 5 | Pass |
| Completed evaluation episodes | 200 | 200 | Pass |
| Aggregate collection fraction | >= 0.80 | 0.8276 | Pass |
| Individual collection fraction | >= 0.75 | Minimum: 0.2060 | **Fail** |
| Models reaching individual threshold | 5 | 4 | **Fail** |
| Aggregate invalid-action rate | < 0.01 | 0.000328 | Pass |
| Bomb action count | 0 | 0 | Pass |
| Deterministic evaluation | Required | Enabled | Pass |
| Models unchanged during evaluation | Required | Verified | Pass |
| Mean episode p95 decision time | < 50 ms | 0.0299 ms | Pass |
| Maximum decision time | < 100 ms | 0.758 ms | Pass |
| Individual steps per coin | < 2.7 | Maximum: 38.835 | **Fail** |
| Aggregate steps per coin | < 3.599 | 4.796 | **Fail** |

## Interpretation
The hypothesis was not supported. Potential-based reward shaping preserved
strong performance in four of the five independently trained models, but it
did not reduce variation between models.

Compared with the immediate lower-learning-rate baseline, the aggregate coin
collection fraction decreased from 0.9812 to 0.8276. The between-model
standard deviation increased from 0.0420 to 0.3318, and the minimum model
collection fraction decreased from 0.9060 to 0.2060.

Run 3 was the main source of this instability. It selected WAIT in 14,703 of
16,000 evaluation steps and collected only 10.3 coins per episode on average.
The other four models collected between 48.175 and 50 coins per episode.
This indicates that the potential-based reward did not consistently guide
training towards the same effective policy.

The potential-based formulation is theoretically policy-preserving when the
potential represents the state appropriately. However, this experiment uses
finite training and a compact feature-based Q-table in which different game
states may share the same feature representation. Therefore, the theoretical
guarantee does not necessarily imply stable learning under this form of state
aggregation.

The experiment passed the aggregate collection threshold and all runtime,
action-safety, and integrity criteria. It failed the individual performance
and steps-per-coin criteria because of Run 3. Therefore, the new formulation
cannot be considered an improvement over the immediate baseline.

## Decision and follow-up
The potential-based reward-shaping implementation is rejected as the new
default for the current agent. The lower-learning-rate agent remains the
preferred Task 1 configuration because it achieved better aggregate
performance and substantially lower between-model variation.
