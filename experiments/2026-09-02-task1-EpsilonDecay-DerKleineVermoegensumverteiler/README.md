# Task 1 Stability improvement through slower epsilon decay

## Metadata
- Issue: [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Experiment run commit: `4faf7bc7ab4ee66babf67b1f7b1d33b8ad439dc5`
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

### Success criteria
| Criterion | Required | Observed | Result |
| --- | ---: | ---: | :---: |
| Completed models | 5 | 5 | Pass |
| Completed evaluation episodes | 200 | 200 | Pass |
| Aggregate collection fraction | >= 0.80 | 0.7301 | **Fail** |
| Models reaching collection fraction >= 0.75 | 5 | 3 | **Fail** |
| Aggregate invalid-action rate | < 0.01 | 0.061336 | **Fail** |
| Bomb action count | 0 | 0 | Pass |
| Deterministic evaluation | Required | Enabled | Pass |
| Models unchanged during evaluation | Required | Verified | Pass |
| Mean episode p95 decision time | < 50 ms | 0.0287 ms | Pass |
| Maximum decision time | < 100 ms | 1.117 ms | Pass |
| Individual steps per coin | < 2.7 | Best model: 3.033 | **Fail** |
| Aggregate steps per coin | < 3.599 | 7.767 | **Fail** |

### Stability comparison
| Stability metric | Immediate baseline | Slower epsilon decay | Result |
| --- | ---: | ---: | :---: |
| Between-model SD collection fraction | 0.1507 | 0.2362 | Worse |
| Minimum model collection fraction | 0.6535 | 0.3505 | Worse |
| Maximum model collection fraction | 1.0000 | 0.9680 | Worse |
| Collection-fraction range | 0.3465 | 0.6175 | Worse |
| Between-model SD steps per coin | 2.547 | 7.285 | Worse |

## Interpretation
The hypothesis was not supperted. Chenging the epsilon made the variation
between independently traing model even worse.

The between-model standard deviation of the mean collection fraction increased
from `0.1507` to `0.2362`, while the worst-model collection fraction decreased
from `0.6535` to `0.3505`. The collection-fraction range increased from
`0.3465` to `0.6175`.

Aggregate performance also decreased. The collection fraction fell from
`0.9196` to `0.7301`, and steps per coin increased from `3.598` to `7.767`.
Only the first paired run improved slightly; the remaining paired runs
performed worse.

A likely explanation is that prolonged exploration prevents the tabular agent
from settling sufficiently early into a reliable policy within the fixed
10,000-episode training budget.

## Decision and follow-up

The slower epsilon decay is rejected. The previous coin-distance reward agent
with an epsilon decay of `0.99` remains the best registered version.
