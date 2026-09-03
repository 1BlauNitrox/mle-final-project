# DerKleineVermoegensumverteiler

> Status: frozen Task 1 development-selected baseline.
> This directory must not be retrained or modified for Task 2.

These are development-selection results, not final held-out Task 1 results.

Task 2 development must use a separately named successor agent. The frozen
source, configuration, and model artifact in this directory remain unchanged
for Task 1 regression evaluation.

## Hypothesis

A compact representation of local movement constraints and coarse information
about the nearest visible coin is sufficient for a tabular Q-learning policy to
learn coin navigation in the `coin-heaven` scenario.

The hypothesis was evaluated through the development experiments in issues
#35 and #36. The selected configuration achieved an aggregate development
coin-collection fraction of `0.9812` across five independently trained models.

These results are development evidence only. The final held-out Task 1
evaluation has not been performed.

## Scope

The agent is limited to the Task 1 visible-coin-navigation milestone. It learns
to select from the ordered action set:

```text
UP, RIGHT, DOWN, LEFT, WAIT
```

`BOMB` is excluded during both training and evaluation.

Bomb placement, crate destruction, opponent modelling, self-play, and
tournament optimization are outside the current scope.

## Learning method

The agent uses feature-based tabular Q-learning. Each encoded state maps to five
floating-point action values.

Unseen state-action values are initialized to zero. Reading an unseen state
during evaluation does not add it to the Q-table.

For ordinary transitions, training applies:

```text
Q(s, a) <- Q(s, a) +
           alpha * (reward + gamma * max Q(s', a') - Q(s, a))
```

Terminal transitions use:

```text
Q(s, a) <- Q(s, a) +
           alpha * (reward - Q(s, a))
```

Terminal updates therefore do not bootstrap from a future state.

## State representation

Each game state is encoded as an eight-element integer tuple:

```text
(
    free_up,
    free_right,
    free_down,
    free_left,
    coin_visible,
    coin_dx,
    coin_dy,
    coin_distance_bin,
)
```

The four movement features are binary:

- `1`: the adjacent tile is currently traversable;
- `0`: the adjacent tile is blocked by the field, a bomb, or another agent.

`coin_visible` is `1` when at least one coin is visible and `0` otherwise.

`coin_dx` and `coin_dy` contain the signs of the coordinate differences to the
nearest visible coin. Their possible values are `-1`, `0`, and `1`.

The Manhattan-distance bin is:

- `0`: no coin is visible;
- `1`: distance 1;
- `2`: distance 2-3;
- `3`: distance 4 or greater.

Ties between equally distant coins are resolved deterministically by
coordinate. The encoder does not perform pathfinding and does not encode an
optimal action.

The independent feature ranges give an upper bound of 1,152 encoded
combinations. Many combinations are unreachable, for example a missing coin
with a non-zero coin direction.

## Hyperparameters

The frozen baseline uses:

| Parameter | Value |
| --- | ---: |
| Learning rate (`alpha`) | `0.05` |
| Discount factor (`gamma`) | `0.9` |
| Initial epsilon | `1.0` |
| Epsilon decay per episode | `0.99` |
| Minimum epsilon | `0.1` |
| Default agent seed | `0` |

Training uses epsilon-greedy exploration. Random exploration and greedy
tie-breaking use a NumPy random generator initialized from
`BOMBERMAN_AGENT_SEED`. Evaluation always uses epsilon `0`.

The learning rate was reduced from `0.1` in the LowerLearningRate experiment,
which cut the between-model standard deviation of the coin collection fraction
from `0.1507` to `0.0420`.

## Rewards

The Task 1 reward mapping is:

| Event | Reward | Rationale |
| --- | ---: | --- |
| `COIN_COLLECTED` | `+10.0` | Provides the primary task signal. |
| `INVALID_ACTION` | `-0.5` | Discourages attempts to enter blocked tiles. |
| `WAITED` | `-0.1` | Discourages unproductive waiting. |
| `MOVED_TOWARDS_COIN`| `+0.1`| Dense signal toward the nearest visible coin. |
| `MOVED_AWAY_FROM_COIN` | `-0.1`| Symmetric penalty; prevemts oscillation. |

All other framework events currently have reward zero.

`MOVED_TOWARDS_COIN` and `MOVED_AWAY_FROM_COIN` are custom events emitted by
`train.py`. For a movement action, the minimum Manhattan distance from the old
position to any coin visible in the old state is compared against the same
distance from the new position. A decrease emits `MOVED_TOWARDS_COIN`, an
increase emits `MOVED_AWAY_FROM_COIN`, and an unchanged distance, an absent coin
list, or a non-movement action emits nothing. Measuring both distances against
the old coin list keeps the event defined on the step a coin is collected.

The symmetric penalty is deliberate: returning to a tile cancels the reward
earned by leaving it, which removes the incentive to oscillate for repeated
approach reward.

## Training and immutability

Training is disabled for this frozen baseline. Calling `setup_training()`
raises an error before any training update can occur.

The committed `model.npz` must not be overwritten or replaced. Deliberate
retraining requires a new issue, a new artifact version, and a separately
named successor agent.

The original producing configuration is preserved in
`baseline-config.yaml`.

## Evaluation

Evaluation requires an existing compatible `model.npz`:

```bash
python -m training.run_experiment \
  --agent DerKleineVermoegensumverteiler \
  --mode evaluation \
  --scenario coin-heaven \
  --rounds 100 \
  --world-seed 3001 \
  --agent-seed 2001
```

Evaluation disables exploration and learning. It does not write or replace the
model.

## Learning metrics

The training callback reports:

- cumulative configured reward;
- epsilon used during the completed episode;
- Q-table size;
- mean absolute TD error.

These values are passed to the repository experiment pipeline without creating
an evaluation-time dependency on `training/`.

## Model persistence

The Q-table is stored as `model.npz` beside the agent source files. The path is
resolved relative to `persistence.py` and does not depend on the current working
directory.

The non-pickle archive contains:

- encoded states;
- Q-values;
- action order;
- model- and feature-schema versions;
- learning rate and discount factor;
- epsilon;
- completed episode count;
- reward configuration.

The producing implementation wrote models atomically during the completed
development experiment. Training is now disabled in this frozen directory.
Evaluation loads the committed artifact without modifying it.

## Frozen artifact

The committed evaluation artifact was produced by Run 1 of
`2026-09-02-task1-LowerLearningRate-DerKleineVermoegensumverteiler`.

| Property | Value |
| --- | --- |
| Artifact | `model.npz` |
| SHA-256 | `4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307` |
| Size | `6845` bytes |
| Model schema | `2` |
| Feature schema | `1` |
| Training world seed | `11006` |
| Agent seed | `21006` |
| Producing agent commit | `0df4eb1b01d1dd6cef5c4111c42544468db1fc28` |
| Producing experiment commit | `5af393d4c2c5b46d04751b270f4d865aec41ccaf` |
| Frozen source commit | `22c91de3c97998d5c70b0109befbdecef3d34e90` |
| Campaign merge commit | `9d58f3260dbdc11bdbe7ec8838acc94bce8d89c3` |
| Framework revision | `0f55c1d` |

The machine-readable version of this information is stored in
`artifact.json`.

## Dependencies

The frozen package documents its Python dependencies in `requirements.txt`.

The producing environment used:

- Python `3.13.15`
- NumPy `2.5.2`
- one CPU thread
- no multiprocessing

## Limitations and next steps

- Rewards and the learning rate were optimized by the #35 campaign; epsilon
  decay and the remaining hyperparameters were not.
- The feature representation contains no bomb-danger or opponent-strategy
  information.
- The agent is limited to `coin-heaven` and is not tournament-ready.
- The shaping is not potential-based, so a small policy bias cannot be excluded.
- Task 1 completion still requires the held-out evaluation defined in
  `docs/0007-task-1-baseline-contract.md`.

## Chosen model

For freezing we choos the model from run1 in the LowerLearningRate experiment.
It has one of the highest mean_collection_fraction with the lowest
steps_per_coin count.

| Metric | Run 1 |
| --- | ---: |
| Collection fraction | 1.0000 |
| Full clear rate | 1.0000 |
| Steps per coin | 2.6065 |
| Invalid action rate | 0.0003836562 |
| BOMB actions | 0 |

- Producing agent implementation commit:
  0df4eb1b01d1dd6cef5c4111c42544468db1fc28
- Producing experiment commit:
  5af393d4c2c5b46d04751b270f4d865aec41ccaf
- Producing agent source SHA-256:
  a74a9efeacc8c772675666f7768a7fbf3ee5bea509d36af93834405976335600
- Campaign merge commit:
  9d58f3260dbdc11bdbe7ec8838acc94bce8d89c3
- Framework revision:
  0f55c1d