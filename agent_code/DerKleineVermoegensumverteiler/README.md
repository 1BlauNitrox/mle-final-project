# DerKleineVermoegensumverteiler

> Status: Task 1 Q-learning baseline with movement-coin shaping and
> learning rate, selected by the #35 optimization campaign.

## Hypothesis

A compact representation of local movement constraints and coarse information
about the nearest visible coin is sufficient for a tabular Q-learning policy to
learn coin navigation in the `coin-heaven` scenario.

This hypothesis has not yet been evaluated scientifically. The completed smoke
runs validate only implementation and pipeline integration.

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

The initial implementation uses:

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

## Training

Run a reproducible headless training job from the repository root:

```bash
python -m training.run_experiment \
  --agent DerKleineVermoegensumverteiler \
  --mode training \
  --scenario coin-heaven \
  --rounds 100 \
  --world-seed 1001 \
  --agent-seed 2001
```

Training updates the Q-table, decays epsilon once after every completed episode,
records learning metrics, and saves the resumable model after each episode.

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

Training writes the model atomically. Evaluation requires a compatible artifact
and leaves it byte-for-byte unchanged.

## Smoke validation

A five-round training smoke run and subsequent five-round evaluation smoke run
completed successfully with recorded world and agent seeds.

The model SHA-256 before and after evaluation was identical:

```text
c29238f5306c83c5882816531f1b45d8d7297aebbded7591cb6ed75762563f93
```

This is pipeline evidence only. It is not scientific evidence of agent
performance or Task 1 completion.

## Dependencies

- NumPy

## Limitations and next steps

- Rewards and the learning rate were optimized by the #35 campaign; epsilon
  decay and the remaining hyperparameters were not.
- The feature representation contains no bomb-danger or opponent-strategy
  information.
- The agent is limited to `coin-heaven` and is not tournament-ready.
- The shaping is not potential-based, so a small policy bias cannot be excluded.
- Task 1 completion still requires the held-out evaluation defined in
  `docs/0007-task-1-baseline-contract.md`.