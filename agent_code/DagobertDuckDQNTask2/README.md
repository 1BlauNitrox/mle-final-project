# DagobertDuckDQNTask2

> Status: behavior-preserving Task 2 successor scaffold.
>
> Current capability: Task 1 visible-coin navigation only.

## Purpose

`DagobertDuckDQNTask2` is the DQN Task 2 successor of `DagobertDuckDQN`.

The successor was created for issue #43 so that Task 2 development can continue
without modifying the frozen Task 1 baseline. The initial successor deliberately
preserves the parent agent's observable behavior.

This change is a structural refactor and artifact migration only. It introduces:

- no Task 2 state features;
- no bomb action;
- no new rewards;
- no scientific training;
- no performance claim.

Future Task 2 behavior must be introduced through separately reviewed and
experimentally evaluated issues (starting with issue #44).

## Parent baseline

The immutable parent agent is:

```text
agent_code/DagobertDuckDQN/
```

Its frozen source, configuration, documentation, and model artifact remain
unchanged.

The successor starts from the parent's selected Task 1 model artifact:

| Property | Value |
| --- | --- |
| Parent agent | `DagobertDuckDQN` |
| Artifact | `checkpoint.pt` |
| SHA-256 | `45e38fa8900acd0783a84c339bf81d7e718de7797fbeeb147b5db94da3e96649` |
| Size | `47280` bytes |
| Checkpoint schema | `1` |
| Model schema | `1` |
| Feature schema | `1` |
| Producing agent commit | `84396eb771a7eb3e3daa028a415e9b00570b20c8` |
| Producing experiment commit | `dbe259721409092448e8594cbeaaca469c4f9835` |
| Frozen parent source commit | `d81eee00ffa4b095edbcf3be6510594dda9a9733` |
| Parent freeze PR tip (PR #72, pending merge) | `7a1d302da78862f4fd1fc6bf6a43522d20979a87` |
| Framework revision | `0f55c1d` |

The successor artifact is copied and verified by:

```text
scripts/migrate_task2_dqn_artifact.py
```

Machine-readable successor lineage is recorded in `artifact.json`. The original
parent manifest is preserved as `parent-artifact.json`.

`baseline-config.yaml` and `reference-results.csv` are inherited Task 1
provenance records. They describe the frozen parent experiment (issue #58,
selected under issue #42) and must not be interpreted as a Task 2 successor
configuration or as new successor evidence.

## Current behavior contract

The successor currently preserves the complete Task 1 behavior contract.

Its ordered action set is:

```text
UP, RIGHT, DOWN, LEFT, WAIT
```

`BOMB` is deliberately excluded. Bomb placement is not implemented by issue #43.

Evaluation:

- loads the committed `checkpoint.pt`;
- uses epsilon `0`;
- performs no learning;
- does not modify the checkpoint;
- uses one PyTorch CPU thread and no multiprocessing;
- selects only actions from the five-action Task 1 action space.

## State representation

The state is encoded as the same eight-element tuple used by the parent:

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

The meaning of these features is unchanged:

- the first four values indicate whether adjacent tiles are traversable;
- `coin_visible` indicates whether at least one coin is visible;
- `coin_dx` and `coin_dy` contain the direction signs to the nearest coin;
- `coin_distance_bin` contains the coarse Manhattan-distance category.

The feature schema remains version `1` and the feature count remains `8`.

The implementation has only been reorganized into:

```text
features/
├── __init__.py
├── assemble.py
└── navigation.py
```

`assemble.py` constructs the complete feature tuple and normalizes it for the
network. `navigation.py` contains the existing local-navigation and
nearest-coin helpers. `features/__init__.py` provides the stable public
feature API.

This reorganization must not change the returned feature tuple or its
normalized form.

## Learning method

The inherited model is a small feed-forward Deep Q-Network:

```text
8 inputs -> Linear(8, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 5)
```

Q-values map to the fixed action order `UP, RIGHT, DOWN, LEFT, WAIT`. The
learning rate remains `0.001` and the discount factor remains `0.9`.

No network architecture, weight, action order, or hyperparameter is changed by
issue #43.

## Rewards

The inherited reward mapping is unchanged:

| Event | Reward |
| --- | ---: |
| `COIN_COLLECTED` | `+10.0` |
| `INVALID_ACTION` | `-0.5` |
| `WAITED` | `-0.1` |
| `MOVED_TOWARDS_COIN` | `+0.1` |
| `MOVED_AWAY_FROM_COIN` | `-0.1` |

These rewards document the inherited Task 1 behavior, including the movement-
coin shaping tested in issue #58. Issue #43 does not add Task 2 reward shaping.

## Training status

Scientific training is not authorized as part of issue #43.

Calling `setup_training()` currently raises a `RuntimeError` before optimizer
updates or checkpoint writes can occur. This prevents the migrated parent
artifact from being accidentally overwritten during the behavior-preserving
scaffold change.

A later Task 2 issue (#44) may deliberately enable training after defining:

- a Task 2 feature contract (bomb, crate, and danger representation);
- an extended six-action contract (`BOMB` appended, preserving the five Task 1
  indices);
- Task 2 rewards;
- controlled checkpoint migration from this scaffold's weights;
- fixed seeds and a curriculum;
- evaluation metrics and success criteria;
- artifact provenance.

## Evaluation

A local read-only smoke evaluation can be run with:

```bash
python -m training.run_experiment \
  --agent DagobertDuckDQNTask2 \
  --mode evaluation \
  --scenario coin-heaven \
  --rounds 2 \
  --world-seed 3001 \
  --agent-seed 2001
```

This is a compatibility and regression check, not a scientific experiment and
not performance evidence.

The checksum before and after evaluation must remain:

```text
45e38fa8900acd0783a84c339bf81d7e718de7797fbeeb147b5db94da3e96649
```

## Validation

Behavior preservation is checked by tests that compare parent and successor:

- artifact bytes and SHA-256 checksum;
- checkpoint, model, and feature schema;
- action order;
- hyperparameters and rewards;
- feature tuples for representative game states;
- Q-values for the migrated network on representative and generated states;
- deterministic read-only actions;
- artifact immutability during evaluation;
- absence of `BOMB`.

The successor package is additionally evaluated after being copied into a clean
framework tree.

## Package structure

```text
DagobertDuckDQNTask2/
├── README.md
├── artifact.json
├── parent-artifact.json
├── baseline-config.yaml
├── reference-results.csv
├── requirements.txt
├── callbacks.py
├── config.py
├── model.py
├── checkpoint.pt
├── persistence.py
├── replay.py
├── rewards.py
├── train.py
└── features/
    ├── __init__.py
    ├── assemble.py
    └── navigation.py
```

Evaluation-time code is self-contained inside this directory. It does not
import from the parent agent, `training/`, `experiments/`, or `scripts/`.

Parent imports are permitted only in repository-level differential tests.

## Limitations and next steps

The current successor:

- cannot place bombs;
- cannot destroy crates;
- has no explosion-danger representation;
- has no escape planning;
- has no opponent model;
- is not a complete Task 2 agent;
- has not produced new experimental results;
- inherits the frozen parent's greedy-evaluation invalid-action failure mode
  documented in issue #58, since no feature or logic changed.

Issue #43 only establishes the behavior-preserving starting point. Task 2
capabilities and experiments follow in separate issues, beginning with the
planned work tracked by issue #44.
