# DerKleineSprengstoffkapitalist

> Status: behavior-preserving Task 2 successor scaffold.
>
> Current capability: Task 1 visible-coin navigation only.

## Purpose

`DerKleineSprengstoffkapitalist` is the tabular Task 2 successor of
`DerKleineVermoegensumverteiler`.

The successor was created for issue #65 so that Task 2 development can continue
without modifying the frozen Task 1 baseline. The initial successor deliberately
preserves the parent agent's observable behavior.

This change is a structural refactor and artifact migration only. It introduces:

- no Task 2 state features;
- no bomb action;
- no new rewards;
- no scientific training;
- no performance claim.

Future Task 2 behavior must be introduced through separately reviewed and
experimentally evaluated issues.

## Parent baseline

The immutable parent agent is:

```text
agent_code/DerKleineVermoegensumverteiler/
```

Its frozen source, configuration, documentation, and model artifact remain
unchanged.

The successor starts from the parent's selected Task 1 model artifact:

| Property | Value |
| --- | --- |
| Parent agent | `DerKleineVermoegensumverteiler` |
| Artifact | `model.npz` |
| SHA-256 | `4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307` |
| Size | `6845` bytes |
| Model schema | `2` |
| Feature schema | `1` |
| Producing agent commit | `0df4eb1b01d1dd6cef5c4111c42544468db1fc28` |
| Producing experiment commit | `5af393d4c2c5b46d04751b270f4d865aec41ccaf` |
| Frozen parent source commit | `22c91de3c97998d5c70b0109befbdecef3d34e90` |
| Parent freeze merge commit | `04f9dab8f6d160984a036a4c846756a12d1a0fb5` |
| Framework revision | `0f55c1d` |

The successor artifact is copied and verified by:

```text
scripts/migrate_task2_tabular_artifact.py
```

Machine-readable successor lineage is recorded in `artifact.json`. The original
parent manifest is preserved as `parent-artifact.json`.

`baseline-config.yaml` and `reference-results.csv` are inherited Task 1
provenance records. They describe the frozen parent experiment and must not be
interpreted as a Task 2 successor configuration or as new successor evidence.

## Current behavior contract

The successor currently preserves the complete Task 1 behavior contract.

Its ordered action set is:

```text
UP, RIGHT, DOWN, LEFT, WAIT
```

`BOMB` is deliberately excluded. Bomb placement is not implemented by issue
#65.

Evaluation:

- loads the committed `model.npz`;
- uses epsilon `0`;
- performs no learning;
- does not modify the model artifact;
- uses no multiprocessing;
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

`assemble.py` constructs the complete feature tuple. `navigation.py` contains
the existing local-navigation and nearest-coin helpers. `features/__init__.py`
provides the stable public feature API.

This reorganization must not change the returned feature tuple.

## Learning method

The inherited model is a tabular Q-learning policy. Each encoded state maps to
five Q-values in the fixed action order:

```text
UP, RIGHT, DOWN, LEFT, WAIT
```

The learning rate remains `0.05` and the discount factor remains `0.9`.

No learning rule, Q-value, action order, or hyperparameter is changed by issue
#65.

## Rewards

The inherited reward mapping is unchanged:

| Event | Reward |
| --- | ---: |
| `COIN_COLLECTED` | `+10.0` |
| `INVALID_ACTION` | `-0.5` |
| `WAITED` | `-0.1` |
| `MOVED_TOWARDS_COIN` | `+0.1` |
| `MOVED_AWAY_FROM_COIN` | `-0.1` |

These rewards document the inherited Task 1 behavior. Issue #65 does not add
Task 2 reward shaping.

## Training status

Scientific training is not authorized as part of issue #65.

Calling `setup_training()` currently raises a `RuntimeError` before model
updates or artifact writes can occur. This prevents the migrated parent artifact
from being accidentally overwritten during the behavior-preserving scaffold
change.

A later Task 2 issue may deliberately enable training after defining:

- a Task 2 feature contract;
- an extended action contract;
- rewards;
- controlled variables;
- fixed seeds;
- evaluation metrics;
- success criteria;
- artifact provenance.

## Evaluation

A local read-only smoke evaluation can be run with:

```bash
python -m training.run_experiment \
  --agent DerKleineSprengstoffkapitalist \
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
4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307
```

## Validation

Behavior preservation is checked by tests that compare parent and successor:

- artifact bytes and SHA-256 checksum;
- model and feature schema;
- action order;
- hyperparameters and rewards;
- feature tuples for representative game states;
- all stored Q-table values;
- deterministic read-only actions;
- artifact immutability during evaluation;
- absence of `BOMB`.

The successor package is additionally evaluated after being copied into a clean
framework tree.

## Package structure

```text
DerKleineSprengstoffkapitalist/
├── README.md
├── artifact.json
├── parent-artifact.json
├── baseline-config.yaml
├── reference-results.cvs
├── requirements.txt
├── callbacks.py
├── config.py
├── model.py
├── model.npz
├── persistence.py
├── rewards.py
├── train.py
└── features/
    ├── __init__.py
    ├── assemble.py
    └── navigation.py
```

Evaluation-time code is self-contained inside this directory. It does not import
from the parent agent, `training/`, `experiments/`, or `scripts/`.

Parent imports are permitted only in repository-level differential tests.

## Limitations and next steps

The current successor:

- cannot place bombs;
- cannot destroy crates;
- has no explosion-danger representation;
- has no escape planning;
- has no opponent model;
- is not a complete Task 2 agent;
- has not produced new experimental results.

Issue #65 only establishes the behavior-preserving starting point. Task 2
capabilities and experiments follow in separate issues, beginning with the
planned work tracked by issue #45.
