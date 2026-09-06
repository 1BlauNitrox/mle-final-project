# DerKleineSprengstoffkapitalist

> Status: implemented tabular Task 2 successor.
>
> Current capability: coin collection, bomb placement, crate destruction,
> explosion-danger representation and escape-aware movement.

## Purpose

`DerKleineSprengstoffkapitalist` is the tabular Task 2 successor of
`DerKleineVermoegensumverteiler`.

The successor was created for issue #65 so that Task 2 development can continue
without modifying the frozen Task 1 baseline. The initial successor deliberately
preserves the parent agent's observable behavior.

Issue #45 extends the behavior-preserving successor created by issue #65 with:

- a six-action space including `BOMB`;
- a 17-value Task 2 feature representation;
- bomb, crate, danger and escape features;
- sparse Task 2 Q-values initialized from the frozen Task 1 prior;
- native Task 2 event rewards;
- resumable Task 2 training;
- per-episode Task 2 training diagnostics.

No scientific training or performance claim is introduced by this
implementation change.

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

Each Task 2 state maps to six Q-values in the fixed action order:

```text
UP, RIGHT, DOWN, LEFT, WAIT, BOMB
```

For a previously unseen Task 2 state, the first five action values are
initialized from the corresponding frozen eight-feature Task 1 state. The
initial BOMB value is set below the minimum parent value by the configured
bomb-prior margin.
The Task 2 Q-table remains sparse and materializes states only when they are
updated during training.

## State representation

Feature schema version `2` contains 17 categorical values. Indices `0-7`
remain the exact Task 1 projection:

| Index | Feature | Domain |
| ---: | --- | --- |
| 0-3 | `free_up/right/down/left` | `{0, 1}` |
| 4 | `coin_visible` | `{0, 1}` |
| 5-6 | `coin_dx`, `coin_dy` | `{-1, 0, 1}` |
| 7 | `coin_distance_bin` | `{0, 1, 2, 3}` |
| 8 | `bomb_available` | `{0, 1}` |
| 9 | `current_danger_bin` | `{0, 1, 2, 3}` |
| 10 | `safe_direction_mask` | `{0, ..., 15}` |
| 11 | `escape_after_bomb` | `{0, 1}` |
| 12 | `crate_visible` | `{0, 1}` |
| 13-14 | `crate_dx`, `crate_dy` | `{-1, 0, 1}` |
| 15 | `crate_distance_bin` | `{0, 1, 2, 3}` |
| 16 | `crates_in_current_blast_bin` | `{0, 1, 2, 3}` |

The safe-direction mask uses bit 0 for `UP`, bit 1 for `RIGHT`, bit 2 for
`DOWN`, and bit 3 for `LEFT`.

Together with the unchanged `free_*` features, it distinguishes blocked,
traversable-but-dangerous, and traversable-and-safe directions.

A useful bomb target is derived as:

```text
crates_in_current_blast_bin > 0
```

It is not stored as a redundant additional feature.

The theoretical Cartesian upper bound is 84,934,656 states. The Q-table
remains sparse and creates entries only for states encountered during training.
Many combinations are inconsistent or unreachable.

## Learning method

The inherited model is a tabular Q-learning policy. Each encoded state maps to
five Q-values in the fixed action order:

```text
UP, RIGHT, DOWN, LEFT, WAIT
```

The learning rate remains `0.05` and the discount factor remains `0.9`.

No learning rule, Q-value, action order, or hyperparameter is changed by issue
#65.

## Learning metrics

At the end of every training episode, `end_of_round()` returns the following
diagnostics:

| Metric | Meaning |
| --- | --- |
| `coins_collected` | Number of `COIN_COLLECTED` events |
| `coins_found` | Number of `COIN_FOUND` events |
| `crates_destroyed` | Number of `CRATE_DESTROYED` events |
| `bombs_dropped` | Number of confirmed `BOMB_DROPPED` events |
| `useful_bombs` | Bombs placed where at least one crate is in the blast range |
| `self_kills` | Number of `KILLED_SELF` events |
| `survived_round` | Number of `SURVIVED_ROUND` events |
| `invalid_actions` | Number of `INVALID_ACTION` events |
| `shaped_reward` | Sum of all rewards used for Q-learning during the episode |
| `q_table_size` | Number of materialized Task 2 states |
| `mean_abs_td_error` | Mean absolute temporal-difference error |
| `epsilon` | Exploration rate used during the completed episode |

`useful_bombs` is a diagnostic count only. It does not introduce an
additional reward. Potential-based reward shaping remains outside the scope
of issue #45.

## Rewards

| Event | Reward |
| --- | ---: |
| `COIN_COLLECTED` | `+10.0` |
| `INVALID_ACTION` | `-0.5` |
| `WAITED` | `-0.1` |
| `MOVED_TOWARDS_COIN` | `+0.1` |
| `MOVED_AWAY_FROM_COIN` | `-0.1` |
| `CRATE_DESTROYED` | `+1.0`|
| `COIN_FOUND` | `+2.0`|
| `KILLED_SELF` | `-10.0` |
| `GOT_KILLED` | `-10.0` |
| `SURVIVED_ROUND` | `+5.0` |

The Task 1 reward mapping, including the existing movement-distance shaping,
is inherited unchanged. Issue #45 adds only native Task 2 framework-event
rewards for destroyed crates, revealed coins, death, and survival.

No direct reward is assigned to `BOMB_DROPPED` or `BOMB_EXPLODED`. Bomb
placement must receive value through its later consequences.

Potential-based reward shaping is deliberately out of scope for issue #45.
Changing the inherited shaping formulation would be a separate controlled
variable and requires a prospectively registered experiment.

## Training status

Task 2 training is implemented and enabled. A newly initialized agent starts
from the frozen Task 1 parent prior and extends it with the `BOMB` action.

The implementation may be exercised with short deterministic smoke tests.
Scientific training runs and hyperparameter changes must be performed as
separately preregistered experiments.

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

The Task 2 implementation is validated by tests covering:

- preservation of the first eight Task 1 feature values;
- the complete 17-value Task 2 feature schema;
- bomb, crate, danger and escape geometry;
- the six-action contract including `BOMB`;
- initialization from the frozen Task 1 prior;
- conservative initialization of unseen `BOMB` values;
- sparse schema-validated model persistence;
- ordinary and terminal Q-learning updates;
- prevention of terminal bootstrapping;
- prevention of duplicate final-transition updates;
- epsilon decay and resumable model persistence;
- native Task 2 rewards;
- complete per-episode training diagnostics;
- read-only evaluation behavior and artifact integrity.

## Package structure

```text
DerKleineSprengstoffkapitalist/
├── README.md
├── artifact.json
├── parent-artifact.json
├── baseline-config.yaml
├── reference-results.csv
├── requirements.txt
├── callbacks.py
├── config.py
├── migration.py
├── model.py
├── model.npz
├── parent-model.npz
├── persistence.py
├── rewards.py
├── train.py
└── features/
    ├── __init__.py
    ├── assemble.py
    ├── bombs_and_crates.py
    └── navigation.py
```

Evaluation-time code is self-contained inside this directory. It does not import
from the parent agent, `training/`, `experiments/`, or `scripts/`.

Parent imports are permitted only in repository-level differential tests.

## Limitations and next steps

The current successor:

- has not yet been scientifically trained for Task 2;
- has not yet produced Task 2 performance evidence;
- uses a categorical sparse state representation;
- does not contain a full opponent strategy;
- retains the inherited non-potential-based coin movement shaping;
- requires separately preregistered experiments for reward or
  hyperparameter changes.

Potential-based reward shaping and further reward tuning must be evaluated as
separate controlled experiments.