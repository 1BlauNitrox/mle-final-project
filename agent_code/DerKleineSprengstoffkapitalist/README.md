# DerKleineSprengstoffkapitalist

> Status: vorläufig eingefrorener tabellarischer Task-2-Agent (Issue #183).
>
> Das eingefrorene Modell ist der Control-Run r2 aus Issue #153.
> Die unabhängige Bestätigung hat nicht alle vorab festgelegten Kriterien
> erfüllt. Dies ist ausdrücklich **kein Nachweis, dass Task 2 abgeschlossen ist**.
> Training dieses Agenten ist gesperrt.

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

The frozen Task 1 artifact is preserved as `parent-model.npz`. Its checksum and
schema are validated by `migration.py` whenever it is loaded as the Task 2
prior. The separate `model.npz` uses the Task 2 model schema.

Machine-readable successor lineage is recorded in `artifact.json`. The original
parent manifest is preserved as `parent-artifact.json`.

`baseline-config.yaml` and `reference-results.csv` are inherited Task 1
provenance records. They describe the frozen parent experiment and must not be
interpreted as a Task 2 successor configuration or as new successor evidence.

## Current behavior contract

The frozen Issue #183 artifact is the Issue #153 control replica r2. It uses
the `compact_decision` representation (five features), zero initialization,
standard exploration, no action masking, and no potential shaping. Its six
Q-values follow the fixed action order `UP, RIGHT, DOWN, LEFT, WAIT, BOMB`.

The frozen `model.npz` is loaded for evaluation only. Training is blocked.
`artifact.json` and `frozen-config.yaml` record its identity and provenance.

## State representation

The frozen model uses compact feature schema version `1` with five values:
`danger_level`, `safe_directions_mask`, `coin_direction`,
`crate_direction`, and `bomb_status`. The 17-value schema described below
belongs to the earlier baseline representation, **not** to the frozen model.
It is retained to document the agent's development history.

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

The successor uses tabular Q-learning. Each encoded Task 2 state maps to six
Q-values in the fixed action order:

```text
UP, RIGHT, DOWN, LEFT, WAIT, BOMB
```

For unseen Task 2 states, the first five Q-values are initialized from the
corresponding frozen Task 1 state. The BOMB value is initialized below the
minimum parent value by the configured bomb-prior margin.

The frozen Issue #183 model uses the five-feature compact_decision state and
zero initialization.

The learning rate remains 0.05 and the discount factor remains 0.9.
Issue #45 extends the state and action contracts without changing the
Q-learning update rule or these hyperparameters.

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

`useful_bombs` remains a diagnostic count by default. Experiment #114 adds an
optional immediate reward for the same event; the registered control uses
`0.0` and the candidate uses `+1.0`.

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

The Issue #114 treatment can additionally reward `USEFUL_BOMB_PLACED`, defined
as a framework-confirmed placement whose blast can destroy at least one crate.
The default is `0.0`, so existing models and evaluation behavior are unchanged.

Potential-based reward shaping is deliberately out of scope for issue #45.
Changing the inherited shaping formulation would be a separate controlled
variable and requires a prospectively registered experiment.

## Action masking

The agent supports two action-selection modes:

- `none`: no action mask is applied;
- `framework_legal`: actions rejected by the framework are excluded during
  exploration, greedy selection, tie-breaking, and Bellman bootstrapping.

The selected mode is stored in the model artifact. Resuming training with a
different mode is rejected to prevent incompatible continuation.

Framework-legal masking prevents invalid actions but does not exclude actions
that are legal yet tactically unsafe.

## Training status

The frozen artifact is evaluation-only. `setup_training()` rejects training.
Further training or a changed policy requires a separate issue and a new,
preregistered experiment. The parent-prior initialization described earlier
was part of the historical development path; it is not the initialization
of the frozen Issue #183 model.

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
93470f92082597b1848f2fa65265c7288dbb1b50ac90370e5747e085e775ea3e
```

## Issue #183 freeze confirmation

The selected artifact is Issue #153 control replica r2, trained for 10,000
episodes. Issue #183 evaluated this unchanged checkpoint on 100 held-out seed
pairs per scenario in Classic, Coin Heaven, and Loot Crate. Each evaluation
was repeated with the same seeds; all 300 primary/repeat pairs were
deterministic. The model SHA-256 remained unchanged.

| Scenario | Mean coin collection | Self-kills | Bomb actions |
| --- | ---: | ---: | ---: |
| Classic | 0.213333 | 13/100 | — |
| Coin Heaven | 0.9716 | 3/100 | 33 |
| Loot Crate | 0.2626 | 28/100 | — |

The confirmation **did not pass**: Classic self-kills exceeded the registered
maximum of 10/100, and Coin Heaven had 33 bomb actions instead of zero.
Classic collection met its minimum of 0.15, and Coin Heaven collection met
its minimum of 0.95. The artifact is therefore frozen only as an interim
checkpoint for further project work, not as a Task 2 completion claim.

Protocol: `training/run_plans/issue183-tabular-task2-freeze-confirmation.yaml`.
Local raw results: `training_outputs/confirmation-issue183/`.

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

## Experimental evidence and limitations

Issue #102 established the historical unmasked Task 2 development baseline.

Issue #110 evaluated optional framework-legal masking. The retained execution
eliminated invalid actions and reduced aggregate self-kills, but did not improve
hidden-coin collection. Because no compute budget was prospectively recorded,
the execution is classified as exploratory and does not constitute a completed
confirmatory experiment.

The unmasked configuration remains the baseline. Framework-legal masking
remains an optional capability.

The agent still:

- uses a categorical sparse state representation;
- does not contain a full opponent strategy;
- retains the inherited non-potential-based coin movement shaping;
- requires separately preregistered experiments for reward or
  hyperparameter changes.

Potential-based reward shaping and further reward tuning must be evaluated as
separate controlled experiments.

Issue #135 found that full-strength time-aware escape-distance shaping reduced
Classic self-kills but caused an unacceptable collection loss. Issue #139 then
tested half strength against a newly trained full-strength control. Half
strength significantly improved Classic collection but failed the registered
safety guard (`0.070` self-kill rate versus the historical `0.055` no-shaping
limit), so neither escape-distance treatment is the default. Both modes remain
available for reproducibility of the experiment lineage.

Issue #142 isolated the no-route bucket by comparing uniform half strength
against half-strength progress with the original `-5` no-route potential. The
candidate reduced mean Classic self-kills from `0.115` to `0.045`, with four of
five replicas improving, but its paired 95% interval still crossed zero and
Classic collection fell by `0.0678`, beyond the registered `0.05` guard. A
single deterministic-repeat mismatch also produced a maximum-latency outlier.
The candidate was rejected, the default remains unchanged, and the mode is
retained only for reproducibility.
Issue #153 tested an opt-in `safe_bomb` exploration mode that excludes an
unsafe `BOMB` only from the random epsilon-greedy branch. It increased Classic
self-kills from `0.045` to `0.075` and reduced Classic collection from `0.1694`
to `0.1489`; no replica improved its Classic self-kill rate. The treatment was
rejected, and standard epsilon-greedy exploration remains the default. The mode
is retained only to reproduce the negative experiment.
Issue #144 tested whether an appended post-bomb escape-status category could
resolve compact-state aliasing. It produced exactly the same Classic collection
(`0.1272`) and self-kill rate (`0.075`) as control, with a paired collection
difference and confidence interval of zero. Post-hoc Q-table inspection found
no compact-state prefix split across multiple status values, so the added value
acted as a relabeling on visited states. The candidate was rejected and
`compact_decision` remains the default.
