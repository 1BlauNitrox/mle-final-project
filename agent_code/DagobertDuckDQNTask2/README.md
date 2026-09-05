# DagobertDuckDQNTask2

> Status: Task 2 capability implemented (issue #44) and evaluated in the
> preregistered Issue #46 development experiment.
>
> Current measured result: the corrected evaluation passed determinism and
> latency checks, but failed the registered Task 2 feasibility and Task 1
> retention gates. It is therefore not a submission candidate.

Issue #46 currently has an incomplete negative/mixed result. Corrected development
evaluation passed determinism and latency checks, but failed the registered
Task 2 feasibility and Task 1 retention gates. See
`experiments/2026-09-05-dqn-task2-development/result.json` and Issue #82 for
the metric correction concerning initially hidden coins.

## Purpose

`DagobertDuckDQNTask2` is the DQN Task 2 successor of `DagobertDuckDQN`.

Issue #43 created it as a byte-identical, behavior-preserving scaffold of the
frozen Task 1 baseline. Issue #44 (this revision) adds the actual Task 2
capability: bomb/crate/danger features, a six-action network, a one-way
checkpoint migration from the frozen parent's weights, Task 2 rewards, and
training diagnostics. It does not run or claim a scientific result -- that is
separate, downstream experiment work. Curriculum orchestration is explicitly
split to issue #50.

## Parent baseline

The immutable parent agent is:

```text
agent_code/DagobertDuckDQN/
```

Its frozen source, configuration, documentation, and model artifact remain
unchanged by this issue; `test_parent_checkpoint_is_unmodified` pins its
checksum.

| Property | Value |
| --- | --- |
| Parent agent | `DagobertDuckDQN` |
| Parent artifact | `checkpoint.pt`, SHA-256 `eb08e3f6...ba8b05e`, `23829` bytes |
| Parent action order | `UP, RIGHT, DOWN, LEFT, WAIT` (5 actions, 8 features) |
| Producing agent commit | `84396eb771a7eb3e3daa028a415e9b00570b20c8` |
| Producing experiment commit | `dbe259721409092448e8594cbeaaca469c4f9835` |
| Frozen parent source commit | `d81eee00ffa4b095edbcf3be6510594dda9a9733` |
| Parent freeze source commit (PR #72) | `9d184efec3e47cf014b5203023fd50b9de62feb5` |
| Parent freeze merge commit (PR #72, squash-merged) | `0e34bc0e93a6ffa4746bb9987962a8f5169e71cd` |
| Framework revision | `0f55c1d` |

Full lineage is in `artifact.json`'s `parent` and `migration` blocks. The
original frozen manifest is preserved as `parent-artifact.json`.

`baseline-config.yaml` and `reference-results.csv` are inherited Task 1
provenance records describing the *parent's* issue #58/#42 history. They are
not Task 2 configuration or evidence.

## Checkpoint migration

`scripts/migrate_task2_dqn_capability.py` builds the committed `checkpoint.pt`
from the parent's frozen network. It supersedes issue #43's byte-copy
placeholder, since the two checkpoints are no longer the same shape.

The rule (`agent_code/DagobertDuckDQNTask2/migration.py`, tested in
`tests/test_DagobertDuckDQNTask2_migration.py`):

- the parent's 8 input-layer columns are copied verbatim into the successor's
  first 8 input columns; the remaining 13 columns come from a network freshly
  seeded with the migration seed (`44`) -- deterministic, not left to chance;
- both hidden layers (`64x64`) are copied verbatim; their shape does not
  change;
- the parent's 5 output rows and biases are copied verbatim into the
  successor's first 5 output rows;
- the new `BOMB` output row's weights are zeroed and its bias fixed at `-1.0`
  -- a deliberately conservative estimate so a freshly migrated policy does
  not select an untested action by chance of initialization (verified
  empirically: a coin-heaven regression smoke over 5 rounds selected `BOMB`
  zero times);
- the target network is initialized from the migrated online network;
- the optimizer and replay buffer are reset to fresh/empty -- neither is
  compatible with the parent's shapes, and neither has any value for a
  network that has not yet trained under the new architecture.

The committed checkpoint is therefore a **fresh, training-ready** artifact
(`completed_episodes=0`), not a trained or evaluated one.

## Action space

```text
UP, RIGHT, DOWN, LEFT, WAIT, BOMB
```

`BOMB` is appended; the first five indices are unchanged from Task 1, so the
migrated output rows keep their meaning exactly.

## Feature schema (version 2, 21 features)

Indices 0-7 are the unchanged Task 1 prefix (identical values on Task 1
states -- pinned by `test_task1_prefix_matches_parent_on_task1_states`):

```text
free_up, free_right, free_down, free_left,
coin_visible, coin_dx, coin_dy, coin_distance_bin
```

Indices 8-20 are the Task 2 additions, computed in
`features/bombs_and_crates.py`:

| # | Feature | Domain |
| --: | --- | --- |
| 8 | `bomb_available` | `{0, 1}` |
| 9 | `danger_countdown_bin` (here) | `{0..3}` (0=safe, 1=lethal now/lingering, 2=soon, 3=later) |
| 10-13 | `safe_up/right/down/left` | `{0, 1}` each |
| 14 | `escape_exists_after_bomb` | `{0, 1}` |
| 15 | `crate_visible` | `{0, 1}` |
| 16-17 | `crate_dx`, `crate_dy` | `{-1, 0, 1}` |
| 18 | `crate_distance_bin` | `{0..3}` |
| 19 | `crates_destroyed_here_bin` | `{0..3}`, capped |
| 20 | `bomb_has_useful_target` | `{0, 1}` |

Normalization divides indices `7`, `9`, `18`, `19` by `3`; everything else is
already in `[-1, 1]`.

**Blast physics** (`blast_footprint`, `build_danger_map`) mirror the framework:
a bomb's blast is a cross of radius `3` that stops only at walls, not crates
(`items.Bomb.get_blast_coords`). Each tile retains every inclusive lethal time
interval from overlapping bombs and active explosions. Timer boundaries follow
the framework update order: an observed positive timer `t` detonates after
`t + 1` arrivals and remains dangerous for `EXPLOSION_TIMER=2` arrivals.

**Escape and safety** (`safe_direction`, `safe_escape_exists`) use a
time-aware breadth-first search: a candidate tile must be enterable (open,
not currently occupied by a bomb or another agent) *and* not lethal at the
exact step the agent would arrive there, for every step of the path, not just
the destination. `safe_escape_exists` additionally simulates a hypothetical
bomb placed at the current position to answer "if I place a bomb here, can I
still get away?"

**Crate targeting** (`nearest_crate_features`, `crates_destroyed_by_bomb_at`)
uses deterministic BFS to the nearest reachable open tile from which a bomb
would destroy at least one crate. It encodes the first path direction and path
distance, or neutral values when no useful bombing tile is reachable. A
separate feature counts crates hit by a bomb at the current position.

There are no opponents in the Task 2 curriculum (`coin-heaven`, `loot-crate`,
`classic` without opponents), so the danger model accounts only for bombs
already on the board; it does not simulate an opponent placing a new one.

## Hyperparameter revision

Task 1's defaults are not reused verbatim (`config.py` has the full,
computed rationale). In short: `DagobertDuckDQN`'s `epsilon_decay=0.99` over
a 10,000-episode budget reaches its exploration floor after roughly 230
episodes -- under 3% of the run -- which issue #58 identified as consistent
with why its greedy evaluation diverged so far from its training performance
(0.989-1.000 training vs. 0.65-0.92 greedy evaluation). `epsilon_decay=0.9997`
reaches the same floor around episode 8,000 instead. `learning_rate`
(`0.001 -> 0.0005`), `target_update_interval` (`250 -> 500`), and
`replay_warmup` (`256 -> 500`) are also revised, more conservatively, for the
larger 21-feature/6-action problem.

**These are implementation defaults, not a validated fix.** None of the four
has been tested as a controlled variable; citing them as evidence that Task 2
training is more stable than Task 1's would be exactly the kind of unearned
claim this repository's process exists to prevent. A registered experiment
comparing them is future work.

## Rewards

| Event | Reward | Source |
| --- | ---: | --- |
| `COIN_COLLECTED` | `+10.0` | inherited |
| `INVALID_ACTION` | `-0.5` | inherited |
| `WAITED` | `-0.1` | inherited |
| `MOVED_TOWARDS_COIN` | `+0.1` | inherited |
| `MOVED_AWAY_FROM_COIN` | `-0.1` | inherited |
| `CRATE_DESTROYED` | `+1.0` | new |
| `COIN_FOUND` | `+2.0` | new |
| `KILLED_SELF` | `-10.0` | new |
| `GOT_KILLED` | `-10.0` | new (unreachable without opponents) |
| `SURVIVED_ROUND` | `+5.0` | new |
| `USEFUL_BOMB_PLACED` | `+0.5` | new, custom shaping |
| `WASTEFUL_BOMB_PLACED` | `-0.5` | new, custom shaping |

`CRATE_DESTROYED`, `COIN_FOUND`, `KILLED_SELF`, `GOT_KILLED`, and
`SURVIVED_ROUND` are native framework events (`events.py`); no custom
derivation is needed for them. The two `*_BOMB_PLACED` events are custom
shaping computed only when the framework confirms `BOMB_DROPPED` actually
happened (an attempted-but-invalid `BOMB` gets `INVALID_ACTION` only).
`BOMB_DROPPED` and `BOMB_EXPLODED` are deliberately left unrewarded on their
own, so bomb placement is learned from its consequences rather than a flat
per-placement bonus.

All values are the pre-exploration implementation defaults. The later reward
variants described below were reverted because they were selected after
observing unregistered runs. They may be tested only in a prospective,
controlled experiment.

## Development history

Full detail, including binned training-curve data and figures for round 3,
is in `experiments/2026-09-04-dqn-task2-capability-development/`.

**2026-09-03/04, unregistered overnight run.** Before the redesign above, an
exploratory training run on `loot-crate` reached 129,091 episodes. A
snapshot evaluation at that point showed real problems, not a clean learning
curve:

- 18/20 episodes ended in self-inflicted death;
- mean coins collected fell to 3.65/50 despite ~77 mean steps per episode;
- a Task 1 regression check on `coin-heaven` (no bombs, no crates) showed
  the invalid-action rate had risen from ~0% at migration to 59%, and `BOMB`
  was now selected 224 times in 10 rounds versus 0 at migration -- training
  on Task 2 was visibly eroding Task 1 behavior, not just leaving it alone.

Investigating this surfaced two real problems, not just "needs more
training":

1. **A bug**, not a design gap: `escape_after_bomb` never actually added a
   hypothetical bomb to the danger map before checking for an escape route,
   so it silently measured "can I escape right now" instead of "would this
   bomb trap me." Fixed in `features/assemble.py`; pinned by
   `test_escape_after_bomb_feature_simulates_a_hypothetical_bomb`.
2. **A candidate hypothesis**: `MOVED_TOWARDS_COIN`/`MOVED_AWAY_FROM_COIN`
   carried over from the frozen parent's config without being reconsidered
   for Task 2. Issue #58's own result for that shaping was inconclusive (a
   95% CI of roughly `+-0.24` on the paired comparison) -- there was never
   evidence it helps, and no Task-2-specific justification was given for
   keeping it either. The development branch temporarily removed it and added
   safe/unsafe bomb shaping. Those outcome-driven reward changes are recorded
   here but are not shipped by this PR.

This run was not re-executed after the fix. Its numbers motivate future
hypotheses; they do not select the shipped reward configuration.

**2026-09-04, follow-up development run.** A second exploratory run with the
fixes above showed Task 1 retention fully recovered (coin-heaven
invalid-action rate back to ~0%, `BOMB` still never selected), which
confirmed the escape-feature fix held up under real training. But Task 2
itself settled into a passive local optimum rather than continuing to
improve: across three snapshots (episodes ~5k, ~12k, ~24k) the `loot-crate`
death rate rose from 40% to 65% to 70%, mean coins plateaued around 2.2-2.25,
and the action mix became lopsided -- roughly 43% `WAIT` and next to no
`LEFT`/`RIGHT`, consistent across different evaluation seeds.

Two contributing factors, not mutually exclusive:

1. That run used `--rounds 500000` against an epsilon schedule computed for
   a 10,000-episode budget (see the hyperparameter revision above), so
   epsilon hit its floor around episode 8,000 -- under 2% of that run's
   target -- leaving little further exploration to escape a bad optimum for
   the remaining 98%. The same shape of problem the schedule was built to
   fix, recreated at the wrong episode count.
2. `SURVIVED_ROUND` (`+5.0`) was large enough relative to
   `CRATE_DESTROYED`/`COIN_FOUND` (`+1.0`/`+2.0`) that passively surviving a
   round was plausibly competitive with actually engaging with it.

The development branch temporarily reduced `SURVIVED_ROUND` to `+2.0` and
strengthened `WAITED` to `-0.3`, then ran another 15,000 episodes. Because the
observations selected those changes post hoc, both changes are reverted in
the shipped implementation. The record remains useful for preregistering a
controlled reward experiment, not for claiming either value is better.

## Training status

Training is implemented and enabled (`setup_training` no longer raises).
Running it produces real diagnostics -- per-episode event counts for every
reward-relevant event, in addition to the inherited loss/TD-error/replay/
target-sync metrics -- but **no scientific training has been run under this
issue**, and none is authorized by it (see `artifact.json`'s
`scientific_training_authorized: false`). A registered Task 2 experiment is
separate, downstream work.

A training-only curriculum mixing `coin-heaven`, `loot-crate`, and `classic`
(without opponents) is expected to live in `training/` orchestration under
issue #50. Issue #44 implements the agent-side capability that curriculum
will drive; the reviewed issue split records the orchestration as downstream.

## Smoke evidence (integration only, not performance evidence)

Run locally against the committed checkpoint:

- a 3-round seeded training smoke on `coin-heaven` (world seed `44001`,
  agent seed `44`) completed 3 episodes and selected `BOMB` during training;
- a 2-round read-only evaluation smoke on `classic` without opponents
  completed and left the checkpoint byte-identical;
- a 5-round read-only evaluation smoke on `coin-heaven` (the Task 1
  regression check) completed and selected `BOMB` **zero** times, and left
  the checkpoint byte-identical.

These are the same three checks CI runs (`dqn-task2-successor-smoke`); CI
restores the clean, untrained checkpoint afterward so the committed artifact
stays the migration's direct output.

## Validation

- `tests/test_DagobertDuckDQNTask2_successor.py`: Task 1 prefix and neutral
  Task 2 feature values are pinned on Task 1 states; parent lineage and
  migration are recorded in the manifest; the migrated network continues the
  parent's weights exactly.
- `tests/test_DagobertDuckDQNTask2_bombs_and_crates.py`: blast propagation
  (wall-blocking, crate pass-through), danger-map timers and lingering,
  per-direction safety, escape existence (including a sealed no-escape case
  and a walk-away case), and crate targeting.
- `tests/test_DagobertDuckDQNTask2_migration.py`: preserved input/hidden/
  output weights, the conservative `BOMB` row, deterministic repeatability,
  and rejection of an incompatible parent or successor config.
- `tests/test_DagobertDuckDQNTask2_train.py`: training re-enabled, the
  bomb-usefulness shaping event (including the invalid-attempt case), event
  counting and its per-round reset, and a full six-action optimizer update.

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
├── migration.py
├── checkpoint.pt
├── persistence.py
├── replay.py
├── rewards.py
├── train.py
└── features/
    ├── __init__.py
    ├── assemble.py
    ├── navigation.py
    └── bombs_and_crates.py
```

Evaluation-time code is self-contained inside this directory. It does not
import from the parent agent, `training/`, `experiments/`, or `scripts/`
(`migration.py` is imported only by the migration script and its tests, never
by `callbacks.py`/`train.py`).

Parent imports are permitted only in repository-level differential tests.

## Limitations and next steps

- No scientific Task 2 training or evaluation has been run; every reward,
  hyperparameter, and feature choice here is an implementation default.
- The danger model does not account for an opponent placing a new bomb,
  matching the opponent-free Task 2 curriculum; it will need revisiting for
  Task 3.
- Escape search is bounded to 10 steps and accepts a destination only outside
  every currently represented blast footprint; intermediate steps are checked
  against all retained lethal intervals.
- The curriculum mixing `coin-heaven`/`loot-crate`/`classic` is not yet
  implemented in `training/` orchestration.
- `KILLED_SELF`/`GOT_KILLED`/`SURVIVED_ROUND` reward magnitudes and the
  epsilon/learning-rate/target-interval/replay-warmup revision are all
  untested defaults pending a registered experiment.

The next issue should register a prospective Task 2 experiment: fixed seeds,
a stopping rule, and success criteria, following the pattern #41/#58
established for Task 1.
