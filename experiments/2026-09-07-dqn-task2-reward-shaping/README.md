# Issue #103 reward-shaping treatments for Task 2 DQN

> Status: **Registered — ready to execute.** Not blocked on #97 (this
> experiment trains direct-classic only, independent of the curriculum
> question) or on anything else.

## Hypothesis and factors

Two independent reward-shaping changes, each isolating one mechanism,
compared against the current (`main`) reward values as a shared control.
They are **not** bundled into one arm, per the owner's preference to test
reward changes separately rather than compounded.

Both treatments use the exact values a 2026-09-04 unregistered development
branch already tried, together and uncontrolled, then reverted (recorded in
`agent_code/DagobertDuckDQNTask2/README.md`'s "Development history"): this
experiment is the controlled, separated test that record was always meant to
lead to, not a new guess.

- **survival_rebalance** (`SURVIVED_ROUND: 5.0 → 2.0`, `WAITED: -0.1 → -0.3`):
  targets a passive-survival-over-engagement imbalance. `SURVIVED_ROUND`
  alone is half of `COIN_COLLECTED` and five to ten times
  `CRATE_DESTROYED`/`USEFUL_BOMB_PLACED`, so merely surviving the round
  without engaging is competitive with actively collecting or bombing.
- **safety_bomb** (new `SAFE_BOMB_PLACED: +0.2`, `UNSAFE_BOMB_PLACED: -1.0`):
  rates every confirmed `BOMB_DROPPED` by whether a safe escape existed at
  the moment of placement. #46 found high self-kill rates; #86 found that
  removing every framework-illegal action (masking) did not fix collection
  or survival -- meaning the problem is not moves the framework forbids, but
  bombing decisions that are framework-*legal* and still fatal. Nothing in
  `main`'s current reward configuration rewards or penalizes bombing by
  escape safety.

## Why the new safety signal is cheap to add

The check reuses the escape search already computed for every state's
feature vector -- `escape_after_bomb`, `features/assemble.py`'s
`ESCAPE_AFTER_BOMB_INDEX` (index 14 of the 21-feature tuple) -- rather than
recomputing `safe_escape_exists`. `train.py`'s new `_bomb_safety_event`
reads that index directly off the already-computed pre-action feature
tuple and returns `SAFE_BOMB_PLACED` or `UNSAFE_BOMB_PLACED` accordingly. It
costs nothing new at inference time and cannot disagree with what the
network already sees.

## Comparison

Three arms, same starting artifact, replicas, and seeds (paired by replica
ID); only `reward_variant` differs:

| Arm | `reward_variant` | Change from control |
| --- | --- | --- |
| Control | `control` | none (`BASE_REWARDS` verbatim) |
| Survival rebalance | `survival_rebalance` | `SURVIVED_ROUND: 2.0`, `WAITED: -0.3` |
| Safety bomb | `safety_bomb` | `SAFE_BOMB_PLACED: 0.2`, `UNSAFE_BOMB_PLACED: -1.0` (both new) |

All three: 5 replicas (`world_seed` 91001-91005, `agent_seed` 92001-92005),
10,000 episodes of `classic` only (direct training, no curriculum --
deliberately independent of #97's still-open curriculum question), starting
from the same reviewed migration artifact as #86/#97
(`checkpoint-issue85-zero-suffix.pt`). Evaluation reuses the same
development seed pairs as #86/#97 across `classic`/`coin-heaven`/
`loot-crate` (primary + repeat, for the determinism check), so results stay
eyeball-comparable across experiments even though formal pairing is only
within this experiment.

## Mechanism: `reward_variant`

`reward_variant` is a new `run_plan.py` field, added the same way
`action_masking` already works: validated against a fixed set
(`control`/`survival_rebalance`/`safety_bomb`), threaded to the training
subprocess as the environment variable `BOMBERMAN_DQN_REWARD_VARIANT`
(`training/run_plan.py`'s `environment_overrides`), and recorded in job
metadata and the plan's configuration fingerprint (the fingerprint hashes
the whole raw plan YAML, so this needed no extra fingerprint code).
`config.py` resolves the process's `REWARDS` dict from that variable once
at import time, merging a per-variant override onto `BASE_REWARDS`.

`persistence.py` already stored the active `REWARDS` mapping in every
checkpoint (for provenance) and rejected loading a checkpoint whose stored
mapping did not match the current process's -- but unconditionally, in
`_load_payload`, before `callbacks.py` can tell a fresh migration (0
completed episodes, empty replay buffer) apart from an actually-resumed
run. That would have made every arm here fail immediately: all three start
from the same `checkpoint-issue85-zero-suffix.pt`, saved under the old flat
`REWARDS`, and `survival_rebalance`/`safety_bomb` intentionally do not match
it. Confirmed by reproducing the crash before fixing it (a fresh checkpoint
saved under one variant, then loaded under another, raised "Checkpoint
reward mapping mismatch" every time).

Fixed by moving the comparison out of `_load_payload` into
`_setup_training_policy`, using the exact same `is_fresh_migration`
leniency `action_masking` already relies on there: a mismatch is only an
error once a replica has actually trained under it.
`LoadedTrainingCheckpoint` now carries the checkpoint's stored `rewards` so
`callbacks.py` can make that comparison itself. Re-verified end-to-end
through the real subprocess path (not just unit tests): a 3-round training
run under `safety_bomb` bakes `SAFE_BOMB_PLACED`/`UNSAFE_BOMB_PLACED` into
the checkpoint as expected, and continuing that now-trained checkpoint
under `control` correctly raises rather than silently drifting.

`SAFE_BOMB_PLACED`/`UNSAFE_BOMB_PLACED` are deliberately absent from
`BASE_REWARDS`: `train.py` always emits one of the two on a confirmed bomb
placement (so both are always tallied as diagnostics, in every arm), but
`reward_from_events`'s `REWARDS.get(event, 0.0)` makes them worth exactly
0.0 under `control` and `survival_rebalance`, and only `safety_bomb` gives
either a nonzero reward.

## Decision rule (Claude's proposal; confirm or adjust before training starts)

For each treatment arm independently, paired against control by replica ID:

1. **Timing** (carried over from #46/#86 unchanged): p95 < 50 ms and max
   < 100 ms decision time in every scenario.
2. **Non-regression**: paired 95% bootstrap CI lower bound for collection
   fraction (treatment − control) ≥ -0.05 in all three scenarios.
3. **Primary effect**:
   - `survival_rebalance`: paired 95% CI lower bound for survival rate
     (treatment − control) ≥ 0 in `classic`.
   - `safety_bomb`: paired 95% CI lower bound for the `UNSAFE_BOMB_PLACED`
     rate reduction (control − treatment) > 0 in `classic` (the mechanism
     actually fires less), **and** paired 95% CI lower bound for survival
     rate (treatment − control) ≥ 0 in `classic` (fewer unsafe placements
     actually reduce deaths, not just bomb attempts).
4. An arm is adopted only if it clears gates 1-3. If both clear, prefer the
   larger `classic` survival-rate improvement. If neither clears, retain
   control and report the negative result, as #86 did.

These numeric criteria are Claude's proposal, made at the owner's explicit
invitation this round -- not a team-ratified threshold. Adjust before
training if the team wants different margins.

## Execution

```bash
python -m training.run_plan training/run_plans/issue103-dqn-task2-reward-control.yaml --dry-run
python -m training.run_plan training/run_plans/issue103-dqn-task2-reward-survival-rebalance.yaml --dry-run
python -m training.run_plan training/run_plans/issue103-dqn-task2-reward-safety-bomb.yaml --dry-run
```

Confirm each dry-run matrix (5 replicas × [1 training stage + 6 evaluation
suites × 10 seed pairs] = 305 jobs, 50,000 training episodes) before
removing `--dry-run`. The three arms do not depend on each other and write
to independent plan directories, so they can run concurrently (each in its
own `tmux` pane) if the host has the resources, or sequentially otherwise:

```bash
tmux new -s issue103-control
python -m training.run_plan training/run_plans/issue103-dqn-task2-reward-control.yaml 2>&1 | tee logs/issue103-control.log
```

```bash
tmux new -s issue103-survival
python -m training.run_plan training/run_plans/issue103-dqn-task2-reward-survival-rebalance.yaml 2>&1 | tee logs/issue103-survival-rebalance.log
```

```bash
tmux new -s issue103-safety
python -m training.run_plan training/run_plans/issue103-dqn-task2-reward-safety-bomb.yaml 2>&1 | tee logs/issue103-safety-bomb.log
```

Each plan uses at most two training workers and 8 GiB RAM. Combined across
all three arms: 915 jobs, 150,000 training episodes total (1.5x #86's
two-arm total), an estimated ~75 CPU-hours / ~45 wall-hours if run serially
-- proportionally less wall-clock if run concurrently, bounded by the host's
own CPU/RAM headroom for three simultaneous `max_parallel_training: 2`
plans. Detach with `Ctrl-b d`; resume an interrupted plan with `--resume`.
Do not alter a plan, source tree, or artifact between a failed run and its
resume.
