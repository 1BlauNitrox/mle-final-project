# Issue #103 reward-shaping treatments for Task 2 DQN

> Status: **Completed — both treatments rejected.** See "Result and
> decision" below. Independent of #97 throughout (direct-classic training,
> not the curriculum question).

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

## Decision rule (Claude's proposal, applied mechanically to the result below)

For each treatment arm independently, paired against control by replica ID:

1. **Timing** (carried over from #46/#86 unchanged): p95 < 50 ms and max
   < 100 ms decision time in every scenario.
2. **Non-regression**: paired 95% bootstrap CI lower bound for collection
   fraction (treatment − control) ≥ -0.05 in all three scenarios.
3. **Primary effect**: paired 95% CI lower bound for survival rate
   (treatment − control) ≥ 0 in `classic`.
4. An arm is adopted only if it clears gates 1-3. If both clear, prefer the
   larger `classic` survival-rate improvement. If neither clears, retain
   control and report the negative result, as #86 did.

**Correction made when writing the analyzer** (`training/analyze_issue103_dqn_task2_reward_shaping.py`):
the originally proposed `safety_bomb` gate 3 ("paired CI lower bound for the
`UNSAFE_BOMB_PLACED` rate reduction > 0") is not computable from the
available data -- that per-event-type diagnostic is only ever exposed on
*training* episodes (`train.py`'s `end_of_round`), never on the *evaluation*
episodes this decision rule is scored against, which only carry the
framework's native per-episode counters (`self_kills`, `crates_destroyed`,
`bombs_dropped`, etc.), not a breakdown by shaping-event type. Replaced with
the plain survival-rate gate above, which both arms now share -- self-kills
without opponents are the sole cause of a `classic`/`loot-crate` death here,
so a survival-rate improvement already is the direct evidence a working
safety mechanism would produce.

These numeric criteria remain Claude's proposal, made at the owner's
explicit invitation -- not a team-ratified threshold. They were applied
mechanically to the result below, unchanged after seeing it.

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

Each plan was drafted for at most two training workers and 8 GiB RAM.
Executed here with `max_parallel_training: 1` in all three plans instead
(committed as actually run): all three arms plus Issue #97's direct arm ran
concurrently on one machine, so bounding each plan to one worker kept total
concurrent training processes at four rather than eight. Combined across
all three arms: 915 jobs, 150,000 training episodes total (1.5x #86's
two-arm total). Detach with `Ctrl-b d`; resume an interrupted plan with
`--resume`. Do not alter a plan, source tree, or artifact between a failed
run and its resume.

## Result and decision

All three arms and Issue #97's direct arm ran concurrently on the owner's
own machine on 2026-09-07 and completed in full (915/915 jobs, 0 failures)
in about 6 hours -- well inside the ~45 wall-hour serial estimate above.
Deterministic repeats matched exactly (identical
`executed_action_sequence_sha256` between every primary/repeat pair; the
only differences were incidental decision-time measurements). Full compact evidence: `result.json`, `summary.csv`
(`training/analyze_issue103_dqn_task2_reward_shaping.py`).

## Evidence: committed, not the ~4.4 GiB raw output tree

The full raw output tree (every job's attempt directory, including a full
agent-source snapshot per job) runs to ~4.4 GiB. The team decided this is
not worth hosting anywhere durable: it is overwhelmingly redundant source
snapshots and per-round framework dumps (`framework_stats.json` alone is
~39 MB *per training job*), not evidence. What actually matters for
verifying the claimed result -- every episode row the analyzer reads -- is
committed instead, at `evidence/<plan-id>/`:

- `evaluation-episodes.csv` and `training-episodes.csv.gz`: every job's
  `episodes.csv` rows, tagged with `plan_id`/`run_id`/`kind`/`replica`/
  `stage_or_suite`/`world_seed`/`agent_seed` (via
  `training/export_evidence.py`). ~19 MiB total across all three arms.
- `manifest.json`: the plan's own configuration/source/framework/agent
  fingerprints (from `resolved_plan.json`), per-job provenance (status,
  seeds, git commit, duration), and the two evidence files' own SHA-256.

This is independently checkable **without the raw tree or any external
archive**: `analyze_issue103_dqn_task2_reward_shaping.py --verify-from-evidence`
rebuilds the exact same `rows` from the committed `evaluation-episodes.csv`
alone, recomputes `result.json` end-to-end (the same `_summaries`/
`_paired_comparisons`/`_criteria` code path as the original run), and
diffs it against the committed one:

```bash
python -m training.analyze_issue103_dqn_task2_reward_shaping \
  --verify-from-evidence experiments/2026-09-07-dqn-task2-reward-shaping/evidence \
  --output experiments/2026-09-07-dqn-task2-reward-shaping
# -> MATCHES committed result.json
```

Confirmed to print exactly that against this experiment's own committed
evidence and `result.json`. A reviewer with only this repository checkout
can run this one command; they do not need to trust that the owner's
machine still holds the same 4.4 GiB it started with. To regenerate
`evidence/` from a fresh raw run instead (e.g. if the experiment is ever
re-run): `python -m training.export_evidence --plan-directory
training_outputs/run-plans/<plan-id> --output experiments/2026-09-07-dqn-task2-reward-shaping/evidence/<plan-id>`
for each of the three plan IDs.

**Both treatments are rejected.** Neither cleared gate 3 (a confirmed
`classic` survival-rate improvement):

| Gate | `survival_rebalance` | `safety_bomb` |
| --- | --- | --- |
| Timing | pass | pass |
| Collection non-regression (all 3 scenarios) | pass | **fail** (coin-heaven -0.027 [-0.072, 0.010], lower bound below -0.05) |
| `classic` survival improved | fail: -0.04 [-0.32, 0.30] | fail: **-0.22** [-0.52, 0.12] |

`survival_rebalance` shows a real, if unregistered-for, behavioral shift:
relative to control, mean `classic` coins rose 0.24→0.32 and mean crates
destroyed 5.6→6.48 (similarly in `loot-crate`: 0.9→1.3 coins, 4.26→5.1
crates) -- the agent visibly engaged more, consistent with the reduced
`SURVIVED_ROUND`/strengthened `WAITED` doing what they were meant to do.
But that extra engagement cost survival: `classic` survival fell 0.48→0.44,
`loot-crate` 0.52→0.32, with self-kill rate rising correspondingly. The
paired confidence intervals are wide enough (n=5 replicas × 10 seeds) that
none of this reaches significance in either direction -- it is a directionally
consistent but not confirmed engagement-for-safety trade-off, not a
confident negative result.

`safety_bomb` looks worse without an offsetting upside: survival and
self-kill rate moved the wrong way in all three scenarios (`classic`
survival 0.48→0.26, `loot-crate` 0.52→0.24), collection-fraction actively
regressed in `coin-heaven`, and the invalid-action rate rose sharply in
`classic` (0.123→0.272) despite the mechanism adding no new action options.
The point estimates suggest the treatment may have made bombing decisions
*more* hesitant or erratic rather than safer, though the wide intervals mean
this is not a statistically confident finding of harm either -- only a clear
absence of the intended benefit. No follow-up is proposed without first
understanding why (see "Known gaps" below).

Per the decision rule: **both `survival_rebalance` and `safety_bomb` are
rejected for this training configuration; `main`'s current reward values
(`control`) are retained.** This does not establish that either mechanism
is universally unhelpful -- it rejects these exact values, this seed
population, and this direct-classic training protocol.

## Known gaps

- **Root cause of `safety_bomb`'s regression is not investigated here.**
  The per-episode `SAFE_BOMB_PLACED`/`UNSAFE_BOMB_PLACED` event counts that
  would show *how often* the mechanism actually fired are only recorded on
  training episodes, not evaluation episodes (see the decision-rule
  correction above) -- so this result cannot say whether the penalty was
  too large, miscalibrated against the escape-search's own limits (e.g. its
  `MAX_ESCAPE_SEARCH_STEPS=10` truncation), or simply insufficient signal at
  `n=5` replicas. A follow-up reading the *training* episodes' event counts
  (not requiring new runs) could narrow this down.
- **No combined arm.** `survival_rebalance` and `safety_bomb` were
  deliberately kept independent (see "Hypothesis and factors"); whether
  combining them compounds or cancels their effects is untested.
- **Not cross-checked against Issue #97's direct arm**, which used
  different replica seeds (`51001`-`51005` vs this experiment's
  `91001`-`91005`) and so cannot be paired against this result, only
  eyeballed for rough consistency of scale.
