# DagobertDuckDQNTask3

## Issue 171 retention pilot

[Issue #171](https://github.com/1BlauNitrox/mle-final-project/issues/171) tested
whether reducing Adam's learning rate from `0.0005` to `0.00005` improves retention
without reducing hunting. Both arms used the same fresh #168 initialization and
80% greedy / 20% random episode schedule. The pilot used the archived 39-input
runtime at `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`, separately from the tracked
34-input provisional fixture described in the remaining implementation sections.
Three paired replicas completed 300 training episodes; all six fixed-final
checkpoints and the unchanged reference completed 280 evaluation episodes,
including exact repeats, across four suites.

**The registered retention screen failed.** Lower-rate Task 1 collection improved
by 6.8 percentage points versus control, below the required 10 (descriptive 95%
interval: -37.47 to +57.20 points). Hunting fell from 0.267 to 0.133 eliminations
per game (paired difference -0.133; interval -0.533 to +0.267). Loot-crate and
classic-empty retention failed, as did the whole-matrix zero-invalid-action gate
because of one control event reproduced in its repeat. Genuine learning,
behavioral repeats and latency passed. The wide intervals limit generalization.

This result does not support adopting the lower rate as a retention remedy.
No checkpoint was selected or promoted, and no agent default changed. Task 2
remains cumulatively incomplete; the original Task 3 gates and coin-collector
continuation requirements remain in force. Review the negative result before
proposing another controlled experiment.

The [experiment record](../../experiments/2026-09-13-task3-learning-rate-retention/README.md)
links the registered configuration, artifact provenance, all effects and paired
uncertainty, resource accounting and reproducible analysis. Durable evidence
publication and human interpretation are still pending in
[PR #172](https://github.com/1BlauNitrox/mle-final-project/pull/172).

## Purpose and hypothesis

Task 3 extends the Task 2 DQN with descriptive awareness of public opponents.
The hypothesis for the downstream experiment is that direction/distance,
public occupancy, bomb attack opportunity, and bomb-escape context provide
useful information for hunting `peaceful_agent` and then
`coin_collector_agent`, while preserving opponent-free Task 1/Task 2 behavior
at initialization. The representation never selects an action or predicts an
opponent's hidden/private state.

The full versioned feature, reward, attribution, and migration contract is in
[`docs/0008-task-3-opponent-awareness-contract.md`](../../docs/0008-task-3-opponent-awareness-contract.md).

## Parent and migration status

The predecessor is provisionally the corrected Task 2 migration artifact from
issue #85:

```text
agent: DagobertDuckDQNTask2
artifact: checkpoint-issue85-zero-suffix.pt
sha256: 3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015
source commit: 933a8fe11440e0f7645254390928da6af5dad46d (current main at branch creation)
```

Issue #107 is the final Task 2 selection campaign. After it selects the
development artifact, the parent source/artifact hashes in `artifact.json`
must be rebound before scientific Task 3 training. This provisional fixture is
not evidence and does not close that dependency.

Migration copies the 21 Task 2 input columns, both 64-unit hidden layers, and
all six output rows. The thirteen Task 3 input columns are zero-initialized, so
initial inherited Q-values and greedy actions are preserved within the
documented float32 tolerance. Optimizer, replay, epsilon, and RNG state are
reset for Task 3 training. Incompatible feature schema, action order, reward
mapping, or configuration fails during checkpoint loading.

## Learning model and evaluation contract

This is a CPU DQN with a `34 -> 64 -> 64 -> 6` multilayer perceptron, seeded
epsilon-greedy training, bounded replay, a target network, and the inherited
Task 2 hyperparameters. The action order is:

```text
UP, RIGHT, DOWN, LEFT, WAIT, BOMB
```

Evaluation loads only the online network, uses one CPU thread, does not create
training objects, does not write the checkpoint, and uses paths relative to
this directory. It uses public `game_state['others']` positions as obstacles.
`OPPONENT_ELIMINATED` is diagnostic-only; the native `KILLED_OPPONENT` event is
rewarded once per attributable kill at `+5.0`, including terminal transitions.

## Planned run path

The issue-specific plan extends the staged runner with this explicit order:

1. `classic` against one `peaceful_agent`;
2. `classic` against one `coin_collector_agent`.

The plan also includes opponent-free `coin-heaven` and `loot-crate` regression
suites. It is a prospective integration path, not a scientific selection rule.
The bounded smoke runs use one episode and are reported only as integration
checks.

## Validation and limitations

Tests cover blocked paths, no opponent, nearest-opponent tie-breaking, blast
opportunities, occupancy, escape context, migration Q-value preservation,
gradient flow, persistence, terminal elimination reward attribution, and
reproducible run-plan expansion. The agent card records the Task 2 campaign's
known mixed/failed gates as inherited context; it does not turn them into a
Task 3 result.

The eventual parent artifact, training seeds, evaluation seeds, numerical
selection rule, and scientific results are intentionally not fixed here. They
belong to the post-#107 Task 3 launch protocol.
