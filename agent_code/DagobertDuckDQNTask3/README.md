# DagobertDuckDQNTask3

> Status: Task 3 implementation successor prepared under issue #108. This PR
> contains contract decisions, tests, migration, and bounded integration
> validation only. It contains no scientific training result or performance
> claim.

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

The migration tool also accepts a 26-input Task 2 control parent whose
`escape_continuation_features` is false. Its five neutral continuation inputs
are removed without changing inherited Q-values; active continuation parents
are rejected. Explicit source/output paths and a required source checksum
prepare the verified #107 handoff without replacing the provisional artifact.
See the #109 experiment's `PREPARATION.md` for remaining launch gates.

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
