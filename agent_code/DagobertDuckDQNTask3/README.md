# DagobertDuckDQNTask3

## Completed Double DQN comparison (#150)

The [registered result](../../experiments/2026-09-12-task3-double-dqn-results/README.md)
compared five matched standard/Double DQN replicas, 10,000 training episodes
each, from provisional #91 A/r3. Both arms achieved 23.5% peaceful elimination;
the Double-minus-standard difference was 0 percentage points, 95% CI
[-18.5, +16.0]. Double DQN's registered benefit and cumulative hunting/retention
gates failed. No arm or replica is selected, and Task 2 remains incomplete.
Runtime and deterministic-repeat gates passed.

The executed 39-input implementation is pinned at
`6f014485a3026cc3707fa2cc3a379880dd0b74bd` in PR #154. This result PR retains
the older 34-input implementation described below; it does not install or
promote the experimental model. Full metrics, parent/checkpoint hashes, seeds
and retrievable evidence are in the linked result record.

The next step is bounded hunting/retention diagnosis before another registered
intervention. This result does not justify adopting Double DQN or choosing a
favorable standard-DQN replica as a fallback.

## Purpose and hypothesis

Task 3 extends the Task 2 DQN with descriptive awareness of public opponents.
The hypothesis for the downstream experiment is that direction/distance,
public occupancy, bomb attack opportunity, and bomb-escape context provide
useful information for hunting peaceful_agent and then
coin_collector_agent, while preserving opponent-free Task 1/Task 2 behavior
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

This is the historical compatibility fixture, not the parent used for the
completed #150 experiment. That experiment used explicitly authorized
provisional #91 A/r3, with its exact binding retained in the result record.
Future training requires a prospectively registered parent; it does not need
to wait for Task 2 to be declared complete.

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

1. classic against one peaceful_agent;
2. classic against one coin_collector_agent.

The plan also includes opponent-free coin-heaven and loot-crate regression
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

## Experiment evidence: archived 39-input runtime

| Experiment | Runtime source | Eliminations per game reference / full fine-tuning / frozen treatment | Treatment minus control (95% CI) | Decision |
| --- | --- | --- | --- | --- |
| [#173](../../experiments/2026-09-13-task3-frozen-inheritance/README.md) | `c4ddfa4` (39 inputs; separate from this checkout's 34-input fixture) | 0.200 / 0.185 / 0.170 | -0.015 [-0.150, +0.125] | No checkpoint promoted |

[Exact gates, uncertainty and retention](../../experiments/2026-09-13-task3-frozen-inheritance/results/analysis.json); [artifact provenance](../../experiments/2026-09-13-task3-frozen-inheritance/results/evidence.json).

The historical implementation and smoke checks below the result summary do
not establish performance. The #150 result record defines the exact executed
implementation, parent, seeds, decision rules and failed scientific gates.
