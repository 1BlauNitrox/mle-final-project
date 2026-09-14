# DagobertDuckDQNTask3

> Status: this checkout retains the original Task 3 implementation fixture.
> Issue #168 tested a separately pinned, escape-preserving runtime and a
> provisional #91-derived parent. Its learning pilot failed; no default model
> or fixture was replaced, and cumulative Task 2/3 success is not established.

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

This is the historical compatibility fixture, not the parent used in #168.
That experiment explicitly bound the provisional #91 A/r3 predecessor and
executed the 39-input, escape-preserving runtime at
`c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`; the implementation described below
is the original 34-input checkout. The exact experimental parent, initialization,
source and six final checkpoint hashes are in the
[phase-D manifest](../../experiments/2026-09-13-task3-exploration-screen/phase-d-evidence-manifest.json).
Neither the fixture nor the experimental parent certifies Task 2 completion.

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

## Exploration pilot result (#168)

Frozen-policy probes motivated a three-replica learning comparison of stepwise
20% exploration against an 80% greedy / 20% random episode mixture. After 300
training and 280 evaluation episodes, mixture hunting was 0.200 eliminations/game
versus 0.133 for stepwise exploration and 0.200 for the unchanged parent.
The +0.067 effect missed the +0.10 pilot threshold; its descriptive 95% interval
was [-0.400, +0.533]. Coin-heaven collection was 71.2% versus the parent's 88.0%.
Coin-heaven and opponent-free classic failed collection and self-kill retention;
loot-crate passed. Learning, repeatability, invalid-action and latency gates passed.

The pilot is negative overall and selected no checkpoint. The proposed follow-up
holds episode exploration fixed and tests a smaller learning rate to investigate
retention. It must use fresh paired seeds and the unchanged initialization;
smaller updates are not yet a validated remedy. Detailed results and limitations
are in the [experiment record](../../experiments/2026-09-13-task3-exploration-screen/README.md).

The historical implementation and smoke checks below the result summary do
not establish performance. The #150 result record defines the exact executed
implementation, parent, seeds, decision rules and failed scientific gates.
