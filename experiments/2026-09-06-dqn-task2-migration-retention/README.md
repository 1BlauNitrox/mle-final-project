# Issue #85 DQN migration-retention evaluation

## Metadata

- Issue: #85
- Branch: `fix/85-preserve-task1-q-values`
- Owner: Waffelmanufaktur
- Reviewer: 1BlauNitrox
- Status: prospective; review required before execution
- Agent: `DagobertDuckDQNTask2`

## Hypothesis

Zeroing the 13 Task 2-only input-layer columns preserves the frozen parent
policy's first five Q-values on Task 1 states within an absolute `1e-5`
float32 tolerance and removes the old migration's pre-learning function
perturbation. This protocol measures only the associated coin-collection
behavior; it does not attribute the full Issue #46 retention deficit to this
defect.

## Controlled comparison

All arms use the same 10 development-only world/agent seed pairs, the
`coin-heaven` scenario, no opponents, one episode per pair, training disabled,
one CPU thread, and a deterministic-repeat pass. No checkpoint selection,
training, or held-out seed is permitted.

| Arm | Artifact | Purpose |
| --- | --- | --- |
| Frozen parent | `DagobertDuckDQN/checkpoint.pt` | Task 1 reference |
| Old migration | `DagobertDuckDQNTask2/checkpoint.pt` | Preserved #46 start |
| Corrected migration | `DagobertDuckDQNTask2/checkpoint-issue85-zero-suffix.pt` | Issue #85 treatment |

The primary seed pairs are world/agent `36001` through `36010` and `46001`
through `46010`, respectively. They are new development seeds, distinct from
Issue #46 training, development-evaluation, and reserved confirmation seeds.
The repeat reuses each exact pair and must reproduce its action digest.

Before those episodes, generate the separate, deterministic Q-value/action
evidence from the committed probe population and checksum-pinned parent and
corrected artifacts:

```bash
python -m training.dqn_task2_migration_contract \
  --output training_outputs/issue85/probe-report.json
```

`probes.json` is the version-one registered probe manifest. The command refuses
changed artifacts, records each probe's inherited five Q-values, BOMB Q-value,
and both selected actions, then rejects incomplete or altered evidence.

## Metrics and decision rule

Report Q-value/action agreement separately from measured collection fraction,
coins, invalid-action rate, BOMB count, and decision-time median/p95/maximum.
The compatibility limits are p95 below 50 ms and maximum below 100 ms on both
passes. The corrected migration must match the parent action on every registered
probe and must not select BOMB on the Task 1 suite. This is a migration-contract
check, not a success criterion for Task 2 learning or a claim that it resolves
the Issue #46 retention result.

The planned budget is 60 evaluation-only episodes (three arms, ten seed pairs,
two passes), serial on one CPU thread, with a 30-minute wall-clock ceiling and
8 GiB memory ceiling. Retain all rows and report variation; do not rerun,
replace, or select checkpoints after inspecting results.
