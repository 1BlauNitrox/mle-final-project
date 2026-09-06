# Tabular Task 2 development baseline

> Status: prospective protocol; no scientific run has started.

## Metadata

- Issue: #102
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Reviewer: 
- Date: 2026-09-06
- Registration base: `8ec66d760e6601f8b82091f572d12a33c044b1bf`
- Experiment commit: ``
- Framework revision: `0f55c1d`

## Research question

Can the unchanged Task 2 defaults learn crate destruction, hidden-coin
collection, and bomb survival while retaining the frozen Task 1 capability?

## Hypothesis

Five independently trained replicas will outperform the untrained migrated
Task 2 agent on `classic` and retain most of the frozen Task 1 agent's
`coin-heaven` performance.

Both the Task 2 and Task 1 gates in `config.yaml` must pass.

## Baselines

- Untrained Task 2 artifact:
  `DerKleineSprengstoffkapitalist/model.npz`,
  SHA-256 `8f2e618bfb38d690b565be1d3034f153d120887a36d90a61f8adcc1a765c1bbb`.
- Frozen Task 1 artifact:
  `DerKleineVermoegensumverteiler/model.npz`,
  SHA-256 `4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307`.

## Training protocol

Train five replicas from the unchanged migrated artifact:

1. 2,000 `coin-heaven` episodes;
2. 2,000 `loot-crate` episodes;
3. 6,000 `classic` episodes without opponents.

Only the final checkpoint after 10,000 episodes is evaluated. Rewards,
features, hyperparameters, and exploration settings remain unchanged.

## Evaluation protocol

Evaluate with training and exploration disabled:

- trained versus untrained Task 2 on `classic` and `loot-crate`;
- trained versus frozen Task 1 on `coin-heaven`;
- 40 paired development seeds per suite;
- one complete deterministic repeat;
- confirmation seeds remain unused.

## Metrics and success criteria

Primary metric: hidden-coin collection fraction on `classic`, including
initially hidden coins in the denominator.

The experiment passes only if:

- mean `classic` collection fraction is at least `0.30`;
- trained-minus-untrained improvement is at least `0.10`;
- its paired 95% CI lower bound is above zero;
- at least four of five replicas improve;
- aggregate self-kill rate is at most `0.20`;
- Task 1 retention CI lower bound is above `-0.05`;
- at least four replicas remain within `0.10` of Task 1;
- Task 1 invalid-action rate is below `0.01`;
- no bombs occur on `coin-heaven`;
- decision-time p95 is below `50 ms` and maximum below `100 ms`;
- evaluation is deterministic and artifacts remain unchanged.

Secondary metrics include crates destroyed, coins found and collected, useful
bombs, survival, invalid actions, action counts, steps per coin, score, Q-table
size, TD error, epsilon, and reward.

## Evidence

Commit the protocol, exact configuration, compact per-seed observations,
aggregates, result JSON, and final figures. Keep raw logs, replays, temporary
checkpoints, and training workspaces outside Git.

Large required evidence must have a durable location, SHA-256, byte size,
schema, retrieval command, and verification command.

## Execution

```bash
python -m training.run_plan training/run_plans/issue102-tabular-task2-trained.yaml --dry-run
python -m training.run_plan training/run_plans/issue102-tabular-task2-untrained.yaml --dry-run
python -m training.run_plan training/run_plans/issue102-tabular-task1-frozen.yaml --dry-run
```

Training starts only after non-author approval of this protocol.

## Results

Not available. No scientific run has started.

## Decision

Pending.