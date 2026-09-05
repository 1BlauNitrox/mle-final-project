# DQN Task 2 development baseline

> Status: preregistered and prepared; no scientific training or evaluation has
> started.

## Metadata

- Issue: #46
- Agent: `DagobertDuckDQNTask2`
- Date registered: 2026-09-05
- Parent Task 1 issue: #42
- Task 2 implementation issue: #44
- Run-plan orchestration issue: #50
- Experiment implementation commit:
  `b5f2024618c772e36481c1298edcc68df62ecbc5`
- Owner: Waffelmanufaktur
- Non-author reviewer: 1BlauNitrox (approval required before the first run)

## Research question and hypothesis

Does the unchanged Task 2 implementation learn reproducible crate destruction
and hidden-coin collection under a staged 10,000-episode curriculum without
materially degrading visible-coin navigation?

The experiment passes only if both the Task 2 feasibility gate and the Task 1
retention gate registered in `config.yaml` pass. Mixed and negative results are
retained.

## Baselines

- Primary learning baseline: untrained Task 2 migration checkpoint from #44,
  SHA-256 `44cd337001b27b8596eaed985cfae1d7f30ecaf0b6b0328b35185395b7b81b6e`.
- Retention baseline: frozen Task 1 DQN checkpoint from #42, SHA-256
  `eb08e3f67b620ac2a253a2af4db3d5b4c6ea9e667a2aaf1d91e3fccf4ba8b05e`.

## Training protocol

Five seed-compatible starts preserve identical migrated online and target
weights while assigning the registered independent action and replay RNG
streams. Each replica trains sequentially for:

1. 2,000 `coin-heaven` episodes;
2. 2,000 `loot-crate` episodes;
3. 6,000 `classic` episodes.

There are no opponents. Only the checkpoint after episode 10,000 is selected.
The exact plans, seeds, evaluation repeats, thresholds, and resource ceiling
are in `config.yaml` and the three referenced run-plan files.

## Execution

From a clean checkout of the reviewed experiment commit:

```bash
python -m training.prepare_dqn_task2_experiment
python -m training.run_plan training/run_plans/issue46-dqn-task2-trained.yaml --dry-run
python -m training.run_plan training/run_plans/issue46-dqn-task2-untrained.yaml --dry-run
python -m training.run_plan training/run_plans/issue46-dqn-task1-frozen.yaml --dry-run
```

After verifying the three dry-run matrices, execute the same plans without
`--dry-run`. An interrupted plan is continued with `--resume`; it must never be
restarted under the same `plan_id` without preserving the failed record.

After all three plans complete:

```bash
python -m training.analyze_dqn_task2_experiment
```

Raw outputs remain under `training_outputs/`. Before transferring them from the
server, archive that directory and record the archive SHA-256.

## Results

Pending. No values may be added here until the registered runs complete.

## Decision

Pending. The next step is selected only from the registered outcome: freeze and
confirm, run an isolated legal-action-masking experiment, register another
single-factor experiment, or stop this lineage.
