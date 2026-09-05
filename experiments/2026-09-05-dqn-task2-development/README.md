# DQN Task 2 development baseline

> Status: completed negative/mixed result; corrected evaluation evidence is
> archived and linked to bug #82.

## Metadata

- Issue: #46
- Agent: `DagobertDuckDQNTask2`
- Date registered: 2026-09-05
- Parent Task 1 issue: #42
- Task 2 implementation issue: #44
- Run-plan orchestration issue: #50
- Experiment implementation commit: `b5f2024618c772e36481c1298edcc68df62ecbc5`
- Metric correction and evaluation-only rerun commit: `50686bc6cfd4e3d861b6612ed083ecd8ba2607f8`
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

The original evaluation output was invalid because hidden coins produced a zero
denominator. It remains preserved outside Git in the server archive. The
corrected evaluation reused the completed final training workspaces and ran all
840 primary plus 840 repeat episodes.

- Deterministic repeats: passed.
- Decision-time p95 and maximum limits: passed.
- Task 2 crate/coin discovery guard and four-of-five improvement guard: passed.
- Task 2 mean classic collection fraction (`0.30`), `+0.10` improvement,
  self-kill, and per-replica invalid-action gates: failed.
- Task 1 retention CI, four-of-five retention, invalid-action, and zero-bomb
  gates: failed.
- Hierarchical paired differences: classic trained minus untrained `+0.0406`
  (95% CI `[+0.0233, +0.0589]`); coin-heaven trained minus Task 1 `-0.6351`
  (95% CI `[-0.7006, -0.5658]`).

The registered overall decision is therefore **fail**. No confirmation or
final held-out population was used. Compact evidence is in `summary.csv` and
`result.json`; the complete corrected raw archive is
`issue46-corrected-results.tar.gz`, SHA-256
`b6f40ed8c4f8e28a7fa2e91635c6a489b75e9698f28abae219bb2eb9e9d40a9b`.

## Figures and evidence limits

The `figures/` directory contains compact plots generated directly from
`summary.csv`: collection fraction by replica, failure-mode rates, and the
classic action distribution. They summarize the registered evaluation rows;
they are not additional observations.

- [Collection fraction by replica](figures/collection_fraction_by_replica.png)
- [Failure modes by replica](figures/failure_modes_by_replica.png)
- [Classic action distribution](figures/classic_action_distribution.png)

The corrected transfer contains evaluation rows, metadata, and checksums. It
does not contain the original training `episodes.csv` files or resource-monitor
traces, so training learning curves and measured CPU/RAM usage cannot be
reconstructed from this archive. Those evidence items are explicitly
unavailable rather than inferred; the corresponding acceptance criteria remain
open for reviewer confirmation from the server-side raw training directory.

## Decision

The Task 2 feasibility and Task 1 retention gates both fail. This lineage must
not be tuned post hoc. A possible next step is a separately registered
single-factor legal-action-masking experiment; confirmation seeds remain
unopened.
