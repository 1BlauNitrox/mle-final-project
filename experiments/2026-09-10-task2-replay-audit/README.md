# Task 2 replay and stage-seed prerequisite

Refs #123 and #124. This is a read-only diagnostic of Issue #107 and a runner
prerequisite, not a new performance experiment or evidence of Task 2 completion.

All thirty snapshots pass the recorded checks in `result.json`. No protected
replay loss or counter/epsilon discontinuity was found at these boundaries.
This does not establish that a fixed historical replay pool is an effective
retention strategy. Full local validation: 713 tests and 11 subtests passed.

The compact result audits C/D's thirty retained stage checkpoints from the
checksum-pinned `issue107-evidence-v1` archive. It checks protected contents,
partition sizes and closure, episode/epsilon counters, monotonic update counts,
and sampling after restoring the persisted replay RNG. At the initial Task 1
boundary the other partition is empty, so the registered backfill rule draws
64 protected transitions; later complete partitions draw 16 protected and 48
other transitions. A checkpoint cannot reconstruct historical minibatches or
prove continuous optimizer state at every episode.

Reproduce from this branch, using the archive documented in
`experiments/2026-09-07-dqn-task2-factorial/EVIDENCE.md`:

```bash
python -m scripts.audit_issue107_replay --archive /path/to/issue107-evidence-v1.tar.gz --output /path/to/replay-audit.json
python -m pytest tests/test_run_plan.py tests/test_DagobertDuckDQNTask2_protected_replay.py tests/test_audit_issue107_replay.py -q
```

The runner now accepts a non-negative integer `world_seed_offset` per training
stage. Its resolved world seed is replica seed plus offset, bounded to the
NumPy seed range and checked against evaluation populations. Omitted offsets
retain existing behavior. Agent seeds remain unchanged so stage continuation
does not reset the learned policy RNG. The new experiment must pair explicit
offsets by scenario occurrence and use equal stage boundaries across arms.
