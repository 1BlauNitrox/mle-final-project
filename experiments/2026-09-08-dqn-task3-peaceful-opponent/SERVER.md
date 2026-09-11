# Task 3 server commands

Use a dedicated clean checkout of the full peer-reviewed PR #120 revision,
Python 3.13 and the documented requirements. Run commands from its root.
Do not reuse, change or interrupt Task 2 #124/#91 checkouts, outputs, launchers,
processes or resource records. No scientific execution occurred in preparation.

## Prepare without scientific execution

Retrieve the resumable parent using the durable source in
[the migration record](../2026-09-10-task3-escape-migration/README.md).
For historical exploratory #107 A/r2:

```bash
python scripts/prepare_task3_baseline.py \
  --parent /srv/task3-inputs/parent.pt \
  --parent-sha256 d571b199bc00a0419d16706cc5f7844364a184384058268b2cc8de5b79bb479d \
  --parent-source-commit 69e2a931e4ebfd358cc0128049516b1a12adc2a8 \
  --output-dir /srv/task3-bindings/issue109-exploratory

python -m training.run_task3_campaign --dry-run \
  --binding-dir /srv/task3-bindings/issue109-exploratory
```

Use a new binding directory. A different selected Task 2 parent requires its
verified SHA/source/selection evidence and a separate binding. Evaluation-only
exports lack the target network; supply the resumable source. Preserve the old
`training_outputs/issue125-task3-baseline` with its ten-pair plans.

Optional isolated integration smoke, two training and four evaluation episodes:

```bash
python -m training.run_plan /srv/task3-bindings/issue109-exploratory/smoke.yaml \
  --output-root /srv/task3-smokes/issue109-new
```

Smokes are not scientific evidence. Never execute the scientific candidate YAML
directly: it bypasses the shared budget and fails campaign-evidence validation.

## Launch only after separate review and compute authorization

Replace the commit, authorizer and hardware values with actual approved values.
Reserve at least 8 GiB for the campaign plus operating-system headroom. These
ceilings do not authorize competition for another session's resources.

```bash
python -m training.run_task3_campaign \
  --binding-dir /srv/task3-bindings/issue109-exploratory \
  --output-root /srv/task3-runs/issue109-exploratory \
  --reviewed-commit FULL_PEER_REVIEWED_COMMIT_SHA \
  --authorized-by HUMAN_COMPUTE_AUTHORIZER \
  --hardware-description ACTUAL_SERVER_CPU_AND_RAM \
  --available-memory-gib 8 \
  --authorize-compute
```

Both plans share one monitor and run sequentially. Resume with the identical
command plus `--resume`, retaining authorization, resources and failed attempts.
Resume never resets CPU consumption or elapsed wall time. A hard-crash lock may
be removed only after verifying this campaign's owner has exited; do not stop
other processes. No second launcher may share these records concurrently.

## Analyze retained evidence

```bash
python -m training.analyze_task3_campaign \
  --campaign-root /srv/task3-runs/issue109-exploratory \
  --binding-dir /srv/task3-bindings/issue109-exploratory \
  --output /srv/task3-analysis/issue109-exploratory

python -m training.analyze_task3_campaign \
  --verify-evidence /srv/task3-analysis/issue109-exploratory
```

Use a new output directory. Publish the necessary observations, artifacts and
provenance durably before a result claim. All peaceful gates must pass before
considering the selected median replica under #137. These commands never
automatically launch coin-collector training or declare Task 2 complete.
