# Issue 168 pilot: training and evaluation on one PC

The registered pilot is complete: 300 training and 280 evaluation episodes,
followed by analysis and compact export. Do not restart its completed root.
The failed decision is recorded in the
[experiment README](../experiments/2026-09-13-task3-exploration-screen/README.md).
For result verification, use the
[archive reproduction instructions](../experiments/2026-09-13-task3-exploration-screen/phase-d-reproduction.md).

Both training arms and every evaluation ran serially on the same PC. Future
device allocations should assign whole experiments, including matched controls
and evaluation, to each device. No laptop transfer is needed for this result.

## Inspect the completed local run

From PowerShell, regardless of the current directory:

```powershell
$repo = Join-Path $env:TEMP 'issue168-exploration'
$run = Join-Path $repo 'training_outputs/phase-d-ready'
Get-Content "$run/pipeline-status.json"
Get-Content "$run/analysis.json" -TotalCount 35
```

Exact source: `dcff79fefe812becbe76e7bdf2d1718a9eda4cea`.
Scientific runtime: `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`.
The original initialization and every final checkpoint remain preserved.
Results: `training_outputs/issue168-pilot-evidence.tar.gz` and its manifest.

## Stage commands for a separately authorized reproduction

Use a fresh root, the unchanged registered protocol and the verified parent.
Do not reuse these seeds for new parameter tuning. New scientific comparisons
need their own prospective configuration and output root.
Set `$py`, `$repo`, `$run` and `$parent` to absolute paths before these commands:

```powershell
Set-Location $repo
& $py -m scripts.pilot_task3_episode_exploration prepare --root $run --parent $parent
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
& $py -m scripts.pilot_task3_episode_exploration dry-run --root $run
if ($LASTEXITCODE -ne 0) { throw 'Dry run failed' }
```

After the concrete run budget is approved, this sequence trains and evaluates
on the same machine. It stops on any failed stage:

```powershell
$env:TASK168_PILOT_AUTHORIZED = 'yes'
& $py -m scripts.pilot_task3_episode_exploration train --root $run
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect retained attempts' }
& $py -m scripts.pilot_task3_episode_exploration evaluate --root $run
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; inspect retained attempts' }
$out = Join-Path $run 'results.tar.gz'
& $py -m scripts.pilot_task3_episode_exploration results --root $run --output $out
if ($LASTEXITCODE -ne 0) { throw 'Analysis/export failed' }
```

The detached single-stage helper remains `scripts/run_issue168_pilot.ps1`,
with `-Stage train` or `-Stage evaluate`, `-RunRoot` and `-Python`. It starts only
that stage. The completed PC run used a local sequential wrapper, retained in
its output root, to invoke all stages automatically.

For a handled interruption, inspect the stage state and attempt before using
`--resume` for that stage. Completed jobs are skipped; failed attempts are
preserved. Never change source, parent, seeds or resource accounting to force a
resume, and never rerun a completed stage. This does not authorize a new run.
