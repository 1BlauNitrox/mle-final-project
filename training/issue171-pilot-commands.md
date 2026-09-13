# Issue 171 execution commands

The executable registration is `experiments/2026-09-13-task3-learning-rate-retention/config.json`.
Runtime source is `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`.
The dedicated PC worktree is `C:/Users/Julius/AppData/Local/Temp/issue171-retention`.
Use the existing repository Python 3.13 environment. The original #168 worktree is read-only input.

```powershell
$py = 'C:/Users/Julius/Desktop/Uni/SS 26/machine-learning-essentials/.BOMBERMAN/mle-final-project/.venv/Scripts/python.exe'
Set-Location 'C:/Users/Julius/AppData/Local/Temp/issue171-retention'
$run = Join-Path $PWD 'training_outputs/pilot-ready'
& $py -m scripts.pilot_task3_learning_rate dry-run --root $run
$env:TASK171_PILOT_AUTHORIZED = 'yes' # Owner explicitly authorized this exact allocation.
./scripts/run_issue171_pilot.ps1 -RunRoot $run -Python $py
```

The hidden supervisor runs training, then evaluation, then analysis/export serially.
It stops after a failed stage; retries and extensions are never automatic. The
stage monitors retain accounting, attempts and every episode checkpoint. Overall
chain deadline is two hours; individual training/evaluation ceilings are 60/45
CPU and wall minutes. Keep the PC awake and leave other sessions untouched.

Read progress without launching another chain:

```powershell
Get-Content "$run/chain-state.json" -Raw
Get-Content "$run/training-state.json" -Raw
Get-Content "$run/evaluation-state.json" -Raw
Get-Content "$run/chain-train.log" -Tail 15
```

On completion, inspect `analysis.json` and `issue171-pilot-evidence.tar.gz.manifest.json`
in the run root. The archive retains source, bindings, raw native observations,
all training checkpoints and stage state. Reproduce analysis without new games:

```powershell
& $py -m scripts.pilot_task3_learning_rate analyze --root $run --output "$run/recomputed-analysis.json"
```

For independent archive verification, check its manifest SHA-256 and byte size,
then import into a **new** directory using `import --root <new-root> --archive
<archive> --sha256 <manifest-sha>`, and run the same analyzer there at the recorded
tool commit. Preserve the original output root and all attempts.

Human handoff: the concise experiment README draft is uncommitted and must be
read, verified and manually modified by Julius before commit. A manual disclosure
entry in `docs/0006-ai-usage.md` is required for protocol implementation, validation,
execution and analysis. No agent may edit that log. No checkpoint is selected or
promoted; the existing cumulative gates and coin-collector helper remain in force.
