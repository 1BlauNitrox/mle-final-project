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

## Result table and figure export (no games)

After importing and verifying the archive, use the result-exporter commit in
addition to the registered runner source. Install `requirements-dev.txt` for
matplotlib. Export into a fresh destination; differing existing outputs are
rejected. This command never writes a README or the AI log.

```powershell
& $py -m scripts.export_task3_retention_results `
  --root training_outputs/result-verification-01 `
  --archive training_outputs/pilot-ready/issue171-pilot-evidence.tar.gz `
  --output training_outputs/issue171-review-tables
```

The exporter recomputes the registered analysis, checks the archive SHA-256 and
size and every member against the extraction, then generates native compact
observations, episode/replica/arm tables, all paired contrasts, gates, checkpoint
hashes and a figure. The manifest explicitly marks durable publication pending;
replace that status only after uploading and verifying retrievable bytes.

The original `setup-verification.json` records preflight state. Post-run
`execution-verification.json`, `chain-completion.json`, `smoke-result.json`,
`smoke-resources.json` and `result-validation.json` retain completion, resource
and verification records. The chain/smoke supplements are separate from the
immutable 966-member experiment archive. Local human-edit/publication handoff:
`training_outputs/RESULTS-HANDOFF.md`.
