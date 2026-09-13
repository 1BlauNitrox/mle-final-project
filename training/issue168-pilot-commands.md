ï»¿# Issue 168 pilot commands

Source for the commands below: `bf4aef4823512ae4b63dd5b20143f55e818c2887`.
The scientific runtime remains `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`.
The exact protocol is [phase-d-config.json](../experiments/2026-09-13-task3-exploration-screen/phase-d-config.json).
Training and evaluation require the owner's explicit approval of the registered
300 training / 280 evaluation episodes, with 30 training CPU/wall minutes and
45 evaluation CPU/wall minutes. Reserve another 15 minutes for transfer.
Do not set the authorization variable until that approval exists.

## This PC

From the original repository directory, reuse its Python 3.13 environment.
The dedicated preparation is separate from the original checkout and its changes.

```powershell
$py = Join-Path $PWD '.venv/Scripts/python.exe'
$repo = Join-Path $env:TEMP 'issue168-exploration'
$run = Join-Path $repo 'training_outputs/phase-d-ready'
Set-Location $repo
& $py -m scripts.pilot_task3_episode_exploration dry-run --root $run
```

After the concrete pilot allocation is approved:

```powershell
$env:TASK168_PILOT_AUTHORIZED = 'yes'
$launch = @{ Stage='train'; RunRoot=$run; Python=$py }
& ./scripts/run_issue168_pilot.ps1 @launch
```

Progress (the state file is written atomically):

```powershell
$state = Get-Content "$run/training-state.json" -Raw | ConvertFrom-Json
$state | Select-Object status,cpu_seconds,wall_seconds,peak_memory_bytes
$state.completed.PSObject.Properties.Name
$state.attempts | Select-Object job,status,output
Get-ChildItem $run -Filter 'train-*.err.log' | Get-Content -Tail 15
```

For a handled failure, inspect the retained error/attempt before explicitly
resuming within the original remaining budget. Completed replicas are skipped;
a failed replica gets a fresh attempt from the same fixed initialization and
seeds. Its previous partial checkpoint and observations remain preserved.
An abrupt supervisor exit with uncertain accounting is rejected for audit.
A partial stage cannot move to another host or change source/parameters.

```powershell
& ./scripts/run_issue168_pilot.ps1 @launch -Resume
```

When all six training jobs are complete, create the laptop input archive:

```powershell
$out = Join-Path $HOME 'Downloads/issue168-pilot-inputs.tar.gz'
& $py -m scripts.pilot_task3_episode_exploration bundle --root $run --output $out
Get-Content "$out.manifest.json"
```

Copy the archive and its `.manifest.json` into the laptop's Downloads folder.
The archive includes every final model, compact training observations, frozen
inputs and source; it excludes raw diagnostic logs and virtual environments.

## Windows laptop

Python 3.13 must be available through `py -3.13`. Use a fresh checkout:

```powershell
$task = Join-Path $HOME 'task3-issue168'
$repo = Join-Path $task 'repo'
$url = 'https://github.com/1BlauNitrox/mle-final-project.git'
New-Item -ItemType Directory -Force $task | Out-Null
git clone $url $repo
Set-Location $repo
$rev = 'bf4aef4823512ae4b63dd5b20143f55e818c2887'
git checkout --detach $rev
py -3.13 -m venv .venv
$py = Join-Path $repo '.venv/Scripts/python.exe'
& $py -m pip install -r requirements-dev.txt
```

After the PC input archive and manifest arrive:

```powershell
$archive = Join-Path $HOME 'Downloads/issue168-pilot-inputs.tar.gz'
$manifest = Get-Content "$archive.manifest.json" -Raw | ConvertFrom-Json
$run = Join-Path $task 'run'
$import = @('import','--root',$run,'--archive',$archive)
$import += @('--sha256',$manifest.sha256)
& $py -m scripts.pilot_task3_episode_exploration @import
& $py -m scripts.pilot_task3_episode_exploration dry-run --root $run
```

Use mains power and keep the laptop awake. Keep evaluation serial in one
recorded environment. After approval of the pilot allocation:

```powershell
$env:TASK168_PILOT_AUTHORIZED = 'yes'
$launch = @{ Stage='evaluate'; RunRoot=$run; Python=$py }
& ./scripts/run_issue168_pilot.ps1 @launch
```

Progress and a handled-failure resume:

```powershell
$state = Get-Content "$run/evaluation-state.json" -Raw | ConvertFrom-Json
$state | Select-Object status,cpu_seconds,wall_seconds,peak_memory_bytes
$state.completed.PSObject.Properties.Name
Get-ChildItem $run -Filter 'evaluate-*.err.log' | Get-Content -Tail 15
```

```powershell
& ./scripts/run_issue168_pilot.ps1 @launch -Resume
```

After all 28 evaluation jobs finish, analyze and export:

```powershell
$out = Join-Path $HOME 'Downloads/issue168-pilot-results.tar.gz'
& $py -m scripts.pilot_task3_episode_exploration results --root $run --output $out
Get-Content "$out.manifest.json"
Get-Content "$run/analysis.json" -TotalCount 35
```

Copy that archive and manifest back to this PC's Downloads. Import them into a
new result root using the same `import` command, then reproduce the analysis:

```powershell
$analysis = Join-Path $run 'recomputed-analysis.json'
& $py -m scripts.pilot_task3_episode_exploration analyze --root $run --output $analysis
```

All results, including failed gates, are retained. These commands do not select
or publish a checkpoint, authorize a larger run, or merge a PR. If the laptop is
unavailable, evaluate on the PC after training using its existing root and the
same `evaluate` stage; do not split compared artifacts between hosts.
