<#
.SYNOPSIS
Execute one prepared, owner-authorized Task 3 approach pilot end to end.

.DESCRIPTION
Runs training, evaluation, the serial latency stage, the registered analysis and
the evidence bundle under the ceilings in the profile's config.json. The run
stops itself when a ceiling is reached and never extends a budget or promotes a
checkpoint on its own.

Authorization is explicit: -Authorize sets TASK4_AUTHORIZED for this
process only. Without it every stage refuses to start.
#>
[CmdletBinding()]
param(
    [ValidateSet('trainable-scope', 'opponent-mixture', 'finetune-dose', 'lineup-trajectory', 'exploration-period')]
    [string]$Profile = 'trainable-scope',
    [string]$Root = "training_outputs/task4-$Profile",
    [string]$Python = '.\.venv\Scripts\python.exe',
    [switch]$Authorize,
    [switch]$KeepAwake
)

$ErrorActionPreference = 'Stop'
$python = $Python
if (-not (Test-Path $python)) { throw "No interpreter at $python; pass -Python <path>" }
if (-not (Test-Path $Root)) { throw "Prepare $Root first with prepare_task4_competition.ps1" }
if (-not $Authorize) {
    throw 'Refusing to start: re-run with -Authorize once you have approved this allocation.'
}

$env:TASK4_PROFILE = $Profile
$env:TASK4_AUTHORIZED = 'yes'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# psutil's available bytes are what the run's own gate reads; the CIM counter
# reports something different in different units and has misled us before.
$free = & $python -c "import psutil;print(round(psutil.virtual_memory().available/2**30,2))"
Write-Host "Free RAM: $free GiB (psutil available, the figure the gate uses). Close other heavy applications before a long run." -ForegroundColor Yellow

if ($KeepAwake) {
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
    Write-Host 'Sleep and hibernate disabled on AC power for this run.' -ForegroundColor Yellow
}

$started = Get-Date
Write-Host "== chain: train -> evaluate -> latency -> results ==" -ForegroundColor Cyan
& $python -m scripts.pilot_task4_competition chain --root $Root
$code = $LASTEXITCODE

Write-Host ""
Write-Host ("Elapsed: {0:hh\:mm\:ss}" -f ((Get-Date) - $started))
if ($code -ne 0) {
    Write-Host 'Chain did not complete. Inspect the retained stage state and logs:' -ForegroundColor Red
    Write-Host "  $Root/chain-state.json, $Root/chain-*.log, $Root/*-state.json"
    exit $code
}
Write-Host "Completed. Registered analysis: $Root/analysis.json" -ForegroundColor Green
Write-Host "Evidence bundle: $Root/task3-approach-evidence.tar.gz"
Write-Host 'Review the result before writing any interpretation; nothing is promoted automatically.'
