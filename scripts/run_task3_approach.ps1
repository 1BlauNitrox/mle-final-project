<#
.SYNOPSIS
Execute one prepared, owner-authorized Task 3 approach pilot end to end.

.DESCRIPTION
Runs training, evaluation, the serial latency stage, the registered analysis and
the evidence bundle under the ceilings in the profile's config.json. The run
stops itself when a ceiling is reached and never extends a budget or promotes a
checkpoint on its own.

Authorization is explicit: -Authorize sets TASK3_APPROACH_AUTHORIZED for this
process only. Without it every stage refuses to start.
#>
[CmdletBinding()]
param(
    [ValidateSet('approach-shaping', 'update-cadence')]
    [string]$Profile = 'approach-shaping',
    [string]$Root = "training_outputs/task3-$Profile",
    [switch]$Authorize,
    [switch]$KeepAwake
)

$ErrorActionPreference = 'Stop'
$python = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Expected a virtual environment at $python" }
if (-not (Test-Path $Root)) { throw "Prepare $Root first with prepare_task3_approach.ps1" }
if (-not $Authorize) {
    throw 'Refusing to start: re-run with -Authorize once you have approved this allocation.'
}

$env:TASK3_APPROACH_PROFILE = $Profile
$env:TASK3_APPROACH_AUTHORIZED = 'yes'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
Write-Host "Free RAM: $free GB. Close other heavy applications before a long run." -ForegroundColor Yellow

if ($KeepAwake) {
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
    Write-Host 'Sleep and hibernate disabled on AC power for this run.' -ForegroundColor Yellow
}

$started = Get-Date
Write-Host "== chain: train -> evaluate -> latency -> results ==" -ForegroundColor Cyan
& $python -m scripts.pilot_task3_approach chain --root $Root
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
