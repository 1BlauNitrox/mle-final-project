<#
.SYNOPSIS
Prepare one registered Task 3 approach pilot without authorizing execution.

.DESCRIPTION
Fetches the checksum-bound checksum-bound inputs, binds the arm initializations, archives
the pinned runtime and runs the mechanical smoke. Preparation never trains: it
prints the measured throughput so the registered budget can be checked against
this host before anyone authorizes the run.
#>
[CmdletBinding()]
param(
    [ValidateSet('trainable-scope', 'opponent-mixture')]
    [string]$Profile = 'trainable-scope',
    [string]$Root = "training_outputs/task4-$Profile",
    [string]$Archive = 'training_outputs/inputs/issue175-pilot-evidence.tar.gz',
    [string]$Python = '.\.venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$python = $Python
if (-not (Test-Path $python)) { throw "No interpreter at $python; pass -Python <path>" }
if (Test-Path $Root) { throw "Output root $Root already exists; inspect and preserve it" }

$env:TASK4_PROFILE = $Profile
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

Write-Host "== seed audit ==" -ForegroundColor Cyan
& $python -m scripts.audit_task4_competition_seeds --profile $Profile
if (-not $?) { throw 'Seed audit failed' }

Write-Host "== inputs ==" -ForegroundColor Cyan
& $python -m scripts.fetch_task4_competition_inputs --archive $Archive --output training_outputs/inputs/$Profile --download
if (-not $?) { throw 'Input retrieval failed' }

Write-Host "== prepare ==" -ForegroundColor Cyan
& $python -m scripts.pilot_task4_competition prepare `
    --root $Root `
    --parent training_outputs/inputs/$Profile/reference.pt `
    --initial training_outputs/inputs/$Profile/initial.pt
if (-not $?) { throw 'Preparation failed' }

Write-Host "== mechanical smoke (not a scientific run) ==" -ForegroundColor Cyan
& $python -m scripts.pilot_task4_competition smoke --root $Root --output "$Root/smoke"
if (-not $?) { throw 'Smoke failed' }

Write-Host "== dry run ==" -ForegroundColor Cyan
& $python -m scripts.pilot_task4_competition dry-run --root $Root

Write-Host ""
Write-Host "Prepared. Execution is NOT authorized yet." -ForegroundColor Yellow
Write-Host "Compare the smoke throughput above with the registered budget, then run:"
Write-Host "  .\scripts\run_task4_competition.ps1 -Profile $Profile -Root $Root"
