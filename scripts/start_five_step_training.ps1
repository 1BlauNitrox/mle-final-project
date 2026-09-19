[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Root,
    [string]$Python = 'C:\task3-cadence\.venv\Scripts\python.exe',
    [ValidateRange(1, 3)]
    [int]$Workers = 3
)

$ErrorActionPreference = 'Stop'
$rootPath = [IO.Path]::GetFullPath($Root)
if (-not (Test-Path -LiteralPath $rootPath)) { throw "Missing prepared root $rootPath" }
$smoke = Join-Path $rootPath 'smoke\report.json'
if (-not (Test-Path -LiteralPath $smoke)) { throw 'Run the registered smoke first' }
$script = Join-Path $PSScriptRoot 'run_five_step_experiment.py'
& $Python $script launch --root $rootPath --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "Launcher failed with exit code $LASTEXITCODE" }
Write-Host "Status: & '$Python' '$script' status --root '$rootPath'"
Write-Host "Resume after interruption: .\scripts\start_five_step_training.ps1 -Root '$rootPath' -Workers $Workers"
