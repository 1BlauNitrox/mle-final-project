param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$RunRoot
)
$ErrorActionPreference = 'Stop'
$sourceRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$runPath = (Resolve-Path -LiteralPath $RunRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'smoke.json'))) {
    throw 'Prepare and pass the smoke check before launch.'
}
if (Test-Path -LiteralPath (Join-Path $runPath 'STOP.json')) {
    throw 'Persistent stop record exists. Preserve it; do not bypass a failed pilot or budget.'
}
$scriptPath = Join-Path $sourceRoot 'scripts/run_hunting_curriculum.py'
$arguments = '"{0}" run --root "{1}"' -f $scriptPath, $runPath
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments `
    -WorkingDirectory $sourceRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runPath 'supervisor.out.log') `
    -RedirectStandardError (Join-Path $runPath 'supervisor.err.log')
Write-Output "Started supervisor PID $($process.Id). Status: $runPath/resources.json"
