param(
    [Parameter(Mandatory=$true)][string]$RunRoot,
    [Parameter(Mandatory=$true)][string]$Python
)
$ErrorActionPreference = 'Stop'
if ($env:TASK175_PILOT_AUTHORIZED -ne 'yes') { throw 'Issue175 authorization required.' }
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runPath = (Resolve-Path -LiteralPath $RunRoot).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
Push-Location $repoRoot
try {
    & $pythonPath -m scripts.pilot_task3_kill_reward dry-run --root $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Dry-run failed.' }
    if (Test-Path -LiteralPath (Join-Path $runPath 'chain-state.json')) { throw 'Existing chain; inspect it.' }
    $arguments = @('-m','scripts.pilot_task3_kill_reward','chain','--root',('"' + $runPath + '"'))
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $outLog = Join-Path $runPath "chain-$stamp.out.log"
    $errLog = Join-Path $runPath "chain-$stamp.err.log"
    $launchProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
    [PSCustomObject]@{ PID=$launchProcess.Id; Output=$outLog; Errors=$errLog; Root=$runPath }
} finally { Pop-Location }
