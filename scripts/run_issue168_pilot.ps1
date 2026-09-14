param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('train','evaluate')]
    [string]$Stage,
    [Parameter(Mandatory=$true)]
    [string]$RunRoot,
    [Parameter(Mandatory=$true)]
    [string]$Python,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
if ($env:TASK168_PILOT_AUTHORIZED -ne 'yes') {
    throw 'Explicit pilot budget approval is required before setting TASK168_PILOT_AUTHORIZED=yes.'
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runPath = (Resolve-Path -LiteralPath $RunRoot).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
Push-Location $repoRoot
try {
    & $pythonPath -m scripts.pilot_task3_episode_exploration dry-run --root $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Pilot dry-run failed.' }
    $arguments = @('-m','scripts.pilot_task3_episode_exploration',$Stage,'--root',('"' + $runPath + '"'))
    if ($Resume) { $arguments += '--resume' }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $outLog = Join-Path $runPath "$Stage-$stamp.out.log"
    $errLog = Join-Path $runPath "$Stage-$stamp.err.log"
    $launchProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
    [PSCustomObject]@{ PID=$launchProcess.Id; Stage=$Stage; Output=$outLog; Errors=$errLog; Root=$runPath }
} finally { Pop-Location }
