param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$Python,
    [ValidateSet('update-frequency','opponent-regularization')]
    [string]$Profile = 'opponent-regularization'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$taskRoot = (Resolve-Path -LiteralPath $Root).Path
$runs = Join-Path $taskRoot 'runs'
$inputs = Join-Path $taskRoot 'inputs'
$archive = Join-Path $taskRoot 'issue175-pilot-evidence.tar.gz'
if (Test-Path -LiteralPath $runs) { throw 'Run root exists; preserve it and inspect before resume.' }
$env:TASK3_STABILITY_PROFILE = $Profile
function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $Arguments" }
}
Push-Location $repo
try {
    Invoke-CheckedPython -Arguments @('-m','scripts.fetch_task3_stability_inputs','--download','--archive',$archive,'--output',$inputs)
    Invoke-CheckedPython -Arguments @('-m','scripts.pilot_task3_stability','prepare','--root',$runs,'--parent',(Join-Path $inputs 'reference.pt'),'--initial',(Join-Path $inputs 'initial.pt'))
    Invoke-CheckedPython -Arguments @('-m','scripts.pilot_task3_stability','dry-run','--root',$runs)
    Invoke-CheckedPython -Arguments @('-m','scripts.pilot_task3_stability','smoke','--root',$runs,'--output',(Join-Path $taskRoot 'smoke'))
    Write-Output 'Preparation and isolated smoke passed. No scientific training started.'
    Write-Output "Run root: $runs"
} finally { Pop-Location }
