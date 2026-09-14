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
    $referencePath = Join-Path $repo "experiments/2026-09-14-task3-$Profile/preflight.json"
    $reference = Get-Content -Raw -LiteralPath $referencePath | ConvertFrom-Json
    $observedPath = Join-Path $taskRoot 'smoke.resources.json'
    $observed = Get-Content -Raw -LiteralPath $observedPath | ConvertFrom-Json
    $config = Get-Content -Raw -LiteralPath (Join-Path $runs 'config.json') | ConvertFrom-Json
    $ratio = $observed.wall_seconds / $reference.smoke_resources.wall_seconds
    $trainEstimate = 7343.21 * $ratio * 1.2
    $evalEstimate = 8506.82 * $ratio * 1.2
    $estimate = [PSCustomObject]@{
        Basis = 'Matched technical smoke scaling of issue175, with 20 percent margin; estimate only'
        TrainingMinutes = [math]::Round($trainEstimate / 60, 1)
        EvaluationMinutes = [math]::Round($evalEstimate / 60, 1)
        FitsTraining = ($trainEstimate -lt $config.training_limits.wall_seconds)
        FitsEvaluation = ($evalEstimate -lt $config.evaluation_limits.wall_seconds)
    }
    $estimate | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $taskRoot 'timing-estimate.json')
    $estimate | Format-List
    if (-not $estimate.FitsTraining -or -not $estimate.FitsEvaluation) {
        throw 'Estimated work exceeds a registered stage budget. Do not launch; share timing-estimate.json for a prospective plan adjustment.'
    }
    Write-Output 'Preparation and isolated smoke passed. No scientific training started.'

    Write-Output "Run root: $runs"
} finally { Pop-Location }
