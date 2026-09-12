param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [switch]$Resume,
    [switch]$OwnerAuthorizedReviewException
)
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputRoot)
Push-Location $repository
try {
    $headSha = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve execution SHA' }
    & $pythonPath -m training.run_issue124_campaign --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Protocol validation failed' }
    $reviewArguments = @()
    if ($OwnerAuthorizedReviewException) {
        $reviewArguments += '--owner-authorized-review-exception'
    }
    & $pythonPath -m training.run_issue124_campaign --check-review --reviewed-commit $headSha @reviewArguments
    if ($LASTEXITCODE -ne 0) { throw 'Exact-commit non-author review validation failed' }
    New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdoutPath = Join-Path $outputPath "supervisor-$stamp.stdout.log"
    $stderrPath = Join-Path $outputPath "supervisor-$stamp.stderr.log"
    $launchArguments = @('-m', 'training.run_issue124_campaign', '--authorize-compute',
        '--reviewed-commit', $headSha, '--output-root', ('"' + $outputPath + '"'))
    $launchArguments += $reviewArguments
    if ($Resume) { $launchArguments += '--resume' }
    $worker = Start-Process -FilePath $pythonPath -ArgumentList $launchArguments `
        -WorkingDirectory $repository -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    [PSCustomObject]@{pid=$worker.Id; reviewed_commit=$headSha; output=$outputPath;
        stdout=$stdoutPath; stderr=$stderrPath} | ConvertTo-Json |
        Set-Content -Encoding UTF8 -LiteralPath (Join-Path $outputPath "launcher-$stamp.json")
    Start-Sleep -Seconds 3
    if ($worker.HasExited) {
        Get-Content -LiteralPath $stderrPath
        throw 'Supervisor exited during startup; no successful launch is claimed'
    }
    Write-Output "Detached campaign supervisor PID: $($worker.Id)"
    Write-Output "Status: $outputPath\campaign-status.json"
    Write-Output "Errors: $stderrPath"
} finally {
    Pop-Location
}
