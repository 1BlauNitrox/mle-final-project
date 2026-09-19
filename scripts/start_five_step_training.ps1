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
$pidFile = Join-Path $rootPath 'supervisor.json'
if (Test-Path -LiteralPath $pidFile) {
    $old = Get-Content -LiteralPath $pidFile | ConvertFrom-Json
    if (Get-Process -Id $old.pid -ErrorAction SilentlyContinue) {
        throw "Supervisor PID $($old.pid) is already running"
    }
}
$script = Join-Path $PSScriptRoot 'run_five_step_experiment.py'
$stdout = Join-Path $rootPath 'supervisor.stdout.log'
$stderr = Join-Path $rootPath 'supervisor.stderr.log'
$arguments = @(
    $script, 'train', '--root', $rootPath, '--workers', $Workers
)
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory (Split-Path $PSScriptRoot) `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -UseNewEnvironment -PassThru
@{
    pid = $process.Id
    started_at = (Get-Date).ToString('o')
    python = $Python
    arguments = $arguments
    stdout = $stdout
    stderr = $stderr
    resume_command = ".\scripts\start_five_step_training.ps1 -Root '$rootPath' -Workers $Workers"
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $pidFile -Encoding utf8
Write-Host "Detached supervisor PID $($process.Id)"
Write-Host "Status: & '$Python' '$script' status --root '$rootPath'"
Write-Host "Resume after interruption: .\scripts\start_five_step_training.ps1 -Root '$rootPath' -Workers $Workers"
