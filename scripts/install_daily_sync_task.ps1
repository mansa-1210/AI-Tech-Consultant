param(
    [string]$TaskName = "AI Tech Consultant Daily Knowledge Sync",
    [string]$Time = "02:00"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runner = Join-Path $projectRoot "scripts\run_daily_sync.ps1"

if (-not (Test-Path $runner)) {
    throw "Daily sync runner not found: $runner"
}

$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runner`""
schtasks.exe /Create /TN $TaskName /TR $taskCommand /SC DAILY /ST $Time /F
