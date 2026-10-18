$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Missing virtual environment. Run .\start.ps1 once first."
}

New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "logs") | Out-Null
& $python -m knowledge_brain.sync *>> (Join-Path $projectRoot "logs\daily-sync.log")
