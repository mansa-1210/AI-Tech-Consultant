$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$port = 8501

$connection = New-Object System.Net.Sockets.TcpClient
try {
    $connection.Connect("127.0.0.1", $port)
    Write-Host "AI Tech Consultant is already running at http://127.0.0.1:$port/."
    exit 0
} catch {
} finally {
    $connection.Dispose()
}

if (-not (Test-Path $python)) {
    py -3 -m venv .venv
}

& $python -m pip install -r requirements.txt
& $python -m app --port $port
