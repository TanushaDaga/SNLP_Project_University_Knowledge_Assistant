# Starts the API (port 8000) and the REST demo page (port 7861), each in its own window.
# Any old copy still holding those ports is stopped first.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Original     # also starts the original UI (port 7860)
#
# Close the two windows (or press Ctrl+C in them) to stop everything.

param([switch]$Original)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Free-Port($port) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Write-Host "Stopping old process on port $port (PID $($c.OwningProcess))"
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

function Start-Window($title, $command) {
    $full = "`$Host.UI.RawUI.WindowTitle = '$title'; Set-Location '$root'; $command"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $full
}

Free-Port 8000
Free-Port 7861
if ($Original) { Free-Port 7860 }

Write-Host "Starting API on http://127.0.0.1:8000 ..."
Start-Window "Knowledge Assistant API" "uvicorn src.api.app:app --host 127.0.0.1 --port 8000"

$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/ready" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Host "The API did not become ready in 2 minutes. Check the API window for errors."
    exit 1
}
Write-Host "API is ready."

Write-Host "Starting REST demo page on http://127.0.0.1:7861 ..."
Start-Window "Knowledge Assistant REST demo" "python src/rest_demo_ui.py"

if ($Original) {
    Write-Host "Starting original UI on http://127.0.0.1:7860 ..."
    Start-Window "Knowledge Assistant original UI" "python src/app.py"
}

Start-Sleep -Seconds 8
Start-Process "http://127.0.0.1:7861"
Write-Host "Done. Close the new windows to stop the servers."
