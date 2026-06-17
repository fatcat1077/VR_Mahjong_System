param(
    [int]$Port = 5000,
    [int]$ClientIdleTimeout = 60,
    [switch]$KeepRunningAfterStop
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing server venv Python: $Python"
}

$env:PYTHONDONTWRITEBYTECODE = "1"

$ArgsList = @(
    "server_A\quest_seg_latency_telemetry_server.py",
    "--output-dir", (Join-Path $Root "seg_latency_results"),
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--client-idle-timeout", "$ClientIdleTimeout",
    "--print-interval", "1"
)

if (-not $KeepRunningAfterStop) {
    $ArgsList += "--exit-after-stop"
}

& $Python @ArgsList
