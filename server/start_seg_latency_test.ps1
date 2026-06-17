param(
    [string]$ModelsZip = "D:\Download\latency_5_seg_models.zip",
    [string]$ModelsDir = "",
    [int]$MaxFrames = 0,
    [string]$Device = "cuda:0",
    [int]$Port = 5000,
    [int]$Warmup = 5,
    [int]$DurationSec = 0,
    [int]$ClientIdleTimeout = 30,
    [switch]$KeepRunningAfterStop,
    [switch]$Half,
    [switch]$RotateOrder
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing server venv Python: $Python"
}

if ([string]::IsNullOrWhiteSpace($ModelsDir) -and -not (Test-Path $ModelsZip)) {
    throw "Missing models zip: $ModelsZip"
}

if (-not [string]::IsNullOrWhiteSpace($ModelsDir) -and -not (Test-Path $ModelsDir)) {
    throw "Missing models directory: $ModelsDir"
}

$env:PYTHONDONTWRITEBYTECODE = "1"

$ArgsList = @(
    "server_A\seg_latency_server.py",
    "--output-dir", (Join-Path $Root "seg_latency_results"),
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--device", $Device,
    "--warmup", "$Warmup",
    "--max-frames", "$MaxFrames",
    "--duration-sec", "$DurationSec",
    "--client-idle-timeout", "$ClientIdleTimeout",
    "--print-interval", "1"
)

if (-not [string]::IsNullOrWhiteSpace($ModelsDir)) {
    $ArgsList += @("--models-dir", $ModelsDir)
}
else {
    $ArgsList += @("--models-zip", $ModelsZip)
}

if ($Half) {
    $ArgsList += "--half"
}

if ($RotateOrder) {
    $ArgsList += "--rotate-order"
}

if (-not $KeepRunningAfterStop) {
    $ArgsList += "--exit-after-stop"
}

& $Python @ArgsList
