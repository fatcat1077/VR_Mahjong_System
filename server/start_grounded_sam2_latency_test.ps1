param(
    [string]$ModelRoot = "",
    [int]$MaxFrames = 0,
    [string]$Device = "cuda:0",
    [int]$Port = 5000,
    [int]$Warmup = 1,
    [int]$DurationSec = 0,
    [int]$ClientIdleTimeout = 90,
    [string]$TextLabels = "mahjong tile,table",
    [string]$AmpDtype = "bfloat16",
    [switch]$KeepRunningAfterStop
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing server venv Python: $Python"
}

if ([string]::IsNullOrWhiteSpace($ModelRoot)) {
    $ModelRoot = Join-Path $Root "models\GroundedSAM2"
}

if (-not (Test-Path $ModelRoot)) {
    throw "Missing Grounded-SAM2 model root: $ModelRoot"
}

$env:PYTHONDONTWRITEBYTECODE = "1"

$ArgsList = @(
    "server_A\grounded_sam2_latency_server.py",
    "--output-dir", (Join-Path $Root "seg_latency_results"),
    "--model-root", $ModelRoot,
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--device", $Device,
    "--warmup", "$Warmup",
    "--max-frames", "$MaxFrames",
    "--duration-sec", "$DurationSec",
    "--client-idle-timeout", "$ClientIdleTimeout",
    "--print-interval", "1",
    "--text-labels", $TextLabels,
    "--amp-dtype", $AmpDtype
)

if (-not $KeepRunningAfterStop) {
    $ArgsList += "--exit-after-stop"
}

& $Python @ArgsList
