$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing server venv Python: $Python"
}

$Yolo = Join-Path $Root "models\segmentation.pt"
$Cls = Join-Path $Root "models\classify_v2.pt"
$Ppo = Join-Path $Root "server_A\models_from_agent\tw16mj_ppo_hand34_claim.zip"
$DebugDir = Join-Path $Root "debug_runtime"

foreach ($Path in @($Yolo, $Cls, $Ppo)) {
    if (-not (Test-Path $Path)) {
        throw "Missing required model file: $Path"
    }
}

& $Python "server_A\main.py" `
    --yolo $Yolo `
    --cls $Cls `
    --ppo-model $Ppo `
    --ppo-device cpu `
    --host 0.0.0.0 `
    --port 5000 `
    --client-idle-timeout 5 `
    --print-interval 1 `
    --det-imgsz 640 `
    --cls-imgsz 96 `
    --debug-vision `
    --debug-dir $DebugDir `
    --debug-interval 1
