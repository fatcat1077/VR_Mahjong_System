$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing server venv Python: $Python"
}

$ModelRoot = "D:\Download\final_models\final_models"
$Yolo = Join-Path $ModelRoot "best_segmentation.pt"
$Cls = Join-Path $ModelRoot "best_classification.pt"
$Ppo = Join-Path $ModelRoot "masked_cont100m_to150m_seed42_plus25m.zip"
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
    --det-imgsz 960 `
    --cls-imgsz 128 `
    --cls-conf 0.5 `
    --debug-vision `
    --debug-dir $DebugDir `
    --debug-interval 1
