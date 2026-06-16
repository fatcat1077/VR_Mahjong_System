$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$PythonCandidates = @(
    (Join-Path $Root "venv\Scripts\python.exe"),
    "C:\Users\user\Desktop\VR_Mahjong_System\server\venv\Scripts\python.exe"
)
$Python = $PythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Python) {
    throw "Missing server venv Python. Tried: $($PythonCandidates -join ', ')"
}

$ModelRoot = "D:\Download\final_models\final_models"
$Yolo = Join-Path $ModelRoot "best_segmentation.pt"
$Cls = Join-Path $ModelRoot "best_classification.pt"
$DebugDir = Join-Path $Root "debug_runtime"
$CaptureDir = Join-Path $Root "segmentation_samples"

foreach ($Path in @($Yolo, $Cls)) {
    if (-not (Test-Path $Path)) {
        throw "Missing required model file: $Path"
    }
}

& $Python "server_A\main.py" `
    --labeling `
    --yolo $Yolo `
    --cls $Cls `
    --host 0.0.0.0 `
    --port 5000 `
    --client-idle-timeout 5 `
    --print-interval 1 `
    --det-imgsz 960 `
    --cls-imgsz 128 `
    --cls-conf 0.5 `
    --debug-vision `
    --debug-dir $DebugDir `
    --debug-interval 1 `
    --capture-dir $CaptureDir
