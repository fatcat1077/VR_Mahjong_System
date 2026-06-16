$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$PythonCandidates = @(
    (Join-Path $Root "venv\Scripts\python.exe"),
    "C:\Users\user\Desktop\VR_Mahjong_System\server\venv\Scripts\python.exe",
    "python"
)
$Python = $PythonCandidates | Where-Object {
    if ($_ -eq "python") { return $true }
    Test-Path $_
} | Select-Object -First 1
if (-not $Python) {
    throw "Missing Python. Tried: $($PythonCandidates -join ', ')"
}

$SamplesDir = Join-Path $Root "segmentation_samples"
$Labels = Join-Path $Root "mahjong_labels.txt"

& $Python "segmentation_annotation_tool.py" `
    --samples-dir $SamplesDir `
    --labels $Labels `
    --host 127.0.0.1 `
    --port 8765 `
    --open
