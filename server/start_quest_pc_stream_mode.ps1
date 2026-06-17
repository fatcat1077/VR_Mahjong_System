param(
    [string]$Adb = "",
    [string]$PackageName = "com.samples.passthroughcamera",
    [string]$ActivityName = "com.unity3d.player.UnityPlayerActivity",
    [int]$Port = 5000,
    [int]$AutoLatencySeconds = 0
)

$ErrorActionPreference = "Stop"

function Resolve-AdbPath {
    param([string]$RequestedPath)

    if ($RequestedPath -and (Test-Path $RequestedPath)) {
        return (Resolve-Path $RequestedPath).Path
    }

    $fromPath = Get-Command adb -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $candidates = @(
        "C:\Program Files\Unity\Hub\Editor\2022.3.52f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe",
        "C:\Program Files\Unity\Hub\Editor\2022.3.21f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe",
        "C:\Program Files (x86)\Android\android-sdk\platform-tools\adb.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "adb.exe not found. Pass -Adb with the full adb.exe path."
}

$AdbPath = Resolve-AdbPath $Adb

Write-Host "[Quest] adb: $AdbPath"
& $AdbPath devices

Write-Host "[Quest] adb reverse tcp:$Port -> tcp:$Port"
& $AdbPath reverse "tcp:$Port" "tcp:$Port"

Write-Host "[Quest] launching $PackageName/$ActivityName in pc_stream mode"
& $AdbPath shell am force-stop $PackageName
& $AdbPath shell input keyevent 224
$StartArgs = @("shell", "am", "start", "-n", "$PackageName/$ActivityName", "--es", "seg_latency_mode", "pc_stream")
if ($AutoLatencySeconds -gt 0) {
    $StartArgs += @("--ei", "latency_auto_duration_sec", "$AutoLatencySeconds")
}
& $AdbPath @StartArgs
