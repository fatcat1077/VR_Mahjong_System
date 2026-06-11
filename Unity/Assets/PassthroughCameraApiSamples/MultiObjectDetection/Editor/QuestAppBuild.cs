using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class QuestAppBuild
{
    private const string DefaultApkPath = "Builds/VRMahjongSystem.apk";

    public static void BuildAndroidApk()
    {
        var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        var apkPath = Environment.GetEnvironmentVariable("QUEST_APK_PATH");
        if (string.IsNullOrWhiteSpace(apkPath))
        {
            apkPath = Path.Combine(projectRoot, DefaultApkPath);
        }
        else if (!Path.IsPathRooted(apkPath))
        {
            apkPath = Path.GetFullPath(Path.Combine(projectRoot, apkPath));
        }

        var sceneOverride = Environment.GetEnvironmentVariable("QUEST_SCENE_PATH");
        var scenes = !string.IsNullOrWhiteSpace(sceneOverride)
            ? new[] { sceneOverride }
            : new[] { "Assets/PassthroughCameraApiSamples/MultiObjectDetection/MultiObjectDetection.unity" };

        if (scenes.Length == 0)
        {
            throw new InvalidOperationException("No enabled scenes found in EditorBuildSettings.");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(apkPath) ?? projectRoot);

        EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);
        EditorUserBuildSettings.buildAppBundle = false;
        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
        PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevel32;

        var ovrConfig = OVRProjectConfig.CachedProjectConfig;
        var originalHandTrackingSupport = ovrConfig.handTrackingSupport;
        var originalHandTrackingFrequency = ovrConfig.handTrackingFrequency;
        var handTrackingSupport = ResolveHandTrackingSupport();

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = apkPath,
            target = BuildTarget.Android,
            targetGroup = BuildTargetGroup.Android,
            options = BuildOptions.None,
        };

        BuildSummary summary = default;
        try
        {
            ovrConfig.handTrackingSupport = handTrackingSupport;
            ovrConfig.handTrackingFrequency = OVRProjectConfig.HandTrackingFrequency.LOW;
            EditorUtility.SetDirty(ovrConfig);
            AssetDatabase.SaveAssets();

            var report = BuildPipeline.BuildPlayer(options);
            summary = report.summary;
            Debug.Log($"[QuestAppBuild] result={summary.result} totalSize={summary.totalSize} output={apkPath}");
        }
        finally
        {
            ovrConfig.handTrackingSupport = originalHandTrackingSupport;
            ovrConfig.handTrackingFrequency = originalHandTrackingFrequency;
            EditorUtility.SetDirty(ovrConfig);
            AssetDatabase.SaveAssets();
        }

        if (summary.result != BuildResult.Succeeded)
        {
            throw new Exception($"Quest Android build failed: {summary.result}");
        }
    }

    private static OVRProjectConfig.HandTrackingSupport ResolveHandTrackingSupport()
    {
        var value = Environment.GetEnvironmentVariable("QUEST_HAND_TRACKING_SUPPORT");
        if (string.IsNullOrWhiteSpace(value))
        {
            return OVRProjectConfig.HandTrackingSupport.HandsOnly;
        }

        if (Enum.TryParse(value, true, out OVRProjectConfig.HandTrackingSupport parsed))
        {
            return parsed;
        }

        throw new ArgumentException(
            "QUEST_HAND_TRACKING_SUPPORT must be ControllersOnly, ControllersAndHands, or HandsOnly.");
    }
}
