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

        var scenes = EditorBuildSettings.scenes
            .Where(scene => scene.enabled)
            .Select(scene => scene.path)
            .ToArray();

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
        ovrConfig.handTrackingSupport = OVRProjectConfig.HandTrackingSupport.ControllersAndHands;
        ovrConfig.handTrackingFrequency = OVRProjectConfig.HandTrackingFrequency.LOW;
        EditorUtility.SetDirty(ovrConfig);
        AssetDatabase.SaveAssets();

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = apkPath,
            target = BuildTarget.Android,
            targetGroup = BuildTargetGroup.Android,
            options = BuildOptions.None,
        };

        var report = BuildPipeline.BuildPlayer(options);
        var summary = report.summary;
        Debug.Log($"[QuestAppBuild] result={summary.result} totalSize={summary.totalSize} output={apkPath}");

        if (summary.result != BuildResult.Succeeded)
        {
            throw new Exception($"Quest Android build failed: {summary.result}");
        }
    }
}
