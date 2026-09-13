using System;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using YARG.Core.Logging;

namespace Editor
{
    /// <summary>
    /// Headless build entry point for the automation workflow.
    ///
    /// The interactive build menu (<see cref="MakeTestBuild"/>) asks for options through
    /// a window, which cannot run under -batchmode. This class builds the same player
    /// with options supplied up front instead.
    ///
    /// Used by tools/automation/build_headless.sh:
    ///   Unity -batchmode -nographics -quit -projectPath &lt;repo&gt; \
    ///         -executeMethod Editor.AutomationBuild.BuildLinux64
    ///
    /// Output directory can be overridden with the FLYHERO_BUILD_OUTPUT environment
    /// variable; it defaults to &lt;repo&gt;/../FlyHero-build/build/Linux64.
    /// </summary>
    public static class AutomationBuild
    {
        private const string DefaultOutputFolder = "FlyHero-build/build/Linux64";

        public static void BuildLinux64()
        {
            var scenes = EditorBuildSettings.scenes
                .Where(s => s.enabled)
                .Select(s => s.path)
                .ToArray();

            if (scenes.Length == 0)
            {
                throw new InvalidOperationException(
                    "No enabled scenes in EditorBuildSettings; refusing to build an empty player.");
            }

            Debug.Log($"Automation build: {scenes.Length} scene(s)");

            var output = ResolveOutputPath();

            var options = new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = output,
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None
            };

            var report = BuildPipeline.BuildPlayer(options);
            var summary = report.summary;

            Debug.Log($"Automation build result: {summary.result}, " +
                      $"size {summary.totalSize} bytes, {summary.totalTime}");

            if (summary.result != BuildResult.Succeeded)
            {
                // A non-zero exit from the editor is what the wrapper script checks.
                throw new Exception($"Build failed: {summary.result}");
            }
        }

        private static string ResolveOutputPath()
        {
            var overrideFolder = Environment.GetEnvironmentVariable("FLYHERO_BUILD_OUTPUT");

            string folder;
            if (!string.IsNullOrWhiteSpace(overrideFolder))
            {
                folder = overrideFolder;
            }
            else
            {
                // Application.dataPath is <repo>/Assets
                var repoRoot = System.IO.Directory.GetParent(Application.dataPath)!.FullName;
                var buildRoot = System.IO.Directory.GetParent(repoRoot)!.FullName;
                folder = System.IO.Path.Combine(buildRoot, DefaultOutputFolder);
            }

            System.IO.Directory.CreateDirectory(folder);

            // Unity appends the extension for us on Linux.
            return System.IO.Path.Combine(folder, "FlyHero");
        }
    }
}
