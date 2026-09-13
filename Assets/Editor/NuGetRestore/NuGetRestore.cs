using UnityEditor;

namespace FlyHero.EditorTools
{
    /// <summary>
    /// Headless equivalent of the "NuGet/Restore Packages" menu item.
    ///
    /// Upstream YARG expects this to be clicked in the editor GUI, which is not
    /// possible in batchmode CI. This wrapper exposes it as a callable
    /// -executeMethod target so a build script can restore first.
    ///
    /// It deliberately lives in its own assembly (FlyHero.NuGetRestore) that
    /// does NOT reference YARG.Core. When the NuGet packages are missing,
    /// YARG.Core fails to compile, and a compile failure blocks Unity from
    /// running any -executeMethod at all. Keeping this isolated means the
    /// restore can still run and fix the very problem that broke compilation.
    /// </summary>
    public static class NuGetRestore
    {
        /// <summary>
        /// Restore every package listed in Assets/packages.config.
        /// slimRestore=false so dependencies are installed too.
        /// </summary>
        public static void RestoreAll()
        {
            NugetForUnity.PackageRestorer.Restore(false);
        }
    }
}
