#!/usr/bin/env python3
"""Bootstrap the NuGet packages FlyHero needs, without the Unity editor.

Why this exists
---------------
Upstream YARG restores its NuGet packages through the NuGet for Unity GUI
("NuGet > Restore Packages"). That is impossible in batchmode, and calling
NugetForUnity.PackageRestorer.Restore via -executeMethod does not work either:
Unity refuses to run ANY -executeMethod while any script has compile errors,
and the missing packages are exactly what cause those errors. So the very first
restore has to happen from outside the editor.

This script reads Assets/packages.config and installs each package into
Assets/Packages/<Id>.<Version>/lib/<tfm>/, which is the layout NuGet for Unity
uses and which makes the DLLs auto-referenced plugins in Unity's eyes.

It uses the local ~/.nuget/packages cache when a package is already there, and
otherwise downloads the .nupkg from nuget.org.

Once it has succeeded you can use tools/automation/restore_nuget.sh for
subsequent restores (those run inside the editor, correctly).

Usage:
    python3 tools/automation/restore_nuget_python.py            # only what is missing
    python3 tools/automation/restore_nuget_python.py --all
    python3 tools/automation/restore_nuget_python.py --only Melanchall.DryWetMidi.Nativeless
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "Assets" / "packages.config"
OUT_ROOT = REPO / "Assets" / "Packages"
NUGET_CACHE = Path.home() / ".nuget" / "packages"
FLAT_CONTAINER = "https://api.nuget.org/v3-flatcontainer/{id}/{ver}/{id}.{ver}.nupkg"

# Packages the compile errors actually pointed at. Fast path: install just these
# first, before touching anything that might collide with a vendored DLL.
KNOWN_MISSING = [
    "Melanchall.DryWetMidi.Nativeless",
    "ZString",
    "System.Runtime.CompilerServices.Unsafe",
    "Microsoft.VisualStudio.SolutionPersistence",
    "Microsoft.IO.Redist",
]

# Unity 6000.3 runs Mono. Prefer netstandard; .NET Framework assemblies (net472
# and friends) are usable, but netN.N ones are NOT - they demand a newer
# System.Runtime than Mono provides, which surfaces as:
#   error CS1705: ... uses 'System.Runtime, Version=8.0.0.0' which has a higher
#   version than referenced assembly 'System.Runtime, Version=4.1.2.0'
# so net6.0/net8.0 must rank LAST, below net472.
TFM_PREFERENCE = [
    "netstandard2.1",
    "netstandard2.0",
    "netstandard1.6",
    "netstandard1.5",
    "netstandard1.4",
    "netstandard1.3",
    "netstandard1.2",
    "netstandard1.1",
    "netstandard1.0",
    "net472",
    "net471",
    "net47",
    "net462",
    "net461",
    "net46",
    "net45",
    "net40",
    "net35",
]


def parse_packages_config(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    out = []
    for m in re.finditer(r'<package\s+id="([^"]+)"\s+version="([^"]+)"', text):
        out.append((m.group(1), m.group(2)))
    return out


def rank_tfm(tfm: str) -> int:
    """Lower is better. Handles netstandard + netNN + net4NN."""
    t = tfm.lower()
    if t in TFM_PREFERENCE:
        return TFM_PREFERENCE.index(t)
    if t.startswith("netstandard"):
        # Unknown but still netstandard: rank just after the known ones.
        return len(TFM_PREFERENCE)
    if re.fullmatch(r"net\d{2,3}", t):  # net45, net461 ...
        return len(TFM_PREFERENCE) + 100
    if re.fullmatch(r"net\d+\.\d+", t):  # net6.0/net8.0 - Mono cannot load these
        return len(TFM_PREFERENCE) + 1000
    return len(TFM_PREFERENCE) + 2000


def fetch_nupkg(pkg_id: str, version: str) -> bytes | None:
    """Return the .nupkg bytes, from the local cache if possible."""
    cached = NUGET_CACHE / pkg_id.lower() / version / f"{pkg_id.lower()}.{version}.nupkg"
    if cached.is_file():
        print(f"    using cache {cached}")
        return cached.read_bytes()

    url = FLAT_CONTAINER.format(id=pkg_id.lower(), ver=version.lower())
    try:
        print(f"    downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        print(f"    !! HTTP {e.code} for {url}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"    !! {e}", file=sys.stderr)
    return None


def install(pkg_id: str, version: str, force: bool = False) -> bool:
    dest = OUT_ROOT / f"{pkg_id}.{version}"
    if dest.is_dir() and any(dest.rglob("*.dll")) and not force:
        print(f"  {pkg_id} {version}: already present")
        return True

    data = fetch_nupkg(pkg_id, version)
    if data is None:
        print(f"  {pkg_id} {version}: FAILED (no package)", file=sys.stderr)
        return False

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = z.namelist()

        # Group lib/<tfm>/*.dll and pick the best tfm.
        by_tfm: dict[str, list[str]] = {}
        for name in members:
            parts = name.split("/")
            if len(parts) >= 3 and parts[0] == "lib" and name.endswith(".dll"):
                by_tfm.setdefault(parts[1], []).append(name)

        if not by_tfm:
            print(f"  {pkg_id} {version}: no lib/*.dll (meta or build-only package)")
            return True

        tfm = sorted(by_tfm, key=rank_tfm)[0]
        dest_lib = dest / "lib" / tfm
        dest_lib.mkdir(parents=True, exist_ok=True)

        for name in by_tfm[tfm]:
            with z.open(name) as src:
                (dest_lib / Path(name).name).write_bytes(src.read())

        # Keep the nupkg next to the DLLs, as NuGet for Unity does.
        (dest / f"{pkg_id}.{version}.nupkg").write_bytes(data)

    print(f"  {pkg_id} {version}: installed {tfm} -> {dest.relative_to(REPO)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="install every package in packages.config")
    ap.add_argument("--only", action="append", default=[], help="install just this package id")
    ap.add_argument("--force", action="store_true", help="reinstall even if present")
    args = ap.parse_args()

    available = parse_packages_config(CONFIG)
    print(f"packages.config: {len(available)} entries")

    if args.only:
        wanted = [(i, v) for i, v in available if i in set(args.only)]
    elif args.all:
        wanted = available
    else:
        wanted = [(i, v) for i, v in available if i in set(KNOWN_MISSING)]

    if not wanted:
        print("nothing to do")
        return 0

    failures = []
    for pkg_id, version in wanted:
        if not install(pkg_id, version, force=args.force):
            failures.append(pkg_id)

    print()
    print(f"installed into {OUT_ROOT.relative_to(REPO)}:")
    for child in sorted(OUT_ROOT.iterdir()) if OUT_ROOT.is_dir() else []:
        if child.is_dir():
            dlls = list(child.rglob("*.dll"))
            print(f"  {child.name:<55} {len(dlls)} dll(s)")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
