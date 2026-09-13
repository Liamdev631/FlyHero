#!/usr/bin/env bash
# Headless NuGet restore for FlyHero.
#
# Upstream YARG does this by hand: "Click on NuGet on the top menu bar, then
# click on Restore Packages" (README). That is a GUI step, so this script calls
# the same code path in batchmode via FlyHero.EditorTools.NuGetRestore.
#
# Run this BEFORE build_headless.sh, and again after adding a package to
# Assets/packages.config. If YARG.Core fails to compile with
# 'error CS0246: The type or namespace name ... could not be found'
# (Melanchall, Cysharp, Microsoft.VisualStudio, ZString...), the NuGet packages
# are missing and this is the fix.
#
# Note: the restore target lives in its own assembly that does not reference
# YARG.Core, precisely so it still runs while those compile errors are present.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/liamb/projects/FlyHero}"
UNITY_BIN="${UNITY_BIN:-/home/liamb/unity/6000.3.5f2/Editor/Unity}"
UNITY_LIBFIX="${UNITY_LIBFIX:-/home/liamb/unity/libfix/usr/lib/x86_64-linux-gnu}"
LOG_DIR="${LOG_DIR:-/home/liamb/projects/FlyHero-build/logs}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nuget-restore-$(date +%Y%m%d-%H%M%S).log"

if [[ ! -x "$UNITY_BIN" ]]; then
  echo "Unity editor not found at $UNITY_BIN" >&2
  exit 2
fi

if [[ -d "$UNITY_LIBFIX" ]]; then
  export LD_LIBRARY_PATH="$UNITY_LIBFIX:${LD_LIBRARY_PATH:-}"
fi

echo "Restoring NuGet packages for $REPO_DIR"
echo "Unity log: $LOG_FILE"

set +e
"$UNITY_BIN" \
  -batchmode \
  -nographics \
  -quit \
  -silent-crashes \
  -projectPath "$REPO_DIR" \
  -executeMethod FlyHero.EditorTools.NuGetRestore.RestoreAll \
  -logFile "$LOG_FILE"
STATUS=$?
set -e

# Unity exits 0 even when the method throws, so check the log too.
if grep -qE "Restoring [0-9]+ packages|Successfully restored|Restored .* package" "$LOG_FILE" 2>/dev/null; then
  grep -oE "Restoring [0-9]+ packages|Restored [^<]*" "$LOG_FILE" | tail -5
else
  echo "No restore activity found in the log. Tail:" >&2
  sed 's/<[^>]*>//g' "$LOG_FILE" | tail -25 >&2
fi

if grep -qE "error CS[0-9]+" "$LOG_FILE" 2>/dev/null; then
  echo
  echo "WARNING: compile errors present during restore:" >&2
  grep -oE "error CS[0-9]+: [^<]*" "$LOG_FILE" | sort -u | head -10 >&2
fi

echo
echo "Installed NuGet packages:"
ls -1 "$REPO_DIR/Assets/Packages" 2>/dev/null | head -25 || echo "  (none yet)"

exit $STATUS
