#!/usr/bin/env bash
# Headless build of FlyHero (Unity 6000.3.5f2).
#
# Requires an activated Unity license on this machine. If Unity reports
# "No valid Unity Editor license found", activate a license first - no build
# can proceed without one.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/liamb/projects/FlyHero}"
UNITY_BIN="${UNITY_BIN:-/home/liamb/unity/6000.3.5f2/Editor/Unity}"
UNITY_LIBFIX="${UNITY_LIBFIX:-/home/liamb/unity/libfix/usr/lib/x86_64-linux-gnu}"
OUT_DIR="${OUT_DIR:-/home/liamb/projects/FlyHero-build/build}"
LOG_DIR="${LOG_DIR:-/home/liamb/projects/FlyHero-build/logs}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/build-$(date +%Y%m%d-%H%M%S).log"

if [[ ! -x "$UNITY_BIN" ]]; then
  echo "Unity editor not found at $UNITY_BIN" >&2
  exit 2
fi

# Unity 6000.3.5f2 expects older libxml2 and ICU than this distribution ships.
if [[ -d "$UNITY_LIBFIX" ]]; then
  export LD_LIBRARY_PATH="$UNITY_LIBFIX:${LD_LIBRARY_PATH:-}"
fi

echo "Building Linux64 player -> $OUT_DIR"
echo "Unity log: $LOG_FILE"

set +e
"$UNITY_BIN" \
  -batchmode \
  -nographics \
  -quit \
  -silent-crashes \
  -projectPath "$REPO_DIR" \
  -executeMethod Editor.AutomationBuild.BuildLinux64 \
  -logFile "$LOG_FILE"
STATUS=$?
set -e

if [[ $STATUS -ne 0 ]]; then
  echo "Build failed (exit $STATUS). Last lines of the Unity log:" >&2
  tail -40 "$LOG_FILE" >&2
  exit $STATUS
fi

echo "Build finished. Output in $OUT_DIR"
ls -la "$OUT_DIR"
