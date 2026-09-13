#!/usr/bin/env bash
# Launches a built FlyHero in unattended automation mode, optionally recording the
# session to a video file.
#
# Usage:
#   run_queue.sh --game /path/to/FlyHero --queue queue.json [options]
#
# Options:
#   --game PATH      Path to the built game executable (required)
#   --queue PATH     Queue JSON (required)
#   --songs DIR      Folder holding the queue's songs (optional if queue sets songDir)
#   --data DIR       Where to write score_log.jsonl / top_scores.json / run_summary.txt
#   --record FILE    Record the session to FILE (mp4). Needs a display to capture.
#   --display :N     X display to run on and capture (default :1)
#   --speed N        Song speed percent (default 100)
#   --instrument X   Instrument (default FiveFretGuitar)
#   --difficulty X   Difficulty (default Expert)
#   --repeat         Loop the queue forever instead of exiting when it finishes
#   --stay-open      Return to the menu when done instead of quitting
#
# Exits with the game's exit code.

set -euo pipefail

# Unity 6000.3.5f2 on this host needs an older libxml2 (so.2) and ICU 74 than the
# system provides; both are unpacked locally here. Harmless if already present.
UNITY_LIBFIX="${UNITY_LIBFIX:-/home/liamb/unity/libfix/usr/lib/x86_64-linux-gnu}"

GAME=""
QUEUE=""
SONGS=""
DATA=""
RECORD=""
DISPLAY_ID="${DISPLAY:-:1}"
SPEED="100"
INSTRUMENT="FiveFretGuitar"
DIFFICULTY="Expert"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --game)       GAME="$2"; shift 2 ;;
    --queue)      QUEUE="$2"; shift 2 ;;
    --songs)      SONGS="$2"; shift 2 ;;
    --data)       DATA="$2"; shift 2 ;;
    --record)     RECORD="$2"; shift 2 ;;
    --display)    DISPLAY_ID="$2"; shift 2 ;;
    --speed)      SPEED="$2"; shift 2 ;;
    --instrument) INSTRUMENT="$2"; shift 2 ;;
    --difficulty) DIFFICULTY="$2"; shift 2 ;;
    --repeat)     EXTRA+=("-autorepeat"); shift ;;
    --stay-open)  EXTRA+=("-autostayopen"); shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$GAME" || -z "$QUEUE" ]]; then
  echo "--game and --queue are required" >&2
  exit 2
fi

if [[ ! -x "$GAME" ]]; then
  echo "Game executable not found or not executable: $GAME" >&2
  exit 2
fi

if [[ -d "$UNITY_LIBFIX" ]]; then
  export LD_LIBRARY_PATH="$UNITY_LIBFIX:${LD_LIBRARY_PATH:-}"
fi

export DISPLAY="$DISPLAY_ID"

LOG_DIR="$(dirname "$(readlink -f "$QUEUE")")/run-logs"
mkdir -p "$LOG_DIR"
GAME_LOG="$LOG_DIR/game-$(date +%Y%m%d-%H%M%S).log"

ARGS=(
  -batchmode
  -nographics
  -offline
  -autoqueue "$QUEUE"
  -autospeed "$SPEED"
  -autoinstrument "$INSTRUMENT"
  -autodifficulty "$DIFFICULTY"
)

[[ -n "$SONGS" ]] && ARGS+=(-autosongdir "$SONGS")
[[ -n "$DATA"  ]] && ARGS+=(-autodata "$DATA")

# batchmode/nographics are right for headless scoring runs but must be dropped when
# recording, since we need a real window to capture.
if [[ -n "$RECORD" ]]; then
  ARGS=("${ARGS[@]/#-batchmode/}")
  ARGS=("${ARGS[@]/#-nographics/}")
  ARGS=(${ARGS[@]})
fi

ARGS+=("${EXTRA[@]:-}")

echo "Launching: $GAME ${ARGS[*]}"
echo "Game log:  $GAME_LOG"

"$GAME" "${ARGS[@]}" -logFile "$GAME_LOG" &
GAME_PID=$!

if [[ -n "$RECORD" ]]; then
  echo "Recording display $DISPLAY_ID to $RECORD"
  # Wait for the window to actually exist before capturing.
  for _ in $(seq 1 60); do
    if ffprobe -v error -f x11grab -video_size 1280x720 -i "$DISPLAY_ID" -t 0.1 - 2>/dev/null; then
      break
    fi
    sleep 1
  done

  ffmpeg -hide_banner -loglevel warning -y \
    -f x11grab -framerate 60 -video_size 1920x1080 -i "$DISPLAY_ID" \
    -f pulse -i default \
    -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
    -c:a aac -b:a 160k \
    -movflags +faststart \
    "$RECORD" &
  REC_PID=$!

  wait "$GAME_PID"
  GAME_STATUS=$?

  sleep 2
  kill -INT "$REC_PID" 2>/dev/null || true
  wait "$REC_PID" 2>/dev/null || true
  echo "Recording written to $RECORD"
else
  wait "$GAME_PID"
  GAME_STATUS=$?
fi

echo "Game exited with status $GAME_STATUS"
exit "$GAME_STATUS"
