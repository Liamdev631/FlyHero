#!/usr/bin/env bash
# Launches the prebuilt YARG on this headless host and records gameplay.
#
# This host has no usable audio hardware and no accessible MIDI sequencer, which
# crashes stock YARG. Three workarounds are applied:
#
#   1. ALSA software device  - all /dev/snd nodes are root:audio and the user is not
#                              in the audio group, so BASS cannot find a PCM and fails
#                              to initialise. ~/.asoundrc defines a `null` PCM that
#                              alsa-lib provides directly, giving BASS a valid device.
#   2. MIDI backend removed  - YARG's Minis MIDI backend calls rtmidi_get_port_count,
#                              which segfaults when /dev/snd/seq is denied. Removing
#                              libRtMidi.so turns that hard crash into a caught
#                              DllNotFoundException (logged once per frame, harmless).
#   3. Large virtual screen  - the game window is placed on the existing Xvfb :1.
#
# Usage:
#   run_prebuilt.sh [--record FILE] [--seconds N] [--data DIR] [--no-record]

set -euo pipefail

BASE=/home/liamb/projects/FlyHero-build
GAME="$BASE/prebuilt/YARG"
BACKUP="$BASE/plugin-backup"
DISPLAY_ID="${DISPLAY:-:1}"
DATA="$BASE/yargdata"
RECORD=""
SECONDS_TO_RUN=0
RES_W=1280
RES_H=720

while [[ $# -gt 0 ]]; do
  case "$1" in
    --record)  RECORD="$2"; shift 2 ;;
    --seconds) SECONDS_TO_RUN="$2"; shift 2 ;;
    --data)    DATA="$2"; shift 2 ;;
    --no-record) RECORD=""; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$GAME" ]]; then
  echo "Game not found: $GAME" >&2
  exit 2
fi

# Workaround 1: software ALSA device.
if ! grep -q 'type null' "$HOME/.asoundrc" 2>/dev/null; then
  cat > "$HOME/.asoundrc" <<'EOF'
# No usable hardware audio on this host; provide a software null PCM so audio
# clients that only need a valid device (BASS inside YARG) can initialise.
pcm.!default {
    type null
}
EOF
  echo "Wrote ~/.asoundrc (software ALSA device)"
fi

# Workaround 2: keep the crashing MIDI backend out of the plugin directory.
if [[ -f "$GAME_Data/Plugins/libRtMidi.so" ]]; then
  mv "$GAME_Data/Plugins/libRtMidi.so" "$BACKUP/libRtMidi.so"
fi
if [[ -f "$(dirname "$GAME")/YARG_Data/Plugins/libRtMidi.so" ]]; then
  mv "$(dirname "$GAME")/YARG_Data/Plugins/libRtMidi.so" "$BACKUP/libRtMidi.so"
fi

export DISPLAY="$DISPLAY_ID"

mkdir -p "$DATA/Setlists"
LOG="$BASE/logs/yarg-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$BASE/logs"

ARGS=(
  -screen-fullscreen 0
  -screen-width "$RES_W"
  -screen-height "$RES_H"
  -offline
  -download-location "$DATA"
  -logFile "$LOG"
)

echo "Launching $GAME on $DISPLAY_ID"
"$GAME" "${ARGS[@]}" &
GAME_PID=$!

if [[ -n "$RECORD" ]]; then
  # Let the window appear before capturing.
  sleep 12
  echo "Recording $DISPLAY_ID at ${RES_W}x${RES_H} -> $RECORD"
  ffmpeg -hide_banner -loglevel warning -y \
    -f x11grab -framerate 30 -video_size "${RES_W}x${RES_H}" -i "$DISPLAY_ID.0" \
    -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p \
    -movflags +faststart \
    "$RECORD" &
  REC_PID=$!
fi

if [[ "$SECONDS_TO_RUN" -gt 0 ]]; then
  sleep "$SECONDS_TO_RUN"
  kill -INT "$GAME_PID" 2>/dev/null || true
fi

wait "$GAME_PID" 2>/dev/null || true
STATUS=$?

if [[ -n "$RECORD" ]]; then
  sleep 2
  kill -INT "${REC_PID:-0}" 2>/dev/null || true
  wait "${REC_PID:-0}" 2>/dev/null || true
  echo "Recording: $RECORD"
fi

echo "Game exited with status $STATUS (log: $LOG)"
exit 0
