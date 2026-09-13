#!/usr/bin/env bash
# Record one song's playthrough with game audio, for the vision dataset.
#
# Why audio: to pair a video frame with the notes that were on screen we need the
# song time at that frame. The game's on-screen timer is 1-second resolution, and
# wall-clock timing carries the load latency, so neither is precise enough. But the
# recording contains the game's own audio, and we have the source song.ogg - so
# cross-correlating the two gives the song start to well under a frame.
#
# Output: capture/song.mp4 (video + audio)

set -euo pipefail

OUT="${OUT:-/home/liamb/projects/FlyHero-build/capture/song.mp4}"
SONG="${SONG:-FlyHero Test Two}"
DURATION="${DURATION:-70}"
DISPLAY_NUM="${DISPLAY_NUM:-:1}"

mkdir -p "$(dirname "$OUT")"
cd "$(dirname "$0")/../.."

export DISPLAY="$DISPLAY_NUM"

echo "recording -> $OUT  (song: $SONG, ${DURATION}s)"
ffmpeg -hide_banner -loglevel warning -y \
  -f x11grab -framerate 30 -video_size 1024x768 -i "${DISPLAY_NUM}.0" \
  -f pulse -i FakeAudio.monitor \
  -t "$DURATION" \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 192k \
  "$OUT" &
REC_PID=$!

sleep 2
cd tools
python3 -m flyhero.cli play "$SONG" --instrument guitar --difficulty easy --timeout "$((DURATION - 10))" \
  > /tmp/vision_play.txt 2>&1 || true
cd ..

# Let the tail of the song land in the recording before stopping.
wait "$REC_PID" || true

echo
echo "=== capture ==="
ls -la "$OUT"
ffprobe -v error -show_entries format=duration -show_entries stream=codec_type,codec_name \
  -of default=nw=1 "$OUT" 2>/dev/null | head -8
echo
echo "=== play result ==="
tail -6 /tmp/vision_play.txt
