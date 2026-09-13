#!/usr/bin/env bash
# Replace the built player's libRtMidi.so with the zero-port stub.
#
# Why: Minis' MIDI backend calls rtmidi_get_port_count on every input update.
# On a headless host with no usable /dev/snd/seq the real library crashes the
# whole player with SIGSEGV (mono stack: Minis.Native.RtMidi:rtmidi_get_port_count
# <- Minis.MidiBackend:OnUpdate), typically during the startup song scan.
#
# The stub implements the same entry points the managed side resolves and reports
# ZERO ports, so enumeration succeeds and no device is ever opened. It needs no
# access to the ALSA sequencer at all.
#
# Must be re-run after every build, since building overwrites the plugin.

set -euo pipefail

PLAYER_DIR="${PLAYER_DIR:-/home/liamb/projects/FlyHero-build/build/Linux64}"
REPO="${REPO:-/home/liamb/projects/FlyHero}"
SRC="$REPO/tools/automation/rtmidi_stub.c"
TARGET="$PLAYER_DIR/FlyHero_Data/Plugins/libRtMidi.so"

if [[ ! -d "$PLAYER_DIR" ]]; then
  echo "player not found at $PLAYER_DIR" >&2
  exit 2
fi
if [[ ! -f "$SRC" ]]; then
  echo "stub source not found at $SRC" >&2
  exit 2
fi

# Keep the real library once, so the change is reversible.
if [[ ! -f "$TARGET.orig" ]]; then
  cp -v "$TARGET" "$TARGET.orig"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

gcc -shared -fPIC -O2 -o "$TMP/libRtMidi.so" "$SRC"
cp -v "$TMP/libRtMidi.so" "$TARGET"

echo
echo "Exported rtmidi_* symbols:"
nm -D --defined-only "$TARGET" | grep -c rtmidi_
echo "Sample:"
nm -D --defined-only "$TARGET" | grep rtmidi_ | head -5

echo
echo "Done. Restore the original with:  cp $TARGET.orig $TARGET"
