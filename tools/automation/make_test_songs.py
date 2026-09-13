#!/usr/bin/env python3
"""
Generates self-contained test songs for the FlyHero automated queue.

Produces, for each song, a folder containing:
    song.ini      - metadata read by YARG
    notes.chart   - a hand-authored Moonscraper-format chart (5-fret guitar)
    song.ogg      - generated audio, matching the chart length

Everything is synthesised from scratch (ffmpeg tones), so there is no third-party
or copyrighted content involved and results are byte-for-byte reproducible.

Usage:
    python3 make_test_songs.py <output_dir> [--notes 240] [--bpm 120]

Notes are placed on a steady eighth-note grid across the five lanes, which gives
the bot something continuous to hit and produces a real, non-zero score.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

# A minor pentatonic-ish ladder so the generated audio is less grating than a
# flat tone, one frequency per lane.
LANE_FREQUENCIES = [220.00, 261.63, 329.63, 392.00, 440.00]

RESOLUTION = 192  # ticks per beat in the .chart format


def build_chart(song_name, artist, resolution, bpm, notes_per_difficulty):
    """Builds the .chart text for a song."""
    lines = []
    add = lines.append

    add("[Song]")
    add("{")
    add(f'  Name = "{song_name}"')
    add(f'  Artist = "{artist}"')
    add('  Charter = "FlyHero automation"')
    add('  Album = "Generated"')
    add('  Year = "2026"')
    add('  Genre = "Test"')
    add('  MediaType = "cd"')
    add('  MusicStream = "song.ogg"')
    add("  Offset = 0")
    add(f"  Resolution = {resolution}")
    add("  Player2 = bass")
    add("  Difficulty = 0")
    add("  PreviewStart = 0")
    add("  PreviewEnd = 0")
    add("}")

    add("[SyncTrack]")
    add("{")
    add(f"  0 = TS 4")
    add(f"  0 = B {int(bpm * 1000)}")
    add("}")

    add("[Events]")
    add("{")
    add("}")

    # One track per difficulty so the song is selectable on any difficulty.
    # Tick spacing doubles as the difficulty drops, keeping them roughly aligned.
    for track, divisor in (("ExpertSingle", 2), ("HardSingle", 3), ("MediumSingle", 4), ("EasySingle", 6)):
        ticks_per_note = RESOLUTION // divisor
        add(f"[{track}]")
        add("{")
        tick = 0
        index = 0
        for _ in range(notes_per_difficulty):
            lane = (index * 2 + divisor) % 5
            # Alternate plain notes and short sustains so both code paths get exercised.
            sustain = ticks_per_note // 2 if index % 4 == 3 else 0
            add(f"  {tick} = N {lane} {sustain}")
            tick += ticks_per_note
            index += 1
        add("}")

    return "\n".join(lines) + "\n"


def song_length_seconds(track, note_count, bpm):
    """Duration of the charted track in seconds."""
    ticks_per_note = RESOLUTION // track
    total_ticks = ticks_per_note * note_count
    beats = total_ticks / RESOLUTION
    return beats * (60.0 / bpm)


def build_audio(path, duration, lane_cycle_length=8):
    """Renders an ogg of the given duration using ffmpeg."""
    # Build a sequence of short tones, one per eighth note, following the lane ladder.
    beat = 0.25
    count = max(1, int(duration / beat))
    freqs = [LANE_FREQUENCIES[i % len(LANE_FREQUENCIES)] for i in range(count)]

    # aevalsrc with a per-segment frequency via a piecewise sine lookup is fiddly, so
    # generate each tone then concatenate.
    tmp_dir = os.path.join(os.path.dirname(path), ".tmp_audio")
    os.makedirs(tmp_dir, exist_ok=True)

    parts = []
    try:
        for i, freq in enumerate(freqs):
            part = os.path.join(tmp_dir, f"p{i:05d}.ogg")
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi",
                    "-i", f"sine=frequency={freq:.2f}:duration={beat}:sample_rate=44100",
                    "-af", "afade=t=out:st=0:d=0.06,volume=0.28",
                    part,
                ],
                check=True,
            )
            parts.append(part)

        list_file = os.path.join(tmp_dir, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as fh:
            for part in parts:
                fh.write(f"file '{os.path.abspath(part)}'\n")

        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", list_file,
                "-c:a", "libvorbis", "-q:a", "3",
                path,
            ],
            check=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def write_song_ini(path, song_name, artist, duration):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("[song]\n")
        fh.write(f"name = {song_name}\n")
        fh.write(f"artist = {artist}\n")
        fh.write("charter = FlyHero automation\n")
        fh.write("album = Generated\n")
        fh.write("year = 2026\n")
        fh.write("genre = Test\n")
        fh.write(f"song_length = {int(duration * 1000)}\n")
        fh.write("diff_guitar = 0\n")
        fh.write("pro_drums = False\n")
        fh.write("five_lane_drums = False\n")


SONGS = [
    ("FlyHero Test One", "Automation", 120, 96),
    ("FlyHero Test Two", "Automation", 132, 112),
    ("FlyHero Test Three", "Automation", 100, 80),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="folder to write song folders into")
    parser.add_argument("--only", help="generate only the song with this name")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required but was not found on PATH", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    created = []

    for song_name, artist, bpm, note_count in SONGS:
        if args.only and args.only != song_name:
            continue

        folder = os.path.join(args.output_dir, song_name)
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, "notes.chart"), "w", encoding="utf-8") as fh:
            fh.write(build_chart(song_name, artist, RESOLUTION, bpm, note_count))

        # Length driven by the Expert track; leave a short tail after the last note
        # so the chart resolves before the audio stops.
        duration = song_length_seconds(2, note_count, bpm) + 2.0

        build_audio(os.path.join(folder, "song.ogg"), duration)
        write_song_ini(os.path.join(folder, "song.ini"), song_name, artist, duration)

        created.append((song_name, folder, duration, note_count))
        print(f"created {folder}  ({duration:.1f}s, {note_count} expert notes, {bpm} bpm)")

    print(f"\n{len(created)} song(s) written to {os.path.abspath(args.output_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
