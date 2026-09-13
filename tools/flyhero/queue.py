"""Build the song queue the game consumes.

Matches the schema read by Assets/Script/Automation/AutomationManager.cs
(LoadQueue): either a bare JSON array of song names, or an object with
``songDir`` and ``songs``.

Each entry may be:
  * a song folder name            ("FlyHero Test One")
  * a "Name - Artist" string      ("FlyHero Test One - Automation")
  * a full path to a song folder
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config


def write_queue(
    songs: list[str],
    song_dir: str | Path | None = None,
    path: str | Path | None = None,
) -> Path:
    """Write a queue file and return its path.

    ``songs`` is the play order. ``song_dir`` is the folder the game scans for
    songs; it is added to the game's song folders at startup.
    """
    if not songs:
        raise ValueError("queue must contain at least one song")

    out = Path(path) if path else config.queue_file()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "songDir": str(song_dir if song_dir is not None else config.songs_dir()),
        "songs": list(songs),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def read_queue(path: str | Path | None = None) -> list[str]:
    """Read back the song list from a queue file."""
    src = Path(path) if path else config.queue_file()
    root = json.loads(src.read_text())

    if isinstance(root, list):
        return [str(s) for s in root]
    if isinstance(root, dict):
        return [str(s) for s in root.get("songs", [])]
    raise ValueError(f"unrecognised queue format in {src}")


def available_songs(song_dir: str | Path | None = None) -> list[str]:
    """List song folders on disk (each folder holding a notes file)."""
    d = Path(song_dir) if song_dir else config.songs_dir()
    if not d.is_dir():
        return []

    out = []
    for child in sorted(d.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.chart")) or any(child.glob("*.mid")):
            out.append(child.name)
    return out
