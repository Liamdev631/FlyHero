"""FlyHero control suite — drive the game, collect scores, feed a model.

Quick start (from tools/):

    from flyhero import api

    api.queue(["FlyHero Test One", "FlyHero Test Two"])
    api.play_queue()
    print(api.best_scores())
    print(api.note_stats())

The heavy lifting happens in the game itself: built with the automation hooks,
it plays a queue end to end, rolls straight into the next song, and writes a
per-attempt log plus a top-score dictionary.
"""
from __future__ import annotations

# Thin, intention-revealing facade over the modules underneath.
from .config import data_dir, game_binary, songs_dir
from .queue import available_songs, read_queue, write_queue
from .results import (
    Attempt,
    PlayerResult,
    best_scores,
    note_stats,
    read_attempts,
)
from .runner import build_args, play, run_queue, stop

__all__ = [
    "Attempt",
    "PlayerResult",
    "available_songs",
    "best_scores",
    "build_args",
    "data_dir",
    "game_binary",
    "note_stats",
    "play",
    "read_attempts",
    "read_queue",
    "run_queue",
    "songs_dir",
    "stop",
    "write_queue",
]
