"""The callable surface of the control suite.

This is the module to import when driving the game from Python or a notebook:

    from flyhero import api

    api.queue(["FlyHero Test One", "FlyHero Test Two"])
    result = api.play_queue(instrument="guitar", difficulty="easy")
    api.best_scores()
    api.note_stats(instrument="guitar")

Everything is a plain function over the files the game writes; nothing is cached
between calls, so a long-running process always sees the latest results.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import config
from . import queue as queue_mod
from . import results as results_mod
from . import runner as runner_mod
from .results import Attempt


# --- where things live ---------------------------------------------------

def data_path() -> Path:
    """Directory holding score_log.jsonl, top_scores.json, run_summary.txt."""
    return config.data_dir()


def songs_path() -> Path:
    return config.songs_dir()


def queue_path() -> Path:
    return config.queue_file()


def binary() -> Path:
    """The game executable that will be run."""
    return config.game_binary()


# --- queueing ------------------------------------------------------------

def queue(
    songs: list[str],
    song_dir: str | Path | None = None,
    path: str | Path | None = None,
) -> Path:
    """Write the song queue (play order) and return the queue file path."""
    return queue_mod.write_queue(songs, song_dir=song_dir, path=path)


def queued(path: str | Path | None = None) -> list[str]:
    """Read back the songs currently in the queue file."""
    return queue_mod.read_queue(path)


def songs(song_dir: str | Path | None = None) -> list[str]:
    """Songs the game will find, without needing the game."""
    return queue_mod.available_songs(song_dir)


# --- running -------------------------------------------------------------

def play(
    songs: list[str] | None = None,
    instrument: str | None = None,
    difficulty: str | None = None,
    speed: float = 1.0,
    repeat: bool = False,
    wait: bool = True,
    timeout: float | None = None,
) -> subprocess.Popen | int:
    """Run the game.

    With ``wait=True`` it blocks and returns the exit status. With
    ``wait=False`` it returns the live process so the caller can drive the
    session (the mode a policy loop needs).
    """
    return runner_mod.play(
        songs=songs,
        instrument=instrument,
        difficulty=difficulty,
        speed=speed,
        repeat=repeat,
        wait=wait,
        timeout=timeout,
    )


def play_queue(
    songs: list[str] | None = None,
    instrument: str = "guitar",
    difficulty: str = "expert",
    speed: float = 1.0,
    timeout: float | None = None,
) -> dict:
    """Play a queue to completion; return the new attempts and best scores."""
    return runner_mod.run_queue(
        songs=songs,
        instrument=instrument,
        difficulty=difficulty,
        speed=speed,
        timeout=timeout,
    )


def stop(proc: subprocess.Popen) -> None:
    """Stop a game started with ``wait=False``."""
    runner_mod.stop(proc)


# --- reading results -----------------------------------------------------

def attempts(data_dir: str | Path | None = None) -> list[Attempt]:
    """Every attempt logged so far, oldest first."""
    return results_mod.read_attempts(data_dir)


def best_scores(data_dir: str | Path | None = None) -> dict[str, dict]:
    """The top-score dictionary: song key -> best result ever achieved."""
    return results_mod.best_scores(data_dir)


def note_stats(
    data_dir: str | Path | None = None,
    instrument: str | None = None,
) -> dict[str, dict]:
    """Note hits/misses/accuracy per song+instrument+difficulty.

    ``instrument`` filters to one instrument (e.g. "guitar"); omit for all.
    """
    return results_mod.note_stats(data_dir, instrument=instrument)


def instruments() -> list[str]:
    """Every instrument seen in the logs so far."""
    seen = {p.instrument for a in results_mod.read_attempts() for p in a.players}
    return sorted(seen)
