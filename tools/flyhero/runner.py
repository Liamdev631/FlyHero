"""Launch and control the game.

The game (built from source) accepts these automation arguments, declared in
Assets/Script/Persistent/CommandLineArgs.cs:

  -autoqueue <file>        queue JSON to play; switches on automation mode
  -autosongdir <dir>       extra song folder for the queue
  -autodata <dir>          where score_log.jsonl / top_scores.json are written
  -autospeed <percent>     song speed, 100 = normal
  -autoinstrument <name>   instrument to play (e.g. guitar, bass, drums, keys)
  -autodifficulty <name>   difficulty (easy, medium, hard, expert)
  -autorepeat              run the queue again when it finishes
  -autostayopen            leave the game running instead of exiting at the end

With no `-autostayopen` the process exits on its own when the queue is done,
which is what makes a "run the whole queue and hand me the scores" call simple.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from . import config, results


def build_args(
    queue_path: str | Path,
    data_dir: str | Path | None = None,
    song_dir: str | Path | None = None,
    speed: float = 1.0,
    instrument: str | None = None,
    difficulty: str | None = None,
    repeat: bool = False,
    stay_open: bool = False,
    offline: bool = True,
) -> list[str]:
    """Assemble the command line for an automated run."""
    args = [
        "-autoqueue", str(queue_path),
        "-autodata", str(data_dir if data_dir is not None else config.data_dir()),
    ]

    if song_dir is not None:
        args += ["-autosongdir", str(song_dir)]
    if instrument:
        args += ["-autoinstrument", str(instrument)]
    if difficulty:
        args += ["-autodifficulty", str(difficulty)]

    # -autospeed is read as a percentage.
    args += ["-autospeed", str(int(round(speed * 100)))]

    if repeat:
        args.append("-autorepeat")
    if stay_open:
        args.append("-autostayopen")
    if offline:
        args.append("-offline")

    return args


def play(
    songs: list[str] | None = None,
    queue_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    song_dir: str | Path | None = None,
    speed: float = 1.0,
    instrument: str | None = None,
    difficulty: str | None = None,
    repeat: bool = False,
    wait: bool = True,
    timeout: float | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen | int:
    """Run the game on a queue.

    Pass ``songs`` to build a queue on the fly, or ``queue_path`` to use one that
    already exists. With ``wait=True`` blocks until the run finishes and returns
    the exit status; otherwise returns the Popen handle so the caller can drive
    the game live (this is the mode the training loop will use).
    """
    if songs is not None:
        from . import queue as queue_mod

        queue_path = queue_mod.write_queue(songs, song_dir=song_dir, path=queue_path)
    if queue_path is None:
        queue_path = config.queue_file()

    argv = [str(config.game_binary())] + build_args(
        queue_path,
        data_dir=data_dir,
        song_dir=song_dir,
        speed=speed,
        instrument=instrument,
        difficulty=difficulty,
        repeat=repeat,
        stay_open=not wait,
    )

    if extra_args:
        argv += [str(a) for a in extra_args]

    env = dict(os.environ)
    env.setdefault("DISPLAY", env.get("FLYHERO_DISPLAY", ":1"))

    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if not wait:
        return proc

    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop(proc)
        raise


def stop(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Ask a running game to quit, escalating to SIGKILL if it will not."""
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def run_queue(
    songs: list[str] | None = None,
    instrument: str = "guitar",
    difficulty: str = "expert",
    speed: float = 1.0,
    timeout: float | None = None,
) -> dict:
    """Play a queue to completion and return the freshly recorded results.

    Convenience wrapper: records the attempt count before the run so the caller
    can tell which attempts are new.
    """
    before = len(results.read_attempts())

    status = play(
        songs=songs,
        instrument=instrument,
        difficulty=difficulty,
        speed=speed,
        wait=True,
        timeout=timeout,
    )

    attempts = results.read_attempts()
    return {
        "exit_status": status,
        "new_attempts": attempts[before:],
        "total_attempts": len(attempts),
        "best_scores": results.best_scores(),
    }
