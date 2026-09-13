"""Paths, defaults and environment for the FlyHero control suite.

Everything here is overridable with environment variables so the suite is not
tied to one machine layout.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- roots ---------------------------------------------------------------

REPO = Path(os.environ.get("FLYHERO_REPO", "/home/liamb/projects/FlyHero"))
WORK = Path(os.environ.get("FLYHERO_WORK", "/home/liamb/projects/FlyHero-build"))
UNITY_EDITOR = Path(os.environ.get("FLYHERO_UNITY", "/home/liamb/unity/6000.3.5f2/Editor/Unity"))


# --- directories ---------------------------------------------------------

def data_dir() -> Path:
    """Where the game writes score_log.jsonl / top_scores.json."""
    d = Path(os.environ.get("FLYHERO_DATA", WORK / "automation-data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def songs_dir() -> Path:
    d = Path(os.environ.get("FLYHERO_SONGS", WORK / "songs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def queue_file() -> Path:
    """Default queue path handed to the game via -autoqueue."""
    return Path(os.environ.get("FLYHERO_QUEUE", data_dir() / "queue.json"))


def build_dir() -> Path:
    return Path(os.environ.get("FLYHERO_BUILD_OUT", WORK / "build"))


# --- the game executable -------------------------------------------------

#: Names the built player is likely to use, in preference order.
_BINARY_NAMES = ("FlyHero", "FlyHero.x86_64", "YARG", "YARG.x86_64")


def game_binary() -> Path:
    """Locate the game executable.

    Order: explicit FLYHERO_GAME_BINARY, then the source build we produce with
    build_headless.sh, then the prebuilt release used before the editor was
    licensed.
    """
    explicit = os.environ.get("FLYHERO_GAME_BINARY")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"FLYHERO_GAME_BINARY does not exist: {p}")
        return p

    candidates: list[Path] = []

    # Source build: <work>/build/**/<name>
    root = build_dir()
    if root.is_dir():
        for name in _BINARY_NAMES:
            candidates += sorted(root.rglob(name))

    # Prebuilt fallback: <work>/prebuilt/<name>
    pre = WORK / "prebuilt"
    if pre.is_dir():
        for name in _BINARY_NAMES:
            candidates.append(pre / name)

    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c

    raise FileNotFoundError(
        "No game executable found. Build it with tools/automation/build_headless.sh "
        "or set FLYHERO_GAME_BINARY."
    )


def unity_libfix() -> Path:
    """Older-soname shims the Unity editor needs on this distro."""
    return Path(os.environ.get("FLYHERO_LIBFIX", "/home/liamb/unity/libfix/usr/lib/x86_64-linux-gnu"))
