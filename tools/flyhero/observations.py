"""Read what the game observed about itself (observations.jsonl).

Written by Assets/Script/Automation/AutomationObserver.cs during an unattended run.
This is the game's own clock and player state, sampled ~20 Hz.

Why it matters: it maps *absolute* wall time to *song* time. A screen recording
started at a known moment can therefore be aligned to song time exactly, which is
what pairing video frames with the notes on screen requires. Inferring that from the
recording's audio did not work (see todo.md) - the game simply knows the answer.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class PlayerState:
    name: str
    instrument: str
    difficulty: str
    is_bot: bool
    score: int
    combo: int
    notes_hit: int
    total_notes: int
    notes_hit_fraction: float
    stars: float
    is_fc: bool


@dataclass
class NoteObservation:
    """One upcoming note, as the game reported it."""
    lane: int
    dt: float           # seconds until it must be hit (negative once it has passed)
    sustain: float
    chord: bool

    @property
    def is_upcoming(self) -> bool:
        return self.dt > 0


@dataclass
class Sample:
    unix_ms: int
    wall_ms: float
    song_time: float
    song_length: float
    players: list[PlayerState] = field(default_factory=list)
    horizon: list[NoteObservation] = field(default_factory=list)

    def player(self, instrument: str | None = None) -> PlayerState | None:
        if not self.players:
            return None
        if instrument:
            want = instrument.lower()
            for p in self.players:
                if want in p.instrument.lower():
                    return p
            return None
        return max(self.players, key=lambda p: p.score)

    def next_note(self) -> NoteObservation | None:
        upcoming = [n for n in self.horizon if n.is_upcoming]
        return min(upcoming, key=lambda n: n.dt) if upcoming else None


def observations_path(data_dir: str | Path | None = None) -> Path:
    d = Path(data_dir) if data_dir else config.data_dir()
    return d / "observations.jsonl"


def read(data_dir: str | Path | None = None) -> list[Sample]:
    """All samples, in order. Tolerates the BOM older builds wrote."""
    src = observations_path(data_dir)
    if not src.exists():
        return []

    out: list[Sample] = []
    with src.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                # A partially written final line is normal mid-run.
                continue

            out.append(Sample(
                unix_ms=int(d.get("unix_ms", 0)),
                wall_ms=float(d.get("wall_ms", 0.0)),
                song_time=float(d.get("song_time", 0.0)),
                song_length=float(d.get("song_length", 0.0)),
                players=[PlayerState(
                    name=p.get("name", ""),
                    instrument=p.get("instrument", ""),
                    difficulty=p.get("difficulty", ""),
                    is_bot=bool(p.get("is_bot", False)),
                    score=int(p.get("score", 0)),
                    combo=int(p.get("combo", 0)),
                    notes_hit=int(p.get("notes_hit", 0)),
                    total_notes=int(p.get("total_notes", 0)),
                    notes_hit_fraction=float(p.get("notes_hit_fraction", 0.0)),
                    stars=float(p.get("stars", 0.0)),
                    is_fc=bool(p.get("is_fc", False)),
                ) for p in (d.get("players") or [])],
                horizon=[NoteObservation(
                    lane=int(n.get("lane", 0)),
                    dt=float(n.get("dt", 0.0)),
                    sustain=float(n.get("sustain", 0.0)),
                    chord=bool(n.get("chord", False)),
                ) for n in (d.get("horizon") or [])],
            ))
    return out


def reconstruct_notes(samples: list[Sample]) -> list[tuple[int, int, float]]:
    """Rebuild the note timeline from the horizon stream.

    Each observation reports a note's lane and its offset from the current song time, so
    the note's absolute song time is `song_time + dt`. A note appears in many consecutive
    samples, so collapse them.

    Returns (hit_unix_ms, lane, song_time) sorted by time.

    NOTE: `lane` is the game's 1-based `GuitarNote.Fret` (green..orange = 1..5), NOT the
    chart's 0-based lane index. Chart lanes 0..4 correspond to reported lanes 1..5. This
    off-by-one is silent and will mislabel training data if forgotten.

    Bucketing is deliberately coarse (50 ms): the same note seen in different samples
    yields slightly different absolute times, and a fine bucket would split one note into
    several. Verified against the chart - all notes matched, no spurious times.
    """
    bucket_ms = 50
    seen: dict[tuple[int, int], float] = {}

    for s in samples:
        for n in s.horizon:
            absolute = s.unix_ms + int(round(n.dt * 1000))
            key = (n.lane, int(round(absolute / bucket_ms)))
            seen.setdefault(key, s.song_time + n.dt)

    return sorted(
        ((lane, bucket * bucket_ms, song_time) for (lane, bucket), song_time in seen.items()),
        key=lambda r: r[1],
    )


def song_time_at(samples: list[Sample], unix_ms: int) -> float | None:
    """Interpolate the song time at an absolute wall-clock time.

    This is the frame-alignment primitive: pass the unix time of a video frame and
    get the song position on screen.
    """
    if not samples:
        return None

    times = [s.unix_ms for s in samples]
    if unix_ms <= times[0]:
        return samples[0].song_time
    if unix_ms >= times[-1]:
        return samples[-1].song_time

    i = bisect.bisect_left(times, unix_ms)
    before, after = samples[i - 1], samples[i]
    span = after.unix_ms - before.unix_ms
    if span <= 0:
        return after.song_time

    t = (unix_ms - before.unix_ms) / span
    return before.song_time + (after.song_time - before.song_time) * t


def summary(samples: list[Sample], instrument: str = "guitar") -> dict:
    if not samples:
        return {"samples": 0}

    intervals = [samples[i + 1].song_time - samples[i].song_time for i in range(len(samples) - 1)]
    intervals = [d for d in intervals if d > 0]

    first = samples[0].player(instrument)
    last = samples[-1].player(instrument)

    return {
        "samples": len(samples),
        "song_time_first": samples[0].song_time,
        "song_time_last": samples[-1].song_time,
        "song_length": samples[0].song_length,
        "sample_hz": (1.0 / (sum(intervals) / len(intervals))) if intervals else 0.0,
        "monotonic": all(
            samples[i + 1].song_time >= samples[i].song_time for i in range(len(samples) - 1)
        ),
        "wall_span_ms": samples[-1].unix_ms - samples[0].unix_ms,
        "song_span_ms": (samples[-1].song_time - samples[0].song_time) * 1000.0,
        "instrument": instrument,
        "score_first": first.score if first else None,
        "score_last": last.score if last else None,
        "notes_hit_last": last.notes_hit if last else None,
        "total_notes": last.total_notes if last else None,
    }


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the game's observation stream")
    ap.add_argument("--instrument", default="guitar")
    ap.add_argument("--at-unix-ms", type=int, default=None,
                    help="resolve this absolute time to a song time (frame alignment)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    samples = read()
    if not samples:
        print(f"no observations in {observations_path()}")
        return 1

    if args.at_unix_ms is not None:
        t = song_time_at(samples, args.at_unix_ms)
        print(f"unix_ms {args.at_unix_ms} -> song_time {t:.3f} s" if t is not None else "no mapping")
        return 0

    s = summary(samples, args.instrument)
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        print(f"file           : {observations_path()}")
        for k, v in s.items():
            print(f"{k:<16}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
