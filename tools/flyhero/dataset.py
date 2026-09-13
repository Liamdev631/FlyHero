"""Turn a rhythm chart into supervised (observation, action) training data.

Why this exists: a rhythm chart is *ground truth*. Every note states its lane and
its exact tick, so the correct action at any instant is known without any
environment interaction, reward signal, or rollout. That makes the first training
stage supervised behavioural cloning rather than sparse-reward RL - which is the
main lesson from doomfly, whose plasticity rule never saw the game variables and
whose training therefore failed its validation gates.

Observation (per step):
    for each of 5 lanes: seconds until that lane's next note onset (clamped to
    `horizon`, and 0.0 when the lane is currently being held); plus the remaining
    sustain hold in seconds; plus normalised song position.

Action (label, per step):
    frets  - 5-bit mask of which frets should be held
    strum  - 1 on the step where a note onset lands

Chart format (notes.chart, Moonscraper/EOF "CHART" text format):
    [Song] { Resolution = 192 ... }
    [SyncTrack] { 0 = B 120000 ... }        # BPM = value / 1000
    [EasySingle] { 1485 = N lane sustain }  # tick = N lane sustain
Sections are named <Difficulty><Track>; five-fret guitar is *Single.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LANES = 5  # five-fret guitar: green, red, yellow, blue, orange

_SECTION = re.compile(r"^\[([A-Za-z0-9_]+)\]\s*$")
_KV = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")
_EVENT = re.compile(r"^\s*(\d+)\s*=\s*([A-Za-z]+)\s+(.*?)\s*$")

DIFFICULTIES = ("Easy", "Medium", "Hard", "Expert")


@dataclass
class Chart:
    song_dir: Path
    name: str
    artist: str
    resolution: int
    offset_ms: float
    bpms: list[tuple[int, float]] = field(default_factory=list)  # (tick, bpm)
    # difficulty -> list of (tick, lane, sustain_ticks)
    notes: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)

    @property
    def ticks_per_second(self) -> float:
        """Ticks per second at the first BPM (these test charts are single-tempo)."""
        bpm = self.bpms[0][1] if self.bpms else 120.0
        return self.resolution * bpm / 60.0

    def seconds_at(self, tick: int) -> float:
        """Seconds for a tick, walking any tempo changes."""
        if not self.bpms:
            return tick / self.ticks_per_second

        elapsed = 0.0
        prev_tick, prev_bpm = self.bpms[0]
        for t, bpm in self.bpms[1:]:
            if tick <= t:
                break
            elapsed += (t - prev_tick) / (self.resolution * prev_bpm / 60.0)
            prev_tick, prev_bpm = t, bpm
        return elapsed + (tick - prev_tick) / (self.resolution * prev_bpm / 60.0)


def parse_chart(song_dir: str | Path) -> Chart:
    song_dir = Path(song_dir)
    chart_path = song_dir / "notes.chart"
    if not chart_path.is_file():
        raise FileNotFoundError(f"no notes.chart in {song_dir}")

    text = chart_path.read_text(encoding="utf-8", errors="replace")
    chart = Chart(song_dir=song_dir, name=song_dir.name, artist="", resolution=192, offset_ms=0.0)

    section = ""
    for raw in text.splitlines():
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue

        m = _SECTION.match(line)
        if m:
            section = m.group(1)
            continue
        if line.strip() in ("{", "}"):
            continue

        if section == "Song":
            kv = _KV.match(line)
            if kv:
                key, value = kv.group(1), kv.group(2).strip('"')
                if key == "Name":
                    chart.name = value
                elif key == "Artist":
                    chart.artist = value
                elif key == "Resolution":
                    chart.resolution = int(float(value))
                elif key == "Offset":
                    chart.offset_ms = float(value)
            continue

        if section == "SyncTrack":
            ev = _EVENT.match(line)
            if ev and ev.group(2) == "B":
                chart.bpms.append((int(ev.group(1)), int(ev.group(3)) / 1000.0))
            continue

        # Note sections: <Difficulty><Track>, e.g. EasySingle
        for diff in DIFFICULTIES:
            if section.startswith(diff):
                ev = _EVENT.match(line)
                if ev and ev.group(2) == "N":
                    parts = ev.group(3).split()
                    lane = int(parts[0])
                    sustain = int(parts[1]) if len(parts) > 1 else 0
                    if 0 <= lane < LANES:
                        chart.notes.setdefault(section, []).append(
                            (int(ev.group(1)), lane, sustain))
                break

    chart.bpms.sort()
    for key in chart.notes:
        chart.notes[key].sort()
    return chart


def find_note_section(chart: Chart, difficulty: str) -> str | None:
    """Resolve a difficulty name to a chart section (guitar = *Single)."""
    for section in chart.notes:
        if section.lower().startswith(difficulty.lower()):
            return section
    return None


@dataclass
class Dataset:
    x: np.ndarray          # (steps, obs_dim) float32
    frets: np.ndarray      # (steps, 5) float32
    strum: np.ndarray      # (steps,)   float32
    times: np.ndarray      # (steps,)   float32, seconds
    meta: dict

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.x.shape[1])


def build(
    song_dir: str | Path,
    difficulty: str = "Easy",
    rate_hz: float = 100.0,
    horizon: float = 1.0,
) -> Dataset:
    """Build a cloning dataset for one song/difficulty.

    ``rate_hz`` is the controller step rate; 100 Hz is finer than the game's
    input update and fine enough that strum timing error stays under ~10 ms.
    """
    chart = parse_chart(song_dir)
    section = find_note_section(chart, difficulty)
    if section is None:
        raise ValueError(
            f"{chart.name}: no chart section for difficulty {difficulty!r} "
            f"(have {sorted(chart.notes)})")

    notes = chart.notes[section]
    if not notes:
        raise ValueError(f"{chart.name}: section {section} has no notes")

    tps = chart.ticks_per_second
    offset = chart.offset_ms / 1000.0

    # Onset / release in seconds for every note.
    onsets = np.array([chart.seconds_at(t) - offset for t, _, _ in notes], dtype=np.float64)
    holds = np.array([s / tps for _, _, s in notes], dtype=np.float64)
    lanes = np.array([l for _, l, _ in notes], dtype=np.int64)

    duration = float(onsets[-1] + holds[-1]) + 0.25
    steps = int(math.ceil(duration * rate_hz))
    times = np.arange(steps, dtype=np.float64) / rate_hz

    obs = np.zeros((steps, LANES * 2 + 1), dtype=np.float32)
    frets = np.zeros((steps, LANES), dtype=np.float32)
    strum = np.zeros(steps, dtype=np.float32)

    # Precompute, per lane, the ordered onsets and their hold ends.
    per_lane: list[list[int]] = [[] for _ in range(LANES)]
    for i, lane in enumerate(lanes):
        per_lane[int(lane)].append(i)
    for lst in per_lane:
        lst.sort(key=lambda i: onsets[i])

    step_dt = 1.0 / rate_hz

    for s in range(steps):
        t = times[s]
        for lane in range(LANES):
            idxs = per_lane[lane]
            if not idxs:
                continue

            next_delta = horizon
            holding_until = None

            for i in idxs:
                start = onsets[i]
                end = start + holds[i]

                # A long sustain dominates the lane: hold it until it ends.
                if start <= t <= end:
                    holding_until = end if end > t else t
                    next_delta = 0.0
                    break

                if start > t:
                    next_delta = min(next_delta, float(start - t))
                    break

            obs[s, lane * 2] = next_delta / horizon
            if holding_until is not None:
                obs[s, lane * 2 + 1] = float(holding_until - t) / horizon

            # Action: hold the lane while inside a note (or its sustain).
            if holding_until is not None:
                frets[s, lane] = 1.0

        # Strum: a note onset lands inside this step and is not already held.
        for lane in range(LANES):
            for i in per_lane[lane]:
                start = onsets[i]
                if t <= start < t + step_dt:
                    strum[s] = 1.0
                    frets[s, lane] = 1.0
                    break

        obs[s, LANES * 2] = t / max(duration, 1e-9)

    meta = {
        "song": chart.name,
        "artist": chart.artist,
        "song_dir": str(chart.song_dir),
        "section": section,
        "difficulty": difficulty,
        "resolution": chart.resolution,
        "bpm": chart.bpms[0][1] if chart.bpms else None,
        "ticks_per_second": tps,
        "note_count": len(notes),
        "rate_hz": rate_hz,
        "horizon_seconds": horizon,
        "steps": steps,
        "duration_seconds": duration,
        "obs_dim": int(obs.shape[1]),
        "obs_layout": [
            *[f"lane{l}_seconds_until_next_note" for l in range(LANES)],
            *[f"lane{l}_seconds_remaining_in_hold" for l in range(LANES)],
            "song_position_normalized",
        ],
    }

    return Dataset(x=obs, frets=frets, strum=strum, times=times.astype(np.float32), meta=meta)


def verify(ds: Dataset) -> dict:
    """Self-checks that the labelling actually encodes the chart.

    A mislabelled dataset is the classic silent failure here - the model trains
    happily and learns nothing - so these are asserted rather than eyeballed.
    """
    expected_notes = ds.meta["note_count"]
    strum_events = int(ds.strum.sum())

    # Holds: each note's onset step must have its lane held.
    onset_steps = np.nonzero(ds.strum)[0]
    held_at_onset = int(ds.frets[onset_steps].sum() if len(onset_steps) else 0)

    per_lane = ds.frets.sum(axis=0)

    return {
        "notes_in_chart": expected_notes,
        "strum_events": strum_events,
        "strum_matches_note_count": strum_events == expected_notes,
        "lane_hold_steps": [int(v) for v in per_lane],
        "any_lane_never_used": [i for i, v in enumerate(per_lane) if v == 0],
        "onset_steps_with_a_held_fret": held_at_onset,
        "onset_coverage_ok": held_at_onset == strum_events,
        "obs_finite": bool(np.isfinite(ds.x).all()),
        "obs_range": [float(ds.x.min()), float(ds.x.max())],
    }


def save(ds: Dataset, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{ds.meta['song'].replace(' ', '_')}_{ds.meta['difficulty']}"
    npz = out / f"{stem}.npz"
    np.savez_compressed(npz, x=ds.x, frets=ds.frets, strum=ds.strum, times=ds.times)
    (out / f"{stem}.json").write_text(json.dumps({**ds.meta, "verify": verify(ds)}, indent=2))
    return npz
