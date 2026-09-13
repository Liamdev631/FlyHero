"""Read what the game wrote down.

The C# side (Assets/Script/Automation/AutomationScoreStore.cs) produces, in the
automation data directory:

  score_log.jsonl  one JSON object per finished attempt (AutomationSongResult)
  score_log.txt    the same events as human-readable lines
  top_scores.json  dict: song key -> best result ever achieved
  run_summary.txt  end-of-run summary

Per-attempt record shape (from AutomationSongResult / AutomationPlayerResult):

  song     : Timestamp SongKey SongName SongArtist SongCharter SongFolder
             QueueIndex AttemptNumber BandScore BandStars SongSpeed
             SongLengthSeconds NewHighScore PreviousBestScore
  players[]: Name Instrument Difficulty IsBot Score Stars NotesHit
             NotesMissed MaxCombo Percent IsFc
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class PlayerResult:
    name: str
    instrument: str
    difficulty: str
    is_bot: bool
    score: int
    stars: int
    notes_hit: int
    notes_missed: int
    max_combo: int
    percent: float
    is_fc: bool

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerResult":
        return cls(
            name=d.get("Name", ""),
            instrument=d.get("Instrument", ""),
            difficulty=d.get("Difficulty", ""),
            is_bot=bool(d.get("IsBot", False)),
            score=int(d.get("Score", 0)),
            stars=int(d.get("Stars", 0)),
            notes_hit=int(d.get("NotesHit", 0)),
            notes_missed=int(d.get("NotesMissed", 0)),
            max_combo=int(d.get("MaxCombo", 0)),
            percent=float(d.get("Percent", 0.0)),
            is_fc=bool(d.get("IsFc", False)),
        )

    @property
    def notes_total(self) -> int:
        return self.notes_hit + self.notes_missed


@dataclass
class Attempt:
    timestamp: str
    song_key: str
    song_name: str
    song_artist: str
    song_charter: str
    song_folder: str
    queue_index: int
    attempt_number: int
    band_score: int
    band_stars: int
    song_speed: float
    song_length_seconds: float
    new_high_score: bool
    previous_best_score: int
    players: list[PlayerResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Attempt":
        return cls(
            timestamp=d.get("Timestamp", ""),
            song_key=d.get("SongKey", ""),
            song_name=d.get("SongName", ""),
            song_artist=d.get("SongArtist", ""),
            song_charter=d.get("SongCharter", ""),
            song_folder=d.get("SongFolder", ""),
            queue_index=int(d.get("QueueIndex", 0)),
            attempt_number=int(d.get("AttemptNumber", 0)),
            band_score=int(d.get("BandScore", 0)),
            band_stars=int(d.get("BandStars", 0)),
            song_speed=float(d.get("SongSpeed", 1.0)),
            song_length_seconds=float(d.get("SongLengthSeconds", 0.0)),
            new_high_score=bool(d.get("NewHighScore", False)),
            previous_best_score=int(d.get("PreviousBestScore", 0)),
            players=[PlayerResult.from_dict(p) for p in d.get("Players") or []],
        )

    def player(self, instrument: str | None = None) -> PlayerResult | None:
        """The result for one instrument (defaults to the highest-scoring one).

        Matching is by substring because the game reports instrument names like
        "FiveFretGuitar" and "FiveLaneDrums" - an exact match on "guitar" would
        silently return nothing.
        """
        if not self.players:
            return None
        if instrument:
            want = instrument.lower()
            for p in self.players:
                if want in p.instrument.lower():
                    return p
            return None
        return max(self.players, key=lambda p: p.score)


def read_attempts(data_dir: str | Path | None = None) -> list[Attempt]:
    """Every logged attempt, in the order the game wrote them."""
    d = Path(data_dir) if data_dir else config.data_dir()
    src = d / "score_log.jsonl"
    if not src.exists():
        return []

    out = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Attempt.from_dict(json.loads(line)))
        except json.JSONDecodeError:
            # A partially-written final line is normal if the game is still running.
            continue
    return out


def best_scores(data_dir: str | Path | None = None) -> dict[str, dict]:
    """The top-score dictionary: song key -> best result ever.

    Falls back to folding the attempt log if the game has not written
    top_scores.json yet.
    """
    d = Path(data_dir) if data_dir else config.data_dir()
    src = d / "top_scores.json"
    if src.exists():
        try:
            return json.loads(src.read_text())
        except json.JSONDecodeError:
            pass

    folded: dict[str, dict] = {}
    for a in read_attempts(d):
        cur = folded.get(a.song_key)
        if cur is None or a.band_score > cur["BestScore"]:
            folded[a.song_key] = {
                "SongName": a.song_name,
                "SongArtist": a.song_artist,
                "BestScore": a.band_score,
                "BestStars": a.band_stars,
                "Plays": (cur or {}).get("Plays", 0) + 1,
            }
        elif cur is not None:
            cur["Plays"] += 1
    return folded


def note_stats(
    data_dir: str | Path | None = None,
    instrument: str | None = None,
) -> dict[str, dict]:
    """Aggregate note hits/misses per song, for one instrument or all of them.

    This is the per-instrument training signal: how many notes the player was
    asked to hit and how many it actually hit, per song.
    """
    stats: dict[str, dict] = {}

    for a in read_attempts(data_dir):
        for p in a.players:
            # Substring match: the game reports "FiveFretGuitar", not "guitar".
            if instrument and instrument.lower() not in p.instrument.lower():
                continue

            key = f"{a.song_key}|{p.instrument}|{p.difficulty}"
            s = stats.setdefault(
                key,
                {
                    "song_key": a.song_key,
                    "song_name": a.song_name,
                    "instrument": p.instrument,
                    "difficulty": p.difficulty,
                    "attempts": 0,
                    "notes_hit": 0,
                    "notes_missed": 0,
                    "notes_total": 0,
                    "best_score": 0,
                    "best_percent": 0.0,
                    "best_combo": 0,
                    "full_combos": 0,
                },
            )

            s["attempts"] += 1
            s["notes_hit"] += p.notes_hit
            s["notes_missed"] += p.notes_missed
            s["notes_total"] += p.notes_total
            s["best_score"] = max(s["best_score"], p.score)
            s["best_percent"] = max(s["best_percent"], p.percent)
            s["best_combo"] = max(s["best_combo"], p.max_combo)
            s["full_combos"] += 1 if p.is_fc else 0

    for s in stats.values():
        total = s["notes_total"]
        s["accuracy"] = (s["notes_hit"] / total) if total else 0.0

    return stats
