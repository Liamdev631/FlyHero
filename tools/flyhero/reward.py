"""Per-note reward from the judgements the game already produces.

The user's spec:
    reward  for hitting a note
    punish  for missing a note
    punish  for hitting the wrong note
    punish  for the right note at the wrong time

Every one of those has a real signal in YARG's results, which we already log:

    hit            -> NotesHit / the note's judgement
    miss           -> NotesMissed
    wrong note     -> Overstrums  (strummed when no note was there)
                      GhostInputs (pressed a fret when no note was there)
    wrong timing   -> the note offset the results screen reports (we measured ~-15 ms)

Using the game's own judgements matters. A hand-rolled reward that re-derives "was that
right?" from positions would disagree with the game's hit windows, and the policy would be
optimising a different game than the one being scored.

CAVEAT ON THE CONSTANTS: PERFECT_MS and HIT_WINDOW_MS are placeholders. YARG's real windows
vary by difficulty and by the player's hit-window settings, so these must be read from the
engine before the reward can be trusted. Until then they are a documented approximation, not
a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

# Placeholder windows - see the caveat above.
PERFECT_MS = 25.0
HIT_WINDOW_MS = 70.0

# Judgement names as the game reports them (case-insensitive on input).
HIT_JUDGEMENTS = ("perfect", "great", "good")
MISS_JUDGEMENTS = ("miss", "missed")
WRONG_NOTE_JUDGEMENTS = ("overstrum", "overhit", "ghost", "ghostinput", "wrong")


@dataclass(frozen=True)
class RewardConfig:
    hit: float = 1.0
    miss: float = -1.0
    wrong_note: float = -1.0
    #: reward at exactly PERFECT_MS of error; falls linearly to `window_edge` at HIT_WINDOW_MS
    perfect: float = 1.0
    window_edge: float = -1.0

    def timing(self, error_ms: float) -> float:
        """Reward for a correct note, scaled by how precisely it was timed.

        Inside PERFECT_MS it is full credit. Beyond HIT_WINDOW_MS it is indistinguishable
        from a miss and scores `miss`. In between it degrades linearly, crossing zero - so
        a right note played sloppily late is a genuine punishment, per the spec.
        """
        a = abs(float(error_ms))
        if a <= PERFECT_MS:
            return self.perfect
        if a >= HIT_WINDOW_MS:
            return self.miss
        t = (a - PERFECT_MS) / (HIT_WINDOW_MS - PERFECT_MS)
        return self.perfect + (self.window_edge - self.perfect) * t


DEFAULT = RewardConfig()


def reward(judgement: str, error_ms: float | None = None, config: RewardConfig = DEFAULT) -> float:
    """Reward for one note event.

    ``judgement`` is the game's verdict; ``error_ms`` is the signed offset for hits.
    """
    j = judgement.strip().lower().replace("_", "").replace(" ", "")

    if j in WRONG_NOTE_JUDGEMENTS:
        # Strummed or fretted with nothing there. Ignoring it would let a policy spam
        # input and collect every hit by accident.
        return config.wrong_note
    if j in MISS_JUDGEMENTS:
        return config.miss
    if j in HIT_JUDGEMENTS:
        if error_ms is None:
            raise ValueError(f"judgement {judgement!r} needs error_ms to score timing")
        return config.timing(error_ms)
    raise ValueError(f"unknown judgement {judgement!r}; expected one of "
                     f"{HIT_JUDGEMENTS + MISS_JUDGEMENTS + WRONG_NOTE_JUDGEMENTS}")


@dataclass
class AttemptReward:
    """Reward summary for a whole song attempt, from one logged attempt record."""

    hits: int
    misses: int
    overstrums: int
    ghost_inputs: int
    mean_error_ms: float | None
    total: float

    @property
    def notes(self) -> int:
        return self.hits + self.misses

    @property
    def per_note(self) -> float:
        return self.total / self.notes if self.notes else 0.0


def attempt_reward(
    notes_hit: int,
    notes_missed: int,
    overstrums: int = 0,
    ghost_inputs: int = 0,
    mean_error_ms: float | None = None,
    config: RewardConfig = DEFAULT,
) -> AttemptReward:
    """Total reward for an attempt.

    Timing is applied as the mean offset because the per-note offsets are not logged;
    with mean_error_ms=None every hit is scored at full credit. This is a summary, not a
    substitute for per-note rewards - see todo.md on the observation/timing stream.
    """
    hit_value = config.perfect if mean_error_ms is None else config.timing(mean_error_ms)

    total = (
        notes_hit * hit_value
        + notes_missed * config.miss
        + overstrums * config.wrong_note
        + ghost_inputs * config.wrong_note
    )
    return AttemptReward(
        hits=notes_hit,
        misses=notes_missed,
        overstrums=overstrums,
        ghost_inputs=ghost_inputs,
        mean_error_ms=mean_error_ms,
        total=total,
    )


def _main() -> int:
    """Preview the reward on real logged attempts."""
    import argparse

    from . import config, results

    ap = argparse.ArgumentParser(description="Per-note reward preview from the score log")
    ap.add_argument("--instrument", default="guitar")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    attempts = results.read_attempts()
    if not attempts:
        print(f"no attempts in {config.data_dir()}")
        return 1

    print(f"{'song':<22} {'hit':>4} {'miss':>4} {'over':>5} {'ghost':>6} {'total':>8} {'/note':>7}")
    for a in attempts[-args.limit:]:
        p = a.player(args.instrument)
        if p is None:
            continue
        ar = attempt_reward(
            notes_hit=p.notes_hit,
            notes_missed=p.notes_missed,
            overstrums=int(getattr(p, "overstrums", 0) or 0),
            ghost_inputs=int(getattr(p, "ghost_inputs", 0) or 0),
        )
        print(f"{a.song_name:<22} {ar.hits:>4} {ar.misses:>4} {ar.overstrums:>5} "
              f"{ar.ghost_inputs:>6} {ar.total:>8.1f} {ar.per_note:>7.3f}")

    print("\nTiming curve (correct note, error in ms -> reward):")
    for ms in (0, 10, 25, 40, 55, 70, 90):
        print(f"  {ms:>3} ms -> {DEFAULT.timing(ms):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
