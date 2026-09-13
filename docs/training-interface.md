# Training interface: what a guitar policy needs

Goal: train a policy that plays five-fret guitar, starting with a plain ANN and
moving to a spiking network after ANN pretraining. This note records what the
game already exposes, what is missing, and the planned hook.

## Status

| Layer | State |
| --- | --- |
| Queue songs + roll to the next song | **done** (`AutomationManager`, `-autoqueue`) |
| Per-attempt scores, per-instrument note hits/misses | **done** (`AutomationScoreStore` → `Players[]`) |
| Top-score dictionary per song | **done** (`top_scores.json`) |
| Python control of the above | **done** (`tools/flyhero/`) |
| Per-note observation/action stream for learning | **not yet** |
| Replaceable controller (policy drives input) | **not yet** |

The existing bot hits every note with a profile-level autoplay. That is fine for
exercising the queue, but a policy needs to *choose* input and see the
consequence, note by note.

## The two hooks to add

### 1. Observation stream (game → Python)

Emitted once per input update (~1 per frame, 60 Hz) and on note events. Enough
for a policy to know what is coming and what it just did:

```
t          song position (seconds, high precision)
notes[]    upcoming notes in a fixed horizon, e.g. 5 lanes x next 16 notes:
             lane, Δt until the hit window, length (sustain), is_hit
score      current score
combo      current combo
multiplier current multiplier
starpower  whether star power is active
health     rock meter value
```

Also useful, cheap to add: the last hit judgement (`Perfect`/`Great`/`Miss`
+ timing error in ms), which gives a dense per-note reward instead of a sparse
end-of-song one.

### 2. Action channel (Python → game)

The policy must be able to press the five frets and strum, and choose star
power. Two options:

- **Preferred:** a `RemoteInputDevice` implementing the same device interface
  the real input devices use, fed by JSON lines over a Unix socket or
  stdin/stdout. The policy then drives the game through its normal input path,
  so hit windows, scoring and timing behave exactly as for a human.
- Fallback: drive the game's virtual keyboard through synthetic X events. Much
  worse — timing jitter, and it needs a display.

Protocol sketch, one JSON object per line:

```
{"t": 12.345, "obs": { ...observation above... }}
```
```json
{"t": 12.401, "act": {"frets": ["green", "red"], "strum": true, "star_power": false}}
```

## Why guitar first

Five-fret guitar is the simplest action space in the game: 5 discrete frets plus
a strum bit, and a note chart with exact expected times. Drums and vocals add
simultaneous-hit and pitch-tracking problems respectively. Get one instrument
working end to end, then reuse the same observation/action plumbing for the
others (`Instrument` is already carried through the whole result path).

## Signal for the model

- **Reward:** per-note judgement (dense), with score delta as a secondary
  signal. `NotesHit`/`NotesMissed` per attempt is the eval metric.
- **Attack angle:** a chart is a schedule, so a policy can start by learning
  *when* to press, given a horizon of upcoming notes — a sequence problem more
  than a perception problem. That makes a plain ANN feasible before pixels are
  involved at all.
- **Next step after that:** add the rendered highway (screen frames) as an
  additional modality if we want the policy to learn from what the screen looks
  like rather than from structured note data. Structured note data is the
  cheaper and more reliable first modality.
