# FlyHero control suite

Python control surface for the FlyHero (YARG) source build. It queues songs,
runs the game unattended, and reads back scores/note statistics. Nothing here
needs a mouse or a visible window — the game drives itself from the queue file.

```
tools/flyhero/
  config.py    paths (env-overridable: FLYHERO_*, see below)
  queue.py     write/read the queue JSON the game consumes
  runner.py    assemble args and launch/stop the game
  results.py   parse score_log.jsonl, top_scores.json, run_summary.txt
  cli.py       `python -m flyhero.cli`
```

## Quick start (run from `tools/`)

```bash
cd /home/liamb/projects/FlyHero/tools

python3 -m flyhero.cli doctor          # is everything wired up?
python3 -m flyhero.cli songs           # what the game will find
python3 -m flyhero.cli queue "FlyHero Test One" "FlyHero Test Two"
python3 -m flyhero.cli play --instrument guitar --difficulty easy
python3 -m flyhero.cli best            # top score per song
python3 -m flyhero.cli notes           # note hits/misses per instrument
python3 -m flyhero.cli attempts --limit 5
```

`play` with no song arguments queues every song found.

## From Python

```python
from flyhero import api, results

api.queue(["FlyHero Test One", "FlyHero Test Two"])   # write the queue
out = api.play_queue(instrument="guitar", difficulty="easy")
for a in out["new_attempts"]:
    p = a.player("guitar")
    print(a.song_name, a.band_score, p.notes_hit, p.notes_total, f"{p.percent:.1f}%")

print(api.best_scores())     # {song_key: {BestScore, BestStars, Plays, ...}}
print(api.note_stats())      # per song/instrument/difficulty accuracy
```

Live driving (for a future policy loop) uses `wait=False`, which leaves the game
running and returns the process handle:

```python
from flyhero import api
proc = api.play(["FlyHero Test One"], wait=False)
...
api.stop(proc)
```

## What the game gives us

The game is built from source with automation hooks
(`Assets/Script/Automation/`), and writes to `-autodata`:

| File | Contents |
| --- | --- |
| `score_log.jsonl` | one JSON object per finished attempt — append-only |
| `score_log.txt` | the same events, human readable |
| `top_scores.json` | dictionary: song key → best result ever achieved |
| `run_summary.txt` | end-of-run summary with the top-score table |

Each attempt carries a `Players[]` array with **one entry per instrument**:
`Instrument`, `Difficulty`, `Score`, `Stars`, `NotesHit`, `NotesMissed`,
`MaxCombo`, `Percent`, `IsFc`. That is the per-instrument training signal.

## Command line arguments

Declared in `Assets/Script/Persistent/CommandLineArgs.cs`:

| Argument | Meaning |
| --- | --- |
| `-autoqueue <file>` | queue JSON; switches on automation mode |
| `-autosongdir <dir>` | song folder to scan |
| `-autodata <dir>` | output directory for logs/scores |
| `-autospeed <pct>` | song speed, 100 = normal |
| `-autoinstrument <name>` | guitar, bass, drums, keys, … |
| `-autodifficulty <name>` | easy, medium, hard, expert |
| `-autorepeat` | run the queue again when it finishes |
| `-autostayopen` | stay running instead of exiting at the end |

## Environment overrides

| Variable | Default |
| --- | --- |
| `FLYHERO_REPO` | `/home/liamb/projects/FlyHero` |
| `FLYHERO_WORK` | `/home/liamb/projects/FlyHero-build` |
| `FLYHERO_DATA` | `<work>/automation-data` |
| `FLYHERO_SONGS` | `<work>/songs` |
| `FLYHERO_GAME_BINARY` | auto-detected (source build, else `prebuilt/YARG`) |
| `FLYHERO_DISPLAY` | `:1` |

## Building the game

```bash
bash tools/automation/build_headless.sh     # -> FlyHero-build/build
```

The suite prefers a source build when `FLYHERO_BUILD_OUT` contains one, and
falls back to `FlyHero-build/prebuilt/YARG`.
