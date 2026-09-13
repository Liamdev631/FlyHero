# FlyHero unattended automation

Runs the game without a human: plays a queue of songs back to back, writes the score of
every attempt to a log, keeps a dictionary of the best score achieved per song, and
starts the next song as soon as one ends.

No input device and no player model is required to exercise the loop — songs are played
by a YARG **bot profile**, which skips input processing in the engine and hits every note
at its exact time (`YARG.Core/Engine/BaseEngine.Generic.cs`). That produces real, non-zero
scores, which is what makes the loop testable today and gives a baseline for the planned
PyTorch player to beat later.

## What was added

| Path | Purpose |
| --- | --- |
| `Assets/Script/Automation/AutomationManager.cs` | Boots straight into the queue, resolves songs, registers the bot player, advances on song end |
| `Assets/Script/Automation/AutomationScoreStore.cs` | Run log + top-score dictionary persistence |
| `Assets/Script/Persistent/CommandLineArgs.cs` | New `-auto*` arguments (modified) |
| `Assets/Script/Persistent/GlobalVariables.cs` | Skips the menu when a queue is supplied (modified) |
| `Assets/Script/Gameplay/GameManager.cs` | `EndSong()` logs the score and rolls into the next song instead of the score screen (modified) |
| `tools/automation/make_test_songs.py` | Generates reproducible test songs |
| `tools/automation/run_queue.sh` | Launches a built game with a queue, optionally recording video |
| `tools/automation/build_headless.sh` | Headless Unity build |

Only three existing files were touched, each with a small, clearly-commented hook.

## Command line arguments

| Argument | Meaning |
| --- | --- |
| `-autoqueue <path>` | JSON queue. **Presence of this argument enables automation mode.** |
| `-autosongdir <dir>` | Folder to scan for songs (otherwise the queue's `songDir`). |
| `-autodata <dir>` | Where logs go. Defaults to `<persistent data>/automation`. |
| `-autospeed <percent>` | Song speed, e.g. `150`. Default `100`. |
| `-autoinstrument <name>` | Default `FiveFretGuitar`. |
| `-autodifficulty <name>` | Default `Expert`. |
| `-autorepeat` | Loop the queue forever instead of stopping at the end. |
| `-autostayopen` | Return to the menu when finished instead of quitting. |

## Queue file

```json
{
  "songDir": "/path/to/songs",
  "songs": ["FlyHero Test One", "FlyHero Test Two"]
}
```

Each entry is matched against, in order: a full song folder path, the song folder's
name, `"Name - Artist"`, then the song title. A bare JSON array of song names is also
accepted (using `-autosongdir`).

## Outputs

Written to the automation data directory:

- **`score_log.jsonl`** — append-only, one JSON object per finished attempt: song
  identity, queue index, band score, stars, per-player score / notes hit / notes missed /
  max combo / percent / full-combo flag, and whether it beat the previous best.
- **`score_log.txt`** — the same events as readable one-liners.
- **`top_scores.json`** — the top-score dictionary, keyed by song hash:
  `{ "<hash>": { "SongName", "SongArtist", "BestScore", "BestPercent", "BestStars", "BestMaxCombo", "BestIsFc", "Plays", "FirstPlayed", "LastPlayed" } }`
- **`run_summary.txt`** — end-of-run table of best scores.

## Running it

```bash
# 1. Generate test songs (no copyrighted content, fully reproducible)
python3 tools/automation/make_test_songs.py /home/liamb/projects/FlyHero-build/songs

# 2. Play the queue, no recording
tools/automation/run_queue.sh \
    --game /path/to/FlyHero \
    --queue tools/automation/queue.example.json

# 3. Play the queue and record gameplay
tools/automation/run_queue.sh \
    --game /path/to/FlyHero \
    --queue tools/automation/queue.example.json \
    --record /home/liamb/projects/FlyHero-build/gameplay.mp4
```

## Building

```bash
tools/automation/build_headless.sh
```

Requires an **activated Unity 6000.3.5f2 license** on this machine. The editor binary is
installed at `/home/liamb/unity/6000.3.5f2/Editor/Unity` but currently reports:

```
No valid Unity Editor license found. Please activate your license.
```

This host runs newer system libraries than the editor expects, so two older ones are
unpacked under `/home/liamb/unity/libfix` and must be on `LD_LIBRARY_PATH` (the scripts
above do this for you):

- `libxml2.so.2` (system provides `so.16`) — Ubuntu 24.04 package
- ICU 74 (system provides 78) — Ubuntu 24.04 package

## Next steps

1. Activate a Unity licence, then run the build script.
2. Run the queue and confirm `score_log.jsonl` fills in and `top_scores.json` updates.
3. Replace the bot with the PyTorch player: `AutomationManager.RegisterBotPlayer()` is the
   single place that decides who is driving, and the logged per-attempt stats are the
   reward signal.
