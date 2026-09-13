# FlyHero — working TODO / handoff document

**Last updated:** 2026-09-13 (evening session)
**Repo:** `/home/liamb/projects/FlyHero` — a fork of `YARC-Official/YARG` (Unity 6000.3.5f2, C#)
**Build workspace:** `/home/liamb/projects/FlyHero-build/`

> This document is the durable record. If context was compacted, read this first —
> it carries the verified state, the exact commands, and the traps that cost the most time.

---

## 1. The goal

Train a neural network to play FlyHero (a Guitar Hero-style rhythm game), **guitar first**.
Start with a plain **ANN**, later migrate to a **spiking** model (after ANN pretraining).

To make that possible we first needed an **unattended automation harness**: queue songs,
play them headlessly, log scores and per-instrument note statistics, and roll automatically
into the next song. That harness is now built and verified.

The user cannot access this filesystem from their chat surface — deliverables must be pushed
to GitHub (`Liamdev631/FlyHero`) or shared via the media-server skill.

---

## 2. STATUS — what is verified working

### 2.1 The automation harness: WORKING (verified end to end)

A full 3-song queue ran headless with automatic rollover:

```
exit status: 0
attempts recorded this run: 3
  FlyHero Test One:   3650 [FiveFretGuitar Easy 100.0% 31/31]
  FlyHero Test Three: 2740 [FiveFretGuitar Easy 100.0% 26/26]
  FlyHero Test Two:   4810 [FiveFretGuitar Easy 100.0% 36/36]
```

Game log sequence proves each stage: `Automation queue: 3 entries` → `created bot profile
"FlyHero AutoBot"` → `Loading song FlyHero Test One` → `recorded ... score 3650` →
**`advancing to queue index 1`** → `... index 2` → `run complete: 3 attempt(s) logged
across 3 song(s)` → `exiting`.

**This is the rollover that mouse-driven UI automation could never achieve.**

### 2.2 The source build: WORKING

`FlyHero-build/build/Linux64/FlyHero` (517 MB) builds from source with the automation
hooks compiled in. Originally blocked by the Unity licence, then by ~272 NuGet errors.

### 2.3 Chart → dataset → ANN pipeline: WORKING (with a known weak spot)

Note counts from my chart parser match the game's own results screens exactly
(31 / 26 / 36), which cross-validates the parser.

Leave-one-song-out ANN results (pure-NumPy MLP, 128 hidden, 600 epochs, pos_weight 10):

| held-out song | frets (green..orange) F1 | strum F1 |
|---|---|---|
| FlyHero Test One | 0.85 – 0.94 | **0.47** |
| FlyHero Test Three | 0.95 – 0.97 | **0.00** (never fired) |
| FlyHero Test Two | 0.87 – 0.95 | 0.66 |

**Frets are learnable; strum is not yet reliable.** Strum is the timing-critical output and
the current blocker for real gameplay. See §8.

### 2.4 Data available for training

`FlyHero-build/automation-data/datasets/*.npz` — 3 datasets, 11-dim observation,
6-dim action (5 frets + strum), 100 Hz, 2,346–2,530 steps each.

---

## 3. Repository / workspace map

### 3.1 C# automation hooks (in the repo, `Assets/Script/Automation/`)

| File | Purpose |
|---|---|
| `AutomationManager.cs` | Queue loading, song resolution, bot player, rollover, run lifecycle |
| `AutomationScoreStore.cs` | Writes `score_log.jsonl`, `score_log.txt`, `top_scores.json`, `run_summary.txt` |

Patched upstream files (keep these patches in mind when rebasing):
- `Assets/Script/Persistent/CommandLineArgs.cs` — added the automation CLI args
- `Assets/Script/Persistent/GlobalVariables.cs` — calls `AutomationManager.Begin()`
- `Assets/Script/Gameplay/GameManager.cs` — calls `AutomationManager.OnSongFinished()`

### 3.2 Build + platform scripts (`tools/automation/`)

| Script | Purpose | ⚠️ Notes |
|---|---|---|
| `build_headless.sh` | Unity batchmode build → `FlyHero-build/build` | Needs the licence |
| `restore_nuget_python.py` | **Bootstrap** NuGet restore from outside the editor | Run FIRST; see §6 |
| `restore_nuget.sh` | In-editor restore via `NugetForUnity.PackageRestorer.Restore(false)` | Only works once compilation is clean |
| `patch_player_midi.sh` | Replaces the player's `libRtMidi.so` with a zero-port stub | **MUST re-run after EVERY build** |
| `rtmidi_stub.c` | The 22-entry-point MIDI stub (reports 0 ports) | Prevents a startup SIGSEGV |
| `make_test_songs.py` | Generates the 3 synthetic test songs | Charts span the audio length |

### 3.3 Python control suite (`tools/flyhero/`)

| Module | Purpose |
|---|---|
| `config.py` | Paths; all overridable via `FLYHERO_*` env vars |
| `queue.py` | Write/read the queue JSON the game consumes |
| `runner.py` | Assemble args, launch/stop the game |
| `results.py` | Parse `score_log.jsonl` / `top_scores.json` → attempts, best scores, note stats |
| `api.py` | The importable facade |
| `dataset.py` | Chart → supervised `(observation, action)` pairs + self-verification |
| `ann.py` | Dependency-free NumPy MLP for the cloning task |
| `cli.py` | `python -m flyhero.cli ...` |
| `README.md` | Usage reference |

### 3.4 Data locations

```
FlyHero-build/
  build/Linux64/FlyHero              ← the built game
  songs/FlyHero Test {One,Two,Three} ← 3 synthetic songs (notes.chart, song.ini, song.ogg)
  automation-data/
    queue.json                       ← current queue
    score_log.jsonl                  ← one JSON object per attempt (the training log)
    score_log.txt                    ← human-readable mirror
    top_scores.json                  ← dictionary: song key → best result ever
    run_summary.txt                  ← end-of-run summary + top-score table
    datasets/*.npz, *.json           ← chart-derived training data
  logs/                              ← Unity build + NuGet restore logs
```

Runtime game data: `~/.config/unity3d/YARC/YARG/` (`Player.log`, `scores/scores.db`, profiles).

---

## 4. How to run everything

### Build (after any C# change)

```bash
cd /home/liamb/projects/FlyHero
bash tools/automation/build_headless.sh      # ~5 min incremental
bash tools/automation/patch_player_midi.sh   # ⚠️ REQUIRED after every build
```

### Run a queue

```bash
cd /home/liamb/projects/FlyHero/tools
python3 -m flyhero.cli doctor                # sanity-check paths + song count
python3 -m flyhero.cli songs                 # list songs the game will find
python3 -m flyhero.cli queue "FlyHero Test One" "FlyHero Test Two"
python3 -m flyhero.cli play --instrument guitar --difficulty easy --timeout 400
```

`play` with no song arguments queues **every song found**. The run exits by itself when the
queue finishes (`-autostayopen` is only passed when `wait=False`).

### Inspect results

```bash
python3 -m flyhero.cli best                  # top score per song
python3 -m flyhero.cli notes                 # note hits/misses per instrument+difficulty
python3 -m flyhero.cli attempts --limit 10   # recent attempts
```

### Training data + ANN

```bash
python3 -m flyhero.cli dataset --difficulty Easy --rate 100
python3 -m flyhero.cli train-ann --epochs 600 --hidden 128
```

---

## 5. Reference: formats, schemas, arguments

### Game CLI arguments (declared in `CommandLineArgs.cs`)

| Argument | Meaning |
|---|---|
| `-autoqueue <file>` | Queue JSON; switches on automation mode |
| `-autosongdir <dir>` | Extra song folder to scan |
| `-autodata <dir>` | Output directory for logs/scores |
| `-autospeed <pct>` | Song speed, 100 = normal |
| `-autoinstrument <name>` | guitar / bass / drums / keys … |
| `-autodifficulty <name>` | easy / medium / hard / expert |
| `-autorepeat` | Re-run the queue when it finishes |
| `-autostayopen` | Stay running instead of exiting at the end |

### Queue JSON

```json
{ "songDir": "/path/to/songs", "songs": ["FlyHero Test One", "FlyHero Test Two"] }
```

Entries may be a folder name, `"Name - Artist"`, or a full folder path.

### Attempt record (`score_log.jsonl`, one object per line)

Song level: `Timestamp, SongKey, SongName, SongArtist, SongCharter, SongFolder, QueueIndex,
AttemptNumber, BandScore, BandStars, SongSpeed, SongLengthSeconds, NewHighScore,
PreviousBestScore, Players[]`

`Players[]` — **one entry per instrument**: `Name, Instrument, Difficulty, IsBot, Score,
Stars, NotesHit, NotesMissed, MaxCombo, Percent, IsFc`

### ⚠️ Data quirks

- **`Percent` is a 0–1 fraction, not 0–100.** `1.0` means 100%. Display code must multiply.
  (The log's own text summary prints `1.00%` — that is the C# formatting, not a real 1%.)
- **`IsFc` at band level requires ALL players to be FC.** A single clean player run gives
  `IsFc: true` on the player but may give `false` on the band.
- `SongSpeed` is a multiplier (`1.0` = normal).

### Chart format (`notes.chart`, Moonscraper/EOF text)

```
[Song]      { Resolution = 192  Offset = 0  ... }
[SyncTrack] { 0 = B 120000 }          # BPM = value/1000
[EasySingle] { 1485 = N lane sustain }  # tick = N lane sustain
```
`ticks_per_second = Resolution * BPM / 60` (here 192 × 120 / 60 = **384**).
Guitar sections are `<Difficulty>Single`; lanes 0–4 = green…orange.

### Observation / action space (current)

- **Observation (11):** for each of 5 lanes, seconds until the next note onset (clamped to a
  1 s horizon, 0 while holding) + seconds remaining in the hold; plus normalised song position.
- **Action (6):** 5 fret bits + 1 strum bit.
- 100 Hz controller rate (finer than the game's input update; keeps strum timing error < ~10 ms).

---

## 6. TRAPS — hard-won, do not rediscover

1. **Stale `songcache.bin` aborts every song.** If the song files change (regenerate charts),
   `~/.config/unity3d/YARC/YARG/release/songcache.bin` no longer matches and every song dies
   at load (`Loading song` → `Exiting song` in ~5 log lines, Gameplay scene torn down before
   it initialises, no score row written). **Fix:** `mv songcache.bin songcache.bin.stale` and
   restart. Verify by **reading the score DB**, not by looking at the screen.

2. **`-executeMethod` cannot run while any script has compile errors** — even a method in an
   assembly that doesn't reference the broken code. This is why the first NuGet restore MUST
   come from outside the editor (`restore_nuget_python.py`).

3. **NuGet TFM preference is backwards from .NET.** Unity runs Mono: `netstandard2.x` and
   `net472` work, but `net6.0`/`net8.0` fail with
   `CS1705: ... System.Runtime 8.0.0.0 ... higher than ... 4.1.2.0`. Rank `netstandard*`
   first, then `net4xx`, and `netN.N` **last**.

4. **NuGet package layouts vary.** `UniRx` ships `lib/UniRx.dll` with **no TFM subfolder**;
   `sqlite-net` is a *source* package (`content/SQLite.cs`) with no DLL. A strict
   `lib/<tfm>/*.dll` matcher silently skips both *while reporting success*. **A restore that
   reports success is not evidence the packages are usable — only compilation is.**

5. **`ManagedBass`'s `netstandard1.4` build omits members** (`Bass.VistaTruePlayPosition`)
   that `net45` has → `CS0117`. Inspect candidates with `b"SymbolName" in dll_bytes`.

6. **The build overwrites `libRtMidi.so`.** Without the stub, Minis' MIDI backend
   segfaults the player at startup (`rtmidi_get_port_count` ← `MidiBackend:OnUpdate`).
   Always run `patch_player_midi.sh` after a build.

7. **`SongContainer.RunRefresh(bool, LoadingContext? = null)`** accepts null but
   `SongSources.LoadSprites` dereferences it (`context.SetLoadingText`) → `NullReferenceException`.
   Automation must construct a `LoadingContext` itself.

8. **`YargLogger.LogFormatXxx<T1>` vs `<T1,T2>` is ambiguous** for multi-argument calls
   (`CS0121`). Use interpolated `YargLogger.LogXxx($"...")` instead. All such calls in the
   automation code were converted.

9. **`PathHelper.PersistentDataPath` is not public** (`CS0122`). Use
   `CommandLineArgs.PersistentDataPath`.

10. **`Convert.ToHexString` is .NET 5+** — absent from Mono. Use
    `BitConverter.ToString(b).Replace("-", "")`.

11. **A leftover player profile pollutes the band.** A previously saved human profile stays
    active and contributes a zero score, corrupting band aggregates and the all-players-FC
    flag. `RegisterBotPlayer` now disposes pre-existing players so the bot runs alone.

12. **Never `pkill -f "FlyHero"`** in a command whose own command line contains that string —
    it kills the shell running it (`exit -15`).

13. The display `:1` is **shared with a live Blender session**; the X server has restarted
    mid-run before. Prefer the headless `-autoqueue` path, which needs no input automation.

---

## 7. The data question — decided

The user directed that **`nftechie/doomfly`** be the basis for the trainable model, and asked
what data is best suited for training. Investigated (cloned at `/home/liamb/projects/doomfly`,
read-only) — conclusions:

**What doomfly is:** full MaleCNS v1.0 simulation (166,700 neurons, 25,582,938 directed edges,
124,177,617 contacts), a fixed hand-designed retina (3,335 brightness + 811 inferred colour
inputs), a **fixed engineered decoder** (DNp20 → turn, DNpe017 → move/fire), and a
dopamine-gated **anti-Hebbian** rule on **4,184 of 25.6 M edges (0.016%)**.

**Why it failed:** `docs/doom-live-training.md` states *"Neither health, enemy coordinates,
rewards nor memory metrics select controls"*, and `rule.py` states *"No game variables enter."*
There is **no credit assignment from the goal to the learning rule**; the decoder is fixed so
no policy is learned; the sensory encoding is an unvalidated proxy. Its v6 candidate failed
its visual, conditioning and survival gates, and `announcement_ready` is `false`.

**Decision — train as supervised behavioural cloning on chart-derived data.**
A rhythm chart is *ground truth*: every note states its lane and exact tick, so the correct
action is known with no environment interaction. Doom has no such labels; FlyHero does, for free.

- **Primary data:** chart-derived `(observation, action)` pairs (implemented in `dataset.py`).
- **Secondary data:** real in-game `(observation, action, judgement)` triples — dense per-note
  reward (`Perfect/Great/Good/Miss` + timing error in ms). Needed to close the sim-to-real gap;
  the results screen already reports **note offset average ≈ −15 ms**. Requires the per-note
  observation hook in §8.
- **Deferred:** pixels. doomfly's frame→retina proxy is its weakest link and only necessary
  because Doom offers nothing structured.

**Take from doomfly:** the full-graph sparse simulator + native C++ kernel, the MaleCNS
importer, and its **provenance discipline** (`datasets.json` + checksum-locked
`source.lock.json`, hashed checkpoints, published negative results).

**Replace:** the fixed motor decoder (make the readout trainable), the non-gradient plasticity
rule (ANN pretraining needs autograd), and the fixed sensory proxy (use structured note state).

---

## 7b. Observation and reward design (user-directed)

Two decisions from the user, both implemented:

### Crop frames to the instrument's highway (`vision.py`)

The game draws one highway **per instrument** plus shared HUD (score, stars, timer,
star-power meter, multiplier). Feeding the whole frame is wrong twice over: most pixels are
irrelevant, and the HUD is *shared* between instruments while the highway is not — so the same
pixels would mean different things for different instruments. Cropping to the instrument's own
highway fixes both and makes the input unambiguous per instrument.

- Measured region for a 1024×768 frame, guitar, 1 player: **x 330..700, y 415..768** (370×353).
  The highway is a perspective trapezoid (x 440–580 at the horizon, 330–700 at the fret line),
  so `Region` carries the top/bottom extents and interpolates the 5 lane centres.
- `to_policy_input()` crops → resizes to 84×84 → normalises to float32 **CHW** in [0,1].
- Verify a region with `python -m flyhero.vision --frame <png>` (writes `*_crop.png` and
  `*_overlay.png` beside the frame).
- **Caveat:** the `3×` multiplier badge is drawn *on the play field*, so it falls inside the
  crop. Either mask it or accept it as legitimate state (it does carry information).
- **Caveat:** regions are resolution/layout-specific. A different resolution, window mode,
  highway tilt or note speed invalidates them — re-measure.

### Reward from the game's own judgements (`reward.py`)

| behaviour | signal in our logs | reward |
|---|---|---|
| hit the note | `NotesHit` | `+1.0`, scaled down by timing error |
| miss the note | `NotesMissed` | `−1.0` |
| hit the wrong note | `Overstrums`, `GhostInputs` | `−1.0` |
| right note, wrong timing | note offset (we measured ≈ **−15 ms**) | falls from `+1.0` and goes negative |

Timing curve: full credit inside **25 ms**; linear falloff to **−1.0** at **70 ms**; crosses
zero at ≈47 ms. So a correct note played sloppily late is a genuine punishment, per the spec.

Using the game's judgements matters: a hand-rolled "was that right?" check would disagree with
the engine's hit windows, and the policy would end up optimising a different game than the one
being scored. `Overstrums`/`GhostInputs` are what stop a policy collecting every hit by
spamming input.

- Preview on real attempts: `python -m flyhero.reward --instrument guitar`.
- **Caveat:** `PERFECT_MS`/`HIT_WINDOW_MS` are placeholders. YARG's real windows vary by
  difficulty and hit-window settings and must be read from the engine before this is trusted.
- **Caveat:** timing is currently applied as the *mean* offset, because per-note offsets are not
  logged. Per-note timing needs the observation/timing stream in P2.

---

## 8. TODO — next steps, in order

### P0 — The synthetic cloning task is a strawman (diagnosed, needs reframing)

**Finding (decisive):** a two-line hand-coded rule scores **F1 0.973** (P 0.989, R 0.957)
on the strum label across all three songs, where the network scores 0.05–0.46:

```
rule: strum when min(lane delta) crosses INTO one step   ->  F1 0.984 / 0.941 / 0.986
net:                                                     ->  F1 0.464 / 0.074 / 0.054
```

Run it: `python3 -m flyhero.rule_baseline`.

**Why this matters:** the observation *contains the answer*. `seconds_until_next_note` has
the label as a direct function of it, so this dataset measures whether a net can rediscover
arithmetic that is already in its inputs — not whether it can learn to play. The earlier
"frets F1 0.85–0.97" numbers are inflated by exactly the same flaw and are not evidence of
learning either.

Two things were tried first and **neither fixed it** (keep the code, they are still correct):
frame stacking (`--stack 2`, because strum is a *transition* and invisible in one frame) and
per-output threshold calibration on a validation split. Stacking is nonetheless *required*:
the rule above needs the previous frame to express the transition.

**What to do instead:**

- [ ] Stop tuning this benchmark. It cannot fail in an informative way.
- [ ] Rebuild the observation from **what the game actually shows a player** — note-highway
      state (or pixels) with realistic latency/jitter — so the label is *not* a closed-form
      function of the input. That is the task a policy must actually solve.
- [ ] Keep the rule as a **deterministic baseline**: F1 0.97 offline means it should play the
      chart near-perfectly in game, which makes it an excellent test of the in-game interface
      (P2) — if the rule scores ~100% in game, the plumbing is right and any shortfall is the
      policy's.
- [ ] Re-evaluate the ANN only against the game-derived observation.

### P0b (original, superseded) — strum reliability

Strum was the suspected blocker: net F1 0.47 / 0.00 / 0.66, and on Test Three it never fired.
Superseded by the finding above — the net's failure was a symptom of the strawman dataset, not
of the strum output being intrinsically hard.

### P1 — Install PyTorch and move to the real ANN

- [ ] Install `torch` (currently absent — the NumPy MLP in `ann.py` is a stopgap).
- [ ] Port the model, add a proper train/val split and per-note metrics.
- [ ] Target: beat the random-input baseline (score 896, 20/112 notes) — see §9.

### P2 — Per-note observation stream + policy-driven controller (the missing hooks)

`docs/training-interface.md` specifies both. Today the bot hits every note by *skipping input
processing in the engine* — that is not a learnable controller.

- [ ] `RemoteInputDevice`: a device implementing the same interface as real input devices, fed
      by JSON lines over a Unix socket (or stdin/stdout), so the policy drives input through the
      game's normal path and hit windows/scoring behave exactly as for a human.
- [ ] Observation emitter: per input update (~60 Hz), emit the note horizon (lane, Δt, sustain),
      score, combo, multiplier, star power, health, and the last hit judgement + timing error.
- [ ] Dense per-note reward from the judgement (this is what doomfly lacks).

Protocol sketch (one JSON object per line):
```
{"t": 12.345, "obs": {...}}
{"t": 12.401, "act": {"frets": ["green","red"], "strum": true, "star_power": false}}
```

### P3 — Training loop

- [ ] Behavioural-clone the chart policy, then fine-tune against in-game judgement data.
- [ ] Evaluate with the automation harness (queue → play → `flyhero best` / `flyhero notes`).
- [ ] Only then consider the SNN migration (surrogate gradients / BPTT through time — note that
      **no repo in the awesome-fly list does this**; doomfly has no gradient path at all).

### P4 — Housekeeping

- [ ] Re-apply `patch_player_midi.sh` after every future build (add it to `build_headless.sh`?).
- [ ] Commit the Python suite + scripts (`tools/flyhero/`, `dataset.py`, `ann.py`, new scripts)
      — currently uncommitted.
- [ ] Clean the junk captures in `FlyHero-build/*.mp4` and stale screenshots.
- [ ] Consider wiring the MIDI stub + NuGet bootstrap into `build_headless.sh` so a fresh clone
      builds in one command.

---

## 9. Baselines to beat

| Source | Result |
|---|---|
| Random input (mouse-spam era) | score **896**, 20/112 notes, 51%, 3 stars |
| Perfect bot (autoplay profile) | **100%**, FC, scores 3650 / 2740 / 4810 on the 3 test songs |
| Current ANN (offline cloning) | frets F1 0.85–0.97; strum F1 0.00–0.66 |

The random-input run is the honest floor for a learned policy; the bot is the ceiling.

---

## 10. Environment facts

- Unity editor: `/home/liamb/unity/6000.3.5f2/Editor/Unity`
  (needs `LD_LIBRARY_PATH=/home/liamb/unity/libfix/usr/lib/x86_64-linux-gnu`)
- Unity licence: **working** — the client log reports `Successfully parsed license`, `found: 2
  groups`, `Found 1 entitlements`.
- Display: `:1` (1024×768), shared with Blender.
- Audio: PulseAudio null sink **FakeAudio**; `~/.asoundrc` routes ALSA→Pulse
  (backup at `~/.asoundrc.pulse.bak`). Capture the monitor to record game audio.
- `sudo` is passwordless for `apt`/`systemctl`.
- Python: `python3` 3.11 with **numpy only** (no torch, no scipy, no pyarrow).
- The game writes `Player.log` to `~/.config/unity3d/YARC/YARG/Player.log` — the first place to
  look when automation misbehaves.

---

## 11. Open questions / risks

1. **Compute scale.** If the connectome substrate is adopted (166,700 neurons at 0.1 ms
   integration, 25.6 M edges), that dominates training cost. A rhythm game runs for minutes
   and we want many episodes — measure before committing.
2. **Sim-to-real gap.** Offline chart cloning ignores the game's actual hit windows, input
   latency and audio offset. The −15 ms measured offset is a real correction to apply.
3. **Test songs are synthetic.** 31–36 notes at Easy. They validate the pipeline but are far
   from real charts; the parser and the model should be checked on real Moonscraper/Clone Hero
   charts before drawing conclusions.
4. **The strum problem is unsolved** (§8 P0) and gates everything downstream.
