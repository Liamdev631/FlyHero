# Gameplay recordings

Captured from the **prebuilt YARG v0.15.0** running headless on beelink (Xvfb `:1`,
AMD GPU via RADV). See `tools/automation/run_prebuilt.sh` for the environment
workarounds required to get the game running on this host.

## gameplay_bot_random.mp4

50 seconds, 1024x768 @ 30fps, video only (this host has no usable hardware audio, so
songs play through a software `null` ALSA device — there is no sound to capture).

The song is one of the generated test tracks, `FlyHero Test Two`, played solo on
GUITAR / BEGINNER with a profile whose player is intended to be driven by an external
random-input policy.

**Caveat — read this before treating it as a baseline:** the score stays at **`0` for
the whole song**. See `frame_song_end_score0.png` (`0:26 / 0:27`, score `0`).

The random inputs did **not** register any hits. The cause is that the profile was
created through the UI but its bindings were never initialised — `profiles/bindings.json`
contains `"Profiles": {}`, so the player has a device assigned but no fret/strum
bindings, and gameplay actions map to nothing. Menu navigation still responded to
keyboard (via YARG's default keyboard *menu* bindings), which is why the menus in the
video react but the note highway does not.

So this recording demonstrates **a song playing end to end on this host**, not a bot
hitting notes. To get a real random-note baseline, initialise the profile's bindings
first (Profiles → the profile → Edit Binds), then re-run with the random policy on
frets `1`-`5` + strum `Up`/`Down` (the default keyboard five-fret bindings).

## Frames

| File | Shows |
| --- | --- |
| `frame_song_playing.png` | Note highway, five frets, timer `0:20 / 0:27`, score `0` |
| `frame_song_end_score0.png` | End of song, timer `0:26 / 0:27`, score still `0` |

These files are here only because the filesystem they were generated on is not
reachable from the chat client. Delete the `recordings/` directory once you've pulled
them.
