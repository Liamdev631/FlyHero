# Gameplay capture

Captured from the **prebuilt YARG v0.15.0** running headless on beelink (Xvfb `:1`,
AMD GPU via RADV). See `tools/automation/run_prebuilt.sh` for the environment
workarounds needed to get the game running on this host.

## gameplay_bot_random.mp4

50 seconds, 1024x768 @ 30fps, video only. This host has no usable hardware audio, so
songs play through a software `null` ALSA device and there is no audio to capture.

The song is `FlyHero Test Two` — one of the generated test tracks — played solo on
GUITAR / BEGINNER, driven by an **external random-input policy**: a stream of random
fret keys (`1`-`5`) each paired with a random strum (`Up`/`Down`), sent over XTEST.
That is the "random notes" bot: ~7.5 note attempts per second, no observation of the
screen at all.

### Result — the random policy scores

Taken from the in-game results screen after the run:

| Stat | Value |
| --- | --- |
| Score | **1,018** |
| Notes hit | **20 / 112** (~18%) |
| Accuracy | 17% |
| Max streak | 3 |
| Overstrums | 0 |

So blind random input lands roughly one in five notes. That is the number worth keeping
as the **random-policy baseline** for the planned PyTorch player to beat.

### A caveat about the video's timestamps

The default keyboard five-fret bindings collide with YARG's keyboard *menu* bindings
(frets `1`-`5` are also menu Green/Red/Yellow/Blue/Orange), so the random stream drives
the UI as well as the notes. In this recording that caused the policy to land on
`RESTART SONG` on the results screen, which restarted the song — the last ~10 seconds of
the video are therefore a second attempt (timer reading `0:08 / 0:27`) with the previous
attempt's score still on the HUD.

A future run should bind the frets to keys with no menu mapping so the policy can never
touch the UI, and must only start sending keys once the note highway is visibly scrolling.

## Frames

| File | Shows |
| --- | --- |
| `frame_song_playing.png` | Mid-song: highway, five frets, timer `0:20 / 0:27` |
| `frame_gameplay_score_1018.png` | Gameplay with **score 1,018**, `0:08 / 0:27` |
| `frame_results_20_of_112.png` | Results screen: score 1,018, notes 20/112, 17%, max streak 3 |

## Note

This directory exists only because the filesystem the capture was made on is not
reachable from the chat client. It can be deleted once the video has been pulled.
