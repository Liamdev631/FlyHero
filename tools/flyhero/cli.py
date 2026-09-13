"""Command line for the FlyHero control suite.

Run manually while wiring things up:

    python -m flyhero.cli songs
    python -m flyhero.cli queue "FlyHero Test One" "FlyHero Test Two"
    python -m flyhero.cli play --instrument guitar --difficulty easy
    python -m flyhero.cli best
    python -m flyhero.cli notes --instrument guitar
    python -m flyhero.cli attempts --limit 5
    python -m flyhero.cli doctor
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import config, queue as queue_mod, results, runner


def _cmd_songs(args: argparse.Namespace) -> int:
    songs = queue_mod.available_songs(args.song_dir)
    if not songs:
        print(f"no songs found in {args.song_dir or config.songs_dir()}")
        return 1
    for s in songs:
        print(s)
    return 0


def _cmd_queue(args: argparse.Namespace) -> int:
    path = queue_mod.write_queue(args.songs, song_dir=args.song_dir, path=args.output)
    print(f"wrote {len(args.songs)} song(s) -> {path}")
    print(path.read_text().rstrip())
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    songs = args.songs or None
    if songs is None and args.queue is None:
        songs = queue_mod.available_songs(args.song_dir)
        if not songs:
            print("no songs available to play", file=sys.stderr)
            return 1
        print(f"playing every song found ({len(songs)})")

    print(f"game: {config.game_binary()}")

    if args.detach:
        proc = runner.play(
            songs=songs,
            queue_path=args.queue,
            song_dir=args.song_dir,
            speed=args.speed,
            instrument=args.instrument,
            difficulty=args.difficulty,
            repeat=args.repeat,
            wait=False,
        )
        # wait=False always hands back a live process handle.
        assert isinstance(proc, subprocess.Popen)
        print(f"started detached, pid {proc.pid}")
        return 0

    out = runner.run_queue(
        songs=songs,
        instrument=args.instrument or "guitar",
        difficulty=args.difficulty or "expert",
        speed=args.speed,
        timeout=args.timeout,
    )
    print(f"exit status: {out['exit_status']}")
    print(f"attempts recorded this run: {len(out['new_attempts'])}")
    for a in out["new_attempts"]:
        p = a.player()
        # Percent is a 0..1 fraction, not 0..100.
        extra = f" [{p.instrument} {p.difficulty} {p.percent * 100:.1f}% {p.notes_hit}/{p.notes_total}]" if p else ""
        print(f"  {a.song_name}: {a.band_score}{extra}")
    return 0 if out["exit_status"] == 0 else out["exit_status"]


def _cmd_best(args: argparse.Namespace) -> int:
    best = results.best_scores()
    if not best:
        print("no scores recorded yet")
        return 1
    if args.json:
        print(json.dumps(best, indent=2))
        return 0
    for key, b in sorted(best.items(), key=lambda kv: -kv[1].get("BestScore", 0)):
        print(
            f"{b.get('SongName', key):<28} {b.get('BestScore', 0):>8} "
            f"{b.get('BestStars', 0)}*  {b.get('Plays', 0)} play(s)"
        )
    return 0


def _cmd_notes(args: argparse.Namespace) -> int:
    stats = results.note_stats(instrument=args.instrument)
    if not stats:
        print("no note statistics yet")
        return 1
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    print(f"{'song':<24} {'instrument':<16} {'difficulty':<9} {'hit':>5} {'miss':>5} {'acc':>7}")
    for s in sorted(stats.values(), key=lambda s: s["song_name"]):
        print(
            f"{s['song_name']:<24} {s['instrument']:<16} {s['difficulty']:<9} "
            f"{s['notes_hit']:>5} {s['notes_missed']:>5} {s['accuracy'] * 100:>6.2f}%"
        )
    return 0


def _cmd_attempts(args: argparse.Namespace) -> int:
    """List recent attempts. Percent values are 0..1 fractions in the log."""
    attempts = results.read_attempts()
    if not attempts:
        print("no attempts logged yet")
        return 1
    for a in attempts[-args.limit:]:
        p = a.player(args.instrument)
        extra = f" [{p.instrument} {p.difficulty} {p.notes_hit}/{p.notes_total}]" if p else ""
        print(f"{a.timestamp}  {a.song_name:<24} {a.band_score:>8}{extra}")
    print(f"\n{len(attempts)} attempt(s) total")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    """Turn charts into supervised (observation, action) pairs.

    Imported lazily so the rest of the CLI still works without numpy.
    """
    from . import dataset as ds_mod

    songs = args.songs or queue_mod.available_songs(args.song_dir)
    if not songs:
        print("no songs found", file=sys.stderr)
        return 1

    song_root = Path(args.song_dir) if args.song_dir else config.songs_dir()
    out = Path(args.output)

    bad = 0
    for name in songs:
        song_dir = song_root / name
        try:
            ds = ds_mod.build(
                song_dir,
                difficulty=args.difficulty,
                rate_hz=args.rate,
                horizon=args.horizon,
            )
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"{name}: SKIPPED ({e})")
            bad += 1
            continue

        v = ds_mod.verify(ds)
        path = ds_mod.save(ds, out)
        ok = v["strum_matches_note_count"] and v["onset_coverage_ok"]
        if not ok:
            bad += 1

        print(
            f"{v['notes_in_chart']:>3} notes -> {len(ds):>6} steps x {ds.obs_dim} obs  "
            f"strums {v['strum_events']:>3}  lanes {v['lane_hold_steps']}  "
            f"{'OK' if ok else 'MISMATCH'}  -> {path.name}"
        )

    return 1 if bad else 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Leave-one-song-out train/evaluate of the ANN on the chart datasets."""
    import numpy as np

    from . import ann as ann_mod

    files = sorted(Path(args.datasets).glob("*.npz"))
    if len(files) < 2:
        print(f"need at least 2 datasets in {args.datasets}; run `flyhero dataset` first",
              file=sys.stderr)
        return 1

    bundles = {}
    for p in files:
        d = np.load(p)
        bundles[p.stem] = (d["x"], np.concatenate([d["frets"], d["strum"][:, None]], axis=1))

    exit_code = 0
    for test_name in bundles:
        train_names = [n for n in bundles if n != test_name]
        train_x = np.concatenate([bundles[n][0] for n in train_names])
        train_y = np.concatenate([bundles[n][1] for n in train_names])
        test_x, test_y = bundles[test_name]

        print(f"\n=== hold out {test_name}  "
              f"(train {len(train_x)} steps on {len(train_names)} songs, test {len(test_x)})")
        metrics, _ = ann_mod.train_and_evaluate(
            train_x, train_y, test_x, test_y,
            hidden=args.hidden, epochs=args.epochs, pos_weight=args.pos_weight,
            seed=args.seed, verbose=args.verbose,
        )

        print(f"  {'output':<8} {'P':>6} {'R':>6} {'F1':>6}  {'acc':>6}  {'always-off acc':>14}  pos")
        for name, m in metrics.items():
            flag = ""
            # A strum model that never fires is worse than useless; say so.
            if name == "strum" and m["recall"] < 0.5:
                flag = "  <-- missed most notes"
            print(f"  {name:<8} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
                  f"{m['accuracy']:>6.3f}  {m['baseline_accuracy']:>14.3f}  {m['positives']:>3}{flag}")

        if metrics["strum"]["recall"] < 0.5:
            exit_code = 1

    return exit_code


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report whether each piece is in place, so failure is obvious up front."""
    ok = True

    print(f"repo:        {config.REPO}")
    print(f"songs dir:   {config.songs_dir()}")
    print(f"data dir:    {config.data_dir()}")
    print(f"queue file:  {config.queue_file()}")

    try:
        binary = config.game_binary()
        print(f"game binary: {binary}")
    except FileNotFoundError as e:
        print(f"game binary: MISSING - {e}")
        ok = False

    if config.UNITY_EDITOR.exists():
        print(f"unity:       {config.UNITY_EDITOR}")
    else:
        print(f"unity:       MISSING at {config.UNITY_EDITOR}")
        ok = False

    songs = queue_mod.available_songs()
    print(f"songs found: {len(songs)}")
    if songs:
        for s in songs:
            print(f"  - {s}")

    attempts = results.read_attempts()
    print(f"attempts logged: {len(attempts)}")
    best = results.best_scores()
    print(f"songs with a best score: {len(best)}")

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flyhero", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("songs", help="list songs the game will find")
    p.add_argument("--song-dir", default=None)
    p.set_defaults(func=_cmd_songs)

    p = sub.add_parser("queue", help="write a queue file")
    p.add_argument("songs", nargs="+")
    p.add_argument("--song-dir", default=None)
    p.add_argument("--output", "-o", default=None)
    p.set_defaults(func=_cmd_queue)

    p = sub.add_parser("play", help="run the game on a queue")
    p.add_argument("songs", nargs="*", help="omit to play every song found")
    p.add_argument("--queue", default=None, help="use an existing queue file")
    p.add_argument("--song-dir", default=None)
    p.add_argument("--instrument", default="guitar")
    p.add_argument("--difficulty", default=None)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--detach", action="store_true", help="do not wait for the run")
    p.add_argument("--timeout", type=float, default=None)
    p.set_defaults(func=_cmd_play)

    p = sub.add_parser("best", help="top score per song")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_best)

    p = sub.add_parser("notes", help="per-instrument note hits/misses")
    p.add_argument("--instrument", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_notes)

    p = sub.add_parser("attempts", help="recent attempts")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--instrument", default=None)
    p.set_defaults(func=_cmd_attempts)

    p = sub.add_parser("dataset", help="build supervised (observation, action) data from charts")
    p.add_argument("songs", nargs="*", help="song folder names; omit for all")
    p.add_argument("--song-dir", default=None)
    p.add_argument("--difficulty", default="Easy", choices=["Easy", "Medium", "Hard", "Expert"])
    p.add_argument("--rate", type=float, default=100.0, help="controller steps per second")
    p.add_argument("--horizon", type=float, default=1.0, help="observation look-ahead seconds")
    p.add_argument("--output", "-o", default=str(config.data_dir() / "datasets"))
    p.set_defaults(func=_cmd_dataset)

    p = sub.add_parser("train-ann", help="leave-one-song-out ANN training on chart datasets")
    p.add_argument("--datasets", default=str(config.data_dir() / "datasets"))
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--pos-weight", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("doctor", help="check the setup")
    p.set_defaults(func=_cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
