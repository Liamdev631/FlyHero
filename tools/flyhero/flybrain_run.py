"""Run fly-brain's whole-brain LIF model driven by the game's instrument highway.

This closes the loop: the cropped game frame selects photoreceptors, those photoreceptors
become Poisson-driven inputs to fly-brain's 138,639-neuron LIF network, and we read the
spikes of the **descending neurons** - the brain's motor-command output to the nerve cord.

Model and API are upstream (`code/paper-phil-drosophila/model.py`):
    run_trial(exc, exc2, slnc, path_comp, path_con, params) -> {neuron_index: spike_times}
    poi(neu, exc, exc2, params) installs PoissonInput on the `exc`/`exc2` index sets.

brian2 incompatibility note: brian2 uses `np.ndarray.ptp`, removed in numpy 2. It must run
from a venv with numpy<2 - `/home/liamb/venvs/flybrain` (numpy 1.26.4 + brian2 2.9.0).

Usage:
    /home/liamb/venvs/flybrain/bin/python -m flyhero.flybrain_run \
        --frame /tmp/verify_14.png --t-run 10 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

FLYBRAIN = Path("/home/liamb/projects/fly-brain")
PAPER_DIR = FLYBRAIN / "code/paper-phil-drosophila"
COMPLETENESS = FLYBRAIN / "data/2025_Completeness_783.csv"
CONNECTIVITY = FLYBRAIN / "data/2025_Connectivity_783.parquet"
ANNOTATIONS = FLYBRAIN / "data/flywire_annotations.tsv"


def load_model():
    """Import fly-brain's paper model (its own directory on sys.path)."""
    if str(PAPER_DIR) not in sys.path:
        sys.path.insert(0, str(PAPER_DIR))
    import model  # type: ignore

    return model


def build_params(model, t_run_ms: float, n_run: int, r_poi_hz: float, r_poi2_hz: float):
    """Default model params with the run duration, plus the two Poisson rates.

    Rates: the bright photoreceptors are driven at r_poi (default 150 Hz, which the
    upstream docs describe as sufficient to spike), and the dim ones at the much lower
    r_poi2 - that two-level split is the entire visual encoding here.
    """
    from brian2 import Hz, ms

    params = dict(model.default_params)
    params["t_run"] = t_run_ms * ms
    params["n_run"] = n_run
    params["r_poi"] = r_poi_hz * Hz
    params["r_poi2"] = r_poi2_hz * Hz
    return params


def descending_indices(ann=None) -> dict:
    """Simulation indices of descending neurons, split by side.

    Uses fly-brain's completeness order: index i in the simulation == row i of the
    completeness CSV, so the annotation join has to go through FlyWire root_id.
    """
    import pandas as pd

    comp = pd.read_csv(COMPLETENESS)
    ids = comp.iloc[:, 0].astype("int64").to_numpy()
    flyid2i = {int(j): i for i, j in enumerate(ids)}

    a = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False,
                    usecols=["root_id", "super_class", "side", "cell_type"])
    dn = a[a["super_class"].astype(str) == "descending"]
    out = {"left": [], "right": [], "center": []}
    for _, row in dn.iterrows():
        i = flyid2i.get(int(row["root_id"]))
        if i is None:
            continue
        side = str(row["side"])
        out.setdefault(side, []).append(i)
    return out


def run_standalone(model, params, exc: list, exc2: list, build_dir: Path) -> dict:
    """Run the model in Brian2 **cpp_standalone** mode: generate C++, build, then execute.

    This is the mode fly-brain's own benchmark uses for its Brian2 (CPU) numbers, and the
    difference is enormous. ``model.run_trial`` uses Brian2's *runtime* mode, which
    interprets the equations per timestep in Python; with ~15M synapses that is hundreds of
    times slower (a 10 ms run was still going after 22 minutes). Compiled standalone builds
    the network once and then runs native code, so the same work takes seconds.

    Order matters: ``net.run(duration=...)`` declares the duration, ``device.build(run=False)``
    compiles without executing, then ``device.run()`` executes - so the build is paid once
    and repeated frames cost only the run.
    """
    from brian2 import Network, device as brian_device, set_device

    brian_device.reinit()
    brian_device.activate()
    set_device("cpp_standalone", build_on_run=False)

    neu, syn, spk_mon = model.create_model(COMPLETENESS, CONNECTIVITY, params)
    poi_inp, neu = model.poi(neu, exc, exc2, params)
    if not spk_mon:
        raise RuntimeError("standalone mode needs the SpikeMonitor for the DN readout")
    net = Network(neu, syn, spk_mon, *poi_inp)
    net.run(duration=params["t_run"])

    if build_dir.exists():
        import shutil

        shutil.rmtree(build_dir)
    brian_device.build(directory=str(build_dir), run=False, with_output=False)
    brian_device.run(with_output=False)
    return model.get_spk_trn(spk_mon)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=Path, required=True, help="full game frame PNG")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t-run", type=float, default=10.0, help="simulated milliseconds")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--r-poi", type=float, default=150.0, help="Poisson Hz for the bright (driven) set")
    ap.add_argument("--r-poi2", type=float, default=15.0, help="Poisson Hz for the dim set")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", choices=("standalone", "runtime"), default="standalone",
                    help="brian2 cpp_standalone (compiled, default) or runtime (interpreted, slow)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. stimulus: which photoreceptors the frame lights up
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flyhero.flybrain_stimulus import (crop_frame_to_array, load_annotations,
                                           load_neuron_index, map_frame_to_neurons,
                                           visual_input_neurons)

    ann = load_annotations()
    ids, flyid2i, _ = load_neuron_index()
    brain_ids = set(int(i) for i in ids)
    vis = visual_input_neurons(ann, brain_ids).reset_index(drop=True)
    finite = np.isfinite(vis["pos_x"].to_numpy()) & np.isfinite(vis["pos_y"].to_numpy())
    vis = vis[finite].reset_index(drop=True)

    lum = crop_frame_to_array(args.frame, 64)
    sides = vis["side"].astype(str).to_numpy()
    bright_ids: list[int] = []
    dim_ids: list[int] = []
    for side in ("left", "right"):
        m = sides == side
        if not m.any():
            continue
        sub = vis[m]
        img = lum[:, ::-1] if side == "right" else lum
        b, d = map_frame_to_neurons(img, sub["pos_x"].to_numpy(), sub["pos_y"].to_numpy(),
                                    sides[m], args.threshold)
        bright_ids += [int(sub["root_id"].iloc[i]) for i in b]
        dim_ids += [int(sub["root_id"].iloc[i]) for i in d]
    print(f"drive: {len(bright_ids)} bright photoreceptors @ {args.r_poi} Hz, "
          f"{len(dim_ids)} dim @ {args.r_poi2} Hz")

    exc = [flyid2i[i] for i in bright_ids if i in flyid2i]
    exc2 = [flyid2i[i] for i in dim_ids if i in flyid2i]
    print(f"simulation indices: exc={len(exc)} exc2={len(exc2)}")

    dn = descending_indices()
    print(f"descending neurons in sim: left {len(dn['left'])}, right {len(dn['right'])}, "
          f"center {len(dn.get('center', []))}")

    if args.dry_run:
        (args.out / "drive.json").write_text(json.dumps(
            {"exc": len(exc), "exc2": len(exc2), "t_run_ms": args.t_run,
             "dn": {k: len(v) for k, v in dn.items()}}, indent=2), encoding="utf-8")
        return 0

    # 2. run the model
    model = load_model()
    params = build_params(model, args.t_run, 1, args.r_poi, args.r_poi2)
    print(f"running fly-brain LIF ({args.mode}): {args.t_run} ms over 138,639 neurons ...")
    import time as _time

    t0 = _time.perf_counter()
    if args.mode == "standalone":
        spk = run_standalone(model, params, exc, exc2, args.out / "standalone-build")
    else:
        spk = model.run_trial(exc, exc2, [], COMPLETENESS, CONNECTIVITY, params)
    wall = _time.perf_counter() - t0
    spk = spk or {}
    print(f"simulation wall time: {wall:.2f}s for {args.t_run} ms of brain time")
    print(f"neurons that spiked: {len(spk) if spk else 0}")

    # 3. read out the descending neurons
    report: dict = {"t_run_ms": args.t_run, "n_spiking": len(spk),
                    "n_exc": len(exc), "n_exc2": len(exc2),
                    "dn": {}, "total_spikes": int(sum(len(v) for v in spk.values()))}
    for side, idxs in dn.items():
        if not idxs:
            continue
        counts = [len(spk[i]) for i in idxs if i in spk]
        report["dn"][side] = {
            "n_neurons": len(idxs),
            "n_active": len(counts),
            "total_spikes": int(sum(counts)),
            "mean_rate_hz": float(sum(counts) / len(idxs) / (args.t_run / 1000.0)) if idxs else 0.0,
        }
        print(f"  DN {side:<6}: {len(counts)}/{len(idxs)} active, {sum(counts)} spikes, "
              f"{report['dn'][side]['mean_rate_hz']:.2f} Hz mean")

    (args.out / "dn_readout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(args.out / "dn_spikes.npy", np.array(
        [(i, len(v)) for i, v in spk.items()], dtype=np.int64))
    print(f"wrote {args.out/'dn_readout.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
