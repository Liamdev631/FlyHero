"""Drive fly-brain's whole-brain LIF model from the game's instrument highway.

WHY THIS EXISTS
---------------
flyvis models the optic lobe only and contains no motor neurons, so it cannot answer
"which button should be pressed". fly-brain (`eonsystemspbc/fly-brain`) is a
**whole-brain** spiking model - a leaky integrate-and-fire network built from the
FlyWire v783 connectome (~138,639 neurons, ~15.1M signed synapses), following Shiu et
al. (Nature 2024), which reports 91% accuracy against experimental recordings.

Its stimulus API (`code/paper-phil-drosophila/model.py::run_exp`) takes sets of neurons
to drive with Poisson input:

    neu_exc  -> Poisson at r_poi  (default 150 Hz)
    neu_exc2 -> Poisson at r_poi2 (a second rate)
    neu_slnc -> silenced
    f_poi    -> synaptic scaling; 250 is documented as "sufficient to cause spiking"

That is the hook for the game: map the cropped highway onto the photoreceptor
('optic') neurons by their soma position and drive the bright ones.

WHAT THE READOUT SHOULD BE
--------------------------
Not the optic lobe. The brain's *output* to the body is the **descending neurons**
(super_class='descending', 1,299 of them in this dataset) - they are the only neurons
that leave the brain for the nerve cord. fly-brain also contains 110 `motor` neurons,
but those are brain motor neurons (eye, neck, antenna, proboscis); **leg** motor neurons
are in the VNC and are not in FlyWire v783. So the full path is:

    game pixels -> photoreceptors -> whole-brain LIF -> descending neurons
      -> [bridge: male-CNS connectome] -> VNC leg motor neurons
      -> left foreleg MNs = 5 fret buttons, right foreleg MNs = strum

`side` ('left'/'right') is available for every neuron, which is what makes the
left-buttons / right-fret split expressible.

Usage:
    python -m flyhero.flybrain_stimulus --frame <png> --out <dir> [--duration 0.1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

# fly-brain's own directory layout
FLYBRAIN_DIR = Path("/home/liamb/projects/fly-brain")
COMPLETENESS = FLYBRAIN_DIR / "data/2025_Completeness_783.csv"
CONNECTIVITY = FLYBRAIN_DIR / "data/2025_Connectivity_783.parquet"
ANNOTATIONS = FLYBRAIN_DIR / "data/flywire_annotations.tsv"


def load_annotations(path: Path = ANNOTATIONS):
    """Load the FlyWire v783 annotations.

    Coordinates: use ``pos_x/pos_y/pos_z`` (the neuron's skeleton position), NOT
    ``soma_*``. Coverage is the reason: pos_* is populated for 100% of neurons, while
    soma_* is populated for only 0.1% of photoreceptors (85% overall) - using soma_*
    silently reduced the retina to 10 neurons.
    """
    import pandas as pd

    ann = pd.read_csv(
        path, sep="\t", low_memory=False,
        usecols=["root_id", "super_class", "cell_class", "cell_sub_class",
                 "cell_type", "side", "pos_x", "pos_y", "pos_z"],
    )
    return ann


def load_neuron_index(path: Path = COMPLETENESS):
    """FlyWire root id <-> position in fly-brain's simulation index.

    fly-brain's completeness CSV defines the simulation order, so index i in the
    simulation corresponds to row i of this file.
    """
    import pandas as pd

    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype("int64").to_numpy()
    flyid2i = {int(j): i for i, j in enumerate(ids)}
    i2flyid = {i: int(j) for i, j in enumerate(ids)}
    return ids, flyid2i, i2flyid


def visual_input_neurons(ann, brain_ids: set[int]):
    """Photoreceptors in the fly-brain simulation - the injection point for the frame.

    Driving the retina (R1-6 broadband, R7/R8 chromatic) is the biologically correct
    entry point: they are the first stage of the visual system, so the frame enters where
    light would. Note the FlyWire 'optic' super_class is large (77k) but those are
    optic-lobe interneurons - driving 77k of them would inject the image *after* the
    retina's own processing, which is a different (and misleading) experiment.
    """
    ct = ann["cell_type"].astype(str)
    is_photo = ct.isin(["R1-6", "R7", "R8"]) | ct.str.startswith(("R1", "R2", "R3", "R4", "R5", "R6"))
    cand = ann[is_photo]
    return cand[cand["root_id"].isin(brain_ids)]


def map_frame_to_neurons(frame_xy: np.ndarray, soma_x: np.ndarray, soma_y: np.ndarray,
                         side: np.ndarray, threshold: float) -> tuple[list[int], list[int]]:
    """Map a 2-D luminance image onto neuron soma positions -> (bright, dim) neuron masks.

    The fly's retina is roughly a 2-D array in (x, y) with z ~ constant, so we project
    soma positions to (x, y), normalise to [0, 1], and sample the image at that point.
    Neurons whose sampled luminance exceeds ``threshold`` go into the 'bright' set
    (driven at the higher Poisson rate), the rest into the 'dim' set.

    This is a deliberately simple encoder: a two-level (bright/dim) spatial drive, which
    is what fly-brain's two Poisson rates support. It is not an attempt to reproduce
    photoreceptor optics - see the note in the module docstring of the report.
    """
    h, w = frame_xy.shape
    # Normalise the image to 0-1 first: the crop arrives in 0-255 units, and comparing
    # 0-255 values against a 0-1 threshold marks every neuron as bright (which silently
    # drives the entire retina at the high rate).
    frame_xy = np.asarray(frame_xy, dtype=np.float64)
    lo, hi = np.nanmin(frame_xy), np.nanmax(frame_xy)
    frame_xy = np.zeros_like(frame_xy) if hi <= lo else (frame_xy - lo) / (hi - lo)
    x = np.asarray(soma_x, dtype=np.float64)
    y = np.asarray(soma_y, dtype=np.float64)
    # normalise each axis to [0,1] across the neuron population
    def norm(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        return np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)

    xi = np.clip((norm(x) * (w - 1)).round().astype(int), 0, w - 1)
    yi = np.clip((norm(y) * (h - 1)).round().astype(int), 0, h - 1)
    sampled = frame_xy[yi, xi]
    bright = sampled >= threshold
    return np.nonzero(bright)[0].tolist(), np.nonzero(~bright)[0].tolist()


def crop_frame_to_array(frame_path: Path, size: int = 64):
    """Crop the game frame to the instrument highway and return a luminance array."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flyhero.flyvis_brain import _crop_resize_grey
    from flyhero.vision import crop_highway, load_frame, region_for

    full = load_frame(frame_path)
    region = region_for(full.shape[1], full.shape[0], "guitar")
    return _crop_resize_grey(full, region, (size, size))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=Path, required=True, help="full game frame PNG")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=0.1, help="simulated seconds")
    ap.add_argument("--threshold", type=float, default=0.35, help="bright/dim split on 0-1 luminance")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="compute the drive but do not simulate")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    ann = load_annotations()
    ids, flyid2i, _ = load_neuron_index()
    brain_ids = set(int(i) for i in ids)
    print(f"fly-brain neurons: {len(ids)}   annotated: {len(ann)}")

    vis = visual_input_neurons(ann, brain_ids)
    print(f"optic neurons present in fly-brain: {len(vis)}")
    print("  by cell_class:", dict(vis["cell_class"].astype(str).value_counts().head(6)))

    lum = crop_frame_to_array(args.frame, args.size)
    print(f"crop: {lum.shape} range {lum.min():.1f}-{lum.max():.1f} (0-255)")

    # lateralise: left retina neurons get the mirrored frame, matching the fly's optics
    vis = vis.reset_index(drop=True)
    # Drop neurons with non-finite soma coordinates BEFORE mapping. A single NaN poisons
    # the min/max normalisation, so every sampled luminance becomes NaN, `NaN >= threshold`
    # is False, and the whole retina silently ends up in the 'dim' set - the fly is shown
    # nothing and nothing raises.
    finite = np.isfinite(vis["pos_x"].to_numpy()) & np.isfinite(vis["pos_y"].to_numpy())
    n_dropped = int((~finite).sum())
    vis = vis[finite].reset_index(drop=True)
    print(f"  photoreceptors with usable soma coords: {len(vis)} (dropped {n_dropped} non-finite)")
    sides = vis["side"].astype(str).to_numpy()

    report: dict = {"n_optic": int(len(vis)), "threshold": args.threshold,
                    "frame_range": [float(lum.min()), float(lum.max())]}

    for side in ("left", "right"):
        m = sides == side
        if not m.any():
            continue
        sub = vis[m]
        img = lum[:, ::-1] if side == "right" else lum
        bright_i, dim_i = map_frame_to_neurons(
            img, sub["pos_x"].to_numpy(), sub["pos_y"].to_numpy(), sides[m], args.threshold)
        bright_ids = [int(sub["root_id"].iloc[i]) for i in bright_i]
        dim_ids = [int(sub["root_id"].iloc[i]) for i in dim_i]
        report[side] = {"bright": len(bright_ids), "dim": len(dim_ids),
                        "bright_example": bright_ids[:5]}
        print(f"  {side}: {len(sub)} optic neurons -> {len(bright_ids)} bright (driven), {len(dim_ids)} dim")

    (args.out / "stimulus_map.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out/'stimulus_map.json'}")
    if args.dry_run:
        return 0
    print("\nNext: pass these bright sets as neu_exc / neu_exc2 into fly-brain's run_exp().")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
