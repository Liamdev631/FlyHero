"""Side-by-side figure: game frame -> encoded receptors -> network output activity.

Follows the custom-stimulus notebook (`examples/07_flyvision_providing_custom_stimuli`)
but lays the stages out in one image:

  1. the game frame, cropped to the instrument's highway (what we choose to show the fly)
  2. the greyscale crop (the actual pixels going in)
  3. the ENCODED image - ``BoxEye`` receptor activations on the fly's hexagonal lattice,
     which is the notebook's "rendered frame", plotted with the same hex convention
     (``get_hex_coords`` / ``hex_to_pixel``) that the notebook itself uses.
  4. INPUT cells - the photoreceptors the connectome declares as inputs (``R1``-``R8``)
  5. OUTPUT cells - the cell types the connectome declares as outputs (34 types incl.
     ``T4a``-``T4d`` and ``T5a``-``T5d``, the direction-selective motion detectors)
  6. output time courses across the stimulus

A NOTE ON "MOTOR": flyvis has **no motor neurons** - grepping the whole package for
"motor" returns nothing. It models the optic lobe, ending at the output types above. In
the real fly T4/T5 are the last stage before the lobula plate tangential cells that drive
optomotor behaviour, so T4/T5 are the closest thing to a "motor-relevant" output the
package provides - but they are visual interneurons, not motor neurons. The authoritative
definition of inputs/outputs is the connectome's own ``input_cell_types`` /
``output_cell_types`` (read from the network spec's ``input_units`` / ``output_units``),
which is what this module uses rather than a hand-picked list.

Usage:
    python -m flyhero.flyvis_figure --run <flyvis-out dir> --out <dir> [--frame N]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

# The 8 direction-selective output types, paired by preferred direction:
# T4 = ON-edge motion, T5 = OFF-edge motion; a/b/c/d = the four directions.
MOTION_OUTPUTS = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def hex_xy(n_hexals: int):
    """Pixel coordinates for a hexagonal lattice of ``n_hexals`` receptors.

    Mirrors the notebook's own convention: radius from the hexal count, integer (u, v)
    coordinates, then to pixel space.
    """
    from flyvis.utils import hex_utils

    radius = hex_utils.get_hextent(n_hexals)
    u, v = hex_utils.get_hex_coords(radius)
    x, y = hex_utils.hex_to_pixel(u, v)
    return np.asarray(x), np.asarray(y)


def hex_scatter(ax, values: np.ndarray, x, y, title: str, vmin=None, vmax=None, cmap="viridis", label=None):
    """Scatter ``values`` (one per receptor) onto the hex lattice."""
    values = np.asarray(values).reshape(-1)
    vmin = float(np.nanmin(values)) if vmin is None else vmin
    vmax = float(np.nanmax(values)) if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-6
    sc = ax.scatter(x, y, c=values, s=26, cmap=cmap, vmin=vmin, vmax=vmax, marker="h", linewidths=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=7.5)
    if label:
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontsize=6, color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=1.4, lw=0))
    return sc


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="flyvis_brain output dir (has responses.npy)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame", type=int, help="frame to show spatially (default: most active)")
    ap.add_argument("--frames-dir", type=Path, help="cropped input frames (for the raw-crop panel)")
    ap.add_argument("--video-frame", type=Path, help="full game frame PNG for panel 1")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import flyvis
    from flyvis.utils.activity_utils import LayerActivity

    from .vision import crop_highway, load_frame, region_for

    args.out.mkdir(parents=True, exist_ok=True)

    responses = np.load(args.run / "responses.npy")          # (n_frames, n_cells)
    stimulus = np.load(args.run / "stimulus.npy")            # (n_frames, 1, 721)

    nview = flyvis.NetworkView(flyvis.results_dir / "flow/0000/000")
    net = nview.init_network()
    act = LayerActivity(torch.tensor(responses)[None], net.connectome, keepref=True)

    input_types = [str(t).split("'")[1] for t in net.connectome.input_cell_types]
    output_types = [str(t).split("'")[1] for t in net.connectome.output_cell_types]
    print(f"connectome declares {len(input_types)} input types and {len(output_types)} output types")
    print(f"  inputs : {input_types}")
    print(f"  outputs: {output_types}")

    outs = np.asarray(act["output"].detach().cpu().numpy())[0]     # (n_frames, 34, 721)
    ins = np.asarray(act["input"].detach().cpu().numpy())[0]       # (n_frames, 8, 721)
    out_by_type = {t: outs[:, i, :] for i, t in enumerate(output_types)}
    in_by_type = {t: ins[:, i, :] for i, t in enumerate(input_types)}

    # Frame to display: the one where the output layer varies most across space,
    # i.e. the moment the motion detectors are most differentiated. Chosen from data
    # rather than hard-coded so it tracks whatever clip is passed in.
    if args.frame is not None:
        f = int(args.frame)
    else:
        motion_stack = np.stack([out_by_type[t] for t in MOTION_OUTPUTS if t in out_by_type])
        f = int(np.argmax(motion_stack.std(axis=(0, 2))))
    print(f"displaying frame {f} of {responses.shape[0]}")

    x, y = hex_xy(stimulus.shape[-1])
    n_out = len(output_types)

    fig = plt.figure(figsize=(17, 12), dpi=120)
    # 4 rows x 8 cols: the 8 direction channels of T4/T5 each get their own column.
    gs = fig.add_gridspec(4, 8, height_ratios=[1.2, 1.0, 0.95, 1.25], hspace=0.30, wspace=0.14)

    # ---- Row 1: the game frame, and the crop that actually enters the pipeline ----
    # Panels 1-3 must show the SAME frame as the encoded/receptor panels, so resolve the
    # exact frame that produced response frame `f`: the stimulus movie was built from the
    # frames in sorted order, so response frame f is file f_{f+1}. Falls back to
    # --video-frame if the frames dir is not available.
    raw_frame = None
    if args.frames_dir:
        files = sorted(args.frames_dir.glob("f_*.png"))
        if files:
            raw_frame = load_frame(files[min(f, len(files) - 1)])
    if raw_frame is None and args.video_frame and Path(args.video_frame).exists():
        raw_frame = load_frame(args.video_frame)
    region = region_for(1024, 768, "guitar")
    if raw_frame is not None:
        region = region_for(raw_frame.shape[1], raw_frame.shape[0], "guitar")

    ax = fig.add_subplot(gs[0, 0:4])
    if raw_frame is not None:
        ax.imshow(crop_highway(raw_frame, region))
        ax.set_title("1. game frame, cropped to the instrument's highway (guitar)\n"
                     f"crop region {region}", fontsize=8)
    else:
        ax.text(0.5, 0.5, "pass --frames-dir or --video-frame", ha="center", va="center", fontsize=8)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 4:8])
    # NB: the PNGs in --frames-dir are the *raw* video frames; the crop+greyscale happens
    # inside the pipeline. So recompute the actual BoxEye input here rather than showing the
    # raw frame - otherwise this panel would be labelled "what the fly receives" while
    # displaying the uncropped 1024x768 frame.
    if raw_frame is not None:
        from .flyvis_brain import _crop_resize_grey

        grey128 = _crop_resize_grey(raw_frame, region, (128, 128))
        # _crop_resize_grey returns 0-255; the pipeline divides by 255 before BoxEye, so
        # scale here too - the panel is labelled "what the fly receives", and the fly sees
        # the normalised range.
        grey01 = grey128 / 255.0
        # The highway is genuinely dark (mean luminance ~0.1), so plotting over the full 0-1
        # range renders almost black and the panel looks empty. Stretch the display to the
        # top of the data's range and say so, rather than silently brightening: the true
        # range is printed in the title.
        hi = float(np.percentile(grey01, 99.5))
        ax.imshow(grey01, cmap="gray", vmin=float(grey01.min()), vmax=max(hi, 1e-3))
        ax.set_title("2. the pixels the fly receives: highway crop,\n"
                     f"greyscale 128x128 (BoxEye input range {grey01.min():.2f}-{grey01.max():.2f}, "
                     "display stretched)", fontsize=8)
    ax.axis("off")

    # ---- Row 2: the ENCODED image (BoxEye receptors on the hex lattice) ----
    ax = fig.add_subplot(gs[1, 0:3])
    sc = hex_scatter(ax, stimulus[f].reshape(-1), x, y,
                     f"3. ENCODED image: {stimulus.shape[-1]} photoreceptors\n(BoxEye, hex lattice)",
                     cmap="gray_r")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)

    ax = fig.add_subplot(gs[1, 3:6])
    r_sum = np.mean([in_by_type[t][f] for t in input_types], axis=0)
    sc = hex_scatter(ax, r_sum, x, y, f"4. INPUT cells: mean of R1-R8 photoreceptors\n(frame {f})", cmap="magma")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)

    ax = fig.add_subplot(gs[1, 6:8])
    mot = np.mean([out_by_type[t][f] for t in MOTION_OUTPUTS if t in out_by_type], axis=0)
    sc = hex_scatter(ax, mot, x, y, f"5. OUTPUT: mean of\nT4a-d / T5a-d (frame {f})", cmap="RdBu_r")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)

    # ---- Row 3: each of the 8 direction-selective output channels separately ----
    # Shared colour scale so the channels are comparable; otherwise each panel autoscales
    # and a weak channel looks as strong as a strong one.
    chan = {t: out_by_type[t][f] for t in MOTION_OUTPUTS if t in out_by_type}
    lo = hi = None
    if chan:
        lo = float(min(np.nanmin(v) for v in chan.values()))
        hi = float(max(np.nanmax(v) for v in chan.values()))
    for j, t in enumerate(MOTION_OUTPUTS):
        if t not in chan:
            continue
        ax = fig.add_subplot(gs[2, j])
        hex_scatter(ax, chan[t], x, y, t, vmin=lo, vmax=hi, cmap="RdBu_r")

    # ---- Row 4: output time courses ----
    ax = fig.add_subplot(gs[3, :])
    t_axis = np.arange(responses.shape[0])
    for name in MOTION_OUTPUTS:
        if name in out_by_type:
            ax.plot(t_axis, out_by_type[name].mean(axis=1), lw=1.4, label=name)
    ax.axvline(f, color="k", ls="--", lw=1, label=f"frame shown ({f})")
    ax.set_xlabel("frame", fontsize=8)
    ax.set_ylabel("mean response over cells", fontsize=8)
    ax.set_title("6. OUTPUT cells (T4/T5) across the stimulus: temporal activity", fontsize=9)
    ax.legend(fontsize=6.5, ncol=9, frameon=False, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "game frame -> encoded receptors -> fly optic-lobe output activity"
        "  (flyvis flow/0000/000, connectome-constrained)",
        fontsize=11.5,
    )
    out_png = args.out / "stages_side_by_side.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")

    # Also dump the numbers behind panel 5/6 so the figure is checkable
    summary = {
        "frame_shown": f,
        "n_receptors": int(stimulus.shape[-1]),
        "input_cell_types": input_types,
        "output_cell_types": output_types,
        "n_output_types": n_out,
        "motion_output_peak_spread": {
            t: float(out_by_type[t].std(axis=(0, 1)).mean()) for t in MOTION_OUTPUTS if t in out_by_type
        },
        "note": "flyvis has no motor neurons; outputs are the connectome's declared output_units",
    }
    import json

    (args.out / "stages.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
