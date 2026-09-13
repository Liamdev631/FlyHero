"""Summarise fly visual system activity and render it as a figure.

Companion to :mod:`flyhero.flyvis_brain`. That module produces the response tensors;
this one turns them into numbers and a picture, and - importantly - runs the
motion control.

THE CONTROL: the same frames in a *shuffled temporal order*. Shuffling keeps every
pixel of the highway (same notes, same lanes, same fret line) and destroys only the
coherent motion. So if a cell type responds to motion rather than to edges being
present, it must respond more to the real ordering than to the shuffled one. Without
this control, "the fly lit up" would be unfalsifiable - everything lights up when you
feed it a high-contrast image.

Usage:
    python -m flyhero.flyvis_analyze --real <dir> --shuffled <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .flyvis_brain import MOTION_CELLS, LAMINA_CELLS, _as_frames_cells

# Cell types shown in the figure, ordered so the pathway reads top-to-bottom:
# photoreceptor targets (lamina) -> medulla -> motion detectors.
DISPLAY = ["L1", "L2", "L5", "Mi1", "Mi4", "Tm3", "T2", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5d"]


def _activity(responses: np.ndarray, connectome):
    import torch
    from flyvis.utils.activity_utils import LayerActivity

    return LayerActivity(torch.tensor(responses)[None], connectome, keepref=True)


def temporal_modulation(activity, names: Sequence[str]) -> dict:
    """Per-cell-type temporal standard deviation, averaged over that type's cells.

    Standard deviation *across frames* per cell, then mean over cells: a measure of how
    much the cell's activity moves during the stimulus. Overall magnitude is deliberately
    not used, because a high-contrast stimulus drives large but static responses - motion
    lives in the variation over time, not the level.
    """
    out = {}
    for name in names:
        try:
            arr = _as_frames_cells(activity[name])
        except (KeyError, AttributeError):
            continue
        if arr.size == 0:
            continue
        out[name] = {
            "n_cells": int(arr.shape[1]),
            "temporal_std": float(arr.std(axis=0).mean()),
        }
    return out


def compare(real: dict, shuffled: dict) -> dict:
    common = sorted(set(real) & set(shuffled))
    rows = []
    for name in common:
        r, s = real[name]["temporal_std"], shuffled[name]["temporal_std"]
        rows.append(
            {
                "cell_type": name,
                "n_cells": real[name]["n_cells"],
                "real": r,
                "shuffled": s,
                "ratio": float(r / s) if s > 0 else None,
            }
        )
    rows.sort(key=lambda x: -(x["ratio"] or 0))

    def _mean_over(group):
        vals = [x["ratio"] for x in rows if x["cell_type"] in group and x["ratio"]]
        return float(np.mean(vals)) if vals else None

    return {
        "per_cell_type": rows,
        "motion_ratio_mean": _mean_over(MOTION_CELLS),
        "lamina_ratio_mean": _mean_over(LAMINA_CELLS),
        "n_cell_types": len(common),
        "caveat": (
            "The shuffled control is not a perfect null: shuffling frame order creates "
            "abrupt frame-to-frame changes rather than smooth motion. That biases *for* "
            "the control, so a ratio above 1 understates rather than overstates any real "
            "motion sensitivity."
        ),
    }


def figure(activity_real, real_mod: dict, shuf_mod: dict, out_png: Path, stimulus_dir: Path | None) -> None:
    """Render the activity: stimulus frames, response traces, and a per-type heatmap.

    ``real_mod``/``shuf_mod`` are the per-cell-type temporal-modulation dicts for the two
    conditions. They are passed in rather than recomputed here, because recomputing both
    from the real activity (the obvious copy-paste slip) would draw identical bars and
    silently erase the control from the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9), dpi=130)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.1, 1.5], hspace=0.45, wspace=0.3)

    # Row 1: what the fly was shown (the cropped highway, before BoxEye)
    if stimulus_dir:
        files = sorted(stimulus_dir.glob("f_*.png"))[:4]
        if files:
            import PIL.Image as I

            for i, f in enumerate(files):
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(I.open(f))
                ax.set_title(f"stimulus frame {i + 1}", fontsize=8)
                ax.axis("off")

    # Row 2: central-cell traces for the motion detectors on the real ordering
    ax = fig.add_subplot(gs[1, :])
    for name, colour in (("T4b", "#d62728"), ("T5d", "#1f77b4"), ("L2", "#2ca02c")):
        try:
            r = np.asarray(activity_real.central[name].detach().cpu().numpy()).reshape(-1)
        except (KeyError, AttributeError):
            continue
        ax.plot(np.arange(r.size), r, color=colour, lw=1.6, label=f"{name} (central cell)")
    ax.set_title("central-cell responses to the cropped instrument highway (real frame order)", fontsize=9)
    ax.set_xlabel("frame", fontsize=8)
    ax.set_ylabel("response (a.u.)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # Row 3: per-cell-type temporal modulation, real vs shuffled control
    names = [n for n in DISPLAY if n in real_mod and n in shuf_mod]
    rv = [real_mod[n]["temporal_std"] for n in names]
    sv = [shuf_mod[n]["temporal_std"] for n in names]
    ax = fig.add_subplot(gs[2, :])
    x = np.arange(len(names))
    ax.bar(x - 0.2, rv, 0.4, label="real order (coherent motion)", color="#1f77b4")
    ax.bar(x + 0.2, sv, 0.4, label="shuffled order (control)", color="#bbbbbb")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("temporal std", fontsize=8)
    ax.set_xlabel("lamina  ->  medulla  ->  motion detectors", fontsize=8)
    ax.set_title("activity variation over time, per cell type", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    # ratio annotation, so the control is readable without a log axis
    for xi, n in zip(x, names):
        ratio = real_mod[n]["temporal_std"] / shuf_mod[n]["temporal_std"] if shuf_mod[n]["temporal_std"] else float("nan")
        ax.text(xi, max(rv[xi], sv[xi]) * 1.04, f"{ratio:.2f}x", ha="center", fontsize=6)
    ax.set_ylim(0, max(rv + sv) * 1.22)

    fig.suptitle(
        "flyvis connectome-constrained visual system driven by the game's note highway",
        fontsize=11,
    )
    fig.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, required=True, help="output dir of the real-order run")
    ap.add_argument("--shuffled", type=Path, required=True, help="output dir of the shuffled run")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stimulus-dir", type=Path, help="dir of cropped input frames for the figure")
    args = ap.parse_args(argv)

    import flyvis

    args.out.mkdir(parents=True, exist_ok=True)
    real_resp = np.load(args.real / "responses.npy")
    shuffled_resp = np.load(args.shuffled / "responses.npy")

    nview = flyvis.NetworkView(flyvis.results_dir / "flow/0000/000")
    net = nview.init_network()

    act_real = _activity(real_resp, net.connectome)
    act_shuf = _activity(shuffled_resp, net.connectome)

    all_names = [str(k) for k in act_real.keys()]
    real_mod = temporal_modulation(act_real, all_names)
    shuf_mod = temporal_modulation(act_shuf, all_names)
    result = compare(real_mod, shuf_mod)
    result["global"] = {
        "real_std": float(real_resp.std()),
        "shuffled_std": float(shuffled_resp.std()),
        "real_mean": float(real_resp.mean()),
        "shuffled_mean": float(shuffled_resp.mean()),
    }
    (args.out / "motion_control.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"cell types compared: {result['n_cell_types']}")
    print(f"motion (T4/T5) mean ratio  : {result['motion_ratio_mean']:.3f}")
    print(f"lamina (L1-L5) mean ratio  : {result['lamina_ratio_mean']:.3f}")
    print(f"global std real/shuffled   : {result['global']['real_std']:.4f} / {result['global']['shuffled_std']:.4f}")

    figure(act_real, real_mod, shuf_mod, args.out / "activity.png", args.stimulus_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
