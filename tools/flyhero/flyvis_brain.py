"""Feed the game's instrument highway to a fly visual system (flyvis).

The job of this module is narrow: take raw frames of the running game, crop out the
part of the frame that pertains to the instrument (the note highway, via
``flyhero.vision``), render that crop through a fly compound-eye front end, and read
out the activity of the connectome-constrained visual system.

Why flyvis for this: flyvis (Turaga lab, `FliesInSilico`) is a *connectome-constrained*
model of the fly visual system - the wiring is fixed to the real optic lobe
connectome and only a small number of scalar parameters are free (the pretrained
networks carry ~734 free params against ~2959 fixed). That is what makes it worth
using as the "fly brain" here: it is a real biological visual system with real
measured cell types (T4/T5 motion detectors, lamina L1-L5, medulla Mi1/Tm1/Tm2/...),
not an arbitrary RNN that merely happens to be small.

Pipeline (mirrors the official custom-stimulus example, `examples/07_*`):

    video frame -> crop to highway -> grey scale float -> BoxEye(extent=15,kernel_size=13)
        -> 721 photoreceptors on a hexagonal lattice -> network.simulate(...)
        -> (n_frames, 45669) cell responses -> LayerActivity -> per-cell-type summary

The 721 receptors and 45669 cells are fixed by the model, not chosen here.

Usage (needs the flyvis venv, see docs/flyvis-backend.md):

    python -m flyhero.flyvis_brain --video <gameplay.mp4> --start 12 --frames 24 \
        --out <dir> [--network flow/0000/000] [--instrument guitar]

Outputs into ``--out``: ``stimulus.npy`` (the receptor movie actually fed in),
``responses.npy`` (cell activity), ``summary.json`` and ``report.md`` (the numbers,
including a notes-vs-no-notes comparison if the window is long enough).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .vision import Region, crop_highway, load_frame, region_for

# Cell types worth reporting by name. These are the classic optic-lobe players, and
# splitting them this way makes the readout interpretable instead of a 45k-wide blob.
# T4/T5 are the direction-selective motion detectors (T4 responds to ON edges moving
# in one of 4 directions, T5 to OFF edges); L1-L5 are lamina cells (photoreceptor
# targets); Mi1/Tm1/Tm2/T3 feed the ON/OFF pathways into the motion detectors.
MOTION_CELLS = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
LAMINA_CELLS = ("L1", "L2", "L3", "L4", "L5")
MEDULLA_CELLS = ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "T2", "T3")


def build_stimulus_movie(
    video: Path,
    region: Region,
    start: float,
    n_frames: int,
    stride: int = 1,
    size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Crop a window of the game video to the highway and return greyscale frames.

    Returns an array shaped ``(n_frames, H, W)`` with values in ``[0, 1]``.

    The crop is the whole point: feeding the full 1024x768 frame would hand the fly
    mostly score, timer and star-power pixels, which are identical across instruments
    and carry no information about the action. ``region_for`` was measured from a real
    frame of the built game and is instrument-specific.

    ``size`` is the resolution the crop is resized to *before* BoxEye. BoxEye itself
    resizes to its receptor layout, so this only sets how much spatial detail survives;
    128x128 keeps the five lanes separable at roughly 2-3 px per lane in the rendered
    lattice.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None  # type: ignore

    frames: list[np.ndarray] = []
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        idx = 0
        while len(frames) < n_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(_crop_resize_grey(rgb, region, size))
            idx += 1
        cap.release()
    else:
        # Fallback: read pre-extracted PNGs named in order (used by `--from-frames`).
        raise RuntimeError(
            "OpenCV (cv2) is required to read the video directly; "
            "install opencv-python-headless or pass --from-frames with extracted PNGs"
        )

    if not frames:
        raise RuntimeError(
            f"no frames read from {video} at start={start}s - is the timestamp past the end?"
        )
    return np.stack(frames).astype(np.float32) / 255.0


def movie_from_frames(paths: Sequence[Path], region: Region, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Same as :func:`build_stimulus_movie` but from already-extracted image files."""
    frames = [_crop_resize_grey(load_frame(Path(p)), region, size) for p in paths]
    if not frames:
        raise RuntimeError("no frames given")
    return np.stack(frames).astype(np.float32) / 255.0


def _crop_resize_grey(rgb: np.ndarray, region: Region, size: tuple[int, int]) -> np.ndarray:
    """Crop to the highway, resize, and convert to luminance."""
    from PIL import Image

    crop = crop_highway(rgb, region)
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", 2)
    img = Image.fromarray(crop).resize((size[1], size[0]), resample)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        # Rec. 601 luma; the fly's photoreceptors see luminance, and the highway's
        # lane colours differ mainly in hue, so luminance alone loses lane identity -
        # see the note in report.md about chroma if that turns out to matter.
        arr = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return arr


def render_receptors(movie: np.ndarray, extent: int = 15, kernel_size: int = 13):
    """Render ``(n_frames, H, W)`` luminance frames to fly photoreceptor activations.

    Returns ``(n_frames, 1, 721)`` - one row per frame, 721 photoreceptors, matching
    the layout the pretrained networks expect. BoxEye places simulated photoreceptors
    on a hexagonal lattice (31 columns across) and takes the mean luminance in the
    kernel_size x kernel_size pixel box around each.
    """
    import torch
    import flyvis
    from flyvis.datasets.rendering import BoxEye

    receptors = BoxEye(extent=extent, kernel_size=kernel_size)
    frames = torch.tensor(movie, device=flyvis.device).float()
    # BoxEye requires exactly 4-D (samples, frames, height, width). A single 2-D frame
    # needs both dims added; an (n_frames, H, W) movie needs only the sample dim. Getting
    # this wrong raises inside eye.py as "too many values to unpack (expected 4)".
    if frames.ndim == 2:
        frames = frames[None, None]
    elif frames.ndim == 3:
        frames = frames[None]
    elif frames.ndim != 4:
        raise ValueError(f"unexpected stimulus rank {frames.ndim}; want (n,H,W) or (1,n,H,W)")
    rendered = receptors(frames)
    # (1, n, 1, 721) -> (n, 1, 721); copy so the result does not alias graph memory
    return rendered.squeeze(0).detach().clone().cpu()


def load_network(network_dir: str = "flow/0000/000"):
    """Initialise a pretrained network from the ensemble.

    Directories are ``flow/0000/NNN`` where ``NNN`` sorts by task error, so ``000`` is
    the best of the 50. We default to it deliberately: the point of this first pass is
    to see what a correctly-trained fly visual system does with game frames, not to
    explore ensemble spread. ``EnsembleView(results_dir / "flow/0000")`` iterates all 50.
    """
    import flyvis

    network_view = flyvis.NetworkView(flyvis.results_dir / network_dir)
    network = network_view.init_network()
    return network_view, network


def run_network(network, movie_receptors, dt: float = 1.0 / 100.0, fade_in: float = 1.0):
    """Simulate the network on a receptor movie and return cell responses.

    ``network.simulate`` runs the forward pass without gradients. A stationary state is
    computed first via ``fade_in_state``: it ramps the first frame up from grey so the
    recorded activity reflects the *stimulus* rather than the network's onset transient.
    That distinction matters here - without it, every readout is dominated by the
    moment the highway first appears.

    Returns ``(n_frames, n_cells)`` on CPU.
    """
    movie = movie_receptors
    if movie.ndim == 3 and movie.shape[-1] == 721:
        pass  # already (n, 1, 721)
    elif movie.ndim == 2:
        movie = movie[:, None, :]

    stationary_state = network.fade_in_state(fade_in, dt, movie[[0]])
    responses = network.simulate(movie[None], dt, initial_state=stationary_state).cpu()
    return responses.squeeze(0)


def _as_frames_cells(block) -> np.ndarray:
    """Normalise a LayerActivity block to ``(n_frames, n_cells)``.

    LayerActivity blocks come back as ``(batch, n_frames, hexals)`` - e.g. ``(1, 24, 721)``
    for a 24-frame single-sample simulation. The batch axis must be squeezed *before* any
    reshaping or slicing: reducing on the wrong axis silently reports ``numel`` (24x721 =
    17304) as the cell count and makes a frame split produce an empty half.
    """
    arr = block.detach().cpu().numpy()
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def summarize(responses, connectome, cell_types: Iterable[str] | None = None) -> dict:
    """Per-cell-type activity statistics, plus the strongest responding types.

    ``LayerActivity`` gives dict-style access to the (n_frames, n_cells) response
    tensor by cell type. Reported per type: cell count, mean/peak depolarisation,
    peak hyperpolarisation, and the temporal standard deviation of the *central* cell
    (a single-cell proxy - cells of a type share parameters, so the central one is the
    canonical readout rather than the spatial average).
    """
    from flyvis.utils.activity_utils import LayerActivity

    activity = LayerActivity(responses[None], connectome, keepref=True)
    wanted = list(cell_types) if cell_types is not None else (
        list(MOTION_CELLS) + list(LAMINA_CELLS) + list(MEDULLA_CELLS)
    )

    out: dict = {"n_frames": int(responses.shape[0]), "n_cells": int(responses.shape[1]), "cell_types": {}}
    for name in wanted:
        try:
            block = activity[name]
        except (KeyError, AttributeError):
            continue
        if block is None or block.numel() == 0:
            continue
        arr = _as_frames_cells(block)
        central = np.asarray(activity.central[name].detach().cpu().numpy()).reshape(-1)
        out["cell_types"][name] = {
            "n_cells": int(arr.shape[1]),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
            "min": float(arr.min()),
            "ptp": float(arr.max() - arr.min()),
            "central_std": float(central.std()),
            "central_ptp": float(central.max() - central.min()),
        }

    ranked = sorted(
        out["cell_types"].items(), key=lambda kv: kv[1]["central_ptp"], reverse=True
    )
    out["strongest_by_central_ptp"] = [
        {"cell_type": k, **{m: v[m] for m in ("n_cells", "central_ptp", "central_std")}}
        for k, v in ranked[:10]
    ]
    return out


def compare_notes_vs_blank(responses, connectome, split_frame: int) -> dict:
    """Compare activity before vs after ``split_frame``.

    This is the cheap, honest sanity check on whether the fly is seeing anything
    stimulus-related at all: if the highway window with notes moving is indistinguishable
    from the run-up before the notes arrive, then the crop or the rendering is wrong and
    any downstream claim about "activity" would be measuring noise.
    """
    from flyvis.utils.activity_utils import LayerActivity

    activity = LayerActivity(responses[None], connectome, keepref=True)
    result: dict = {"split_frame": int(split_frame), "cell_types": {}}
    for name in list(MOTION_CELLS) + list(LAMINA_CELLS) + list(MEDULLA_CELLS):
        try:
            block = activity[name]
        except (KeyError, AttributeError):
            continue
        if block is None or block.numel() == 0:
            continue
        arr = _as_frames_cells(block)
        a, b = arr[:split_frame], arr[split_frame:]
        if a.size == 0 or b.size == 0:
            continue
        result["cell_types"][name] = {
            "pre_mean": float(a.mean()),
            "post_mean": float(b.mean()),
            "delta": float(b.mean() - a.mean()),
            "post_std": float(b.std()),
        }
    ranked = sorted(
        result["cell_types"].items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True
    )
    result["largest_shifts"] = [
        {"cell_type": k, **v} for k, v in ranked[:10]
    ]
    return result


def _write_report(out_dir: Path, meta: dict, summary: dict, comparison: dict | None) -> None:
    lines = ["# Fly visual system response to the game's instrument highway", ""]
    lines += [
        f"- stimulus: `{meta['video']}` from {meta['start']}s, {meta['n_frames']} frames",
        f"- instrument/region: `{meta['instrument']}` crop {meta['region']}",
        f"- network: `{meta['network']}` (flyvis {meta['flyvis_version']})",
        f"- integration dt: {meta['dt']} s over {meta['n_frames']} frames "
        f"({meta['n_frames'] * meta['dt']:.2f} s of simulated time)",
        f"- receptors: {meta['n_receptors']} (extent={meta['extent']}, kernel={meta['kernel_size']})",
        f"- cells: {summary['n_cells']}",
        "",
        "## Strongest responding cell types (central cell peak-to-peak)",
        "",
        "| cell type | cells | central ptp | central sd |",
        "|---|---|---|---|",
    ]
    for row in summary.get("strongest_by_central_ptp", []):
        lines.append(
            f"| {row['cell_type']} | {row['n_cells']} | {row['central_ptp']:.4f} | {row['central_std']:.4f} |"
        )
    lines += [
        "",
        "Cell-type reference: **T4/T5** = direction-selective motion detectors (the classic",
        "readout for visual motion); **L1-L5** = lamina, fed directly by photoreceptors;",
        "**Mi/Tm/T2/T3** = medulla cells that shape the ON/OFF pathways into T4/T5.",
        "",
    ]
    if comparison:
        lines += [
            f"## Notes vs pre-stimulus (split at frame {comparison['split_frame']})",
            "",
            "| cell type | pre mean | post mean | delta | post sd |",
            "|---|---|---|---|---|",
        ]
        for row in comparison.get("largest_shifts", []):
            lines.append(
                f"| {row['cell_type']} | {row['pre_mean']:.4f} | {row['post_mean']:.4f} | "
                f"{row['delta']:+.4f} | {row['post_std']:.4f} |"
            )
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- The crop is luminance-only. Lane colours (green/red/yellow/blue/orange) differ",
        "  mainly in hue, so two different lanes can render to the same luminance. If lane",
        "  identity turns out to matter, feed per-lane channels or a luminance-weighted",
        "  colour projection rather than changing the crop.",
        "- The fly's photoreceptors are far coarser than the crop: 721 receptors at 13 px",
        "  spacing against a 1024x768 game frame means several lanes fall inside one",
        "  receptor. Expect motion/edge signals, not lane-resolved ones, at this resolution.",
        "- No training happens here - this is the pretrained visual system's response to",
        "  game frames, i.e. a measurement, not a policy.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, help="gameplay video to sample frames from")
    ap.add_argument("--from-frames", type=Path, nargs="*", help="use these image files instead of a video")
    ap.add_argument("--start", type=float, default=12.0, help="seconds into the video to start")
    ap.add_argument("--frames", type=int, default=24, help="how many frames to feed")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth frame")
    ap.add_argument("--instrument", default="guitar", help="which highway to crop")
    ap.add_argument("--network", default="flow/0000/000", help="pretrained network dir under results")
    ap.add_argument("--dt", type=float, default=1.0 / 100.0, help="integration time step (s)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args(argv)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import flyvis
        flyvis_version = getattr(flyvis, "__version__", "?")
    except ImportError as exc:  # pragma: no cover - environment-level failure
        print(f"flyvis is not importable: {exc}")
        print("Set up the flyvis venv first - see docs/flyvis-backend.md")
        return 2

    region = region_for(1024, 768, args.instrument)
    print(f"crop region for {args.instrument}: {region}")

    if args.from_frames:
        movie = movie_from_frames(args.from_frames, region)
    else:
        if not args.video:
            print("need --video or --from-frames")
            return 2
        movie = build_stimulus_movie(args.video, region, args.start, args.frames, args.stride)
    print(f"stimulus movie: {movie.shape} luminance range [{movie.min():.3f}, {movie.max():.3f}]")

    receptors = render_receptors(movie)
    print(f"receptors: {tuple(receptors.shape)}")
    np.save(out_dir / "stimulus.npy", receptors.numpy())

    _view, network = load_network(args.network)
    responses = run_network(network, receptors, dt=args.dt)
    print(f"responses: {tuple(responses.shape)}")
    np.save(out_dir / "responses.npy", responses.numpy())

    split = max(1, int(responses.shape[0] * 0.4))
    summary = summarize(responses, network.connectome)
    comparison = compare_notes_vs_blank(responses, network.connectome, split)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    meta = {
        "video": str(args.video) if args.video else "pre-extracted frames",
        "start": args.start,
        "n_frames": int(movie.shape[0]),
        "instrument": args.instrument,
        "region": str(region),
        "network": args.network,
        "flyvis_version": flyvis_version,
        "dt": args.dt,
        "n_receptors": int(receptors.shape[-1]),
        "extent": 15,
        "kernel_size": 13,
    }
    _write_report(out_dir, meta, summary, comparison)
    print(f"wrote {out_dir}/report.md, summary.json, comparison.json, stimulus.npy, responses.npy")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
