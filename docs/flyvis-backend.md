# flyvis backend: driving the fly visual system from game frames

## Why flyvis

flyvis (Turaga lab, *FliesInSilico* — Lappalainen et al. 2024) is a
**connectome-constrained** model of the *Drosophila* visual system. The optic-lobe
wiring is fixed to the real connectome and only a small number of scalar parameters are
free — the pretrained networks report `NumberOfParams(free=734, fixed=2959)`. That
property is why it is the right backend for "wire the fly brain to the visual inputs":
the thing receiving the game frames is a real invertebrate visual system with real
named cell types, not an arbitrary RNN that happens to be small.

It replaces the earlier axonweave plan for the *visual* path. axonweave remains
interesting for the whole-brain/connectome-training question, but it has no pretrained
model, its substrate needs a 1.1 GB download, and its `ConnectomeBlock` detaches
gradients (`AXW007`) — so it cannot be trained as shipped. flyvis has published
pretrained weights, a documented custom-stimulus path, and a cell-type-addressable
readout.

## Install (done once)

Kept in its own venv so it never collides with the agent's environment or the game
tooling. CPU-only torch deliberately — the host GPU is an AMD 680M and there is no CUDA,
so the default CUDA wheel would be multiple GB of unusable payload.

```bash
python3 -m venv /home/liamb/venvs/flyvis
V=/home/liamb/venvs/flyvis/bin
$V/python -m pip install --upgrade pip
$V/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
$V/python -m pip install flyvis
FLYVIS_ROOT_DIR=/home/liamb/flyvis-data $V/flyvis download-pretrained
```

Versions in place: flyvis 1.2.0, torch 2.14.0+cpu.

`FLYVIS_ROOT_DIR` selects where models and caches live (default would be a `data/`
folder inside the package). Set it in the environment for **every** flyvis invocation,
including the pipeline below — a different value means "no pretrained models found".

## The stimulus path

Documented example used: `examples/07_flyvision_providing_custom_stimuli/`. The
structure that matters:

- **Custom stimuli are movies,** shaped `(n_frames, height, width)` in luminance `[0, 1]`.
  Any cropped game frame sequence meets this.
- **`BoxEye(extent=15, kernel_size=13)`** renders cartesian frames into photoreceptor
  activations on a hexagonal lattice: 31 columns across, **721 receptors**, spaced 13 px,
  each reading the mean luminance of its 13x13 pixel box. Input must be 4-D
  `(samples, frames, H, W)`; output is `(samples, frames, 1, 721)`.
- **`network.fade_in_state(intensity, dt, first_frame)`** computes a stationary state by
  ramping the first frame from grey. This is not optional in practice: without it the
  readout is dominated by the onset transient when the highway first appears, rather than
  by the stimulus.
- **`network.simulate(movie[None], dt, initial_state=...)`** returns
  `(1, n_frames, 45669)` — **45,669 cells**. `LayerActivity(responses, connectome)`
  makes them addressable by cell type (`T4c`, `L5`, `Mi1`, ...) and exposes `.central`.
- **Ensemble:** `results_dir/flow/0000/NNN`, `NNN` sorted best-to-worst by task error.
  `000` is the best; `EnsembleView(results_dir / "flow/0000")` iterates all 50.

## Cropping to the instrument

The crop is the instrument's own note highway, from `flyhero.vision.region_for()` /
`crop_highway()` — measured from a real 1024x768 frame of the built game
(`x0=330, y0=415, x1=700, y1=768` for guitar, i.e. 370x353). Cropping matters for two
reasons: the HUD (score, timer, star meter, multiplier) is *shared* across instruments
so those pixels would mean different things depending on context, and the highway is
what actually determines the action.

Verified on a real gameplay frame: the crop contains the 5 lanes, the falling notes and
the fret line. After downscaling to 128x128 greyscale the stimulus still resolves the
highway and the notes — horizontal neighbour correlation 0.93 (noise would be ~0) with
~5% bright pixels matching the notes.

## Running it

```bash
cd /home/liamb/projects/FlyHero/tools
FLYVIS_ROOT_DIR=/home/liamb/flyvis-data \
  /home/liamb/venvs/flyvis/bin/python -m flyhero.flyvis_brain \
    --video /home/liamb/projects/FlyHero-build/capture/flyhero_gameplay_notes.mp4 \
    --start 12 --frames 24 --out /home/liamb/projects/FlyHero-build/flyvis-out
```

Writes `stimulus.npy` (the receptor movie actually fed in), `responses.npy`, plus
`summary.json`, `comparison.json` and `report.md`.

## Pitfalls

- **Luminance-only.** The lanes differ mainly in *hue* (green/red/yellow/blue/orange), so
  two lanes can render to the same luminance. If lane identity matters, feed per-lane
  channels or a colour-weighted projection — do **not** "fix" it by changing the crop.
- **Receptor spacing is coarser than the lanes.** 721 receptors over a 1024x768 frame at
  13 px spacing means several lanes fall inside one receptor. Expect motion and edge
  signals, not lane-resolved ones.
- **Never feed the full frame.** It is mostly HUD and background gradient; the fly would
  mostly see the timer.
- `BoxEye` runs `torchvision` resize internally and warns about the future `antialias`
  default. Harmless.
- flyvis needs Python <3.13 — the 3.11 venv is correct.
