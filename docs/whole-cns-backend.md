# Whole-CNS backend: fly-brain (whole brain) + male-CNS (motor neurons)

> **Settled by D-010:** fly-brain (FlyWire v783) is the trainable SNN substrate, with GeNN as
> the rollout backend. FlyWire has no leg motor neurons, so the two-volume bridge described
> below — descending neurons → male-CNS VNC motor neurons — is the permanent design, not a
> workaround. Alternatives and their rejection reasons are in `docs/brain-repo-evaluation.md`.

## Why the backend changed

flyvis is **optic-lobe only** — its outputs are visual interneurons (`T4`/`T5` etc.) and
the package contains no motor neuron of any kind (grepping it for `motor` returns
nothing). That makes it the wrong tool for "the outputs should be decisions about which
buttons to press".

**`eonsystemspbc/fly-brain`** is a **whole-brain** spiking model: a leaky integrate-and-fire
network built from the **FlyWire v783** connectome (~138,639 neurons, ~15.1M signed
synapses), following Shiu et al. (Nature 2024), which reports 91% accuracy against
experimental recordings. 0.1 ms timestep, six backends (Brian2/Brian2CUDA/PyTorch/NEST
GPU/GeNN/Brian2GeNN), and the connectome ships in the repo.

## Where the motor neurons actually are

Brain and nerve cord are separate volumes, and this matters:

| volume | contains | motor neurons? |
|---|---|---|
| FlyWire v783 (fly-brain) | brain, 138,639 neurons | **110, but brain-only**: ingestion 28, neck 26, proboscis 24, antennal 10, crop 8, eye 4, haustellum 4, salivary 2 — **no leg MNs** |
| male-CNS (`flyem-male-cns/v1.0`) | brain **+ VNC**, 211,577 neurons | **708 `vnc_motor`**, incl. **135 foreleg (`fl`)**, all in neuromere **T1** |

This matches the published anatomy (Nature 2026, brain-and-cord connectome): the brain
holds the motor neurons of the eyes, antennae, mouthparts and foregut; **the VNC holds
the motor neurons of the legs**.

So the model is deliberately **two-volume**, and that is a feature rather than a
workaround: the brain does the sensing and the decision (fly-brain, spiking), and the
nerve cord supplies the motor neurons the decoder writes to.

## The readout: descending neurons

The brain's output to the body is the **descending neurons** — the only neurons that
leave the brain for the nerve cord. In fly-brain:

    super_class='descending'   1,299 neurons   side: left 645 | right 646 | center 8

Perfectly balanced left/right, which is what makes the required split expressible.
Readout order:

```
game pixels -> photoreceptors -> whole-brain LIF -> DESCENDING NEURONS
  -> [bridge] -> VNC leg motor neurons
  -> LEFT foreleg MNs  = 5 fret buttons
  -> RIGHT foreleg MNs = strum
```

## Stimulus injection point

Drive the **photoreceptors**, not the optic lobe. `R1-6` (7,932), `R7` (1,336),
`R8` (1,314) — ~10,582, all `cell_class='visual'`. The FlyWire `optic` super_class is
~77k neurons, but those are optic-lobe interneurons: driving them injects the image
*after* the retina has already processed it, which is a different experiment.

fly-brain's own API (`code/paper-phil-drosophila/model.py::run_exp`) takes:

    neu_exc   -> Poisson drive at r_poi  (default 150 Hz)
    neu_exc2  -> Poisson drive at r_poi2 (second rate)
    neu_slnc  -> silenced
    f_poi = 250   # documented as "sufficient to cause spiking"

That gives a two-level (bright/dim) spatial drive, which is what
`flyhero.flybrain_stimulus` produces from the cropped highway.

## Verified on this machine

fly-brain runs here on CPU (no CUDA — host GPU is an AMD Radeon 680M):

```
python main.py --pytorch --t_run 0.1 --n_run 1 --no_log_file
  setup 8.05s  simulation 22.10s  ->  success
  active neurons 329   total spikes 1601         (0.005x realtime)
```

Stimulus mapping (game frame -> retina):

```
10,582 photoreceptors (left 5,374 | right 5,208)
  left : 327 driven / 5,047 dim
  right: 492 driven / 4,716 dim
```

## Pitfalls (each cost real debugging)

- **Use `pos_x/pos_y/pos_z`, NOT `soma_*`.** pos_* covers 100% of neurons; soma_* covers
  only **0.1% of photoreceptors** (85% overall). Using soma_* silently reduced the retina
  to **10 neurons** and drove nothing.
- **Normalise the crop before thresholding.** The crop is 0-255; comparing it to a 0-1
  threshold marks *every* neuron bright and drives the whole retina at the high rate.
- **Drop non-finite coordinates before mapping.** One NaN poisons the min/max
  normalisation, so every sampled luminance is NaN, `NaN >= threshold` is False, and the
  retina silently ends up entirely in the 'dim' set.
- **brian2 is incompatible with numpy 2.x** (`ndarray.ptp` was removed). The Brian2
  paper path needs its own venv with `numpy<2`; the PyTorch backend runs fine on numpy 2.
- No conda on this host, and no CUDA: use `--pytorch` or a `numpy<2` venv for Brian2 CPU.

## Relationships between the pieces

- `flyhero.flybrain_stimulus` — game frame -> photoreceptor drive sets (`neu_exc`/`neu_exc2`)
- `flyhero.motor_decoder` — resolves the 6 action units from the male-CNS annotations and
  reports the premotor drive onto them
- `flyhero.flyvis_brain` / `flyvis_analyze` / `flyvis_figure` — the optic-lobe work,
  retained: useful for visual-system questions, but *not* the action path
