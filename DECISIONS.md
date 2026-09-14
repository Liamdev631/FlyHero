# FlyHero — Design Decisions

**This file is the authoritative record of the user's design decisions.** Per the standing
convention in `SOUL.md`: *record first, implement after.* When a design choice is made it is
written here **before** the code that implements it. Implementation choices must not exist
only in code, commit messages, or chat.

Each entry records the decision, the reasoning, and the date. Superseded decisions are kept
and marked, never silently edited away.

---

## D-008 — ANN→SNN migration by conversion: train ReLU, convert to IF (2026-09-14)

**Decision.** The ANN→SNN migration is done by **conversion**, not surrogate-gradient
BPTT. Train the policy as an ANN whose activations are **ReLU**, then convert the
trained weights to an **IF** (integrate-and-fire, leak term removed) spiking network
by rate coding.

**Reasoning (user's).** IF with no leak term is the unit whose firing rate equals the
ReLU of its input, so the trained ANN transfers without retraining the spiking net.

**Verified before recording** (torch 2.14.0+cpu; `tools/axonweave/ann2snn.py`):

- **The equivalence holds.** IF rate matches ReLU(I) to within 0.001 for I ∈ [0,1]
  over T=1000 steps. Above I=1 the rate saturates at 1 spike/step, so activations
  *must* be normalised into [0,1] — a requirement, not a refinement.
- **LIF is not equivalent and cannot be tuned into IF.** `lif_step` hardcodes the
  leak as `dv = (-(v - v_rest) + I)*dt/tau`; steady state is `v_ss = v_rest + I`, so
  it fires only when `I ≥ v_th − v_rest`. Measured rate at I=0.5 was exactly **0** for
  tau = 1, 5, 100 and 1e6. Raising tau cannot remove the dead zone — the leak term has
  to be deleted. `LIF ≈ ReLU(I − 1)`, `IF ≈ ReLU(I)`.
- Full MLP conversion works: XOR task, ANN test acc 0.960 → SNN **0.822** at T=500.
  Error falls to ~T=100 then plateaus; the residual gap does not close with more
  timesteps (systematic quantisation/offset bias, the known ANN→SNN conversion error —
  needs bias correction to close).
- Soft reset beats hard reset at every T (0.822 vs 0.780 at T=500).

**Consequence — this changes the simulated biology.** Dropping the leak term removes
membrane time-constant dynamics from every neuron: the substrate stops being a LIF
network and becomes a bank of pure integrators. Recorded explicitly because this is a
change to the model's biology, not an implementation detail.

**What AxonWeave lacks — all four items are needed:**
1. No IF unit exists — `DYNAMICS = {lif, adaptive_lif, rate}`, those three only.
2. `Rate` is linear (`baseline + gain*I`), *not* ReLU — "train with ReLU" requires an
   explicit `nn.ReLU` after the connectome layer.
3. Only hard reset (`v_reset`); no reset-by-subtraction.
4. Reaching IF requires a kernel change (delete the leak term): ~5 lines Rust, or
   ~3 lines in a torch port.

**Status.** Recorded, not yet implemented.

---

## D-007 — Output decoding: motor-neuron clusters, linearly decoded (2026-09-14)

**Decision.** The decoded output is **not** single neurons. Take **distinct clusters of
motor neurons** and **linearly decode each ensemble** to produce the final output.

**Neurons per cluster: left to the agent's convenience** (explicit delegation).

**Status.** Recorded, not yet implemented.

**Rationale (user's).** A single motor neuron is a noisy, high-variance readout; individual
cells can be silent or fire spuriously without the ensemble's behaviour changing. Decoding
the ensemble linearly is the standard premotor readout and is far more robust.

**Open questions to settle at implementation time (not yet decided by the user):**
- Which clustering criterion defines a "distinct cluster" — annotated motor-neuron *type*
  (`Acc. ti flexor MN`, `Ti flexor MN`, …), muscle target, or a data-driven cluster?
- Cluster size: the foreleg MN types currently range 1–10 cells per type, so "cluster"
  cannot simply mean "one annotated type" without some types having a single member.
- Whether the linear decoder's weights are trained or fixed from connectome path strengths.

---

## D-006 — Readout is descending neurons, not optic-lobe cells (2026-09-14)

**Decision.** The action readout is taken from the **descending neurons** — the only neurons
that leave the brain for the nerve cord.

**Reasoning.** flyvis's optic-lobe outputs are visual interneurons and cannot express a
button press. The descending neurons are the brain's actual motor-command output.

**Data.** 1,299 descending neurons in FlyWire v783, side-resolved: left 645, right 646,
centre 8.

---

## D-005 — Motor neurons come from the VNC; left/right split (2026-09-14)

**Decision.** Use **left-side** motor control for the five fret buttons and **right-side**
motor control for the strum.

**Reasoning.** The user wants whole-brain involvement and motor-neuron outputs. The brain
and nerve cord hold motor neurons in different places (brain: eye, antenna, mouthpart,
neck, proboscis; VNC: legs), so the leg motor neurons are taken from the male-CNS
connectome and split by side.

**Data.** `vnc_motor` = 708 neurons; foreleg (`fl` subclass) = 135, all in neuromere T1,
split L 68 / R 67. Six action units resolved: green (L, 10), red (L, 8), yellow (L, 5),
blue (L, 5), orange (L, 4), strum (R, 9).

---

## D-004 — Whole-brain backend: fly-brain (2026-09-14)

**Decision.** Use `eonsystemspbc/fly-brain` for the brain simulation, in preference to
flyvis.

**Reasoning (user's).** Whole-brain involvement makes the project substantially more
interesting. fly-brain is a whole-brain LIF model from FlyWire v783 (138,639 neurons,
15.1M synapses; Shiu et al., *Nature* 2024). flyvis remains useful for visual-system
questions but is optic-lobe only and contains no motor neuron.

---

## D-003 — Visual input enters at the retina (2026-09-14)

**Decision.** Feed the game frame to the **photoreceptors** (`R1-6`, `R7`, `R8`; 10,582
cells), not to optic-lobe interneurons.

**Reasoning.** Photoreceptors are the first stage of the visual system, so the frame enters
where light would. Driving the ~77k `optic` super-class would inject the image *after* the
retina had processed it — a different experiment.

**Encoding.** Two-level spatial drive through fly-brain's Poisson API: bright
photoreceptors → `neu_exc` at `r_poi` (150 Hz); dim → `neu_exc2` at `r_poi2` (15 Hz).

---

## D-002 — Input is the instrument's cropped highway (2026-09-14)

**Decision.** The policy's visual input is the **game frame cropped to that instrument's
note highway**, not the full frame.

**Reasoning (user's, from the earlier phase).** The HUD (score, timer, star meter,
multiplier) is shared across instruments and carries no information about the action, while
the highway is what determines it. Crop region for guitar on a 1024×768 frame:
`x0=330 y0=415 x1=700 y1=768`.

**Known limitation.** The crop is luminance-only, so lane colours (which differ mainly in
hue) can collapse to the same value. If lane identity matters, add per-lane channels —
do not change the crop.

---

## D-001 — Start with guitar, 5-fret (2026-09-14)

**Decision.** Begin with the 5-fret guitar: 5 fret buttons + strum as the action space.

**Reasoning (user's).** Simplest action space; other instruments (6-fret guitar, drums,
vocals, keys) can follow.

---

## E-001 — AxonWeave evaluated as a candidate simulator (2026-09-14)

**Status: evaluated, NOT decided.** No design choice has been made. This entry
records measurements so the finding is not lost; if a decision follows it gets its
own `D-` entry.

**What it is.** `dhakalnirajan/axonweave` v0.1.0 — a Python/Rust library wrapping
the Janelia **MaleCNS v1.0** connectome (166,700 neurons, ~25.6M edges) as a
trainable substrate. This is the same connectome D-005 already draws the VNC
foreleg motor neurons from; it is *not* FlyWire v783, so it does not replace
fly-brain (D-004).

**Computational units — three, configurable, not fixed:**
- `Rate` — `baseline + gain * I`. No state, no time. **This is the default.**
- `LIF` / `AdaptiveLIF` — genuine spiking: threshold crossing, reset, refractory.
- `ConnectomeLayer` — pure sparse matmul, no nonlinearity.

ANN-practical and SNN-capable, but not SNN-trainable as shipped.

**Measured 2026-09-14** (torch 2.14.0+cpu, Ryzen 7 6800U; scripts in
`tools/axonweave/`):

1. Surrogate gradients are computed in Rust and wired to nothing. `spike_gradient`
   appears only in `dynamics/spiking.py` and `tests/`. The project says so itself,
   `docs/development/IMPLEMENTATION_GAP.md` §7: *"the surrogate-gradient kernels
   exist (`surrogate.rs`), but wiring them into a differentiable framework forward
   pass ... is still pending."*
2. Both framework adapters detach — `frameworks/torch/block.py:82` and
   `jax/block.py:118` round-trip through NumPy, so the connectome receives zero
   gradient. Their own warning AXW007 admits it.
3. **Surrogate-gradient BPTT works once implemented.** A 1:1 port of
   `dynamics.rs` / `surrogate.rs` into PyTorch with a custom `autograd.Function`
   trains to 100% accuracy (loss 0.694 → 0.051), with gradients reaching the sparse
   edge weights.
4. **`torch.sparse.mm` is 26–50× slower than a CSR kernel** (32.4 ms vs 0.64 ms at
   1.0M nnz; 363 ms vs 14.0 ms at 10.0M). Training must stay in native code rather
   than torch's sparse path.
5. **Input-current scaling is a hard requirement, not a tuning detail.** LIF steady
   state is `v_ss = v_rest + I`, so firing needs `I > v_th − v_rest = 15 mV`. Below
   that the substrate is *completely silent* (measured rate 0.0000) and nothing
   learns. AxonWeave's stock `ImageEncoder` normalises to O(1) — i.e. a silent brain.
6. The Rust kernels are **serial**: no `rayon`/`par_iter` anywhere in `rust/src/*.rs`,
   and `py.allow_threads` only releases the GIL. ~8× is unused on this 8C/16T box,
   and parallelising them would speed the *existing* forward path independently of
   any training work.

**Effort for surrogate-gradient learning.** ~250–400 lines Rust (LIF recurrence
backward, `dL/dW` scatter-accumulate), ~150 lines PyO3 + `autograd.Function` glue,
~150 lines finite-difference gradient tests. The surrogate derivative math already
exists and is unit-tested. Roughly a focused week. Design that keeps speed: wrap
*per step* and let torch's autograd chain the timesteps — do not write BPTT in Rust.

**Open question (not decided).** Whether to adopt AxonWeave at all, and if so:
(a) use it frozen as a fixed sparse layer, (b) implement the Rust adjoint kernels,
or (c) use snnTorch/Norse with the connectome loaded as a sparse weight matrix.

**Unverified.** No Rust toolchain on this box (`cargo` absent), so the Rust kernels
were read rather than built; the CSR comparison uses scipy as a stand-in proxy.

---

## Rejected / not pursued

- **Pruning the brain by reachability (2026-09-14).** Proposed as a speed-up, then measured
  and dropped: 99.5% of neurons (137,903 of 138,639) lie on some photoreceptor →
  descending-neuron path, so pruning would remove only 736 neurons (0.5%) — none of them
  isolating a region. It would buy essentially nothing while making the simulated brain a
  different one. **Not doing it.**
- **ROCm + GeNN on this host (2026-09-14).** The HIP workaround was proven to work
  (`HSA_OVERRIDE_GFX_VERSION=10.3.0` → `gfx1030`, HIP vadd kernel ran correctly), but the
  measured effective bandwidth is **38.8 GB/s**, ~13× below the reference hardware's
  504 GB/s. GeNN is bandwidth-bound, so the expected gain over the CPU path is ~1.25× for a
  source build of GeNN on an unsupported iGPU. Dropped on measurement.
