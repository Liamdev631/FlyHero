# FlyHero — Design Decisions

**This file is the authoritative record of the user's design decisions.** Per the standing
convention in `SOUL.md`: *record first, implement after.* When a design choice is made it is
written here **before** the code that implements it. Implementation choices must not exist
only in code, commit messages, or chat.

Each entry records the decision, the reasoning, and the date. Superseded decisions are kept
and marked, never silently edited away.

---

## D-009 — MNIST test task on the fly CNS: photoreceptor in, motor neuron out (2026-09-14)

**Decision.** Train a classifier for **MNIST** on the fly connectome, using **LIF**
neurons, with two hard architectural constraints:

1. **The input must be projected into the fly brain's visual input** — not into
   arbitrary or all neurons.
2. **The output must be read from motor neurons** — explicitly *not* from all
   neurons. The user's stated concern: a readout that samples everything would
   "short circuit the entire fly brain from input to output".

And one hard requirement: **the data must flow through a significant portion of the
fly brain before the output is generated.** A task that can be solved without the
brain doing the work does not count as a result.

Node selection was **delegated to the agent** ("I'll leave it up to you to figure out
which nodes to sample"). The agent's choice:

| Role | Node set | Count |
|---|---|---|
| Input (visual) | `ol_sensory` — photoreceptors R1-R6, R7, R8 | 6,098 |
| Output (motor) | `vnc_motor` | 708 |

**Substrate: male-CNS v1.0** (local copy, `FlyHero-build/connectome/`), *not* FlyWire
v783 / fly-brain. Reason: male-CNS is a single **whole-CNS** graph — brain *and* nerve
cord — so it is the only available substrate containing photoreceptors and motor
neurons in one connectivity matrix. FlyWire v783 is brain-only and has no motor
neuron, which is why D-005 had to borrow motor neurons from male-CNS in the first
place.

**Why this satisfies the "must cross the brain" requirement.** The node sets are
disjoint from each other and are the *only* points where the model touches the graph.
The forced route is:

```
photoreceptors (ol_sensory, 6,098)
  -> optic lobe intrinsic        (ol_intrinsic,  89,403)
  -> visual projection           (visual_projection, 9,201)
  -> central brain intrinsic     (cb_intrinsic,  32,164)
  -> descending neurons          (descending_neuron, 1,314)
  -> VNC intrinsic               (vnc_intrinsic, 13,161)
  -> motor neurons               (vnc_motor, 708)
```

Reachability verified by graph closure (`tools/axonweave/`): **94.6% of the CNS
(181,337 of 191,696 neurons) lies on some photoreceptor -> motor-neuron path**, and
**all 708 motor neurons** are forward-reachable from the photoreceptors (28 hops
forward, 23 hops back). The optic lobe intrinsic population is traversed 89,396 of
89,403; all 1,314 descending neurons are reached.

**Dynamics.** LIF, AxonWeave's parameterisation (`tau=20 ms`, `v_rest=-65 mV`,
`v_th=-50 mV`, `v_reset=-70 mV`, `2 ms` refractory, `dt=1 ms`). Learning uses
**surrogate gradients** through the spike (sigmoid surrogate, k=5.0) — the path E-001
identified as missing from AxonWeave — because a hard spike has zero gradient and
would leave the input projection untrainable.

**Status.** Implementing.

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

## E-002 — All 14 awesome-fly "brain" repos surveyed; fly-brain recommended as the trainable substrate (2026-09-14)

**Status: evaluated, NOT decided.** The user asked which of the brain-section repos is
most suitable for training an entire fly-brain SNN to play FlyHero, with training to run
later on an **RTX 3050** rather than this box. This entry records the survey and the
recommendation; adoption is the user's call because it would change the substrate the
project trains on.

**All 14 repos downloaded** (shallow) to `/home/liamb/projects/awesome-fly-repos/`
(2.4 GB). Re-runnable via `tools/brain-repos/fetch_brain_repos.sh`; metadata in
`tools/brain-repos/repo-metadata.tsv`; full reasoning in `docs/brain-repo-evaluation.md`.
Two repos share the name `fly-brain` — `eonsystemspbc/fly-brain` is upstream, and
`erojasoficial-byte/fly-brain` was cloned as `fly-brain-erojasoficial`.

**Recommendation: `eonsystemspbc/fly-brain`, as a library to convert — not as shipped.**
It is the only repo that is whole-brain *and* spiking *and* peer-reviewed *and* already
PyTorch `nn.Module`s *and* **already carrying a working surrogate gradient**:

```python
class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradient (ATan)."""
    self.spike_gradient = self.ATan.apply
    class ATan(torch.autograd.Function):
        def forward(ctx, v):   spike = (v > 0).float()          # hard spike
        def backward(ctx, g):  return 1/(1 + (np.pi*v).pow_(2)) * g   # soft surrogate
```

This is the piece E-001 found **missing** from AxonWeave (which needs ~250–400 lines of
Rust to get it). Here it already exists, in pure Python — and `cargo` is absent on this
box, so that difference is decisive. AlphaLIF over FlyWire v783: 138,639 neurons,
15,091,983 signed synapses, `dt = 0.1 ms`, Shiu et al. *Nature* 2024. GPL-2.0, 666 stars,
pushed 2026-08-29.

**Measured backends** (upstream's own 720-row `benchmark-results.csv`, all
`status=success`; `realtime_ratio` > 1.0 is faster than realtime):

| backend | 1.0 s | 10 s | 100 s |
|---|---|---|---|
| **GeNN (GPU)** | 1.840 | 2.040 | 1.916 |
| Brian2GeNN (GPU) | 0.838 | 0.812 | 0.810 |
| NEST GPU | 0.782 | 0.783 | 0.790 |
| Brian2 (CPU, compiled) | 0.346 | 0.403 | 0.335 |
| Brian2CUDA (GPU) | 0.092 | 0.294 | 0.373 |
| PyTorch (CUDA) | 0.153 | 0.154 | 0.154 |

Two findings that generalise: **GeNN is the only backend above realtime and it is stable
from 0.1 s to 100 s** (not a short-run artefact) — so GeNN for rollout. And **CUDA is not
automatically fast**: both CUDA backends are *slower than the compiled CPU path*, because
`dt = 0.1 ms` means 10,000 timesteps per simulated second and per-step launch overhead
dominates. Do not reach for CUDA by default on this model class.

**What fly-brain lacks, stated plainly.**
1. **Nothing is trainable as shipped** — grepping `code/` for `nn.Parameter` or
   `requires_grad` returns nothing. The connectome is a frozen `torch.sparse_csr_tensor`;
   the surrogate is present but unused. Converting it is the work (see the RTX 3050 note).
2. **No leg motor neurons** — FlyWire v783 is brain-only; its 110 MNs are ingestion/neck/
   proboscis/antennal/eye. Same two-volume bridge as `docs/whole-cns-backend.md` and D-009.
3. **It is a benchmark harness, not a training repo.**
4. Activity under the benchmark stimulus is very sparse: **386 of 138,639 neurons (0.28%)**.
   Measure firing rate under visual drive before diagnosing any training failure.

**Hardware consequence for the RTX 3050 — the fit is poor, and this is the main caveat.**
The relevant measured figure is the fork's single-row CSV: PyTorch+CUDA **0.0804** on its
tested box (an RTX 5090 Laptop per its README), i.e. 1 s of brain time = 12.4 s wall.
Bandwidth-scaled to a 3050 (~224 GB/s vs ~896) that is **~0.02×**, so an estimated
**0.02–0.10× realtime (10–50× slower than realtime)**. Then:

- **One 30 fps frame is 333 timesteps** (33.3 ms ÷ 0.1 ms) and costs ~415 ms wall on the
  *author's* GPU — 12× over a 33 ms budget before any 3050 penalty. **The whole-brain model
  at `dt = 0.1 ms` cannot drive the game live**; it is an offline/training substrate.
- **A 3-minute song = 180 s brain time ≈ 37 min wall on the author's GPU, ~2.5 h scaled to
  a 3050, per episode.** Online RL is not a plan; behavioural cloning from the existing
  automation captures is.
- **BPTT hits an 8 GB memory wall:** ~12.8 MB of state per batch item per timestep (the
  1.8 ms delay buffer is 19 deep = 10.5 MB of it). batch 8 × T=100 → 10.2 GB stored,
  over budget before gradients; batch 8 × T=10 → 1.0 GB, fits. **The 3050 forces truncated
  BPTT (T ≈ 10, batch ≤ 8) or gradient checkpointing**, against a 333-step frame horizon.

**Runner-ups, by role.** `TuragaLab/flyvis` has the only mature gradient-training harness
(`MultiTaskSolver`) — copy the engineering, but it is **rate-based, not spiking**, and
optic-lobe only, so it is not the substrate. `dhakalnirajan/axonweave` has the right
anatomy (male-CNS whole CNS, incl. the 708 VNC motor neurons D-005 wants) but its training
is still unwired — **re-checked and unchanged since E-001** (v0.1.0; `torch/block.py:82`
still `detach()`s and still warns). `eonfathom/FastFly` is the fastest forward-only CUDA/CuPy
path for consumer NVIDIA but has no autograd and no license. The body simulators
(`flybody`, `flygym`, `chimera`, `NeuroFly`, `webgpu-fly`) are **not needed** — a rhythm
game's action space is five buttons plus a strum; there is no locomotion to simulate.
Six repos (`NeuroFly`, `fruit-fly-lab`, `chimera`, `Connectome-OS`, `FastFly`,
`mps-malecns-model`) ship **no license** and cannot be a code base;
`mps-malecns-model` is also Apple-MPS-only and so disqualified by the target hardware.

**Open question (not decided).** Whether to adopt fly-brain as the trainable substrate in
place of the D-009 male-CNS/AxonWeave direction, and if so: (a) promote the connectome to a
trainable parameter in the existing torch backend and add optimizer + loss + motor readout,
(b) keep D-009's male-CNS substrate and port the ATan surrogate into it (male-CNS has the
leg motor neurons in one graph, but the training path is unwired), or (c) train in fly-brain
and bridge descending neurons → male-CNS VNC motor neurons per `docs/whole-cns-backend.md`.

**Unverified.** The upstream benchmark CSV does **not** record which GPU produced its
numbers, so "~1.9× GeNN" is attributable to the author's paper-grid machine, not to a named
device. The fork's CSV likewise omits the device; the RTX 5090 Laptop attribution comes
from that repo's README "Tested" row only. The 3050 figures are **scaled from spec-sheet
bandwidth, not measured on a 3050** — and the APU measurement in E-001 landed at ~half its
spec sheet, so the low end is optimistic. Nothing here was executed: no GPU is present on
this box and `torch` is not installed in the system interpreter.

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
