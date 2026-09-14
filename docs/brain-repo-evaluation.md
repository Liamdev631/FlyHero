# Brain-section repo evaluation: which substrate can we *train*?

> **Status: DECIDED.** `eonsystemspbc/fly-brain` is adopted as the trainable substrate —
> recorded as **D-010** in `DECISIONS.md`, which holds the declaration, the per-alternative
> reasons, and the rubric. This document is the survey behind that decision: the evidence,
> the measurements, and the comparison framework to score any **future** candidate against.

**Question.** From the "Brain models and embodied simulation" section of
[`cobanov/awesome-fly`](https://github.com/cobanov/awesome-fly) (14 repos), which is most
suitable for training an **entire spiking neural network of the fly brain to play FlyHero**,
with training to happen later on an **RTX 3050** (not this server)?

**Answer.** **`eonsystemspbc/fly-brain`** — as a *library to be converted*, not as shipped.
It is the only repo that is simultaneously whole-brain (138,639 neurons), spiking,
peer-reviewed, already written as PyTorch `nn.Module`s, **already carrying a working
surrogate-gradient spike function**, and CUDA-activable on the target hardware. Every other
candidate fails at least one hard requirement. The full reasoning and the evidence are below.

---

## What was downloaded

All 14 repos, shallow (`--depth 1 --single-branch`), to **`/home/liamb/projects/awesome-fly-repos/`**
(2.4 GB total). Fetch script committed as `tools/brain-repos/fetch_brain_repos.sh`.

One naming collision had to be resolved: `eonsystemspbc/fly-brain` and
`erojasoficial-byte/fly-brain` share a basename, so the second landed as
`fly-brain-erojasoficial/`. Both are evaluated below — they are different projects.

| repo | stars | language | license | last push |
|---|---|---|---|---|
| philshiu/Drosophila_brain_model | 244 | Jupyter | MIT | 2024-09-14 |
| eonsystemspbc/fly-brain | 666 | Python | GPL-2.0 | 2026-08-29 |
| TuragaLab/flybody | 828 | Python | Apache-2.0 | 2026-02-07 |
| NeLy-EPFL/flygym | 303 | Python | Apache-2.0 | 2026-08-24 |
| TuragaLab/flyvis | 151 | Python | MIT | 2026-08-18 |
| abgnydn/webgpu-fly | 4 | TypeScript | MIT | 2026-09-04 |
| seven-monarchs/NeuroFly | 7 | Python | **none** | 2026-04-05 |
| vaibhavkedarisetti/fruit-fly-lab | 10 | Python | **none** | 2026-08-28 |
| caparison1234/chimera | 2 | Python | **none** | 2026-03-21 |
| ruvnet/Connectome-OS | 10 | Rust | **none** | 2026-04-23 |
| eonfathom/FastFly | 8 | Python/CUDA | **none** | 2026-02-27 |
| seohyunjun/mps-malecns-model | 1 | Python | **none** | 2026-09-12 |
| dhakalnirajan/axonweave | 0 | Python/Rust | NOASSERTION | 2026-09-12 |
| erojasoficial-byte/fly-brain | 53 | Python | MIT | 2026-03-21 |

## The criteria that actually decide it

1. **Does it spike?** A rate-based network cannot be "trained as an SNN".
2. **Is the connectome trainable with gradients?** A surrogate-gradient spike function
   (`autograd.Function` with a smooth backward) plus weights that can *be* parameters.
3. **Is it whole-brain?** The project wants the entire brain, not a subsystem.
4. **Does it run on NVIDIA/CUDA?** The training box is an RTX 3050; Apple-only or
   browser-only code is disqualified outright.
5. **Is it licensed?** Four candidates have *no license at all* — legally unusable as a base.
6. **Does it touch a motor output?** Needed to reach the five fret buttons.

## Verdict table

| repo | spiking? | grad-trainable? | whole-brain? | CUDA / 3050? | motor out? | verdict |
|---|---|---|---|---|---|---|
| **eonsystemspbc/fly-brain** | **yes** (AlphaLIF, 138,639) | **surrogate exists**; weights are not parameters | **yes** | **yes** (torch + GeNN) | descending neurons | **BEST SUBSTRATE** |
| erojasoficial-byte/fly-brain | yes | no — but adds GPU Hebbian plasticity | yes | yes (torch) | DN → flygym body | best *embodied* reference |
| TuragaLab/flyvis | **no** (PPNeuron, rate-based) | **yes** — real training harness | no — optic lobe only | yes | **none** | best *training harness* to copy |
| dhakalnirajan/axonweave | yes (LIF/IF) | **no — adapter detaches** | whole **CNS** (brain+VNC) | yes, but Rust-gated | **yes** (708 VNC MNs) | right anatomy, training unwired |
| eonfathom/FastFly | yes | **no** (no autograd anywhere) | yes (FlyWire) | **yes — built for consumer NVIDIA** | no | fastest forward-only sim |
| ruvnet/Connectome-OS | yes (Rust LIF) | **no — "not inferred from gradients"** | 115k FlyWire subset | no (CUDA-free by design) | no | debugging tool, not training |
| philshiu/Drosophila_brain_model | yes (Brian2) | no | yes | via cpp_standalone | no | original paper code, dormant 2024 |
| seohyunjun/mps-malecns-model | yes | no (has a toy PPO file) | male-CNS | **no — Apple MPS only** | no | **disqualified: wrong GPU vendor** |
| TuragaLab/flybody | no | RL (PPO), non-spiking | body only | yes | body joints | body, not brain |
| NeLy-EPFL/flygym | no | RL, non-spiking | body only | yes (Warp) | body joints | body, not brain |
| abgnydn/webgpu-fly | yes | no | FlyWire + MANC | no (browser WebGPU) | tripod gait | demo, self-declared non-scientific |
| seven-monarchs/NeuroFly | yes (Brian2) | no | FlyWire + body | CPU | navigation | unlicensed prototype |
| vaibhavkedarisetti/fruit-fly-lab | yes | no | yes | CPU | motor channels | unlicensed prototype |
| caparison1234/chimera | yes (1,373 larval) | no | larval subset | CPU | MuJoCo body | unlicensed, larval |

## Why `eonsystemspbc/fly-brain` wins

**(a) It is literally "an entire SNN model of the fly brain".** AlphaLIF built from
FlyWire v783 — **138,639 neurons, 15,091,983 signed synapses**, following Shiu et al.
(*Nature* 2024, 91% match against experimental recordings). `dt = 0.1 ms`.

**(b) The surrogate gradient already exists — this is the decisive finding.** E-001
concluded that AxonWeave, the project's current substrate, *lacks* surrogate-gradient
learning and would need ~250–400 lines of Rust to get it. fly-brain's PyTorch backend
**already ships it**:

```python
class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradient (ATan)."""
    self.spike_gradient = self.ATan.apply
    ...
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):            # hard spike
            spike = (v > 0).float()
            ctx.save_for_backward(v)
            return spike
        @staticmethod
        def backward(ctx, grad_output):  # soft, differentiable surrogate
            (v,) = ctx.saved_tensors
            return 1 / (1 + (np.pi * v).pow_(2)) * grad_output
```

That is the arctan surrogate, correctly implemented and correctly detached at the reset
(`reset = ((v - self.v_reset) * spike).detach()`). It is pure Python/PyTorch — **no Rust
toolchain required**, which matters because `cargo` is absent on this box.

**(c) It is a real, maintained, licensed project.** GPL-2.0, 666 stars, pushed 2026-08-29,
6 backends, connectome shipped in-repo.

**(d) The fast runtime and the trainable runtime are the same model.** Six backends, so a
policy can be *trained* in PyTorch and *rolled out / deployed* in GeNN without changing the
model definition.

### What it does not have, stated plainly

Per the E-001 convention, the unverified and the missing:

1. **Nothing is trainable as shipped.** Grepping the whole `code/` tree for `nn.Parameter`
   or `requires_grad` returns **nothing**. The connectome is a frozen
   `torch.sparse_csr_tensor`; the surrogate exists but no optimizer, loss, or parameter
   registration uses it. Converting it is the work: promote the weights to a trainable
   parameter (with the `torch.sparse.mm` slowness from E-001 in mind — 26–50× slower than
   CSR), add an optimizer + loss, and add the motor readout.
2. **No leg motor neurons.** FlyWire v783 is brain-only; its 110 MNs are ingestion/neck/
   proboscis/antennal/eye — **no leg MNs**. This is the two-volume problem already recorded
   in `docs/whole-cns-backend.md` and D-009, and bridged the same way (descending neurons →
   male-CNS VNC motor neurons).
3. **The repo is a benchmark harness, not a training repo.** Its README is a grid of
   backend timings and spike-train comparisons.

## Measured backend performance (upstream's own 720-row benchmark)

`fly-brain/data/benchmark-results.csv`, run `nature_2026_07`, stimulus "Sugar GRNs (200 Hz)".
`realtime_ratio` = simulated_time / wall_time; **> 1.0 is faster than realtime**. Every row
`status=success`, so no backend is being flattered by exclusions.

| backend | 0.1 s | 1.0 s | 10 s | 100 s |
|---|---|---|---|---|
| **GeNN (GPU)** | 1.974 | **1.840** | 2.040 | 1.916 |
| Brian2GeNN (GPU) | 1.230 | 0.838 | 0.812 | 0.810 |
| NEST GPU | 0.593 | 0.782 | 0.783 | 0.790 |
| Brian2 (CPU, compiled) | 0.127 | 0.346 | 0.403 | 0.335 |
| Brian2CUDA (GPU) | 0.011 | 0.092 | 0.294 | 0.373 |
| PyTorch (CUDA) | 0.140 | 0.153 | 0.154 | 0.154 |

Two conclusions:

- **GeNN is the only backend that beats realtime, and it is stable across durations**
  (1.9–2.0 at every window from 0.1 s to 100 s) — not a short-run artefact. Use GeNN for
  rollout.
- **CUDA is not automatically fast:** `Brian2CUDA` (0.01–0.37) and `PyTorch (CUDA)`
  (~0.15) are both *slower than the plain compiled CPU path* (0.35). This is the pattern the
  backend-performance notes warn about: at `dt = 0.1 ms` there are **10,000 timesteps per
  simulated second**, each a small kernel, so per-step launch overhead dominates and the GPU
  sits idle. **A CUDA backend is the wrong thing to reach for by default here.**

**Activity is extremely sparse.** Under this stimulus only **386 of 138,639 neurons (0.28%)**
fire, ~17,000 spikes/s. Worth knowing before training: a near-silent substrate gives a
surrogate gradient almost nothing to act on, so *measure the firing rate under visual drive
before diagnosing any training failure* (the pitfall this project has already hit once).

**Unverified:** the CSV does **not** record which GPU produced these numbers (the
`spike_path` column shows a Linux host, nothing more). The ~1.9× GeNN figure is therefore
attributable to the machine the author ran the paper grid on — **not** specifically to the
RTX 4070 in the older backend notes. Treat "~2× realtime GeNN" as *GPU-class* rather than
device-specific.

## Hardware reality check for the RTX 3050

The training target is 8 GB VRAM and, being a mid-range part, far less memory bandwidth than
the machines these numbers come from. Since these simulators are bandwidth-bound, scaling is
by bandwidth ratio, not FLOPS:

| device | bandwidth (spec) | scaled ratio from the ~0.10 PyTorch-CUDA figure |
|---|---|---|
| RTX 5090 laptop (author's tested box) | ~896 GB/s | 0.080 (measured) |
| RTX 4070 | ~504 GB/s | ~0.045 |
| **RTX 3050 8 GB** | **~224 GB/s** | **~0.020** |
| RTX 3050 6 GB (dGPU / laptop) | 168 / 144 GB/s | ~0.015 / ~0.013 |

**Estimated range, not a measurement: ~0.02–0.10× realtime on the 3050 (10–50× slower than
realtime).** The spread is honest — the low end assumes bandwidth-bound behaviour, the high
end assumes per-timestep launch overhead dominates (which the `dt = 0.1 ms` structure
suggests, and overhead does *not* shrink with bandwidth). Bandwidths are spec-sheet ceilings;
the APU measurement in E-001 came in at ~half its spec sheet, so the low end may be optimistic.

Consequences that matter more than the ratio:

- **One 30 fps frame is 333 timesteps** (33.3 ms ÷ 0.1 ms). At the measured 0.08 ratio that
  frame costs ~415 ms of wall clock — **12× over a 33 ms frame budget** before any 3050
  penalty. **The whole-brain model at `dt = 0.1 ms` will not drive the game live.** It is an
  offline/training substrate; the game loop needs a cheaper policy head or a coarser brain
  timestep.
- **A 3-minute song is 180 s of brain time** → **~37 min wall on the author's GPU, ~2.5 h
  on a bandwidth-scaled 3050, per episode.** Online RL at one episode per 2.5 h is not a
  plan; behavioural cloning from the existing automation captures is.
- **BPTT has a memory wall on 8 GB.** Each timestep of state is ~12.8 MB per batch item
  (the 1.8 ms synaptic delay buffer is 19 deep, and is 10.5 MB of that). Stored over a
  horizon: batch 8 × T=100 → **10.2 GB**, i.e. over budget before gradients (typically ×2–3);
  batch 8 × T=10 → 1.0 GB, which fits. **So the 3050 forces truncated BPTT (T ≈ 10 steps,
  batch ≤ 8) or gradient checkpointing** — relevant because a 333-step frame horizon is
  ~33× larger than what fits.

## The stack this implies

No single repo does the whole job. Layered:

| role | use | why |
|---|---|---|
| **substrate to train** | `eonsystemspbc/fly-brain` (`code/run_pytorch.py`) | whole-brain spiking, surrogate grad present, torch, CUDA |
| **rollout / deployment** | same repo, **GeNN backend** | only backend > realtime, stable across durations |
| **anatomy fix (leg MNs)** | male-CNS VNC via the D-009 path | FlyWire has no leg motor neurons |
| **training-harness patterns** | `TuragaLab/flyvis` — `MultiTaskSolver` | the only mature gradient-training rig here (optimizer, multi-task loss, checkpoints, NaN overflow guard) — **copy the engineering, not the model: flyvis is rate-based and optic-lobe only** |
| **plasticity reference** | `erojasoficial-byte/fly-brain` | a working GPU Hebbian rule over the same CSR weights; MIT |
| **if a fast 3050-native rollout is needed** | `eonfathom/FastFly` | CUDA/CuPy LIF, FP16 CSR, forward-only — no learning, no license |

**Not needed at all:** the body simulators (`flybody`, `flygym`, `chimera`, `NeuroFly`,
`webgpu-fly`). A rhythm game's action space is five buttons plus a strum; there is no
locomotion to simulate. Their brain→motor bridging is interesting reading, not a dependency.

## Things worth knowing before committing

- **`erojasoficial-byte/fly-brain` vendors an *older* copy** of fly-brain's PyTorch backend
  (upstream has since added `perf_counter`, `voltage_stim`, `spike_io_enabled`). Same ATan
  surrogate, but do not treat the fork as the model definition — upstream is.
- Its README "Tested" row is an **RTX 5090 Laptop (24 GB), 64 GB RAM, Windows 11**; its own
  single-row benchmark CSV records PyTorch+CUDA at 0.0804 (1 s brain = 12.4 s wall). That CSV
  also omits the device, same caveat as above. It is a **single-author Zenodo preprint**, not
  peer-reviewed, and it ships `generate_paper.py` — read it as a pattern source, not as
  validated science.
- **`dhakalnirajan/axonweave` is unchanged since E-001.** Still v0.1.0 (its CHANGELOG is
  documentation and release engineering only); `frameworks/torch/block.py:82` still does
  `x.detach().cpu().numpy()` and the adapter still warns *"surrogate-gradient enabled yet;
  gradients through the spiking..."*. Its surrogate kernels remain unwired.
- **Licensing is a real filter:** `NeuroFly`, `fruit-fly-lab`, `chimera`, `Connectome-OS`,
  `FastFly`, and `mps-malecns-model` ship **no license**, so they cannot be a code base.
  `fly-brain` is GPL-2.0 (copyleft — fine for a local model, worth noting if FlyHero
  artifacts are ever distributed); `flyvis` and `erojasoficial-byte/fly-brain` are MIT.

---

## Cross-comparison: scoring a future candidate

When a new substrate appears, score it on the same eight criteria so the comparison stays
apples-to-apples rather than being re-argued from scratch. Criteria 1–4 are pass/fail; 5–8
separate survivors.

| # | criterion | pass looks like |
|---|---|---|
| 1 | **Spiking units** | real threshold/reset dynamics — not a rate network |
| 2 | **Gradient-trainable** | a surrogate spike function **and** weights declarable as parameters |
| 3 | **Scope** | whole-brain, or whole-CNS |
| 4 | **Target hardware** | runs on NVIDIA/CUDA (the RTX 3050 training box) |
| 5 | **Motor output** | motor neurons present, or a documented bridge to them |
| 6 | **LICENSE present** | any OSI license; absent disqualifies outright |
| 7 | **Maintenance** | recent commits, adoption, responsive maintainer |
| 8 | **Evidence** | peer-reviewed, or at minimum ships its own measured benchmark |

The incumbent, scored, as the reference row:

| candidate | 1 spiking | 2 trainable | 3 whole-brain | 4 CUDA | 5 motor | 6 license | 7 maint. | 8 evidence |
|---|---|---|---|---|---|---|---|---|
| **`eonsystemspbc/fly-brain`** | ✅ | ⚠️ surrogate ships, params not declared | ✅ 138,639 | ✅ torch + GeNN | ⚠️ DNs only — leg MNs need the bridge | ✅ GPL-2.0 | ✅ 666★, 2026-08-29 | ✅ *Nature* 2024 |

**Practical checks, before reading any code.** These three greps are what actually decided
this survey, and they take under a minute:

```bash
# 1. Does it train at all, or is it a simulator?  (the cheapest discriminator)
grep -rn "nn.Parameter\|requires_grad" --include='*.py' <repo>/    # empty => simulator

# 2. Is it spiking?  Check the MODEL dir, not README hits in analysis/.
grep -rln "spike\|threshold\|LIF\|surrogate" <repo>/<model_dir>/

# 3. Is there a license?
ls <repo>/LICENSE <repo>/LICENSE.md <repo>/COPYING 2>/dev/null     # missing => disqualified
```

Two traps worth re-checking on any new candidate, both of which caught this survey: a repo
can **ship a working surrogate gradient and still be untrainable** (criterion 2 needs both
halves), and a **fork's vendored backend can be an older revision** than upstream — diff it
before quoting it as current.

If a future candidate passes all eight, the thing to compare against is not a feature list
but these measured quantities: **realtime ratio per backend** (and whether the fast backend
is differentiable — train and deploy may differ), **BPTT activation memory per timestep**,
and **activity sparsity under a real stimulus**, which on this model class is ~0.3% of
neurons.
