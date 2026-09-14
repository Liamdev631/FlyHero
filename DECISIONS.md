# FlyHero — Design Decisions

**This file is the authoritative record of the user's design decisions.** Per the standing
convention in `SOUL.md`: *record first, implement after.* When a design choice is made it is
written here **before** the code that implements it. Implementation choices must not exist
only in code, commit messages, or chat.

Each entry records the decision, the reasoning, and the date. Superseded decisions are kept
and marked, never silently edited away.

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
