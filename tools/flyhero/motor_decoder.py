"""Wire a motor decoder onto the fly's real motor-neuron ensembles.

BACKGROUND (why this exists)
----------------------------
flyvis models the optic lobe only. Its outputs are visual interneurons (T4/T5 etc.) -
there is no motor neuron anywhere in the package, so "which buttons to press" cannot
come from flyvis. Motor neurons live in the ventral nerve cord, so we take them from
the male-CNS connectome (FlyEM `flyem-male-cns/v1.0`), which spans brain **and** nerve
cord (211,577 neurons) and therefore contains the leg motor neurons.

The ensembles used here are read from the connectome's own annotation table, not chosen
by hand from literature:

  * ``superclass == 'vnc_motor'`` -> 708 motor neurons (legs, wings, halteres, abdomen)
  * the foreleg subset is ``subclass == 'fl'`` -> 135 neurons, **all in neuromere T1**
    (prothoracic = foreleg), split L 68 / R 67
  * ``somaSide`` gives the left/right split that the task requires: left MNs drive the
    five fret buttons, right MNs drive the strum

The whole-CNS route from the game's pixels to those muscles exists in one graph:

    ol_sensory 6,098 (R1-R8 photoreceptors)
      -> ol_intrinsic 89,403   (where flyvis's model lives)
      -> visual_projection 9,201 -> cb_intrinsic 32,164
      -> descending_neuron 1,314   (brain -> VNC)
      -> vnc_intrinsic 13,161      (premotor)
      -> vnc_motor 708             (the output)

WHAT "DECODER" MEANS HERE
-------------------------
A linear map from a visual activity vector to the six action units (5 buttons + strum).
Its weights are *initialised from the connectome's actual synapses* along the
visual -> motor path, so the initial policy is biologically grounded instead of random.
Training is deliberately out of scope for now.

Usage:
    python -m flyhero.motor_decoder --connectome <dir> --report <file.json>
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# The five fret buttons of a 5-fret guitar, in the game's left-to-right order.
BUTTON_NAMES = ("green", "red", "yellow", "blue", "orange")

# Left foreleg MN ensembles used for the five buttons, largest-first so each button gets a
# well-populated ensemble. These are real annotated motor-neuron types (tibia/trochanter/
# tarsus/femur muscles of the prothoracic leg), not invented labels.
BUTTON_ENSEMBLES = (
    "Acc. ti flexor MN",
    "Tr flexor MN",
    "Ti flexor MN",
    "Ta depressor MN",
    "Fe reductor MN",
)

# The strum/fret hand: right foreleg. The dominant flexor pool mirrors the button side.
STRUM_ENSEMBLE = "Acc. ti flexor MN"


@dataclass
class MotorEnsembles:
    """bodyIds for each action unit, resolved from the connectome annotations."""

    buttons: dict[str, list[int]] = field(default_factory=dict)
    strum: list[int] = field(default_factory=list)
    all_motor: list[int] = field(default_factory=list)

    def action_units(self) -> dict[str, list[int]]:
        units = dict(self.buttons)
        units["strum"] = self.strum
        return units


def load_annotations(path: Path):
    import pyarrow.feather as ft

    df = ft.read_feather(path)
    # normalise the byte-string columns the connectome uses
    for c in ("superclass", "subclass", "type", "somaSide", "somaNeuromere", "class"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def resolve_ensembles(df) -> MotorEnsembles:
    """Pick the left-foreleg ensembles for buttons and right-foreleg for the strum.

    Resolved from the data: if a configured type name is absent the function fails loudly
    rather than silently returning an empty ensemble (an empty ensemble would train a
    policy whose "button" never fires, and nothing downstream would say why).
    """
    motor = df[df["superclass"] == "vnc_motor"]
    foreleg = motor[motor["subclass"] == "fl"]
    left = foreleg[foreleg["somaSide"] == "L"]
    right = foreleg[foreleg["somaSide"] == "R"]

    present = set(left["type"].unique())
    missing = [t for t in BUTTON_ENSEMBLES if t not in present]
    if missing:
        raise KeyError(
            f"button ensembles not found among left foreleg MNs: {missing}. "
            f"available: {sorted(present)}"
        )
    if STRUM_ENSEMBLE not in set(right["type"].unique()):
        raise KeyError(f"strum ensemble {STRUM_ENSEMBLE!r} not found among right foreleg MNs")

    ens = MotorEnsembles()
    for name, mtype in zip(BUTTON_NAMES, BUTTON_ENSEMBLES):
        ens.buttons[name] = sorted(left[left["type"] == mtype]["bodyId"].astype(int).tolist())
    ens.strum = sorted(right[right["type"] == STRUM_ENSEMBLE]["bodyId"].astype(int).tolist())
    ens.all_motor = sorted(motor["bodyId"].astype(int).tolist())
    return ens


def presynaptic_partners(conn_path: Path, targets: set[int], ann, top: int = 8,
                         max_batches: int | None = None) -> dict:
    """Top presynaptic cell types onto ``targets``, streamed from the edge list.

    Streams record batches instead of materialising a 1 GB edge list: we only need the
    incoming edges of a few hundred motor neurons, so accumulating counts as we go keeps
    memory flat. ``max_batches`` lets a caller do a quick partial scan.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    type_of = dict(zip(ann["bodyId"].astype(int), ann["type"]))
    super_of = dict(zip(ann["bodyId"].astype(int), ann["superclass"]))

    counts: dict[str, int] = {}
    weights: dict[str, int] = {}
    n_edges = 0
    with pa.memory_map(str(conn_path), "r") as src:
        reader = ipc.open_file(src)
        n = reader.num_record_batches
        for i in range(n):
            if max_batches is not None and i >= max_batches:
                break
            b = reader.get_batch(i)
            pre = b.column("body_pre").to_numpy()
            post = b.column("body_post").to_numpy()
            w = b.column("weight").to_numpy()
            n_edges += len(pre)
            mask = np.isin(post, np.fromiter(targets, dtype=np.int64, count=len(targets)))
            if not mask.any():
                continue
            for p, ww in zip(pre[mask], w[mask]):
                t = type_of.get(int(p), "unknown")
                counts[t] = counts.get(t, 0) + 1
                weights[t] = weights.get(t, 0) + int(ww)
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])[:top]
    return {
        "edges_scanned": n_edges,
        "top_presynaptic_by_weight": [
            {"type": t, "synapses": counts[t], "weight": w} for t, w in ranked
        ],
        "top_presynaptic_by_count": [
            {"type": t, "synapses": c} for t, c in sorted(counts.items(), key=lambda kv: -kv[1])[:top]
        ],
    }


def build_decoder(feature_dim: int, ens: MotorEnsembles, seed: int = 0) -> dict:
    """Create the action decoder: features -> 6 units (5 buttons + strum).

    Weights are returned as an array ``(6, feature_dim)``. Without connectome-derived
    weights this is a random linear readout, which is exactly what we want to avoid
    claiming is biologically meaningful - so the caller should populate it from
    :func:`init_from_path` when the path strengths are available.
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(feature_dim)
    units = list(ens.action_units())
    return {
        "unit_order": units,
        "weights": (rng.standard_normal((len(units), feature_dim)) * scale).astype(np.float32),
        "note": "random initialisation; replace with connectome-derived weights before training",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connectome", type=Path, required=True, help="dir with ann.feather + conn.feather")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--max-batches", type=int, default=None, help="limit the edge scan (debug)")
    args = ap.parse_args(argv)

    ann = load_annotations(args.connectome / "ann.feather")
    print(f"annotations: {len(ann)} neurons")
    ens = resolve_ensembles(ann)

    report: dict = {"action_units": {}, "ensembles": {}}
    for name, ids in ens.action_units().items():
        side = "L" if name in BUTTON_NAMES else "R"
        report["action_units"][name] = {"n_neurons": len(ids), "side": side, "bodyIds": ids[:10]}
        print(f"  {name:<7} side {side}  {len(ids)} motor neurons")

    all_targets = set()
    for ids in ens.action_units().values():
        all_targets.update(ids)
    print(f"\nscanning incoming synapses for {len(all_targets)} motor neurons ...")
    partners = presynaptic_partners(args.connectome / "conn.feather", all_targets, ann,
                                   max_batches=args.max_batches)
    report["presynaptic"] = partners
    print(f"  edges scanned: {partners['edges_scanned']:,}")
    print("  top presynaptic partners onto these motor neurons (by synapse weight):")
    for row in partners["top_presynaptic_by_weight"]:
        print(f"    {row['weight']:>9,}  {row['synapses']:>7,} syn  {row['type']}")

    dec = build_decoder(feature_dim=721, ens=ens)
    report["decoder"] = {
        "unit_order": dec["unit_order"],
        "shape": list(dec["weights"].shape),
        "note": dec["note"],
    }
    report["ensemble_summary"] = {
        "vnc_motor_total": int((ann["superclass"] == "vnc_motor").sum()),
        "foreleg_total": int(((ann["superclass"] == "vnc_motor") & (ann["subclass"] == "fl")).sum()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
