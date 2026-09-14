"""Prove, empirically, that the only input path is the photoreceptors.

Ablation logic: LIF with v initialised at v_rest and no input current has
dv = (-(v - v_rest) + 0) * dt/tau = 0, so v never reaches threshold and nothing
spikes. Therefore if zeroing the external drive silences the ENTIRE network,
nothing except the photoreceptors can be receiving input.
"""
import pickle
import sys
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, "/home/liamb/projects/FlyHero/tools/axonweave")
from mnist_cns import FlyCNS, SparseProp, PropFn, load_mnist  # noqa: E402

GRAPH = "/tmp/awnn"
G = np.load(f"{GRAPH}/graph.npz")
A = sp.csr_matrix((G["data"], G["indices"], G["indptr"]), shape=tuple(G["shape"]))
n = A.shape[0]
with open(f"{GRAPH}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
photo, motor, sc = meta["photo"], meta["motor"], meta["superclass"]
row_mean = float(np.asarray(A.sum(axis=1)).ravel().mean())
A.data = (A.data / row_mean * 800.0).astype(np.float32)
prop = SparseProp(A, A.T.tocsr())
PropFn._prop = prop

xtr, ytr, _, _ = load_mnist(f"{GRAPH}/mnist")
xb = torch.from_numpy(xtr[:8])

torch.manual_seed(0)
model = FlyCNS(n, photo, motor, T=10, gain_init=1000.0, width=1.0, prop=prop)

print("=" * 68)
print("1. WHAT ARE THE ONLY TRAINABLE PARAMETERS?")
print("=" * 68)
for name, p in model.named_parameters():
    print(f"  {name:<16} shape={tuple(p.shape)}  n={p.numel():,}  trainable={p.requires_grad}")
print(f"\n  photoreceptor indices used as injection site: {len(photo):,}")
print(f"  motor neuron indices used as readout source  : {len(motor):,}")

print()
print("=" * 68)
print("2. ABLATION: zero the input drive -> does ANYTHING still spike?")
print("=" * 68)
with torch.no_grad():
    _, rates_on = model(xb, collect_rates=True)
    _, rates_off = model(torch.zeros_like(xb), collect_rates=True)

on = sum(float(r.mean()) for r in rates_on) / len(rates_on)
off = sum(float(r.mean()) for r in rates_off) / len(rates_off)
print(f"  spike rate, normal input : {on:.6f}")
print(f"  spike rate, ZERO input   : {off:.6f}")
print(f"  residual activity        : {100*off/on if on else 0:.4f}% of normal")
print()
if off == 0.0:
    print("  => the network is EXACTLY silent with no input.")
    print("  => the photoreceptors are the ONLY input path into the CNS.")
else:
    print("  => there is residual activity; another input path may exist.")

print()
print("=" * 68)
print("3. IS THE READOUT TOUCHING ONLY MOTOR NEURONS?")
print("=" * 68)
n_motor = len(motor)
print(f"  readout Linear in_features = {model.readout.in_features} "
      f"(motor neurons = {n_motor})  match={model.readout.in_features == n_motor}")
print(f"  readout in_features vs total neurons: {model.readout.in_features:,} "
      f"of {n:,}  ({100*model.readout.in_features/n:.2f}% of the CNS)")

print()
print("=" * 68)
print("4. INPUT BOTTLENECK: how big is the input projection?")
print("=" * 68)
ip = model.inp
print(f"  input projection: {ip.in_features} -> {ip.out_features} "
      f"({ip.weight.numel():,} params)")
print(f"    out_features == photoreceptors? "
      f"{ip.out_features == len(photo)}")
print(f"  readout          : {model.readout.in_features} -> "
      f"{model.readout.out_features} ({model.readout.weight.numel():,} params)")
print(f"  TOTAL trainable  : "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"  Frozen connectome: {A.nnz:,} connections, 0 trainable")
