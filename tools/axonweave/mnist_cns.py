#!/usr/bin/env python3
"""MNIST classification through the male-CNS connectome.

Architecture (per DECISIONS.md D-009 and the user's hard constraints):

    MNIST 28x28
      -> input projection (learnable)        784 -> 6,098
      -> PHOTORECEPTORS  (ol_sensory)        the fly's visual input
      ==================== forced ====================
      -> optic lobe intrinsic                89,403
      -> visual projection                    9,201
      -> central brain intrinsic             32,164
      -> descending neurons                   1,314
      -> VNC intrinsic                       13,161
      -> MOTOR NEURONS  (vnc_motor)             708
      ===============================================
      -> linear readout                       708 -> 10

The only entry point is photoreceptors and the only exit is motor neurons, so
the readout cannot short-circuit the brain: 94.6% of the CNS lies on some
photoreceptor -> motor-neuron path.

Dynamics are LIF (AxonWeave's parameterisation) with surrogate gradients
through the spike, since a hard spike has zero gradient.

Sparse propagation uses scipy's sparse-sparse product on the *spike* vector:
at ~1% activity this is ~4-8x cheaper than a dense SpMM, which is what makes
the full CNS tractable on CPU.
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import time
import urllib.request

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

# LIF parameters — AxonWeave's defaults
TAU = 20.0
V_REST = -65.0
V_TH = -50.0
V_RESET = -70.0
REFRACTORY = 2.0
DT = 1.0


# ----------------------------------------------------------------- MNIST
def _fetch(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    for name, url in MNIST_URLS.items():
        p = os.path.join(root, name)
        if not os.path.exists(p):
            print(f"  downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, p)


def _imgs(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)


def _labs(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def load_mnist(root: str):
    _fetch(root)
    xtr = _imgs(os.path.join(root, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    ytr = _labs(os.path.join(root, "train-labels-idx1-ubyte.gz")).astype(np.int64)
    xte = _imgs(os.path.join(root, "t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    yte = _labs(os.path.join(root, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    return xtr, ytr, xte, yte


# ----------------------------------------------------------------- surrogate
class SpikeSTE(torch.autograd.Function):
    """Hard spike forward, straight-through surrogate backward.

    `width` is the half-width of the surrogate's support around threshold, so
    the backward is exactly zero (not merely small) away from threshold. That
    matters: it keeps the gradient tensor sparse, which is what lets the
    connectome's adjoint use the cheap sparse-sparse path too.
    """

    @staticmethod
    def forward(ctx, v_pre, v_th, width):
        ctx.save_for_backward(v_pre)
        ctx.v_th = float(v_th)
        ctx.width = float(width)
        return (v_pre >= v_th).to(v_pre.dtype)

    @staticmethod
    def backward(ctx, g):
        (v_pre,) = ctx.saved_tensors
        x = (v_pre - ctx.v_th).abs()
        return g * (x <= ctx.width).to(g.dtype), None, None


# ----------------------------------------------------------------- sparse prop
class SparseProp:
    """Y = X @ A with dX = G @ A^T, over scipy CSR matrices.

    X is the spike tensor (mostly zeros). Rather than a dense SpMM we build a
    CSR from the spikes and use scipy's sparse-sparse product, whose cost
    scales with the number of *active* neurons rather than with all n.
    """

    def __init__(self, A: sp.csr_matrix, At: sp.csr_matrix):
        self.A, self.At = A, At
        self.last_fwd_ms = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t0 = time.time()
        s = sp.csr_matrix(x.detach().numpy())
        y = np.asarray((s @ self.A).todense(), dtype=np.float32)
        self.last_fwd_ms = (time.time() - t0) * 1e3
        return torch.from_numpy(y)

    def backward(self, g: torch.Tensor) -> torch.Tensor:
        gs = sp.csr_matrix(g.detach().numpy())
        return torch.from_numpy(
            np.asarray((gs @ self.At).todense(), dtype=np.float32))


class PropFn(torch.autograd.Function):
    _prop: SparseProp = None

    @staticmethod
    def forward(ctx, x):
        return PropFn._prop.forward(x)

    @staticmethod
    def backward(ctx, g):
        return PropFn._prop.backward(g)


# ----------------------------------------------------------------- LIF
def lif_step(v, refrac, current, t, width):
    """Port of AxonWeave's lif_step, made differentiable via the surrogate."""
    dt_over_tau = DT / TAU
    t_new = t + DT
    can_spike = (refrac <= t_new).to(v.dtype)

    dv = (-(v - V_REST) + current) * dt_over_tau
    v_step = torch.where(can_spike.bool(), v + dv, v)

    raw = SpikeSTE.apply(v_step, V_TH, width)
    spike = raw * can_spike

    spiked = spike.bool()
    v_new = torch.where(spiked, torch.full_like(v, V_RESET), v_step)
    refrac_new = torch.where(spiked, torch.full_like(refrac, t_new + REFRACTORY),
                             refrac)
    return spike, v_new, refrac_new


# ----------------------------------------------------------------- model
class FlyCNS(nn.Module):
    def __init__(self, n, photo, motor, T=10, gain_init=40.0, width=1.0,
                 prop: SparseProp = None, region_masks=None):
        super().__init__()
        self.n, self.T, self.width = n, T, width
        self.photo = torch.as_tensor(photo, dtype=torch.long)
        self.motor = torch.as_tensor(motor, dtype=torch.long)
        self.prop = prop
        self.region_masks = region_masks or {}
        self._photo_mask = None

        self.inp = nn.Linear(784, len(photo))
        self.gain = nn.Parameter(torch.tensor(gain_init))
        self.readout = nn.Linear(len(motor), 10)

    def forward(self, x, collect_rates=False):
        B = x.shape[0]
        drive = self.inp(x) * self.gain          # (B, n_photo) currents

        v = torch.full((B, self.n), V_REST)
        refrac = torch.zeros(B, self.n)
        t = 0.0
        counts = torch.zeros(B, self.n)
        rates = []

        for _ in range(self.T):
            syn = PropFn.apply(counts)            # recurrent: spikes @ A
            cur = torch.zeros_like(v)
            cur[:, self.photo] = drive
            cur = cur + syn
            spike, v, refrac = lif_step(v, refrac, cur, t, self.width)
            counts = counts + spike
            t += DT
            if collect_rates:
                rates.append(spike.detach())

        logits = self.readout(counts[:, self.motor])
        return (logits, rates) if collect_rates else logits


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/awnn")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=40.0)
    ap.add_argument("--wscale", type=float, default=1.0,
                    help="recurrent weight scale (applied after row-sum normalisation)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- graph
    G = np.load(f"{args.workdir}/graph.npz")
    A = sp.csr_matrix((G["data"], G["indices"], G["indptr"]),
                      shape=tuple(G["shape"]))
    n = A.shape[0]
    with open(f"{args.workdir}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    photo, motor, sc = meta["photo"], meta["motor"], meta["superclass"]

    # scale synapse counts so a fully-active row delivers ~1.0 of "drive"
    row_mean = float(np.asarray(A.sum(axis=1)).ravel().mean())
    A = A.copy()
    A.data = (A.data / row_mean * args.wscale).astype(np.float32)

    print(f"graph        : {n:,} neurons, {A.nnz:,} edges")
    print(f"row_mean     : {row_mean:.1f} synapses -> normalised to 1.0")
    print(f"photoreceptor: {len(photo):,}   motor: {len(motor):,}")

    prop = SparseProp(A, A.T.tocsr())
    PropFn._prop = prop

    regions = {}
    for name in ("ol_sensory", "ol_intrinsic", "visual_projection",
                 "cb_intrinsic", "descending_neuron", "vnc_intrinsic",
                 "vnc_motor"):
        m = np.where(sc == name)[0]
        if len(m):
            regions[name] = torch.as_tensor(m, dtype=torch.long)

    # ---- data
    print("\nloading MNIST ...")
    xtr, ytr, xte, yte = load_mnist(f"{args.workdir}/mnist")
    xtr, ytr = xtr[:args.n_train], ytr[:args.n_train]
    xte, yte = xte[:args.n_test], yte[:args.n_test]
    print(f"train {len(xtr):,}  test {len(xte):,}  T={args.T}  batch={args.batch}")

    model = FlyCNS(n, photo, motor, T=args.T, gain_init=args.gain,
                   width=args.width, prop=prop, region_masks=regions)

    # ---- dry run: is the network even alive?
    if args.dry_run:
        xb = torch.from_numpy(xtr[:8])
        with torch.no_grad():
            logits, rates = model(xb, collect_rates=True)
        print("\n=== spike rate by region (mean over T, 8 samples) ===")
        for name, idx in regions.items():
            tot = sum(float(r[:, idx].mean()) for r in rates) / len(rates)
            print(f"  {name:<22} {tot:.5f}")
        tot = sum(float(r.mean()) for r in rates) / len(rates)
        print(f"  {'ALL NEURONS':<22} {tot:.5f}")
        print(f"forward time (8 samples, T={args.T}): "
              f"{prop.last_fwd_ms:.1f} ms/last step")
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ntr = len(xtr)
    t_start = time.time()

    for ep in range(args.epochs):
        perm = np.random.permutation(ntr)
        model.train()
        tot_loss = 0.0
        nb = 0
        t0 = time.time()
        for i in range(0, ntr - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(xtr[idx])
            yb = torch.from_numpy(ytr[idx])
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss)
            nb += 1
        tr_time = time.time() - t0

        # eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(xte) - args.batch + 1, args.batch):
                xb = torch.from_numpy(xte[i:i + args.batch])
                yb = torch.from_numpy(yte[i:i + args.batch])
                correct += int((model(xb).argmax(1) == yb).sum())
        acc = correct / (len(xte) // args.batch * args.batch)
        el = time.time() - t_start
        print(f"epoch {ep+1}/{args.epochs}  loss {tot_loss/max(nb,1):.4f}  "
              f"test acc {acc:.4f}  ({tr_time:.0f}s train, {el/60:.1f} min total)",
              flush=True)

    # ---- where did the data flow?
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(xte[:16])
        _, rates = model(xb, collect_rates=True)
    print("\n=== spike rate by region (final) ===")
    for name, idx in regions.items():
        tot = sum(float(r[:, idx].mean()) for r in rates) / len(rates)
        print(f"  {name:<22} {tot:.5f}")

    torch.save({"model": model.state_dict(), "args": vars(args)},
               f"{args.workdir}/mnist_cns.pt")
    print(f"\nsaved -> {args.workdir}/mnist_cns.pt")


if __name__ == "__main__":
    main()
