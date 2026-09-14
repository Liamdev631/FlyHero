"""
Standalone feasibility prototype: surrogate-gradient (BPTT) learning on a
sparse connectome substrate, mirroring AxonWeave's kernels exactly.

Ported 1:1 from:
  rust/src/dynamics.rs       lif_step
  rust/src/surrogate.rs      surrogate_grad + surrogate_lif_step

The only change is that the spike nonlinearity is wrapped in a torch
autograd.Function (hard forward, surrogate backward) and the whole thing is
unrolled in framework-native torch ops instead of detaching to NumPy.
"""
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------- surrogate
# Exact port of rust/src/surrogate.rs surrogate_grad()
KIND_SIGMOID, KIND_ATAN, KIND_PIECEWISE, KIND_STE = 0, 1, 2, 3


def surrogate_grad(kind, x, k, width):
    if kind == KIND_SIGMOID:
        clipped = (x * k).clamp(-20.0, 20.0)
        sigma = torch.sigmoid(clipped)
        return k * sigma * (1.0 - sigma)
    if kind == KIND_ATAN:
        t = np.pi * k * x
        return k / (1.0 + t * t)
    if kind == KIND_PIECEWISE:
        return torch.where(x.abs() <= 1.0 / k,
                           torch.full_like(x, k), torch.zeros_like(x))
    if kind == KIND_STE:
        return torch.where(x.abs() <= width,
                           torch.ones_like(x), torch.zeros_like(x))
    return torch.zeros_like(x)


class SpikeFunction(torch.autograd.Function):
    """Heaviside forward, surrogate derivative backward."""

    @staticmethod
    def forward(ctx, v_pre, v_th, kind, k, width):
        ctx.save_for_backward(v_pre)
        ctx.v_th, ctx.kind, ctx.k, ctx.width = float(v_th), kind, k, width
        return (v_pre >= v_th).to(v_pre.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (v_pre,) = ctx.saved_tensors
        x = v_pre - ctx.v_th
        sg = surrogate_grad(ctx.kind, x, ctx.k, ctx.width)
        return grad_out * sg, None, None, None, None


# ---------------------------------------------------------------- LIF step
def lif_step(v, refrac_until, current, t, tau, v_rest, v_th, v_reset,
             refractory, dt):
    """Port of rust/src/dynamics.rs lif_step, made differentiable."""
    dt_over_tau = dt / tau
    t_new = t + dt
    can_spike = (refrac_until <= t_new).to(v.dtype)

    dv = (-(v - v_rest) + current) * dt_over_tau
    v_step = torch.where(can_spike.bool(), v + dv, v)

    raw_spike = SpikeFunction.apply(v_step, v_th, KIND_SIGMOID, 5.0, 0.5)
    spike = raw_spike * can_spike                      # grad 0 while refractory

    spiked = spike.bool()
    v_new = torch.where(spiked, torch.full_like(v, float(v_reset)), v_step)
    refrac_new = torch.where(spiked, torch.full_like(refrac_until, float(t_new + refractory)),
                             refrac_until)
    return spike, v_new, refrac_new


# ---------------------------------------------------------------- block
class SpikingConnectome(nn.Module):
    """Differentiable sparse connectome + LIF, unrolled over T steps."""

    def __init__(self, n, edge_index, edge_weight, T=20, trainable_edges=True):
        super().__init__()
        self.n, self.T = n, T
        self.register_buffer("edge_index", edge_index)
        w = torch.tensor(edge_weight, dtype=torch.float32)
        if trainable_edges:
            self.edge_weight = nn.Parameter(w)
        else:
            self.register_buffer("edge_weight", w)
        self.gain = nn.Parameter(torch.ones(()))

    def _propagate(self, x):
        """y = W^T x  -- same op as torch/layer.py:41-43, but autograd-live."""
        w = torch.sparse_coo_tensor(
            self.edge_index, self.edge_weight, (self.n, self.n)
        ).coalesce()
        return torch.sparse.mm(w.transpose(0, 1), x.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, current_seq):
        """current_seq: (T, B, N) -> (T, B, N) spikes."""
        T, B, N = current_seq.shape
        v = torch.full((B, N), -65.0)
        refrac = torch.zeros(B, N)
        t = 0.0
        spikes = []
        for i in range(T):
            syn = self._propagate(current_seq[i]) * self.gain
            s, v, refrac = lif_step(v, refrac, syn, t, 20.0, -65.0, -50.0,
                                    -70.0, 2.0, 1.0)
            spikes.append(s)
            t += 1.0
        return torch.stack(spikes)


# ---------------------------------------------------------------- experiment
def build_problem(N=512, density=0.02, T=20, seed=1):
    rng = np.random.default_rng(seed)
    W = sp.random(N, N, density=density, format="coo", dtype=np.float32,
                  random_state=rng)
    W.data = rng.standard_normal(W.nnz).astype(np.float32) * 0.5
    edge_index = torch.tensor(np.vstack([W.row, W.col]), dtype=torch.long)
    return N, T, edge_index, W.data


def make_data(N, T, n=64, batch=32, n_in=16, seed=2):
    """Two classes of input current pattern. Target is decodable only from the
    spike-count response, which sits behind the spike nonlinearity."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (batch,))
    base = torch.randn(batch, n_in, generator=g)
    cur = torch.zeros(T, batch, N)
    for b in range(batch):
        cur[:, b, :n_in] = base[b] * (2.0 if y[b] == 1 else 0.0) + 0.5
    return cur, y


def run(N=512, density=0.02, T=20, steps=150, lr=3e-3, label=""):
    n_in = 16
    N, T, edge_index, ew = build_problem(N, density, T)
    net = SpikingConnectome(N, edge_index, ew, T=T, trainable_edges=True)
    inp = nn.Linear(N, N)
    head = nn.Linear(N, 2)
    params = list(net.parameters()) + list(inp.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.CrossEntropyLoss()

    cur, y = make_data(N, T, n_in=n_in)

    t0 = time.time()
    first = last = None
    for i in range(steps):
        opt.zero_grad()
        seq = inp(cur)                       # project currents into neuron space
        spk = net(seq)                       # (T,B,N)
        counts = spk.sum(0)                  # spike-count readout
        logits = head(counts)
        loss = lossf(logits, y)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    dt = time.time() - t0

    with torch.no_grad():
        spk = net(inp(cur))
        acc = float((head(spk.sum(0)).argmax(1) == y).float().mean())

    print(f"[{label}] N={N} T={T} steps={steps} "
          f"loss {first:.3f} -> {last:.3f}  acc={acc:.2f}  "
          f"{dt:.1f}s ({dt/steps*1000:.0f} ms/step)")
    return first, last, acc, dt


if __name__ == "__main__":
    print("=== gradient-path sanity check ===")
    N, T, ei, ew = build_problem(64, 0.05, 10)
    net = SpikingConnectome(N, ei, ew, T=T, trainable_edges=True)
    seq = torch.randn(T, 4, N, requires_grad=True)
    out = net(seq)
    out.sum().backward()
    print("edge_weight grad norm :", float(net.edge_weight.grad.norm()))
    print("input grad norm       :", float(seq.grad.norm()))
    print("grads nonzero        :",
          bool(net.edge_weight.grad.abs().sum() > 0 and seq.grad.abs().sum() > 0))

    print("\n=== does it actually learn? ===")
    run(N=512, density=0.02, T=20, steps=200, label="train")
    run(N=512, density=0.02, T=20, steps=200, lr=0.0, label="frozen(control)")

    print("\n=== scaling ===")
    for n in (1024, 4096):
        run(N=n, density=0.01, T=20, steps=10, label=f"scale N={n}")
