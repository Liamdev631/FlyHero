"""Is the weak learning explained by the current-scale / threshold mismatch?

LIF steady state:  v_ss = v_rest + I   (from dv = (-(v - v_rest) + I) * dt/tau)
So a neuron fires only if  I > v_th - v_rest = -50 - (-65) = 15 mV.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import surrogate_proto as P

torch.manual_seed(0)
N, T, DENS = 1024, 20, 0.01


def build():
    rng = np.random.default_rng(1)
    W = sp.random(N, N, density=DENS, format="coo", dtype=np.float32, random_state=rng)
    W.data = rng.standard_normal(W.nnz).astype(np.float32) * 0.5
    ei = torch.tensor(np.vstack([W.row, W.col]), dtype=torch.long)
    return ei, W.data


print("=== spike rate vs input current scale ===")
ei, ew = build()
net = P.SpikingConnectome(N, ei, ew, T=T, trainable_edges=False)
print(f"{'gain':>8} {'mean rate (Hz-ish)':>20} {'active frac':>12}")
for gain in (1, 5, 10, 20, 40, 80, 160):
    cur = torch.full((T, 4, N), float(gain))
    with torch.no_grad():
        s = net(cur)
    print(f"{gain:>8} {float(s.mean()):>20.4f} {float((s.sum(0)>0).float().mean()):>12.4f}")

print("\n=== learning vs input current scale (500 steps) ===")
for gain in (10, 40, 160):
    torch.manual_seed(0)
    ei, ew = build()
    net = P.SpikingConnectome(N, ei, ew, T=T, trainable_edges=True)
    inp = nn.Linear(N, N)
    head = nn.Linear(N, 2)
    opt = torch.optim.Adam(list(net.parameters()) + list(inp.parameters())
                           + list(head.parameters()), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(2)
    y = torch.randint(0, 2, (32,))
    base = torch.randn(32, 16, generator=g)
    cur = torch.zeros(T, 32, N)
    for b in range(32):
        cur[:, b, :16] = base[b] * (2.0 if y[b] == 1 else 0.0)
    cur = cur * gain
    first = last = None
    for i in range(500):
        opt.zero_grad()
        spk = net(inp(cur))
        loss = lossf(head(spk.sum(0)), y)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    with torch.no_grad():
        s = net(inp(cur))
        acc = float((head(s.sum(0)).argmax(1) == y).float().mean())
    print(f"gain={gain:>4}  loss {first:.4f} -> {last:.4f}  acc={acc:.2f}")
