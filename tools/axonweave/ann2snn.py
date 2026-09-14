"""
ANN(ReLU) -> IF conversion feasibility, tested against AxonWeave's actual
LIF kernel formulation.

Part A: why you cannot reach IF by tuning tau in AxonWeave's lif_step.
Part B: rate-vs-I curve, IF vs LIF vs ReLU.
Part C: full MLP conversion, accuracy vs T, soft vs hard reset.
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

V_TH = 1.0
V_RESET = 0.0
DT_TAU = 1.0     # dt/tau as it appears in AxonWeave's lif_step


# ---------------------------------------------------------------- kernels
def rate_of(fn, I, T=200):
    """Mean firing rate (spikes/timestep) of a single neuron under constant I."""
    n = I.shape[0]
    v = np.zeros(n)
    count = np.zeros(n)
    for _ in range(T):
        v, s = fn(v, I)
        count += s
    return count / T


def make_if(v_reset=0.0, soft=True):
    def step(v, I):
        v = v + I * DT_TAU                      # leak term removed
        spiked = v >= V_TH
        s = spiked.astype(np.float64)
        v = np.where(spiked, v - V_TH if soft else v_reset, v)
        return v, s
    return step


def make_lif(tau, v_rest=0.0, soft=True):
    """AxonWeave's lif_step: dv = (-(v - v_rest) + I) * dt/tau."""
    def step(v, I):
        dt_over_tau = DT_TAU / tau
        v = v + (-(v - v_rest) + I) * dt_over_tau
        spiked = v >= V_TH
        s = spiked.astype(np.float64)
        v = np.where(spiked, v - V_TH if soft else v_reset, v)
        return v, s
    return step


print("=" * 70)
print("PART A -- can tau be tuned to reach IF in AxonWeave's lif_step?")
print("=" * 70)
print("AxonWeave lif_step hardcodes the leak:  dv = (-(v - v_rest) + I)*dt/tau")
print("Setting tau -> inf makes dt/tau -> 0, so dv -> 0: the neuron FREEZES.")
print("The leak term must be REMOVED, not its time constant increased.\n")

I_test = np.full(16, 0.5)
print(f"constant I = 0.5, T = 200 steps, v_th = 1.0")
print(f"{'dynamics':>28} {'firing rate':>14}")
for tau in (1.0, 5.0, 100.0, 1e6):
    r = rate_of(make_lif(tau), I_test).mean()
    print(f"{'LIF tau=' + str(tau):>28} {r:>14.6f}")
r = rate_of(make_if(), I_test).mean()
print(f"{'IF (leak removed)':>28} {r:>14.6f}")
print(f"{'ReLU(I) reference':>28} {0.5:>14.6f}")


print()
print("=" * 70)
print("PART B -- rate-vs-I curve: IF vs LIF vs ReLU")
print("=" * 70)
Is = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5])
print(f"{'I':>6} {'ReLU(I)':>9} {'IF rate':>9} {'IF err':>9} "
      f"{'LIF tau=1':>10} {'LIF tau=10':>11}")
if_r = rate_of(make_if(), Is, T=1000)
lif1 = rate_of(make_lif(1.0), Is, T=1000)
lif10 = rate_of(make_lif(10.0), Is, T=1000)
for i, x in enumerate(Is):
    print(f"{x:>6.2f} {max(0.0, x):>9.3f} {if_r[i]:>9.3f} "
          f"{abs(if_r[i] - max(0.0, x)):>9.4f} {lif1[i]:>10.3f} {lif10[i]:>11.3f}")

print("\nLIF's dead zone: steady state is v_ss = v_rest + I, so it fires only")
print("when I >= v_th - v_rest = 1.0. Below that the rate is exactly 0.")
print("=> LIF ~ ReLU(I - 1)  [shifted]      IF ~ ReLU(I)  [unshifted]")
print("That shift is why conversion targets IF, not LIF -- the user is right.")


print()
print("=" * 70)
print("PART C -- full MLP conversion: train ReLU net, convert to IF")
print("=" * 70)

D, H, C, N = 20, 64, 2, 4000
g = torch.Generator().manual_seed(1)
X = torch.randn(N, D, generator=g)
# XOR-ish: non-linearly separable (a linear model gets ~50%)
ylab = (X[:, 0] * X[:, 1] > 0).long()
g2 = torch.Generator().manual_seed(99)
Xte = torch.randn(2000, D, generator=g2)
yte = (Xte[:, 0] * Xte[:, 1] > 0).long()

ann = nn.Sequential(nn.Linear(D, H), nn.ReLU(),
                    nn.Linear(H, H), nn.ReLU(),
                    nn.Linear(H, C))
opt = torch.optim.Adam(ann.parameters(), lr=1e-2)
lossf = nn.CrossEntropyLoss()
for ep in range(400):
    opt.zero_grad()
    loss = lossf(ann(X), ylab)
    loss.backward()
    opt.step()
with torch.no_grad():
    ann_acc = float((ann(X).argmax(1) == ylab).float().mean())
    ann_acc_te = float((ann(Xte).argmax(1) == yte).float().mean())
print(f"ANN trained: loss={float(loss):.4f}  train acc={ann_acc:.3f}  "
      f"test acc={ann_acc_te:.3f}")


def convert(ann, X):
    """Rueckauer-style: scale each layer by its own output activation scale so
    activations land in [0,1]; then ReLU -> IF is a drop-in swap."""
    layers, acts = [], [X]
    a = X
    for m in ann:
        if isinstance(m, nn.Linear):
            a = m(a)
            layers.append(m)
            acts.append(torch.relu(a))
    L = len(layers)
    # lam[i] = 99.9th pct of layer i's *output* activation
    lam = [float(torch.quantile(acts[i + 1].flatten().float(),
                                torch.tensor(0.999))) for i in range(L)]
    lam = [max(l, 1e-6) for l in lam]
    params = []
    for i, m in enumerate(layers):
        W, b = m.weight.data.clone(), m.bias.data.clone()
        if i < L - 1:                     # last layer left unnormalised
            W, b = W / lam[i], b / lam[i]
        params.append((W, b))
    return params, lam


def snn_forward(params, X, T, soft=True):
    B = X.shape[0]
    r = X
    for i, (W, b) in enumerate(params):
        n_out = W.shape[0]
        v = torch.zeros(B, n_out)
        acc = torch.zeros(B, n_out)
        spikes = torch.zeros(B, n_out)
        for t in range(T):
            v = v + r @ W.t() + b
            spiked = v >= V_TH
            spikes = spikes + spiked.float()
            v = torch.where(spiked, v - V_TH if soft else torch.zeros_like(v), v)
            acc = acc + v
        if i == len(params) - 1:
            return acc                    # last layer: use accumulated potential
        r = spikes / T                    # rate-coded output to next layer
    return r


params, lam = convert(ann, X)
print(f"\nnormalisation lambdas (99.9pct per layer): "
      f"{[round(l, 3) for l in lam]}")
print(f"\n{'T':>6} {'soft-reset acc':>16} {'hard-reset acc':>16}")
for T in (10, 25, 50, 100, 200, 500):
    a_soft = float((snn_forward(params, Xte, T, soft=True).argmax(1) == yte)
                   .float().mean())
    a_hard = float((snn_forward(params, Xte, T, soft=False).argmax(1) == yte)
                   .float().mean())
    print(f"{T:>6} {a_soft:>16.3f} {a_hard:>16.3f}")
print(f"\n(ANN test ceiling = {ann_acc_te:.3f})")
