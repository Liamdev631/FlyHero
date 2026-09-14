"""Head-to-head sparse kernel benchmark: torch.sparse.mm vs scipy CSR.

AxonWeave's Rust sparse_matmul_transpose is a *serial* CSR loop (no rayon),
so scipy's CSR (also serial C) is the fairest local proxy for it. This tells
us whether "keeping Rust" retains any speed advantage over torch at all.
"""
import time
import numpy as np
import scipy.sparse as sp
import torch

torch.manual_seed(0)
rng = np.random.default_rng(0)


def bench(N, density, T=20, reps=3):
    nnz_target = int(N * N * density)
    W = sp.random(N, N, density=density, format="csr", dtype=np.float32,
                  random_state=rng)
    W.data = rng.standard_normal(W.nnz).astype(np.float32) * 0.5
    nnz = W.nnz

    coo = W.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    val = torch.tensor(coo.data, dtype=torch.float32)
    wt = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()

    x = torch.randn(N)
    xn = np.asarray(x)

    # warmup
    torch.sparse.mm(wt.transpose(0, 1), x.unsqueeze(1)).squeeze(1)
    W.T @ xn

    t0 = time.time()
    for _ in range(reps):
        for _ in range(T):
            torch.sparse.mm(wt.transpose(0, 1), x.unsqueeze(1)).squeeze(1)
    t_torch = (time.time() - t0) / (reps * T)

    t0 = time.time()
    for _ in range(reps):
        for _ in range(T):
            W.T @ xn
    t_scipy = (time.time() - t0) / (reps * T)

    print(f"N={N:>7,} nnz={nnz:>10,} | torch.sparse.mm {t_torch*1e3:8.2f} ms "
          f"({nnz/t_torch/1e6:7.1f} M-edge/s) | scipy CSR {t_scipy*1e3:8.2f} ms "
          f"({nnz/t_scipy/1e6:7.1f} M-edge/s) | torch/scipy {t_torch/t_scipy:5.2f}x")
    return t_torch, t_scipy, nnz


if __name__ == "__main__":
    print("torch threads:", torch.get_num_threads())
    print()
    rows = []
    for N, d in ((5000, 0.04), (20000, 0.01), (50000, 0.004)):
        rows.append(bench(N, d))

    print("\n--- extrapolation to the full male-cns:v1.0 substrate ---")
    print("  166,700 neurons / ~25.6M edges, T=100 timesteps, batch=1")
    # linear in nnz
    tt, ts, nnz = rows[-1]
    scale = 25_600_000 / nnz
    print(f"  forward-only, one sample, T=100:")
    print(f"    torch.sparse.mm : {tt*100*scale:7.2f} s")
    print(f"    scipy/rust CSR  : {ts*100*scale:7.2f} s")
    print(f"  NOTE: training needs forward + backward (dL/dx matmul) + the")
    print(f"  dL/dW scatter-accumulate over all {nnz*scale/1e6:.0f}M edges x 100 steps.")
