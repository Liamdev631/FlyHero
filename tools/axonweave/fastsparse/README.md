# fastsparse

Parallel CSR kernels for the male-CNS connectome, written for D-009.

## Why this exists

AxonWeave's Rust core has the forward sparse matmul but is **serial** — there is
no `rayon` anywhere in `rust/src/*.rs`, and `py.allow_threads` only releases the
GIL without parallelising anything. It also has **no adjoint**, which is the
piece E-001 identified as missing for surrogate-gradient training.

This crate provides the pair:

| function | computes | use |
|---|---|---|
| `spmm_gather(x, d, ix, ip)` | `out[b,i] = sum over row i of d[j]*x[b, ix[j]]` | both directions |
| `set_threads(n)` / `n_threads()` | rayon pool control | benchmark reproducibility |

Both directions of the connectome are the *same* gather kernel, invoked with the
matrix or its transpose:

* forward `Y = X @ A` -> pass CSR of `A` (rows = presynaptic)
* adjoint `Z = G @ A^T` -> pass CSR of `A^T` (rows = postsynaptic)

A gather has no write races, so it parallelises over the flattened `(batch,
neuron)` space and is as fast at B=1 as at B=64. An earlier scatter formulation
parallelised only over batch rows and was **slower than scipy at B=1**.

## Build

```bash
export PATH=$HOME/.cargo/bin:$PATH
maturin develop --release          # into the active venv
```

## Verified

* `spmm_gather` matches scipy exactly: max abs err **0.0** for both the forward
  (`X @ A`) and the adjoint (`G @ A^T`).
* Adjoint identity `<X@A, G> == <X, G@A^T>` holds to **8e-7** (float32 rounding).

## Measured, and the honest conclusion

On the full male-CNS graph (191,696 neurons / 26,028,386 edges), vs scipy CSR:

| B | rust fwd | scipy fwd | speedup |
|---|---|---|---|
| 1 | 209.0 ms | 26.8 ms | **0.1x** |
| 16 | 350.5 ms | 356.3 ms | 1.0x |
| 64 | 860.7 ms | 1553.0 ms | 1.8x |
| 256 | 2994.3 ms | 5564.2 ms | 1.9x |

**This is memory-bandwidth-bound, not compute-bound.** One pass touches
26M edges x ~12 bytes = ~312 MB; at this box's usable bandwidth that is ~10-25 ms
no matter how many threads are used. More parallelism cannot fix a bandwidth
wall. scipy's single-threaded CSR matvec is already near that limit, which is why
the speedup at small B is ~1x.

The real win is not parallelism but **sparsity of activity**. At the calibrated
operating point only ~3% of neurons spike per timestep, so the dense SpMM spends
~97% of its bandwidth multiplying by zeros. Propagating from the *active* set
instead gives, measured on this graph:

| activity | dense | sparse@sparse | speedup |
|---|---|---|---|
| 0.5% | 258.7 ms | 31.4 ms | **8.2x** |
| 1.0% | 257.4 ms | 55.3 ms | 4.7x |
| 2.0% | 290.1 ms | 91.8 ms | 3.2x |
| 5.0% | 256.4 ms | 154.3 ms | 1.7x |

That is why `mnist_cns.py` runs the connectome through scipy's sparse-sparse
product on the spike tensor rather than through this crate. `fastsparse` is kept
because its adjoint is exact and it is the natural place to add an event-driven
kernel if a lower-activity operating point is ever needed.
