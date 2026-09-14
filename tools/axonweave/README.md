# AxonWeave evaluation — surrogate-gradient feasibility

Scripts written while evaluating `https://github.com/dhakalnirajan/axonweave`
as a simulator for the FlyHero policy. They are standalone: they do **not**
import axonweave and do not need the multi-GB MaleCNS substrate. Instead they
re-implement AxonWeave's kernels 1:1 in PyTorch so the questions ("is it an
ANN or an SNN?", "how hard is surrogate-gradient learning?", "does keeping the
Rust kernels retain speed?") could be answered by measurement.

Ported from:
- `rust/src/dynamics.rs`  — `lif_step`
- `rust/src/surrogate.rs` — `surrogate_grad`, `surrogate_lif_step`

## Environment

```
python3 -m venv /tmp/awvev
/tmp/awvev/bin/pip install numpy scipy
/tmp/awvev/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Note: the PyTorch CPU index does not host `numpy`/`scipy`; install those from
PyPI (passing `--index-url` for the whole command fails).

Verified against torch 2.14.0+cpu / scipy 1.17.1 / numpy 2.4.6, Python 3.11.16.

## Scripts

### `surrogate_proto.py`

A differentiable LIF + sparse-connectome block. The spike nonlinearity is a
`torch.autograd.Function` — hard `(v >= v_th)` forward, surrogate derivative
backward (sigmoid k=5.0, matching AxonWeave's `SigmoidSurrogate` default).
Unrolled over T timesteps in framework-native ops instead of detaching to NumPy.

Demonstrates the thing AxonWeave's `IMPLEMENTATION_GAP.md` §7 says is *pending*:
"wiring them into a differentiable framework forward pass (e.g. a torch
custom-autograd path)".

Run: `/tmp/awvev/bin/python surrogate_proto.py`

### `sweep_scale.py`

Two measurements that matter for integration:

1. **Spike rate vs input current scale.** LIF steady state is `v_ss = v_rest + I`,
   so a neuron fires only when `I > v_th - v_rest = 15 mV`. Below that the
   substrate is *completely silent* (measured rate 0.0000) and nothing learns.
   AxonWeave's stock `ImageEncoder` normalises by `1/sqrt(n_input)` and produces
   O(1) values — so the stock encoder feeds a silent brain.
2. **Learning vs input scale.** Above the threshold gap it learns cleanly to
   100% accuracy.

Run: `/tmp/awvev/bin/python sweep_scale.py`

### `bench_sparse.py`

`torch.sparse.mm` vs a CSR kernel (scipy is the fair local proxy for
AxonWeave's Rust kernel, which is a serial CSR loop — there is no `rayon`
anywhere in `rust/src/*.rs`, and `py.allow_threads` only releases the GIL).
This is the measurement that decides whether training must stay in native code.

Run: `/tmp/awvev/bin/python bench_sparse.py`

## Headline results

| Measurement | Result |
|---|---|
| Surrogate gradient reaches sparse edge weights | yes — `edge_weight.grad` nonzero |
| Learning with correct input scaling | loss 0.694 → 0.051, 100% accuracy |
| Learning with O(1) input currents | none — network totally silent |
| `torch.sparse.mm` vs CSR (1.0M nnz) | 32.4 ms vs 0.64 ms — **50x slower** |
| `torch.sparse.mm` vs CSR (10.0M nnz) | 363 ms vs 14.0 ms — **26x slower** |

Conclusion recorded in `DECISIONS.md` as E-001 (evaluated, not decided).
