//! Parallel CSR kernels for the male-CNS connectome.
//!
//! AxonWeave's Rust core has a **serial** forward sparse matmul (no rayon
//! anywhere in `rust/src/*.rs`; `py.allow_threads` only releases the GIL) and no
//! adjoint at all. This module provides the pair needed for surrogate-gradient
//! training, in a form that parallelises regardless of batch size.
//!
//! # Formulation
//!
//! A single **gather** kernel computes `out[b, i] = sum_{j in row i} d[j] * x[b, idx[j]]`.
//! Both directions of the connectome are expressed with it by passing the matrix
//! or its transpose:
//!
//! * forward  (`presynaptic drive -> postsynaptic current`)
//!       `Y = X @ W`   ->  call with CSR of `W`   (rows = presynaptic)
//! * adjoint  (`dL/dX` from `dL/dY`)
//!       `Z = G @ W^T` ->  call with CSR of `W^T` (rows = postsynaptic)
//!
//! A gather has no write races, so it parallelises over the flattened
//! `(batch, neuron)` index space and is equally fast at B=1 and B=64 — unlike
//! the scatter form, which needs atomics or private accumulators.

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// out[b, i] = sum over nonzeros j of row i: data[j] * x[b, indices[j]]
///
/// x: (B, n) f32 -> out: (B, n) f32. `indptr` has n+1 entries.
#[pyfunction]
fn spmm_gather<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i64>,
    indptr: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let xv = x.as_array();
    let b = xv.shape()[0];
    let n = xv.shape()[1];
    let d = data.as_slice()?.to_vec();
    let idx = indices.as_slice()?.to_vec();
    let ip = indptr.as_slice()?.to_vec();
    if ip.len() != n + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr has {} entries but expected n+1 = {}",
            ip.len(),
            n + 1
        )));
    }
    let xf: Vec<f32> = xv.iter().copied().collect();

    let out: Vec<f32> = py.allow_threads(|| {
        (0..b * n)
            .into_par_iter()
            .map(|t| {
                let bi = t / n;
                let i = t % n;
                let base = bi * n;
                let mut acc = 0.0f32;
                for j in ip[i] as usize..ip[i + 1] as usize {
                    acc += d[j] * xf[base + idx[j] as usize];
                }
                acc
            })
            .collect()
    });

    Ok(out.into_pyarray_bound(py).reshape([b, n])?)
}

/// Configure the rayon global pool (0 = leave rayon's default). Call once.
#[pyfunction]
fn set_threads(n: usize) {
    if n > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
}

/// Number of threads rayon will use.
#[pyfunction]
fn n_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn fastsparse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spmm_gather, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(n_threads, m)?)?;
    Ok(())
}
