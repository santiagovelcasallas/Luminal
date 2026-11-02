// Luminal Quant — Copyright (C) 2026 Santiago Vélez Casallas. AGPL-3.0-only.
// See LICENSE at the repository root.

//! PyO3 surface: a thin, allocation-light bridge. All heavy math stays in the Rust core so the
//! engine is usable from Rust, C (via the cdylib) or Python without reimplementation.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::hadamard;

/// rotation_signs(n, seed, matrix_index) -> np.ndarray[float64] of ±1.
#[pyfunction]
fn rotation_signs<'py>(py: Python<'py>, n: usize, seed: u64, matrix_index: u64) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, hadamard::rotation_signs(n, seed, matrix_index))
}

/// rotate(weight, signs) -> np.ndarray: returns W' with each row rotated by V = H_n · diag(signs).
/// `weight` is (rows, n), row-major; `signs` is length n. Raises ValueError on bad shapes/dims.
#[pyfunction]
fn rotate<'py>(
    py: Python<'py>,
    weight: PyReadonlyArray2<'py, f64>,
    signs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let w = weight.as_array();
    let (rows, n) = (w.shape()[0], w.shape()[1]);
    let mut buf: Vec<f64> = w.iter().copied().collect();
    let s = signs.as_slice()?;
    hadamard::rotate_rows(&mut buf, n, s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let arr = numpy::PyArray2::from_vec2(py, &buf.chunks(n).map(|c| c.to_vec()).collect::<Vec<_>>())?;
    debug_assert_eq!(arr.dims(), [rows, n]);
    Ok(arr)
}

use numpy::PyReadonlyArray1;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(rotation_signs, m)?)?;
    m.add_function(wrap_pyfunction!(rotate, m)?)?;
    Ok(())
}
