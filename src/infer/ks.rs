//! One-sample Kolmogorov–Smirnov test.
//! v0.3 plan: named CDFs only ("norm", "uniform", "expon") with asymptotic p-value.
//!
//! Stub for now: compiles and exports, but raises NotImplementedError.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn ks_1samp_np(
    _py: Python<'_>,
    _x: numpy::PyReadonlyArray1<f64>,
    _cdf: &str,
    _params: Vec<f64>,
    _alternative: &str,
) -> PyResult<Py<PyDict>> {
    Err(PyNotImplementedError::new_err(
        "ks_1samp_np not implemented yet (Pillar C step 4)",
    ))
}
