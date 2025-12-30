//! Mann–Whitney U (rank-sum) test.
//! v0.3 plan: asymptotic p-value with tie correction.
//!
//! Stub for now: compiles and exports, but raises NotImplementedError.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn mann_whitney_u_np(
    _py: Python<'_>,
    _x: numpy::PyReadonlyArray1<f64>,
    _y: numpy::PyReadonlyArray1<f64>,
    _alternative: &str,
) -> PyResult<Py<PyDict>> {
    Err(PyNotImplementedError::new_err(
        "mann_whitney_u_np not implemented yet (Pillar C step 3)",
    ))
}
