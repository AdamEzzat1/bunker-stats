use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;

/// Exponential PDF with rate λ: 1D x -> 1D pdf(x)
///
/// Python signature:
///     exp_pdf(x, lam=1.0)
#[pyfunction(signature = (x, lam=1.0))]
pub fn exp_pdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    lam: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }
    let x = x.as_slice()?;
    let vals: Vec<f64> = x
        .iter()
        .copied()
        .map(|v| {
            if v < 0.0 {
                0.0
            } else {
                lam * (-lam * v).exp()
            }
        })
        .collect();
    Ok(PyArray1::from_vec_bound(py, vals))
}

/// Exponential CDF with rate λ: 1D x -> 1D cdf(x)
///
/// Python signature:
///     exp_cdf(x, lam=1.0)
#[pyfunction(signature = (x, lam=1.0))]
pub fn exp_cdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    lam: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }
    let x = x.as_slice()?;
    let vals: Vec<f64> = x
        .iter()
        .copied()
        .map(|v| {
            if v < 0.0 {
                0.0
            } else {
                1.0 - (-lam * v).exp()
            }
        })
        .collect();
    Ok(PyArray1::from_vec_bound(py, vals))
}
