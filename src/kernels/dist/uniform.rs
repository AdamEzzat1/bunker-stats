use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;

/// Uniform PDF on [low, high]: 1D x -> 1D pdf(x)
///
/// Python signature:
///     unif_pdf(x, a=0.0, b=1.0)
#[pyfunction(signature = (x, a=0.0, b=1.0))]
pub fn unif_pdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(b > a) {
        return Err(PyValueError::new_err("b must be greater than a"));
    }
    let width = b - a;
    let x = x.as_slice()?;
    let vals: Vec<f64> = x
        .iter()
        .copied()
        .map(|v| {
            if v < a || v > b {
                0.0
            } else {
                1.0 / width
            }
        })
        .collect();
    Ok(PyArray1::from_vec_bound(py, vals))
}

/// Uniform CDF on [low, high]: 1D x -> 1D cdf(x)
///
/// Python signature:
///     unif_cdf(x, a=0.0, b=1.0)
#[pyfunction(signature = (x, a=0.0, b=1.0))]
pub fn unif_cdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(b > a) {
        return Err(PyValueError::new_err("b must be greater than a"));
    }
    let width = b - a;
    let x = x.as_slice()?;
    let vals: Vec<f64> = x
        .iter()
        .copied()
        .map(|v| {
            if v <= a {
                0.0
            } else if v >= b {
                1.0
            } else {
                (v - a) / width
            }
        })
        .collect();
    Ok(PyArray1::from_vec_bound(py, vals))
}
