use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

// 3) Distribution helpers
// ======================

/// Normal PDF: 1D x -> 1D pdf(x)
///
/// Python signature:
///     norm_pdf(x, mu=0.0, sigma=1.0)
#[pyfunction(signature = (x, mu=0.0, sigma=1.0))]
pub fn norm_pdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    mu: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive"));
    }
    let x = x.as_slice()?;
    let dist = Normal::new(mu, sigma)
        .map_err(|e| PyValueError::new_err(format!("Invalid normal params: {e}")))?;
    let vals: Vec<f64> = x.iter().copied().map(|v| dist.pdf(v)).collect();
    Ok(PyArray1::from_vec_bound(py, vals))
}

/// Normal CDF: 1D x -> 1D cdf(x)
///
/// Python signature:
///     norm_cdf(x, mu=0.0, sigma=1.0)
#[pyfunction(signature = (x, mu=0.0, sigma=1.0))]
pub fn norm_cdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    mu: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive"));
    }
    let x = x.as_slice()?;
    let dist = Normal::new(mu, sigma)
        .map_err(|e| PyValueError::new_err(format!("Invalid normal params: {e}")))?;
    let vals: Vec<f64> = x.iter().copied().map(|v| dist.cdf(v)).collect();
    Ok(PyArray1::from_vec_bound(py, vals))
}

/// Normal PPF (inverse CDF): 1D q -> 1D x
///
/// Python signature:
///     norm_ppf(q, mu=0.0, sigma=1.0)
#[pyfunction(signature = (q, mu=0.0, sigma=1.0))]
pub fn norm_ppf<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<f64>,
    mu: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive"));
    }
    let q = q.as_slice()?;
    let dist = Normal::new(mu, sigma)
        .map_err(|e| PyValueError::new_err(format!("Invalid normal params: {e}")))?;
    let vals: Vec<f64> = q.iter().copied().map(|p| dist.inverse_cdf(p)).collect();
    Ok(PyArray1::from_vec_bound(py, vals))
}
