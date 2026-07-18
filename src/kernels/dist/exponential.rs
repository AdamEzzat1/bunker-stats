use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;

// ======================
// Distribution helpers: Exponential (rate λ)
// ======================

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
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = if v < 0.0 { 0.0 } else { lam * (-lam * v).exp() };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Exponential log-PDF with rate λ: 1D x -> 1D logpdf(x)
///
/// Python signature:
///     exp_logpdf(x, lam=1.0)
#[pyfunction(signature = (x, lam=1.0))]
pub fn exp_logpdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    lam: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let log_lam = lam.ln();
    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            out[i] = f64::NAN;
        } else if v < 0.0 {
            out[i] = f64::NEG_INFINITY;
        } else {
            out[i] = log_lam - lam * v;
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
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
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        // -expm1(-lam*x) keeps full precision for tiny lam*x, where
        // 1 - exp(-lam*x) would round to 0.
        out[i] = if v < 0.0 { 0.0 } else { -(-lam * v).exp_m1() };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Exponential SF (survival function): sf(x) = exp(-lam*x) for x>=0, else 1
///
/// Python signature:
///     exp_sf(x, lam=1.0)
#[pyfunction(signature = (x, lam=1.0))]
pub fn exp_sf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    lam: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = if v < 0.0 { 1.0 } else { (-lam * v).exp() };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Exponential log-SF: log(sf(x)) = -lam*x for x>=0, else 0
///
/// Python signature:
///     exp_logsf(x, lam=1.0)
#[pyfunction(signature = (x, lam=1.0))]
pub fn exp_logsf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    lam: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            out[i] = f64::NAN;
        } else if v < 0.0 {
            out[i] = 0.0;
        } else {
            out[i] = -lam * v;
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Exponential cumulative hazard: H(x) = lam*x for x>=0, else 0
///
/// Python signature:
///     exp_cumhazard(x, lam=1.0)
#[pyfunction(signature = (x, lam=1.0))]
pub fn exp_cumhazard<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    lam: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            out[i] = f64::NAN;
        } else if v < 0.0 {
            out[i] = 0.0;
        } else {
            out[i] = lam * v;
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Exponential PPF (inverse CDF): 1D q -> 1D x
///
/// Python signature:
///     exp_ppf(q, lam=1.0)
#[pyfunction(signature = (q, lam=1.0))]
pub fn exp_ppf<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<f64>,
    lam: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if lam <= 0.0 {
        return Err(PyValueError::new_err("lam must be positive"));
    }
    let q = q.as_slice()?;

    // strict range check like SciPy
    for &p in q {
        if p.is_nan() {
            continue;
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err("q values must be in [0, 1]"));
        }
    }

    let n = q.len();
    let mut out = vec![0.0_f64; n];
    for (i, &p) in q.iter().enumerate() {
        if p.is_nan() {
            out[i] = f64::NAN;
        } else if p == 1.0 {
            out[i] = f64::INFINITY;
        } else {
            // stable: -ln(1-p)/lam using ln_1p
            out[i] = -(-p).ln_1p() / lam;
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}
