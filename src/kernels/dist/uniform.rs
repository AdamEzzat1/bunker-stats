use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;

#[inline]
fn validate_uniform(a: f64, b: f64) -> PyResult<f64> {
    if !a.is_finite() || !b.is_finite() {
        return Err(PyValueError::new_err("a and b must be finite"));
    }
    let width = b - a;
    if width <= 0.0 {
        return Err(PyValueError::new_err("b must be greater than a"));
    }
    Ok(width)
}

// ======================
// Distribution helpers: Uniform(a, b)
// ======================

/// Uniform PDF: 1D x -> 1D pdf(x)
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
    if b <= a {
        return Err(PyValueError::new_err("b must be greater than a"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let width = b - a;
    let inv_w = 1.0 / width;

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = if v < a || v > b { 0.0 } else { inv_w };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Uniform log-PDF: 1D x -> 1D logpdf(x)
///
/// Python signature:
///     unif_logpdf(x, a=0.0, b=1.0)
#[pyfunction(signature = (x, a=0.0, b=1.0))]
pub fn unif_logpdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if b <= a {
        return Err(PyValueError::new_err("b must be greater than a"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let width = b - a;
    let log_inv_w = -width.ln();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            out[i] = f64::NAN;
        } else if v < a || v > b {
            out[i] = f64::NEG_INFINITY;
        } else {
            out[i] = log_inv_w;
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Uniform CDF: 1D x -> 1D cdf(x)
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
    if b <= a {
        return Err(PyValueError::new_err("b must be greater than a"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let width = b - a;
    let inv_w = 1.0 / width;

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = if v <= a {
            0.0
        } else if v >= b {
            1.0
        } else {
            (v - a) * inv_w
        };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Uniform SF: sf(x) = 1 - cdf(x)
///
/// Python signature:
///     unif_sf(x, a=0.0, b=1.0)
#[pyfunction(signature = (x, a=0.0, b=1.0))]
pub fn unif_sf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if b <= a {
        return Err(PyValueError::new_err("b must be greater than a"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let width = b - a;
    let inv_w = 1.0 / width;

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = if v <= a {
            1.0
        } else if v >= b {
            0.0
        } else {
            (b - v) * inv_w
        };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Uniform log-SF: log(sf(x))
///
/// CRITICAL: Uses piecewise logic to avoid ln(negative)
///
/// Python signature:
///     unif_logsf(x, a=0.0, b=1.0)
#[pyfunction(signature = (x, a=0.0, b=1.0))]
pub fn unif_logsf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let width = validate_uniform(a, b)?;
    let x = x.as_slice()?;
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = if v.is_nan() {
            f64::NAN
        } else if v <= a {
            // S(x) = 1 for x <= a, so log(1) = 0
            0.0
        } else if v >= b {
            // S(x) = 0 for x >= b, so log(0) = -inf
            f64::NEG_INFINITY
        } else {
            // a < v < b: S(x) = (b-v)/(b-a), guaranteed positive
            let sf = (b - v) / width;
            sf.ln()
        };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Uniform cumulative hazard: H(x) = -ln(sf(x))
///
/// CRITICAL: Computes logsf locally to ensure H(x) = +inf for x >= b (not nan)
///
/// Python signature:
/// Uniform cumulative hazard: H(x) = -ln(sf(x))
///
/// BULLETPROOF VERSION - Direct computation, impossible to return nan
///
/// Python signature:
///     unif_cumhazard(x, a=0.0, b=1.0)
#[pyfunction(signature = (x, a=0.0, b=1.0))]
pub fn unif_cumhazard<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let width = validate_uniform(a, b)?;
    let x = x.as_slice()?;
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        // Direct piecewise computation - no intermediate logsf variable
        out[i] = if v.is_nan() {
            f64::NAN
        } else if v <= a {
            // H(x) = -log(S(x)) = -log(1) = 0
            0.0
        } else if v >= b {
            // H(x) = -log(S(x)) = -log(0) = +inf
            // CRITICAL: Direct assignment, not -(-inf)
            f64::INFINITY
        } else {
            // a < v < b
            // H(x) = -log((b-v)/(b-a)) = log((b-a)/(b-v))
            // Compute directly to avoid any ln(negative) possibility
            let sf = (b - v) / width;
            // sf is guaranteed positive here: a < v < b means b > v, so b - v > 0
            -sf.ln()
        };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}
/// Uniform PPF (inverse CDF): 1D q -> 1D x
///
/// Python signature:
///     unif_ppf(q, a=0.0, b=1.0)
#[pyfunction(signature = (q, a=0.0, b=1.0))]
pub fn unif_ppf<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if b <= a {
        return Err(PyValueError::new_err("b must be greater than a"));
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
    let width = b - a;

    let mut out = vec![0.0_f64; n];
    for (i, &p) in q.iter().enumerate() {
        if p.is_nan() {
            out[i] = f64::NAN;
        } else {
            out[i] = a + p * width;
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}
