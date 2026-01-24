use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;

// Mathematical constants
const LN_2PI: f64 = 1.8378770664093453; // ln(2π)
const SQRT_2: f64 = 1.4142135623730951; // √2
const INV_SQRT_2: f64 = 0.7071067811865476; // 1/√2

// ======================
// Distribution helpers: Normal
// ======================
//
// IMPLEMENTATION NOTES:
// - Uses libm::erfc for CDF/SF/LogSF to achieve SciPy-level precision
// - statrs Normal::cdf has ~1e-11 precision vs SciPy's ~1e-14
// - erfc-based approach matches SciPy's ndtr implementation
// - All functions handle NaN correctly and use stable numerics
// - PPF uses statrs but with explicit NaN and boundary handling

/// Numerically stable Normal CDF using complementary error function
///
/// Φ(x) = 0.5 * erfc(-z/√2) where z = (x-μ)/σ
///
/// This matches SciPy's implementation and provides ~1e-14 precision
#[inline]
fn norm_cdf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let z = (x - mu) / sigma;
    0.5 * libm::erfc(-z * INV_SQRT_2)
}

/// Numerically stable Normal SF (survival function)
///
/// S(x) = 0.5 * erfc(z/√2) where z = (x-μ)/σ
///
/// More stable than 1 - cdf(x) for large positive x
#[inline]
fn norm_sf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let z = (x - mu) / sigma;
    0.5 * libm::erfc(z * INV_SQRT_2)
}

/// Numerically stable Normal log-SF
///
/// log(S(x)) computed via stable SF
#[inline]
fn norm_logsf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let sf = norm_sf_scalar(x, mu, sigma);
    if sf <= 0.0 {
        f64::NEG_INFINITY
    } else {
        libm::log(sf)
    }
}

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
    let n = x.len();

    let inv_sigma = 1.0 / sigma;
    let norm_const = inv_sigma / libm::sqrt(2.0 * std::f64::consts::PI);

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            out[i] = f64::NAN;
            continue;
        }
        let z = (v - mu) * inv_sigma;
        out[i] = norm_const * libm::exp(-0.5 * z * z);
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Normal log-PDF: 1D x -> 1D logpdf(x)
///
/// Python signature:
///     norm_logpdf(x, mu=0.0, sigma=1.0)
#[pyfunction(signature = (x, mu=0.0, sigma=1.0))]
pub fn norm_logpdf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    mu: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let inv_sigma = 1.0 / sigma;
    let log_sigma = libm::log(sigma);

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            out[i] = f64::NAN;
            continue;
        }
        let z = (v - mu) * inv_sigma;
        out[i] = -0.5 * LN_2PI - log_sigma - 0.5 * z * z;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Normal CDF: 1D x -> 1D cdf(x)
///
/// Uses erfc for SciPy-level precision (~1e-14 vs statrs ~1e-11)
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
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = norm_cdf_scalar(v, mu, sigma);
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Normal SF (survival function): 1D x -> 1D sf(x) = 1 - cdf(x)
///
/// Uses erfc directly for better numerical stability in tails
///
/// Python signature:
///     norm_sf(x, mu=0.0, sigma=1.0)
#[pyfunction(signature = (x, mu=0.0, sigma=1.0))]
pub fn norm_sf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    mu: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = norm_sf_scalar(v, mu, sigma);
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Normal log-SF: 1D x -> 1D log(sf(x))
///
/// Computed via stable SF to avoid log(1 - cdf(x)) cancellation
///
/// Python signature:
///     norm_logsf(x, mu=0.0, sigma=1.0)
#[pyfunction(signature = (x, mu=0.0, sigma=1.0))]
pub fn norm_logsf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    mu: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = norm_logsf_scalar(v, mu, sigma);
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Normal cumulative hazard: 1D x -> 1D H(x) = -ln(sf(x))
///
/// Python signature:
///     norm_cumhazard(x, mu=0.0, sigma=1.0)
#[pyfunction(signature = (x, mu=0.0, sigma=1.0))]
pub fn norm_cumhazard<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    mu: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive"));
    }
    let x = x.as_slice()?;
    let n = x.len();

    let mut out = vec![0.0_f64; n];
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            out[i] = f64::NAN;
            continue;
        }
        let sf = norm_sf_scalar(v, mu, sigma);
        out[i] = if sf <= 0.0 {
            f64::INFINITY
        } else {
            -libm::log(sf)
        };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Normal PPF (inverse CDF): 1D q -> 1D x
///
/// Uses statrs for inverse but with explicit NaN and boundary handling
/// to prevent crashes and match SciPy semantics
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

    // Strict range check like SciPy - validates before any computation
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

    // Use statrs for inverse but handle special cases explicitly
    use statrs::distribution::{ContinuousCDF, Normal};
    let dist = Normal::new(mu, sigma)
        .map_err(|e| PyValueError::new_err(format!("Invalid normal params: {e}")))?;

    for (i, &p) in q.iter().enumerate() {
        if p.is_nan() {
            // NaN input -> NaN output (prevents crashes)
            out[i] = f64::NAN;
        } else if p == 0.0 {
            // q=0 -> -inf (explicit to avoid statrs edge case)
            out[i] = f64::NEG_INFINITY;
        } else if p == 1.0 {
            // q=1 -> +inf (explicit to avoid statrs edge case)
            out[i] = f64::INFINITY;
        } else {
            // Safe to call inverse_cdf - we've validated range
            out[i] = dist.inverse_cdf(p);
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}
