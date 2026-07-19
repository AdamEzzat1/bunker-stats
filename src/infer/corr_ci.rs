// ========================================================================
// Correlation confidence interval (Fisher z-transform)
// ========================================================================
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, Normal};

use super::common::{mean, reject_nonfinite, rankdata_average};

/// Pearson or Spearman correlation confidence interval via Fisher z-transform.
///
/// The Fisher z-transform provides an approximate CI for correlation
/// coefficients by exploiting the fact that z = atanh(r) is approximately
/// normal with variance 1/(n-3).
///
/// Parameters:
///   x, y: paired arrays
///   method: "pearson" or "spearman"
///   confidence: confidence level (default 0.95)
///
/// Returns dict: {correlation, ci_lower, ci_upper, z, se}
#[pyfunction]
#[pyo3(signature = (x, y, method="pearson", confidence=0.95))]
pub fn corr_ci_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
    method: &str,
    confidence: f64,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    let n = xs.len();
    if n < 4 {
        return Err(PyValueError::new_err("corr_ci requires n >= 4 (n-3 > 0 for Fisher z)"));
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(PyValueError::new_err("confidence must be in (0, 1)"));
    }

    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;

    // Compute correlation based on method
    let r = match method.to_lowercase().as_str() {
        "pearson" => pearson_r(xs, ys)?,
        "spearman" => {
            let rx = rankdata_average(xs);
            let ry = rankdata_average(ys);
            pearson_r(&rx, &ry)?
        }
        _ => {
            return Err(PyValueError::new_err(
                "method must be 'pearson' or 'spearman'",
            ))
        }
    };

    // Fisher z-transform: z = atanh(r)
    let r_clamped = r.max(-0.9999999).min(0.9999999);
    let z = r_clamped.atanh();

    // SE of z = 1 / sqrt(n - 3)
    let se = 1.0 / ((n as f64 - 3.0).sqrt());

    // z-critical for the given confidence level
    let alpha = 1.0 - confidence;
    let norm = Normal::new(0.0, 1.0).unwrap();
    let z_crit = norm.inverse_cdf(1.0 - alpha / 2.0);

    // CI in z-space, then back-transform
    let z_lo = z - z_crit * se;
    let z_hi = z + z_crit * se;
    let ci_lo = z_lo.tanh();
    let ci_hi = z_hi.tanh();

    let d = PyDict::new_bound(py);
    d.set_item("correlation", r)?;
    d.set_item("ci_lower", ci_lo)?;
    d.set_item("ci_upper", ci_hi)?;
    d.set_item("z", z)?;
    d.set_item("se", se)?;
    Ok(d.unbind())
}

/// Compute Pearson r from two slices.
fn pearson_r(xs: &[f64], ys: &[f64]) -> PyResult<f64> {
    let n = xs.len();
    let mx = mean(xs);
    let my = mean(ys);

    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;

    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    if sxx == 0.0 || syy == 0.0 {
        return Err(PyValueError::new_err("zero variance in x or y"));
    }

    Ok(sxy / (sxx * syy).sqrt())
}
