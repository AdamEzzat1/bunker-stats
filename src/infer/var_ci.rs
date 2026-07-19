// ========================================================================
// Variance confidence interval (chi-square method)
// ========================================================================
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::common::{mean, var_sample, reject_nonfinite};

/// Confidence interval for the population variance using the chi-square method.
///
/// Assumes the data is normally distributed.
/// CI: [(n-1)*s^2 / chi2_upper,  (n-1)*s^2 / chi2_lower]
///
/// Parameters:
///   x: sample data
///   confidence: confidence level (default 0.95)
///
/// Returns dict: {variance, ci_lower, ci_upper, df, std, std_ci_lower, std_ci_upper}
#[pyfunction]
#[pyo3(signature = (x, confidence=0.95))]
pub fn var_ci_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    confidence: f64,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;

    if xs.len() < 2 {
        return Err(PyValueError::new_err("var_ci requires n >= 2"));
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(PyValueError::new_err("confidence must be in (0, 1)"));
    }

    reject_nonfinite(xs, "x")?;

    let n = xs.len();
    let m = mean(xs);
    let s2 = var_sample(xs, m);

    if s2 <= 0.0 {
        return Err(PyValueError::new_err("sample variance is zero"));
    }

    let df = (n - 1) as f64;
    let alpha = 1.0 - confidence;

    let chi2 = ChiSquared::new(df).unwrap();
    let chi2_lower = chi2.inverse_cdf(alpha / 2.0);
    let chi2_upper = chi2.inverse_cdf(1.0 - alpha / 2.0);

    // CI for variance: (df * s^2) / chi2_quantile
    let var_lo = df * s2 / chi2_upper;
    let var_hi = df * s2 / chi2_lower;

    let d = PyDict::new_bound(py);
    d.set_item("variance", s2)?;
    d.set_item("ci_lower", var_lo)?;
    d.set_item("ci_upper", var_hi)?;
    d.set_item("df", df)?;
    d.set_item("std", s2.sqrt())?;
    d.set_item("std_ci_lower", var_lo.sqrt())?;
    d.set_item("std_ci_upper", var_hi.sqrt())?;
    Ok(d.unbind())
}
