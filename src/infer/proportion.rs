// ========================================================================
// Proportion z-tests
// ========================================================================
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, Normal};

use super::common::Alternative;

/// One-sample proportion z-test.
///
/// Tests H0: p = value (population proportion equals hypothesized value).
///
/// Uses normal approximation with optional continuity correction.
///
/// Parameters:
///   count: number of successes
///   nobs: number of observations
///   value: hypothesized proportion (default 0.5)
///   alternative: "two-sided", "less", "greater"
///
/// Returns dict: {statistic, pvalue, proportion}
#[pyfunction]
#[pyo3(signature = (count, nobs, value=0.5, alternative="two-sided"))]
pub fn proportion_ztest_np(
    py: Python<'_>,
    count: f64,
    nobs: f64,
    value: f64,
    alternative: &str,
) -> PyResult<Py<PyDict>> {
    if nobs <= 0.0 || !nobs.is_finite() {
        return Err(PyValueError::new_err("nobs must be a positive finite number"));
    }
    if count < 0.0 || count > nobs || !count.is_finite() {
        return Err(PyValueError::new_err("count must be in [0, nobs]"));
    }
    if value <= 0.0 || value >= 1.0 {
        return Err(PyValueError::new_err("value must be in (0, 1)"));
    }

    let alt = Alternative::from_str(alternative)?;

    let p_hat = count / nobs;
    let se = (value * (1.0 - value) / nobs).sqrt();

    if se == 0.0 {
        return Err(PyValueError::new_err("standard error is zero (degenerate case)"));
    }

    let z = (p_hat - value) / se;

    let norm = Normal::new(0.0, 1.0).unwrap();
    let p = match alt {
        Alternative::TwoSided => (2.0 * norm.sf(z.abs())).clamp(0.0, 1.0),
        Alternative::Greater => norm.sf(z).clamp(0.0, 1.0),
        Alternative::Less => norm.cdf(z).clamp(0.0, 1.0),
    };

    let d = PyDict::new_bound(py);
    d.set_item("statistic", z)?;
    d.set_item("pvalue", p)?;
    d.set_item("proportion", p_hat)?;
    Ok(d.unbind())
}

/// Two-sample proportion z-test.
///
/// Tests H0: p1 = p2 (two population proportions are equal).
///
/// Uses pooled proportion estimate under H0.
///
/// Parameters:
///   count1, nobs1: successes and observations in sample 1
///   count2, nobs2: successes and observations in sample 2
///   alternative: "two-sided", "less", "greater"
///
/// Returns dict: {statistic, pvalue, proportion1, proportion2, diff}
#[pyfunction]
#[pyo3(signature = (count1, nobs1, count2, nobs2, alternative="two-sided"))]
pub fn two_proportions_ztest_np(
    py: Python<'_>,
    count1: f64,
    nobs1: f64,
    count2: f64,
    nobs2: f64,
    alternative: &str,
) -> PyResult<Py<PyDict>> {
    if nobs1 <= 0.0 || nobs2 <= 0.0 || !nobs1.is_finite() || !nobs2.is_finite() {
        return Err(PyValueError::new_err("nobs must be positive finite numbers"));
    }
    if count1 < 0.0 || count1 > nobs1 || !count1.is_finite() {
        return Err(PyValueError::new_err("count1 must be in [0, nobs1]"));
    }
    if count2 < 0.0 || count2 > nobs2 || !count2.is_finite() {
        return Err(PyValueError::new_err("count2 must be in [0, nobs2]"));
    }

    let alt = Alternative::from_str(alternative)?;

    let p1 = count1 / nobs1;
    let p2 = count2 / nobs2;

    // Pooled proportion under H0
    let p_pool = (count1 + count2) / (nobs1 + nobs2);

    if p_pool <= 0.0 || p_pool >= 1.0 {
        // All successes or no successes in both groups
        let d = PyDict::new_bound(py);
        d.set_item("statistic", 0.0)?;
        d.set_item("pvalue", 1.0)?;
        d.set_item("proportion1", p1)?;
        d.set_item("proportion2", p2)?;
        d.set_item("diff", p1 - p2)?;
        return Ok(d.unbind());
    }

    let se = (p_pool * (1.0 - p_pool) * (1.0 / nobs1 + 1.0 / nobs2)).sqrt();

    if se == 0.0 {
        return Err(PyValueError::new_err("standard error is zero"));
    }

    let z = (p1 - p2) / se;

    let norm = Normal::new(0.0, 1.0).unwrap();
    let p = match alt {
        Alternative::TwoSided => (2.0 * norm.sf(z.abs())).clamp(0.0, 1.0),
        Alternative::Greater => norm.sf(z).clamp(0.0, 1.0),
        Alternative::Less => norm.cdf(z).clamp(0.0, 1.0),
    };

    let d = PyDict::new_bound(py);
    d.set_item("statistic", z)?;
    d.set_item("pvalue", p)?;
    d.set_item("proportion1", p1)?;
    d.set_item("proportion2", p2)?;
    d.set_item("diff", p1 - p2)?;
    Ok(d.unbind())
}
