// ========================================================================
// Odds ratio and CI for 2x2 contingency tables
// ========================================================================
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, Normal};

/// Odds ratio and confidence interval for a 2x2 contingency table.
///
/// Table layout:
///   [[a, b],
///    [c, d]]
///
/// OR = (a*d) / (b*c)
///
/// CI uses log(OR) ± z * SE(log(OR)), where SE = sqrt(1/a + 1/b + 1/c + 1/d).
///
/// Parameters:
///   table: 2x2 array (2D numpy array)
///   confidence: confidence level (default 0.95)
///
/// Returns dict: {odds_ratio, log_odds_ratio, ci_lower, ci_upper, se}
#[pyfunction]
#[pyo3(signature = (table, confidence=0.95))]
pub fn odds_ratio_np(
    py: Python<'_>,
    table: numpy::PyReadonlyArray2<f64>,
    confidence: f64,
) -> PyResult<Py<PyDict>> {
    let arr = table.as_array();
    let shape = arr.shape();

    if shape[0] != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err("table must be a 2x2 array"));
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(PyValueError::new_err("confidence must be in (0, 1)"));
    }

    let x = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("table must be C-contiguous"))?;

    let a = x[0];
    let b = x[1];
    let c = x[2];
    let d = x[3];

    // Validate non-negative
    if a < 0.0 || b < 0.0 || c < 0.0 || d < 0.0 {
        return Err(PyValueError::new_err("table entries must be non-negative"));
    }

    // Handle zero cells: Haldane-Anscombe correction (add 0.5 to all cells)
    let (a, b, c, d) = if a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0 {
        (a + 0.5, b + 0.5, c + 0.5, d + 0.5)
    } else {
        (a, b, c, d)
    };

    let or = (a * d) / (b * c);
    let log_or = or.ln();

    // SE of log(OR) = sqrt(1/a + 1/b + 1/c + 1/d)
    let se = (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();

    let alpha = 1.0 - confidence;
    let norm = Normal::new(0.0, 1.0).unwrap();
    let z_crit = norm.inverse_cdf(1.0 - alpha / 2.0);

    let ci_lo = (log_or - z_crit * se).exp();
    let ci_hi = (log_or + z_crit * se).exp();

    let dict = PyDict::new_bound(py);
    dict.set_item("odds_ratio", or)?;
    dict.set_item("log_odds_ratio", log_or)?;
    dict.set_item("ci_lower", ci_lo)?;
    dict.set_item("ci_upper", ci_hi)?;
    dict.set_item("se", se)?;
    Ok(dict.unbind())
}
