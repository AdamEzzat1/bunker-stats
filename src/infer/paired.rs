// ========================================================================
// Paired t-test — reuses common primitives (mean, var_sample)
// ========================================================================
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, StudentsT};

use super::common::{mean, var_sample, reject_nonfinite, Alternative};

/// Paired (dependent) t-test.
///
/// Equivalent to a one-sample t-test on `x - y` against `popmean = 0`.
/// This is the standard approach for before/after or matched-pair designs.
///
/// Returns dict with keys: statistic, pvalue, df, mean (of differences).
#[pyfunction]
#[pyo3(signature = (x, y, alternative="two-sided"))]
pub fn t_test_paired_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
    alternative: &str,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    let n = xs.len();
    if n < 2 {
        return Err(PyValueError::new_err("t_test_paired requires n >= 2"));
    }

    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;

    // Compute differences
    let diffs: Vec<f64> = xs.iter().zip(ys.iter()).map(|(&a, &b)| a - b).collect();

    // One-sample t-test on differences vs popmean=0
    let m = mean(&diffs);
    let v = var_sample(&diffs, m);

    let alt = Alternative::from_str(alternative)?;

    let se = (v / (n as f64)).sqrt();
    let t = if se == 0.0 {
        if m == 0.0 { 0.0 } else { f64::INFINITY * m.signum() }
    } else {
        m / se
    };

    let df = (n - 1) as f64;

    let p = if t.is_finite() {
        let dist = StudentsT::new(0.0, 1.0, df).unwrap();
        let cdf = dist.cdf(t);
        match alt {
            Alternative::TwoSided => {
                let tail = if cdf < 0.5 { cdf } else { 1.0 - cdf };
                (2.0 * tail).clamp(0.0, 1.0)
            }
            Alternative::Less => cdf.clamp(0.0, 1.0),
            Alternative::Greater => (1.0 - cdf).clamp(0.0, 1.0),
        }
    } else {
        0.0
    };

    let d = PyDict::new_bound(py);
    d.set_item("statistic", t)?;
    d.set_item("pvalue", p)?;
    d.set_item("df", df)?;
    d.set_item("mean", m)?;
    Ok(d.unbind())
}
