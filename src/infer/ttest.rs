use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;


use super::common::{mean_var_sample, reject_nonfinite, Alternative};

fn p_from_t(t: f64, df: f64, alt: Alternative) -> f64 {
    super::common::t_pvalue(t, df, alt)
}

/// One-sample t-test vs `popmean`.
///
/// Python signature:
/// t_test_1samp_np(x: np.ndarray, popmean: float, alternative="two-sided") -> dict
#[pyfunction]
#[pyo3(signature = (x, popmean, alternative="two-sided"))]
pub fn t_test_1samp_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    popmean: f64,
    alternative: &str,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    reject_nonfinite(xs, "x")?;

    let n = xs.len();
    if n < 2 {
        return Err(PyValueError::new_err("t_test_1samp requires n >= 2"));
    }

    let (m, v) = mean_var_sample(xs);

    if !v.is_finite() || v < 0.0 {
        return Err(PyValueError::new_err(format!(
            "variance is not finite: var={:.6e}, mean={:.6e}, n={}",
v, m, n
        )));
    }

    let se = (v / (n as f64)).sqrt();
    let t = if se == 0.0 {
        // Degenerate variance: if difference is 0 => t=0, else +/- inf
        if (m - popmean) == 0.0 {
            0.0
        } else {
            f64::INFINITY * (m - popmean).signum()
        }
    } else {
        (m - popmean) / se
    };

    let df = (n - 1) as f64;
    let alt = Alternative::from_str(alternative)?;
    // t_pvalue handles an infinite or NaN statistic itself.
    let p = p_from_t(t, df, alt);

    let d = PyDict::new_bound(py);
    d.set_item("statistic", t)?;
    d.set_item("pvalue", p)?;
    d.set_item("df", df)?;
    d.set_item("mean", m)?;
    Ok(d.unbind())
}

/// Two-sample t-test.
///
/// If `equal_var=True`, performs the pooled-variance Student t-test.
/// If `equal_var=False`, performs Welch's t-test.
///
/// Python signature:
/// t_test_2samp_np(x: np.ndarray, y: np.ndarray, equal_var: bool, alternative="two-sided") -> dict
#[pyfunction]
#[pyo3(signature = (x, y, equal_var, alternative="two-sided"))]
pub fn t_test_2samp_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
    equal_var: bool,
    alternative: &str,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;

    let n1 = xs.len();
    let n2 = ys.len();
    if n1 < 2 || n2 < 2 {
        return Err(PyValueError::new_err(
            "t_test_2samp requires n >= 2 for both samples",
        ));
    }

    let (m1, v1) = mean_var_sample(xs);
    let (m2, v2) = mean_var_sample(ys);

    if !v1.is_finite() || !v2.is_finite() || v1 < 0.0 || v2 < 0.0 {
        return Err(PyValueError::new_err(format!(
            "variance is not finite: v1={:.6e}, v2={:.6e}",
            v1, v2
        )));
    }

    let alt = Alternative::from_str(alternative)?;

    let (t, df) = if equal_var {
        // Pooled variance t-test
        let df = (n1 + n2 - 2) as f64;
        let sp2 = (((n1 - 1) as f64) * v1 + ((n2 - 1) as f64) * v2) / df;

        let se = (sp2 * (1.0 / (n1 as f64) + 1.0 / (n2 as f64))).sqrt();
        let t = if se == 0.0 { 
            // Both samples have zero variance. Compare the means exactly: an
            // absolute tolerance (this was `< 1e-15`) calls 1e-20 and 2e-20
            // equal, and the means of constant samples are exact.
            if m1 == m2 {
0.0
            } else {
                f64::INFINITY * (m1 - m2).signum()
            }
        } else { 
            (m1 - m2) / se 
        };
        (t, df)
    } else {
        // Welch t-test
        let se2 = v1 / (n1 as f64) + v2 / (n2 as f64);
        let se = se2.sqrt();
        let t = if se == 0.0 { 
            // Both samples have zero variance. Compare the means exactly: an
            // absolute tolerance (this was `< 1e-15`) calls 1e-20 and 2e-20
            // equal, and the means of constant samples are exact.
            if m1 == m2 {
0.0
            } else {
                f64::INFINITY * (m1 - m2).signum()
            }
        } else { 
            (m1 - m2) / se 
        };

        // Welch-Satterthwaite df with improved edge case handling
        let num = se2 * se2;
        let den = (v1 * v1)
            / (((n1 as f64) * (n1 as f64)) * ((n1 - 1) as f64))
            + (v2 * v2) / (((n2 as f64) * (n2 as f64)) * ((n2 - 1) as f64));

        let df = if den == 0.0 {
            // Both samples have zero variance - use pooled df
            (n1 + n2 - 2) as f64
        } else if den.is_finite() {
            let welch_df = num / den;
            // Ensure df is positive and reasonable
            welch_df.max(1.0).min((n1 + n2 - 2) as f64)
        } else {
            // Numerical issues - fall back to pooled
            (n1 + n2 - 2) as f64
        };

        (t, df)
    };

    let p = p_from_t(t, df, alt);

    let d = PyDict::new_bound(py);
    d.set_item("statistic", t)?;
    d.set_item("pvalue", p)?;
    d.set_item("df", df)?;
    d.set_item("mean_x", m1)?;
    d.set_item("mean_y", m2)?;
    d.set_item("equal_var", equal_var)?;
    Ok(d.unbind())
}
