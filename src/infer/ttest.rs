use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, StudentsT};

use super::common::{mean, reject_nonfinite, var_sample, Alternative};

fn p_from_t(t: f64, df: f64, alt: Alternative) -> f64 {
    // Student-t with mean 0, scale 1
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let cdf = dist.cdf(t);

    match alt {
        Alternative::TwoSided => {
            // two-sided p = 2 * min(CDF(t), 1-CDF(t))
            let tail = if cdf < 0.5 { cdf } else { 1.0 - cdf };
            (2.0 * tail).clamp(0.0, 1.0)
        }
        Alternative::Less => cdf.clamp(0.0, 1.0),
        Alternative::Greater => (1.0 - cdf).clamp(0.0, 1.0),
    }
}

/// One-sample t-test vs `popmean`.
///
/// Python signature:
/// t_test_1samp_np(x: np.ndarray, popmean: float, alternative="two-sided") -> dict
#[pyfunction]
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

    let m = mean(xs);
    let v = var_sample(xs, m);

    if !v.is_finite() || v < 0.0 {
        return Err(PyValueError::new_err("variance is not finite"));
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
    let p = if t.is_finite() { p_from_t(t, df, alt) } else { 0.0 };

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
/// If `equal_var=False`, performs Welch’s t-test.
///
/// Python signature:
/// t_test_2samp_np(x: np.ndarray, y: np.ndarray, equal_var: bool, alternative="two-sided") -> dict
#[pyfunction]
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

    let m1 = mean(xs);
    let m2 = mean(ys);
    let v1 = var_sample(xs, m1);
    let v2 = var_sample(ys, m2);

    if !v1.is_finite() || !v2.is_finite() || v1 < 0.0 || v2 < 0.0 {
        return Err(PyValueError::new_err("variance is not finite"));
    }

    let alt = Alternative::from_str(alternative)?;

    let (t, df) = if equal_var {
        // Pooled variance t-test
        let df = (n1 + n2 - 2) as f64;
        let sp2 = (((n1 - 1) as f64) * v1 + ((n2 - 1) as f64) * v2) / df;

        let se = (sp2 * (1.0 / (n1 as f64) + 1.0 / (n2 as f64))).sqrt();
        let t = if se == 0.0 { 0.0 } else { (m1 - m2) / se };
        (t, df)
    } else {
        // Welch t-test
        let se2 = v1 / (n1 as f64) + v2 / (n2 as f64);
        let se = se2.sqrt();
        let t = if se == 0.0 { 0.0 } else { (m1 - m2) / se };

        // Welch–Satterthwaite df
        let num = se2 * se2;
        let den = (v1 * v1)
            / (((n1 as f64) * (n1 as f64)) * ((n1 - 1) as f64))
            + (v2 * v2) / (((n2 as f64) * (n2 as f64)) * ((n2 - 1) as f64));

        let df = if den == 0.0 {
            (n1 + n2 - 2) as f64
        } else {
            num / den
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
