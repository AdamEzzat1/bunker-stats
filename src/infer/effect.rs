use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use statrs::distribution::{ContinuousCDF, StudentsT};

use super::common::{mean, reject_nonfinite, var_sample};

#[pyfunction]
pub fn cohens_d_2samp_np(
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
    pooled: bool,
) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;

    let n1 = xs.len();
    let n2 = ys.len();
    if n1 < 2 || n2 < 2 {
        return Err(PyValueError::new_err(
            "cohens_d requires n>=2 for both samples",
        ));
    }

    let m1 = mean(xs);
    let m2 = mean(ys);
    let v1 = var_sample(xs, m1);
    let v2 = var_sample(ys, m2);

    let s = if pooled {
        let df = (n1 + n2 - 2) as f64;
        let sp2 = (((n1 - 1) as f64) * v1 + ((n2 - 1) as f64) * v2) / df;
        sp2.sqrt()
    } else {
        ((v1 + v2) * 0.5).sqrt()
    };

    if s == 0.0 {
        Ok(0.0)
    } else {
        Ok((m1 - m2) / s)
    }
}

#[pyfunction]
pub fn hedges_g_2samp_np2(
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
    pooled: bool,
) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;

    let n1 = xs.len();
    let n2 = ys.len();
    if n1 < 2 || n2 < 2 {
        return Err(PyValueError::new_err(
            "hedges_g requires n>=2 for both samples",
        ));
    }

    let d = cohens_d_2samp_np(x, y, pooled)?;
    let df = (n1 + n2 - 2) as f64;
    let j = 1.0 - 3.0 / (4.0 * df - 1.0);
    Ok(d * j)
}

/// Analytic CI for:
/// - mean(x) if y is None
/// - mean(x) - mean(y) if y is Some
///
/// Uses t critical value (no bootstrap in v0.3).
///
/// IMPORTANT: PyO3 requires an explicit signature when an Option argument is not trailing.
#[pyfunction]
#[pyo3(signature = (x, y=None, alpha=0.05, equal_var=true))]
pub fn mean_diff_ci_np(
    x: numpy::PyReadonlyArray1<f64>,
    y: Option<numpy::PyReadonlyArray1<f64>>,
    alpha: f64,
    equal_var: bool,
) -> PyResult<(f64, f64)> {
    let xs = x.as_slice()?;
    reject_nonfinite(xs, "x")?;
    if xs.len() < 2 {
        return Err(PyValueError::new_err("x must have n>=2"));
    }

    let (diff, se, df) = if let Some(yarr) = y {
        let ys = yarr.as_slice()?;
        reject_nonfinite(ys, "y")?;
        if ys.len() < 2 {
            return Err(PyValueError::new_err("y must have n>=2"));
        }

        let n1 = xs.len();
        let n2 = ys.len();
        let m1 = mean(xs);
        let m2 = mean(ys);
        let v1 = var_sample(xs, m1);
        let v2 = var_sample(ys, m2);

        let diff = m1 - m2;

        if equal_var {
            let df = (n1 + n2 - 2) as f64;
            let sp2 = (((n1 - 1) as f64) * v1 + ((n2 - 1) as f64) * v2) / df;
            let se = (sp2 * (1.0 / (n1 as f64) + 1.0 / (n2 as f64))).sqrt();
            (diff, se, df)
        } else {
            let se2 = v1 / (n1 as f64) + v2 / (n2 as f64);
            let se = se2.sqrt();

            let num = se2 * se2;
            let den = (v1 * v1)
                / (((n1 as f64) * (n1 as f64)) * ((n1 - 1) as f64))
                + (v2 * v2)
                    / (((n2 as f64) * (n2 as f64)) * ((n2 - 1) as f64));

            let df = if den == 0.0 {
                (n1 + n2 - 2) as f64
            } else {
                num / den
            };

            (diff, se, df)
        }
    } else {
        let m = mean(xs);
        let v = var_sample(xs, m);
        let se = (v / (xs.len() as f64)).sqrt();
        let df = (xs.len() - 1) as f64;
        (m, se, df)
    };

    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let tcrit = dist.inverse_cdf(1.0 - alpha / 2.0);

    let lo = diff - tcrit * se;
    let hi = diff + tcrit * se;

    Ok((lo, hi))
}
