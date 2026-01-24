use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::common::reject_nonfinite;

/// Chi-square survival function with improved numerical stability.
fn chi2_sf(stat: f64, df: f64) -> f64 {
    if !stat.is_finite() || !df.is_finite() || stat < 0.0 || df <= 0.0 {
        return f64::NAN;
    }
    
    let dist = ChiSquared::new(df).unwrap();
    let cdf = dist.cdf(stat);
    
    // Better handling of numerical precision at extremes
    if cdf >= 1.0 {
        return 0.0;
    }
    if cdf <= 0.0 {
        return 1.0;
    }
    
    (1.0 - cdf).max(0.0).min(1.0)
}

#[pyfunction]
#[pyo3(signature = (observed, expected=None, *, sum_check=true))]
pub fn chi2_gof_np(
    py: Python<'_>,
    observed: numpy::PyReadonlyArray1<f64>,
    expected: Option<numpy::PyReadonlyArray1<f64>>,
    sum_check: bool,
) -> PyResult<Py<PyDict>> {
    let obs = observed.as_slice()?;
    reject_nonfinite(obs, "observed")?;

    if obs.iter().any(|&v| v < 0.0) {
        return Err(PyValueError::new_err("observed must be non-negative counts"));
    }
    
    let k = obs.len();
    if k < 2 {
        return Err(PyValueError::new_err("chi2_gof requires at least 2 categories"));
    }

    let sum_obs: f64 = obs.iter().sum();
    if sum_obs <= 0.0 {
        return Err(PyValueError::new_err("sum(observed) must be > 0"));
    }

    // Single allocation with capacity
    let mut exp_vec: Vec<f64> = Vec::with_capacity(k);
    
    if let Some(exp_arr) = expected {
        let exp = exp_arr.as_slice()?;
        reject_nonfinite(exp, "expected")?;
        
        if exp.len() != k {
            return Err(PyValueError::new_err(
                "expected must have same length as observed",
            ));
        }
        if exp.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err("expected must be strictly positive"));
        }
        
        // Check sum agreement if sum_check is enabled
        if sum_check {
            let sum_exp: f64 = exp.iter().sum();
            if sum_exp <= 0.0 {
                return Err(PyValueError::new_err("sum(expected) must be > 0"));
            }

            // SciPy-style tolerance: sqrt(eps) for float64
            let rtol = f64::EPSILON.sqrt();
            let rel_diff = (sum_obs - sum_exp).abs() / sum_obs.min(sum_exp);

            if rel_diff > rtol {
                return Err(PyValueError::new_err(format!(
                    "For each axis slice, the sum of the observed frequencies must agree with the sum of the expected frequencies to a relative tolerance of {rtol}, but the percent differences are:\n{rel_diff}"
                )));
            }
        }
        
        exp_vec.extend_from_slice(exp);
    } else {
        // Uniform expected counts - use resize for efficiency
        let e = sum_obs / (k as f64);
        exp_vec.resize(k, e);
    }

    // Compute statistic with Kahan summation for better precision
    let mut stat = 0.0;
    let mut c = 0.0;
    
    for i in 0..k {
        let e = exp_vec[i];
        let o = obs[i];
        let d = o - e;
        let term = d * d / e;
        
        let y = term - c;
        let t = stat + y;
        c = (t - stat) - y;
        stat = t;
    }

    let df = (k - 1) as f64;
    let p = chi2_sf(stat, df);

    let d = PyDict::new_bound(py);
    d.set_item("statistic", stat)?;
    d.set_item("pvalue", p)?;
    d.set_item("df", df)?;
    Ok(d.unbind())
}

#[pyfunction]
pub fn chi2_independence_np(
    py: Python<'_>,
    table: numpy::PyReadonlyArray2<f64>,
) -> PyResult<Py<PyDict>> {
    let arr = table.as_array();
    let r = arr.shape()[0];
    let c = arr.shape()[1];

    if r < 2 || c < 2 {
        return Err(PyValueError::new_err("contingency table must be at least 2x2"));
    }

    let x = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("table must be C-contiguous float64"))?;
    reject_nonfinite(x, "table")?;

    if x.iter().any(|&v| v < 0.0) {
        return Err(PyValueError::new_err("table must be non-negative counts"));
    }

    let mut row_s = vec![0.0; r];
    let mut col_s = vec![0.0; c];
    let mut total = 0.0;

    for i in 0..r {
        for j in 0..c {
            let v = x[i * c + j];
            row_s[i] += v;
            col_s[j] += v;
            total += v;
        }
    }

    if total <= 0.0 {
        return Err(PyValueError::new_err("sum(table) must be > 0"));
    }

    // Compute statistic with Kahan summation
    let mut stat = 0.0;
    let mut comp = 0.0;
    
    for i in 0..r {
        for j in 0..c {
            let e = row_s[i] * col_s[j] / total;
            if e <= 0.0 {
                continue;
            }
            let o = x[i * c + j];
            let d = o - e;
            let term = d * d / e;
            
            let y = term - comp;
            let t = stat + y;
            comp = (t - stat) - y;
            stat = t;
        }
    }

    let df = ((r - 1) * (c - 1)) as f64;
    let p = chi2_sf(stat, df);

    let d = PyDict::new_bound(py);
    d.set_item("statistic", stat)?;
    d.set_item("pvalue", p)?;
    d.set_item("df", df)?;
    Ok(d.unbind())
}