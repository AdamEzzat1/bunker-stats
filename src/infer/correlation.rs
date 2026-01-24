use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, StudentsT};

use super::common::{mean, reject_nonfinite, rankdata_average};

/// Pearson correlation test.
///
/// Tests null hypothesis that correlation is zero.
#[pyfunction]
pub fn pearson_corr_test_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    
    let n = xs.len();
    if n < 3 {
        return Err(PyValueError::new_err("pearson_corr_test requires n >= 3"));
    }
    
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;
    
    // Compute correlation
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
    
    let r = sxy / (sxx * syy).sqrt();
    
    // t-test for r != 0
    let df = (n - 2) as f64;
    let t = if r.abs() >= 1.0 {
        f64::INFINITY * r.signum()
    } else {
        r * (df / (1.0 - r * r)).sqrt()
    };
    
    let p = if t.is_finite() {
        let dist = StudentsT::new(0.0, 1.0, df).unwrap();
        let cdf = dist.cdf(t);
        let tail = if cdf < 0.5 { cdf } else { 1.0 - cdf };
        (2.0 * tail).clamp(0.0, 1.0)
    } else {
        0.0
    };
    
    let d = PyDict::new_bound(py);
    d.set_item("correlation", r)?;
    d.set_item("statistic", t)?;
    d.set_item("pvalue", p)?;
    d.set_item("df", df)?;
    Ok(d.unbind())
}

/// Spearman rank correlation test.
///
/// Computes correlation on ranks, then tests for significance.
#[pyfunction]
pub fn spearman_corr_test_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    
    let n = xs.len();
    if n < 3 {
        return Err(PyValueError::new_err("spearman_corr_test requires n >= 3"));
    }
    
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;
    
    // Compute ranks (average ranks for ties) using shared utility
    let rx = rankdata_average(xs);
    let ry = rankdata_average(ys);
    
    // Compute correlation on ranks
    let mx = mean(&rx);
    let my = mean(&ry);
    
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    
    for i in 0..n {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    
    if sxx == 0.0 || syy == 0.0 {
        return Err(PyValueError::new_err("zero variance in ranks"));
    }
    
    let rho = sxy / (sxx * syy).sqrt();
    
    // t-test for rho != 0
    let df = (n - 2) as f64;
    let t = if rho.abs() >= 1.0 {
        f64::INFINITY * rho.signum()
    } else {
        rho * (df / (1.0 - rho * rho)).sqrt()
    };
    
    let p = if t.is_finite() {
        let dist = StudentsT::new(0.0, 1.0, df).unwrap();
        let cdf = dist.cdf(t);
        let tail = if cdf < 0.5 { cdf } else { 1.0 - cdf };
        (2.0 * tail).clamp(0.0, 1.0)
    } else {
        0.0
    };
    
    let d = PyDict::new_bound(py);
    d.set_item("correlation", rho)?;
    d.set_item("statistic", t)?;
    d.set_item("pvalue", p)?;
    d.set_item("df", df)?;
    Ok(d.unbind())
}
