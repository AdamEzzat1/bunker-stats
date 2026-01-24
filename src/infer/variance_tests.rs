use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use super::common::{mean, var_sample, reject_nonfinite};

/// F-test for equality of two variances.
///
/// Tests null hypothesis that var(x) == var(y).
#[pyfunction]
pub fn f_test_var_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    
    if xs.len() < 2 || ys.len() < 2 {
        return Err(PyValueError::new_err("f_test_var requires n >= 2 for both samples"));
    }
    
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;
    
    let mx = mean(xs);
    let my = mean(ys);
    
    let vx = var_sample(xs, mx);
    let vy = var_sample(ys, my);
    
    if vx <= 0.0 || vy <= 0.0 {
        return Err(PyValueError::new_err("variances must be positive"));
    }
    
    // F = larger_var / smaller_var (always >= 1)
    let (f_stat, df1, df2) = if vx >= vy {
        (vx / vy, (xs.len() - 1) as f64, (ys.len() - 1) as f64)
    } else {
        (vy / vx, (ys.len() - 1) as f64, (xs.len() - 1) as f64)
    };
    
    let dist = FisherSnedecor::new(df1, df2).unwrap();
    
    // Two-sided p-value: 2 * min(P(F >= f), P(F <= f))
    let upper_tail = 1.0 - dist.cdf(f_stat);
    let p = (2.0 * upper_tail).clamp(0.0, 1.0);
    
    let d = PyDict::new_bound(py);
    d.set_item("statistic", f_stat)?;
    d.set_item("pvalue", p)?;
    d.set_item("df1", df1)?;
    d.set_item("df2", df2)?;
    Ok(d.unbind())
}

/// Bartlett's test for equality of variances across multiple groups.
///
/// More sensitive to departures from normality than Levene's test.
#[pyfunction]
pub fn bartlett_test_np(
    py: Python<'_>,
    groups: Vec<numpy::PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyDict>> {
    if groups.len() < 2 {
        return Err(PyValueError::new_err("bartlett_test requires at least 2 groups"));
    }
    
    let mut group_vars = Vec::with_capacity(groups.len());
    let mut group_ns = Vec::with_capacity(groups.len());
    let mut n_total = 0usize;
    
    for g in groups {
        let xs = g.as_slice()?;
        if xs.len() < 2 {
            return Err(PyValueError::new_err("each group must have at least 2 observations"));
        }
        reject_nonfinite(xs, "group")?;
        
        let n = xs.len();
        let m = mean(xs);
        let v = var_sample(xs, m);
        
        if v <= 0.0 {
            return Err(PyValueError::new_err("all group variances must be positive"));
        }
        
        group_vars.push(v);
        group_ns.push(n);
        n_total += n;
    }
    
    let k = group_vars.len();
    
    // Pooled variance
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..k {
        let ni = group_ns[i] as f64;
        numerator += (ni - 1.0) * group_vars[i];
        denominator += ni - 1.0;
    }
    
    let s2_pooled = numerator / denominator;
    
    if s2_pooled <= 0.0 {
        return Err(PyValueError::new_err("pooled variance is zero"));
    }
    
    // Bartlett statistic
    let mut sum_log_s2 = 0.0;
    for i in 0..k {
        let ni = group_ns[i] as f64;
        sum_log_s2 += (ni - 1.0) * group_vars[i].ln();
    }
    
    let t = (n_total - k) as f64 * s2_pooled.ln() - sum_log_s2;
    
    // Correction factor
    let mut sum_inv = 0.0;
    for &ni in &group_ns {
        sum_inv += 1.0 / ((ni - 1) as f64);
    }
    
    let c = 1.0 + (sum_inv - 1.0 / denominator) / (3.0 * (k - 1) as f64);
    
    let chi2_stat = t / c;
    
    // Chi-square with df = k-1
    use statrs::distribution::ChiSquared;
    let df = (k - 1) as f64;
    let dist = ChiSquared::new(df).unwrap();
    let p = (1.0 - dist.cdf(chi2_stat)).clamp(0.0, 1.0);
    
    let d = PyDict::new_bound(py);
    d.set_item("statistic", chi2_stat)?;
    d.set_item("pvalue", p)?;
    d.set_item("df", df)?;
    Ok(d.unbind())
}
