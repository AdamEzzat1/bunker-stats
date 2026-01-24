use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use super::common::{mean, reject_nonfinite, median, sum_of_squares_kahan};

/// One-way ANOVA F-test.
///
/// Tests the null hypothesis that all groups have equal means.
/// 
/// Returns:
/// - statistic: F-statistic
/// - pvalue: p-value
/// - df_between: degrees of freedom between groups
/// - df_within: degrees of freedom within groups
#[pyfunction]
pub fn f_test_oneway_np(
    py: Python<'_>,
    groups: Vec<numpy::PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyDict>> {
    if groups.len() < 2 {
        return Err(PyValueError::new_err("f_test_oneway requires at least 2 groups"));
    }

    let mut group_data: Vec<Vec<f64>> = Vec::with_capacity(groups.len());
    let mut n_total = 0usize;
    
    // Validate and collect data
    for g in groups {
        let xs = g.as_slice()?;
        if xs.len() < 2 {
            return Err(PyValueError::new_err("each group must have at least 2 observations"));
        }
        reject_nonfinite(xs, "group")?;
        n_total += xs.len();
        group_data.push(xs.to_vec());
    }

    let k = group_data.len();
    
    // Compute group means and grand mean
    let mut group_means = Vec::with_capacity(k);
    let mut grand_sum = 0.0;
    
    for group in &group_data {
        let m = mean(group);
        group_means.push(m);
        grand_sum += group.iter().sum::<f64>();
    }
    
    let grand_mean = grand_sum / (n_total as f64);
    
    // Between-group sum of squares
    let mut ss_between = 0.0;
    for (i, group) in group_data.iter().enumerate() {
        let n = group.len() as f64;
        let diff = group_means[i] - grand_mean;
        ss_between += n * diff * diff;
    }
    
    // Within-group sum of squares with Kahan summation
    let mut ss_within = 0.0;
    for (i, group) in group_data.iter().enumerate() {
        let m = group_means[i];
        ss_within += sum_of_squares_kahan(group, m);
    }
    
    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;
    
    if df_within <= 0.0 {
        return Err(PyValueError::new_err("insufficient degrees of freedom"));
    }
    
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    
    let f_stat = if ms_within > 0.0 {
        ms_between / ms_within
    } else if ms_between == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    
    let p = if f_stat.is_finite() && f_stat >= 0.0 {
        let dist = FisherSnedecor::new(df_between, df_within).unwrap();
        (1.0 - dist.cdf(f_stat)).clamp(0.0, 1.0)
    } else if f_stat.is_infinite() {
        0.0
    } else {
        f64::NAN
    };
    
    let d = PyDict::new_bound(py);
    d.set_item("statistic", f_stat)?;
    d.set_item("pvalue", p)?;
    d.set_item("df_between", df_between)?;
    d.set_item("df_within", df_within)?;
    Ok(d.unbind())
}

/// Levene's test for equality of variances.
///
/// Tests the null hypothesis that all groups have equal variances.
/// Uses the median as the center (robust version).
#[pyfunction]
pub fn levene_test_np(
    py: Python<'_>,
    groups: Vec<numpy::PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyDict>> {
    if groups.len() < 2 {
        return Err(PyValueError::new_err("levene_test requires at least 2 groups"));
    }

    let mut transformed_groups = Vec::with_capacity(groups.len());
    
    for g in groups {
        let xs = g.as_slice()?;
        if xs.is_empty() {
            return Err(PyValueError::new_err("each group must be non-empty"));
        }
        reject_nonfinite(xs, "group")?;
        
        // Compute median
        let med = median(xs);
        
        // Transform: |x - median|
        let transformed: Vec<f64> = xs.iter().map(|&x| (x - med).abs()).collect();
        transformed_groups.push(transformed);
    }
    
    // Run one-way ANOVA on transformed data
    let mut n_total = 0usize;
    let mut group_means = Vec::with_capacity(transformed_groups.len());
    let mut grand_sum = 0.0;
    
    for group in &transformed_groups {
        n_total += group.len();
        let m = mean(group);
        group_means.push(m);
        grand_sum += group.iter().sum::<f64>();
    }
    
    let k = transformed_groups.len();
    let grand_mean = grand_sum / (n_total as f64);
    
    let mut ss_between = 0.0;
    for (i, group) in transformed_groups.iter().enumerate() {
        let n = group.len() as f64;
        let diff = group_means[i] - grand_mean;
        ss_between += n * diff * diff;
    }
    
    let mut ss_within = 0.0;
    for (i, group) in transformed_groups.iter().enumerate() {
        let m = group_means[i];
        ss_within += sum_of_squares_kahan(group, m);
    }
    
    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;
    
    if df_within <= 0.0 {
        return Err(PyValueError::new_err("insufficient degrees of freedom"));
    }
    
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    
    let w_stat = if ms_within > 0.0 {
        ms_between / ms_within
    } else {
        0.0
    };
    
    let p = if w_stat.is_finite() && w_stat >= 0.0 {
        let dist = FisherSnedecor::new(df_between, df_within).unwrap();
        (1.0 - dist.cdf(w_stat)).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    
    let d = PyDict::new_bound(py);
    d.set_item("statistic", w_stat)?;
    d.set_item("pvalue", p)?;
    d.set_item("df_between", df_between)?;
    d.set_item("df_within", df_within)?;
    Ok(d.unbind())
}
