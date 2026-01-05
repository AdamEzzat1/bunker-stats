use numpy::PyReadonlyArray1;
use pyo3::prelude::*;


// 2) Jackknife helpers
// ======================

/// Jackknife mean: returns (jackknife_estimate, bias, standard_error)
///
/// Python signature:
///     jackknife_mean(x)
#[pyfunction]
pub fn jackknife_mean(x: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();

    if n <= 1 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let n_f = n as f64;
    let total: f64 = x.iter().copied().sum();
    let theta_full = total / n_f;

    let mut loo_means = Vec::with_capacity(n);
    for i in 0..n {
        let loo_sum = total - x[i];
        let loo_mean = loo_sum / (n_f - 1.0);
        loo_means.push(loo_mean);
    }

    let mean_loo = loo_means.iter().copied().sum::<f64>() / n_f;
    let theta_jack = n_f * theta_full - (n_f - 1.0) * mean_loo;
    let bias = theta_jack - theta_full;

    let mut sum_sq = 0.0;
    for v in &loo_means {
        let d = *v - mean_loo;
        sum_sq += d * d;
    }
    let se = ((n_f - 1.0) / n_f * sum_sq).sqrt();

    Ok((theta_jack, bias, se))
}

/// Jackknife mean with simple percentile CI over leave-one-out estimates
///
/// Python signature:
///     jackknife_mean_ci(x, conf=0.95)
#[pyfunction(signature = (x, conf=0.95))]
pub fn jackknife_mean_ci(x: PyReadonlyArray1<f64>, conf: f64) -> PyResult<(f64, f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();

    if n <= 1 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let n_f = n as f64;
    let total: f64 = x.iter().copied().sum();
    let theta_full = total / n_f;

    let mut loo_means = Vec::with_capacity(n);
    for i in 0..n {
        let loo_sum = total - x[i];
        let loo_mean = loo_sum / (n_f - 1.0);
        loo_means.push(loo_mean);
    }

    let mean_loo = loo_means.iter().copied().sum::<f64>() / n_f;
    let theta_jack = n_f * theta_full - (n_f - 1.0) * mean_loo;

    // percentile CI over LOO estimates (simple, sandbox-friendly)
    let mut sorted = loo_means.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - conf;  // Convert confidence level to alpha
    let lower_q = alpha / 2.0;
    let upper_q = 1.0 - alpha / 2.0;
    let n_b = sorted.len() as f64;

    let mut lower_idx = (n_b * lower_q).floor() as isize;
    let mut upper_idx = (n_b * upper_q).floor() as isize;

    if lower_idx < 0 {
        lower_idx = 0;
    }
    if upper_idx < 0 {
        upper_idx = 0;
    }
    if lower_idx >= sorted.len() as isize {
        lower_idx = (sorted.len() - 1) as isize;
    }
    if upper_idx >= sorted.len() as isize {
        upper_idx = (sorted.len() - 1) as isize;
    }

    let lower = sorted[lower_idx as usize];
    let upper = sorted[upper_idx as usize];

    Ok((theta_jack, lower, upper))
}
