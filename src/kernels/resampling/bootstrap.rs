use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// 1) Bootstrap helpers
// ======================

/// Bootstrap mean (basic version)
///
/// Python signature:
///     bootstrap_mean(x, n_resamples, random_state=None)
#[pyfunction(signature = (x, n_resamples, random_state=None))]
pub fn bootstrap_mean(
    x: PyReadonlyArray1<f64>,
    n_resamples: usize,
    random_state: Option<u64>,
) -> PyResult<f64> {
    let x = x.as_slice()?;
    let n = x.len();

    if n == 0 || n_resamples == 0 {
        return Ok(f64::NAN);
    }

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut acc = 0.0;
    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sum += x[idx];
        }
        acc += sum / (n as f64);
    }

    Ok(acc / (n_resamples as f64))
}

/// Bootstrap mean + CI (percentile CI)
///
/// Python signature:
///     bootstrap_mean_ci(x, n_resamples, conf=0.95, random_state=None)
#[pyfunction(signature = (x, n_resamples, conf=0.95, random_state=None))]
pub fn bootstrap_mean_ci(
    x: PyReadonlyArray1<f64>,
    n_resamples: usize,
    conf: f64,
    random_state: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();

    if n == 0 || n_resamples == 0 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut boots = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sum += x[idx];
        }
        boots.push(sum / (n as f64));
    }

    // point estimate (mean of bootstrap means)
    let mean_hat = boots.iter().copied().sum::<f64>() / (boots.len() as f64);

    // percentile CI
    boots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - conf;  // Convert confidence level to alpha
    let lower_q = alpha / 2.0;
    let upper_q = 1.0 - alpha / 2.0;
    let n_b = boots.len() as f64;

    let mut lower_idx = (n_b * lower_q).floor() as isize;
    let mut upper_idx = (n_b * upper_q).floor() as isize;

    if lower_idx < 0 {
        lower_idx = 0;
    }
    if upper_idx < 0 {
        upper_idx = 0;
    }
    if lower_idx >= boots.len() as isize {
        lower_idx = (boots.len() - 1) as isize;
    }
    if upper_idx >= boots.len() as isize {
        upper_idx = (boots.len() - 1) as isize;
    }

    let lower = boots[lower_idx as usize];
    let upper = boots[upper_idx as usize];

    Ok((mean_hat, lower, upper))
}

/// Generic bootstrap CI for simple stats: "mean", "median", "std"
///
/// Python signature:
///     bootstrap_ci(x, stat="mean", n_resamples=1000, conf=0.95, random_state=None)
#[pyfunction(signature = (x, stat="mean", n_resamples=1000, conf=0.95, random_state=None))]
pub fn bootstrap_ci(
    x: PyReadonlyArray1<f64>,
    stat: &str,
    n_resamples: usize,
    conf: f64,
    random_state: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();

    if n == 0 || n_resamples == 0 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut vals = Vec::with_capacity(n_resamples);

    match stat {
        "mean" => {
            for _ in 0..n_resamples {
                let mut sum = 0.0;
                for _ in 0..n {
                    let idx = rng.gen_range(0..n);
                    sum += x[idx];
                }
                vals.push(sum / (n as f64));
            }
        }
        "median" => {
            let mut scratch = vec![0.0_f64; n];
            for _ in 0..n_resamples {
                for j in 0..n {
                    let idx = rng.gen_range(0..n);
                    scratch[j] = x[idx];
                }
                scratch.sort_by(|a, b| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let med = if n % 2 == 1 {
                    scratch[n / 2]
                } else {
                    (scratch[n / 2 - 1] + scratch[n / 2]) * 0.5
                };
                vals.push(med);
            }
        }
        "std" => {
            for _ in 0..n_resamples {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                for _ in 0..n {
                    let idx = rng.gen_range(0..n);
                    let v = x[idx];
                    sum += v;
                    sum_sq += v * v;
                }
                let nf = n as f64;
                let mean = sum / nf;
                let var = (sum_sq / nf) - mean * mean;
                let std = if var <= 0.0 { 0.0 } else { var.sqrt() };
                vals.push(std);
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "Unsupported stat. Use 'mean', 'median', or 'std'.",
            ));
        }
    }

    // point estimate
    let est = vals.iter().copied().sum::<f64>() / (vals.len() as f64);

    // percentile CI
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - conf;  // Convert confidence level to alpha
    let lower_q = alpha / 2.0;
    let upper_q = 1.0 - alpha / 2.0;
    let n_b = vals.len() as f64;

    let mut lower_idx = (n_b * lower_q).floor() as isize;
    let mut upper_idx = (n_b * upper_q).floor() as isize;

    if lower_idx < 0 {
        lower_idx = 0;
    }
    if upper_idx < 0 {
        upper_idx = 0;
    }
    if lower_idx >= vals.len() as isize {
        lower_idx = (vals.len() - 1) as isize;
    }
    if upper_idx >= vals.len() as isize {
        upper_idx = (vals.len() - 1) as isize;
    }

    let lower = vals[lower_idx as usize];
    let upper = vals[upper_idx as usize];

    Ok((est, lower, upper))
}

/// Bootstrap correlation with CI (percentile CI)
///
/// Python signature:
///     bootstrap_corr(x, y, n_resamples, conf=0.95, random_state=None)
#[pyfunction(signature = (x, y, n_resamples, conf=0.95, random_state=None))]
pub fn bootstrap_corr(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    n_resamples: usize,
    conf: f64,
    random_state: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    let n = x.len();

    if n == 0 || n_resamples == 0 || y.len() != n {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut boots = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        let mut sum_xy = 0.0;

        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            let xi = x[idx];
            let yi = y[idx];

            sum_x += xi;
            sum_y += yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
            sum_xy += xi * yi;
        }

        let nf = n as f64;
        let mean_x = sum_x / nf;
        let mean_y = sum_y / nf;
        let var_x = sum_x2 / nf - mean_x * mean_x;
        let var_y = sum_y2 / nf - mean_y * mean_y;

        if var_x <= 0.0 || var_y <= 0.0 {
            boots.push(f64::NAN);
        } else {
            let cov = sum_xy / nf - mean_x * mean_y;
            let corr = cov / (var_x * var_y).sqrt();
            boots.push(corr);
        }
    }

    boots.retain(|v| v.is_finite());
    if boots.is_empty() {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let corr_hat = boots.iter().copied().sum::<f64>() / (boots.len() as f64);

    boots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - conf;  // Convert confidence level to alpha
    let lower_q = alpha / 2.0;
    let upper_q = 1.0 - alpha / 2.0;
    let n_b = boots.len() as f64;

    let mut lower_idx = (n_b * lower_q).floor() as isize;
    let mut upper_idx = (n_b * upper_q).floor() as isize;

    if lower_idx < 0 {
        lower_idx = 0;
    }
    if upper_idx < 0 {
        upper_idx = 0;
    }
    if lower_idx >= boots.len() as isize {
        lower_idx = (boots.len() - 1) as isize;
    }
    if upper_idx >= boots.len() as isize {
        upper_idx = (boots.len() - 1) as isize;
    }

    let lower = boots[lower_idx as usize];
    let upper = boots[upper_idx as usize];

    Ok((corr_hat, lower, upper))
}
