use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};


// 6) Time-series diagnostics: ADF, KPSS, PP
// ======================

/// Augmented Dickey–Fuller test (simplified: DF with intercept only; no extra lags by default).
///
/// Python signature:
///     adf_test(x, max_lag=None) -> (statistic, pvalue)
#[pyfunction(signature = (x, regression="c", max_lag=None))]
pub fn adf_test(x: PyReadonlyArray1<f64>, regression: &str, max_lag: Option<usize>) -> PyResult<(f64, f64)> {

    let x = x.as_slice()?;
    let n = x.len();

    if n < 3 {
        return Ok((f64::NAN, f64::NAN));
    }

    let lag = max_lag.unwrap_or(0);

    if lag > 0 {
        return Err(PyValueError::new_err(
            "adf_test currently only supports max_lag=None or 0 in this sandbox",
        ));
    }

    // Δy_t = y_t - y_{t-1}, t=1..n-1
    let m = n - 1;
    if m <= 2 {
        return Ok((f64::NAN, f64::NAN));
    }

    let mut dy = Vec::with_capacity(m);
    let mut y_lag = Vec::with_capacity(m);
    for t in 1..n {
        dy.push(x[t] - x[t - 1]);
        y_lag.push(x[t - 1]);
    }

    let m_f = m as f64;
    let mean_y_lag = y_lag.iter().copied().sum::<f64>() / m_f;
    let mean_dy = dy.iter().copied().sum::<f64>() / m_f;

    // simple OLS: dy ~ α + β y_{t-1}
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..m {
        let xi = y_lag[i] - mean_y_lag;
        let yi = dy[i] - mean_dy;
        sxx += xi * xi;
        sxy += xi * yi;
    }
    if sxx <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let beta = sxy / sxx;
    let alpha = mean_dy - beta * mean_y_lag;

    // residual variance
    let mut rss = 0.0;
    for i in 0..m {
        let yi = dy[i];
        let xi = y_lag[i];
        let y_hat = alpha + beta * xi;
        let e = yi - y_hat;
        rss += e * e;
    }

    let dof = m as i64 - 2;
    if dof <= 0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let sigma2 = rss / (dof as f64);
    let se_beta = (sigma2 / sxx).sqrt();

    if se_beta <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let t_stat = beta / se_beta;

    // Approximate p-value using standard normal (sandbox-level, not MacKinnon)
    let z = Normal::new(0.0, 1.0).unwrap();
    let p_val = 2.0 * (1.0 - z.cdf(t_stat.abs()));

    Ok((t_stat, p_val))
}

/// Calculate KPSS p-value using critical value tables
/// Critical values from Kwiatkowski et al. (1992) Table 1
fn kpss_pvalue(stat: f64, regression: &str) -> f64 {
    // Critical values: (critical_value, p_value)
    let critical_values = match regression {
        "c" => {
            // Level stationarity critical values
            vec![
                (0.347, 0.10),
                (0.463, 0.05),
                (0.574, 0.025),
                (0.739, 0.01),
            ]
        }
        "ct" => {
            // Trend stationarity critical values
            vec![
                (0.119, 0.10),
                (0.146, 0.05),
                (0.176, 0.025),
                (0.216, 0.01),
            ]
        }
        _ => {
            // Fallback - should not happen due to earlier validation
            vec![(0.463, 0.05)]
        }
    };

    // Find p-value by interpolation
    if stat < critical_values[0].0 {
        // Statistic smaller than 10% critical value -> p > 0.10
        return 0.10;
    }

    for i in 0..(critical_values.len() - 1) {
        let (cv_low, p_low) = critical_values[i];
        let (cv_high, p_high) = critical_values[i + 1];
        
        if stat >= cv_low && stat < cv_high {
            // Linear interpolation
            let weight = (stat - cv_low) / (cv_high - cv_low);
            return p_low - weight * (p_low - p_high);
        }
    }

    // Statistic larger than 1% critical value -> p < 0.01
    0.01
}

/// KPSS test for stationarity.
///
/// Python signature:
///     kpss_test(x, regression="c") -> (statistic, pvalue)
#[pyfunction(signature = (x, regression="c"))]
pub fn kpss_test(x: PyReadonlyArray1<f64>, regression: &str) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();

    if n < 3 {
        return Ok((f64::NAN, f64::NAN));
    }

    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        t.push((i + 1) as f64);
    }

    let (resid, _) = match regression {
        "c" => {
            // level stationarity: y_t = μ + e_t
            let mean = x.iter().copied().sum::<f64>() / (n as f64);
            (x.iter().map(|v| v - mean).collect::<Vec<f64>>(), 1)
        }
        "ct" => {
            // trend stationarity: y_t = μ + β t + e_t
            let mut xtx = [[0.0_f64; 2]; 2];
            let mut xty = [0.0_f64; 2];
            for i in 0..n {
                let v = x[i];
                let tt = t[i];
                xtx[0][0] += 1.0;
                xtx[0][1] += tt;
                xtx[1][0] += tt;
                xtx[1][1] += tt * tt;
                xty[0] += v;
                xty[1] += v * tt;
            }
            let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
            if det.abs() < 1e-12 {
                return Ok((f64::NAN, f64::NAN));
            }
            let inv00 = xtx[1][1] / det;
            let inv01 = -xtx[0][1] / det;
            let inv10 = -xtx[1][0] / det;
            let inv11 = xtx[0][0] / det;
            let mu = inv00 * xty[0] + inv01 * xty[1];
            let beta = inv10 * xty[0] + inv11 * xty[1];

            let mut resid = Vec::with_capacity(n);
            for i in 0..n {
                let y_hat = mu + beta * t[i];
                resid.push(x[i] - y_hat);
            }
            (resid, 2)
        }
        _ => {
            return Err(PyValueError::new_err(
                "regression must be 'c' or 'ct'",
            ));
        }
    };

    let mut s = Vec::with_capacity(n);
    let mut cum = 0.0;
    for e in &resid {
        cum += *e;
        s.push(cum);
    }

    let var = super::variance(&resid);
    if var <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let mut eta = 0.0;
    for val in &s {
        eta += val * val;
    }
    let n_f = n as f64;
    let stat = eta / (n_f * n_f * var);

    // FIXED: Use proper KPSS critical value tables instead of bogus normal approximation
    let p_val = kpss_pvalue(stat, regression);

    Ok((stat, p_val))
}

/// Phillips–Perron (PP) test (simplified).
///
/// Python signature:
///     pp_test(x) -> (statistic, pvalue)
#[pyfunction(signature = (x, regression="c"))]
pub fn pp_test(x: PyReadonlyArray1<f64>, regression: &str) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();
    if n < 3 {
        return Ok((f64::NAN, f64::NAN));
    }

    let mut dy = Vec::with_capacity(n - 1);
    for t in 1..n {
        dy.push(x[t] - x[t - 1]);
    }

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mean_y_lag = x[..n - 1].iter().copied().sum::<f64>() / ((n - 1) as f64);
    let mean_dy = dy.iter().copied().sum::<f64>() / ((n - 1) as f64);

    for t in 1..n {
        let y_lag = x[t - 1] - mean_y_lag;
        let dy_t = dy[t - 1] - mean_dy;
        sxx += y_lag * y_lag;
        sxy += y_lag * dy_t;
    }

    if sxx <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let beta = sxy / sxx;

    let mut rss = 0.0;
    for t in 1..n {
        let y_hat = mean_dy + beta * (x[t - 1] - mean_y_lag);
        let e = dy[t - 1] - y_hat;
        rss += e * e;
    }

    let dof = (n - 1) as i64 - 2;
    if dof <= 0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let sigma2 = rss / (dof as f64);
    let se_beta = (sigma2 / sxx).sqrt();
    if se_beta <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let t_stat = beta / se_beta;
    let z = Normal::new(0.0, 1.0).unwrap();
    let p_val = 2.0 * (1.0 - z.cdf(t_stat.abs()));

    Ok((t_stat, p_val))
}
