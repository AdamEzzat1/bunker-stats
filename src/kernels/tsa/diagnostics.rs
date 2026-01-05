use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::cmp::min;
use nalgebra::{DMatrix, DVector};

// 7) Serial-correlation diagnostics
// ======================

/// Ljung–Box test for autocorrelation up to a given lag.
///
/// Python signature:
///     ljung_box(x, lags=20) -> (statistic, pvalue)
#[pyfunction(signature = (x, lags=20))]
pub fn ljung_box(x: PyReadonlyArray1<f64>, lags: usize) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();

    if n == 0 || lags == 0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let mean = x.iter().copied().sum::<f64>() / (n as f64);
    let mut denom = 0.0;
    for &v in x {
        let d = v - mean;
        denom += d * d;
    }
    if denom <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let max_lag = min(lags, n.saturating_sub(1));
    if max_lag == 0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let mut acf = vec![0.0f64; max_lag + 1];
    for k in 1..=max_lag {
        let mut num = 0.0;
        for t in k..n {
            num += (x[t] - mean) * (x[t - k] - mean);
        }
        acf[k] = num / denom;
    }

    let n_f = n as f64;
    let mut q = 0.0;
    for k in 1..=max_lag {
        let rk = acf[k];
        q += rk * rk / (n_f - k as f64);
    }
    q *= n_f * (n_f + 2.0);

    let chi = ChiSquared::new(max_lag as f64).unwrap();
    let p_val = 1.0 - chi.cdf(q);

    Ok((q, p_val))
}

/// Durbin–Watson statistic for first-order autocorrelation.
///
/// Python signature:
///     durbin_watson(x) -> scalar in [0,4]
#[pyfunction]
pub fn durbin_watson(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x = x.as_slice()?;
    let n = x.len();
    if n < 2 {
        return Ok(f64::NAN);
    }

    let mut num = 0.0;
    let mut denom = 0.0;
    for t in 1..n {
        let diff = x[t] - x[t - 1];
        num += diff * diff;
    }
    for t in 0..n {
        denom += x[t] * x[t];
    }
    if denom <= 0.0 {
        return Ok(f64::NAN);
    }
    Ok(num / denom)
}

/// Breusch–Godfrey test (LM test for serial correlation in residuals).
///
/// Python signature:
///     bg_test(resid, max_lag=5) -> (statistic, pvalue)
#[pyfunction(signature = (resid, max_lag=5))]
pub fn bg_test(resid: PyReadonlyArray1<f64>, max_lag: usize) -> PyResult<(f64, f64)> {
    let e = resid.as_slice()?;
    let n = e.len();
    if n <= max_lag + 1 || max_lag == 0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let t0 = max_lag;
    let t_len = n - t0;

    // Dependent variable: current residuals e[t] for t >= max_lag
    let mut y = Vec::with_capacity(t_len);
    for t in t0..n {
        y.push(e[t]);
    }

    // Independent variables: intercept + lagged residuals e[t-1], ..., e[t-max_lag]
    let mut x_mat = Vec::with_capacity(t_len * (max_lag + 1));
    for t in t0..n {
        x_mat.push(1.0);  // intercept
        for j in 1..=max_lag {
            x_mat.push(e[t - j]);  // e[t-1], e[t-2], ..., e[t-max_lag]
        }
    }

    let p = max_lag + 1;
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];

    for t in 0..t_len {
        for i in 0..p {
            let xi = x_mat[t * p + i];
            xty[i] += xi * y[t];
            for j in 0..p {
                xtx[i * p + j] += xi * x_mat[t * p + j];
            }
        }
    }

    let xtx_mat = DMatrix::from_vec(p, p, xtx.clone());
    let xty_vec = DVector::from_vec(xty.clone());

    let beta = if let Some(sol) = xtx_mat.lu().solve(&xty_vec) {
        sol
    } else {
        return Ok((f64::NAN, f64::NAN));
    };

    // Calculate RSS (Residual Sum of Squares from auxiliary regression)
    let mut rss = 0.0;
    for t in 0..t_len {
        let mut y_hat = 0.0;
        for i in 0..p {
            y_hat += beta[i] * x_mat[t * p + i];
        }
        let residual = y[t] - y_hat;
        rss += residual * residual;
    }

    // FIXED: Calculate TSS (Total Sum of Squares) correctly
    // TSS = sum of squared deviations from mean of y (the current residuals)
    let mean_y: f64 = y.iter().sum::<f64>() / (t_len as f64);
    let mut tss = 0.0;
    for &yi in &y {
        let dev = yi - mean_y;
        tss += dev * dev;
    }

    // R² from auxiliary regression
    let r2 = if tss > 0.0 {
        1.0 - rss / tss
    } else {
        0.0
    };
	
	// LM statistic: (T - p) * R² where T is sample size, p is number of lags
    let t_len_f = t_len as f64;
    let max_lag_f = max_lag as f64;  // ADD THIS LINE
    let stat = (t_len_f - max_lag_f) * r2;

    // P-value from chi-squared distribution with max_lag degrees of freedom
    let chi = ChiSquared::new(max_lag as f64).unwrap();
    let p_val = 1.0 - chi.cdf(stat);

    Ok((stat, p_val))
}
