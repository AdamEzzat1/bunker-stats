use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::cmp::min;
use nalgebra::{DMatrix, DVector};

// ======================
// Serial-correlation diagnostics
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

    // OPTIMIZATION: Single-pass mean and variance
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

    // CRITICAL: Use biased ACF for Ljung-Box (matches statsmodels)
    // r_k = sum(x[t] * x[t+k]) / sum(x[t]^2)
    // Sum over t=0..(n-k-1) for the numerator, divide by TOTAL variance
    let n_f = n as f64;
    let mut q = 0.0;
    
    for k in 1..=max_lag {
        let mut num = 0.0;
        // Biased ACF: sum from t=0 to t=(n-k-1), i.e., n-k terms
        for t in 0..(n - k) {
            num += (x[t] - mean) * (x[t + k] - mean);
        }
        
        // Divide by TOTAL variance (all n terms), not (n-k)
        let rk = num / denom;
        
        // Ljung-Box formula
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
/// DEBUGGING VERSION - prints intermediate values
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
            x_mat.push(e[t - j]);
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

    let xtx_mat = DMatrix::from_vec(p, p, xtx);
    let xty_vec = DVector::from_vec(xty);

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

    // Calculate TSS
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

    // Breusch-Godfrey LM statistic = T · R², where T is the number of
    // observations in the auxiliary regression (here the reduced sample
    // t_len = n - max_lag, since the first max_lag rows have no full lag set).
    // Asymptotically ~ χ²(max_lag) under the null of no serial correlation.
    // NOTE: statsmodels' `acorr_breusch_godfrey` zero-pads the pre-sample lags
    // to keep all n rows and reports n · R²; that is a different finite-sample
    // convention, not a discrepancy in this statistic.
    let stat = (t_len as f64) * r2;

    // P-value from chi-squared distribution with max_lag degrees of freedom.
    // max_lag >= 1 is guaranteed above, so ChiSquared::new cannot fail; fall
    // back to NaN defensively rather than risk an abort under panic="abort".
    let p_val = match ChiSquared::new(max_lag as f64) {
        Ok(chi) => 1.0 - chi.cdf(stat),
        Err(_) => f64::NAN,
    };

    Ok((stat, p_val))
}


// ======================
// NEW CHEAP FUNCTIONS
// ======================

/// Box-Pierce test (simpler alternative to Ljung-Box)
///
/// Python signature:
///     box_pierce(x, lags=20) -> (statistic, pvalue)
#[pyfunction(signature = (x, lags=20))]
pub fn box_pierce(x: PyReadonlyArray1<f64>, lags: usize) -> PyResult<(f64, f64)> {
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

    let mut q = 0.0;
    for k in 1..=max_lag {
        let mut num = 0.0;
        for t in k..n {
            num += (x[t] - mean) * (x[t - k] - mean);
        }
        let rk = num / denom;
        q += rk * rk;
    }
    q *= n as f64;

    let chi = ChiSquared::new(max_lag as f64).unwrap();
    let p_val = 1.0 - chi.cdf(q);

    Ok((q, p_val))
}

/// Runs test for randomness (Wald-Wolfowitz)
///
/// Tests if values above/below median occur randomly
/// Python signature:
///     runs_test(x) -> (n_runs, z_score, pvalue)
#[pyfunction]
pub fn runs_test(x: PyReadonlyArray1<f64>) -> PyResult<(usize, f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();
    
    if n < 2 {
        return Ok((0, f64::NAN, f64::NAN));
    }
    
    // Calculate median
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    
    // Count runs and n+ / n-
    let mut runs = 1;
    let mut n_plus = 0;
    let mut n_minus = 0;
    let mut prev_sign = if x[0] > median { 
        n_plus += 1; 
        1 
    } else if x[0] < median { 
        n_minus += 1; 
        -1 
    } else { 
        0 
    };
    
    for &val in &x[1..] {
        let sign = if val > median { 
            n_plus += 1; 
            1 
        } else if val < median { 
            n_minus += 1; 
            -1 
        } else { 
            0 
        };
        
        if sign != 0 && prev_sign != 0 && sign != prev_sign {
            runs += 1;
        }
        if sign != 0 {
            prev_sign = sign;
        }
    }
    
    if n_plus == 0 || n_minus == 0 {
        return Ok((runs, f64::NAN, f64::NAN));
    }
    
    // Expected runs and variance under null
    let n1 = n_plus as f64;
    let n2 = n_minus as f64;
    let n_total = n1 + n2;
    
    let expected_runs = (2.0 * n1 * n2) / n_total + 1.0;
    let var_runs = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n_total)) / 
                   (n_total * n_total * (n_total - 1.0));
    
    if var_runs <= 0.0 {
        return Ok((runs, f64::NAN, f64::NAN));
    }
    
    // Z-score with continuity correction
    let z = if runs as f64 > expected_runs {
        (runs as f64 - expected_runs - 0.5) / var_runs.sqrt()
    } else {
        (runs as f64 - expected_runs + 0.5) / var_runs.sqrt()
    };
    
    // Two-tailed p-value (approximate with normal)
    use statrs::distribution::Normal;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_val = 2.0 * (1.0 - normal.cdf(z.abs()));
    
    Ok((runs, z, p_val))
}

/// Calculate lag at which ACF first crosses zero
///
/// Useful for identifying correlation structure
/// Python signature:
///     acf_zero_crossing(x, max_lag=100) -> Optional[int]
#[pyfunction(signature = (x, max_lag=100))]
pub fn acf_zero_crossing(x: PyReadonlyArray1<f64>, max_lag: usize) -> PyResult<Option<usize>> {
    let x = x.as_slice()?;
    let n = x.len();
    
    if n < 2 {
        return Ok(None);
    }
    
    let max_lag = max_lag.min(n - 1);
    
    // Calculate ACF values
    let mean = x.iter().copied().sum::<f64>() / (n as f64);
    let mut var = 0.0;
    for &v in x {
        let d = v - mean;
        var += d * d;
    }
    var /= n as f64;
    
    if var <= 0.0 {
        return Ok(None);
    }
    
    let mut prev_acf = 1.0;  // ACF at lag 0
    
    for k in 1..=max_lag {
        let mut num = 0.0;
        for t in k..n {
            num += (x[t] - mean) * (x[t - k] - mean);
        }
        let acf = num / (var * (n as f64));
        
        // Check for sign change
        if prev_acf > 0.0 && acf <= 0.0 {
            return Ok(Some(k));
        }
        
        prev_acf = acf;
    }
    
    Ok(None)
}