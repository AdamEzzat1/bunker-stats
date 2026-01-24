use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::Bound;

// ======================
// ACF / PACF (Levinson-Durbin optimized)
// ======================

/// Core ACF calculation (optimized, no allocation for demeaning)
pub fn acf_raw(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Inline demean + variance calculation (single pass optimization)
    let mean = x.iter().sum::<f64>() / (n as f64);
    let mut var = 0.0;
    for &v in x {
        let d = v - mean;
        var += d * d;
    }
    var /= n as f64;
    
    if var <= 0.0 {
        return vec![1.0; max_lag + 1];
    }

    let mut acf = vec![0.0_f64; max_lag + 1];
    acf[0] = 1.0;

    for k in 1..=max_lag {
        let mut num = 0.0;
        for t in k..n {
            num += (x[t] - mean) * (x[t - k] - mean);
        }
        acf[k] = num / (var * (n as f64));
    }

    acf
}

/// Autocorrelation function up to `nlags`.
///
/// Python signature:
///     acf(x, nlags=40) -> 1D array
#[pyfunction(signature = (x, nlags=40))]
pub fn acf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let n = x.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let nlags = nlags.min(n - 1);
    let vals = acf_raw(x, nlags);
    Ok(PyArray1::from_vec_bound(py, vals))
}

/// Partial autocorrelation function using Levinson-Durbin recursion
///
/// O(k²) instead of O(k³) for Yule-Walker
/// Python signature:
///     pacf(x, nlags=40) -> 1D array
#[pyfunction(signature = (x, nlags=40))]
pub fn pacf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let n = x.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let nlags = nlags.min(n - 1);
    let vals = pacf_levinson_durbin(x, nlags);
    Ok(PyArray1::from_vec_bound(py, vals))
}

/// Levinson-Durbin recursion for PACF
///
/// Exploits Toeplitz structure of autocorrelation matrix
/// Complexity: O(k²) vs O(k³) for generic matrix solve
fn pacf_levinson_durbin(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let max_lag = max_lag.min(n - 1);
    if max_lag == 0 {
        return vec![1.0];
    }

    // Compute ACF
    let r = acf_raw(x, max_lag);
    
    let mut pacf = vec![0.0_f64; max_lag + 1];
    pacf[0] = 1.0;
    
    if max_lag == 0 || r[1].is_nan() {
        return pacf;
    }
    
    // Initialize for k=1
    let mut phi = vec![0.0; max_lag];
    let mut phi_new = vec![0.0; max_lag];
    
    phi[0] = r[1];
    pacf[1] = r[1];
    let mut v = 1.0 - r[1] * r[1];
    
    if v <= 0.0 || !v.is_finite() {
        // Variance exhausted, fill remaining with NaN
        for i in 2..=max_lag {
            pacf[i] = f64::NAN;
        }
        return pacf;
    }
    
    // Durbin recursion for k = 2, 3, ..., max_lag
    for k in 1..max_lag {
        // Calculate new PACF coefficient
        let mut sum = 0.0;
        for j in 0..k {
            sum += phi[j] * r[k - j];
        }
        
        let a = (r[k + 1] - sum) / v;
        
        if !a.is_finite() {
            // Numerical breakdown
            for i in (k + 1)..=max_lag {
                pacf[i] = f64::NAN;
            }
            break;
        }
        
        // Update AR coefficients using Levinson-Durbin recursion
        for j in 0..k {
            phi_new[j] = phi[j] - a * phi[k - 1 - j];
        }
        phi_new[k] = a;
        
        // Update prediction error variance
        v *= 1.0 - a * a;
        
        pacf[k + 1] = a;
        
        // Swap buffers for next iteration
        std::mem::swap(&mut phi, &mut phi_new);
        
        if v <= 0.0 || !v.is_finite() {
            // Variance exhausted or numerical instability
            for i in (k + 2)..=max_lag {
                pacf[i] = f64::NAN;
            }
            break;
        }
    }
    
    pacf
}

/// Fallback to Yule-Walker (for compatibility/testing)
///
/// Python signature:
///     pacf_yw(x, nlags=40) -> 1D array
#[pyfunction(signature = (x, nlags=40))]
pub fn pacf_yw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use nalgebra::{DMatrix, DVector};
    
    let x = x.as_slice()?;
    let n = x.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let max_lag = nlags.min(n - 1);
    if max_lag == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![1.0]));
    }

    let r = acf_raw(x, max_lag);
    let mut pacf = vec![0.0_f64; max_lag + 1];
    pacf[0] = 1.0;

    for k in 1..=max_lag {
        let mut r_vec = DVector::zeros(k);
        for i in 0..k {
            r_vec[i] = r[i + 1];
        }

        let mut r_mat = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let idx = (i as isize - j as isize).unsigned_abs();
                r_mat[(i, j)] = r[idx];
            }
        }

        if let Some(phi) = r_mat.lu().solve(&r_vec) {
            pacf[k] = phi[k - 1];
        } else {
            pacf[k] = f64::NAN;
        }
    }

    Ok(PyArray1::from_vec_bound(py, pacf))
}

// ======================
// NEW CHEAP FUNCTIONS
// ======================

/// Autocovariance function (unnormalized ACF)
///
/// Python signature:
///     acovf(x, nlags=40) -> 1D array
#[pyfunction(signature = (x, nlags=40))]
pub fn acovf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let n = x.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let nlags = nlags.min(n - 1);
    
    let mean = x.iter().sum::<f64>() / (n as f64);
    
    let mut acovf = vec![0.0_f64; nlags + 1];
    for k in 0..=nlags {
        let mut sum = 0.0;
        for t in k..n {
            sum += (x[t] - mean) * (x[t - k] - mean);
        }
        acovf[k] = sum / (n as f64);
    }
    
    Ok(PyArray1::from_vec_bound(py, acovf))
}

/// ACF with confidence bands using Bartlett's formula
///
/// Python signature:
///     acf_with_ci(x, nlags=40, alpha=0.05) -> (acf, lower, upper)
#[pyfunction(signature = (x, nlags=40, alpha=0.05))]
pub fn acf_with_ci<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
    alpha: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let x = x.as_slice()?;
    let n = x.len();
    if n == 0 {
        return Ok((
            PyArray1::from_vec_bound(py, vec![]),
            PyArray1::from_vec_bound(py, vec![]),
            PyArray1::from_vec_bound(py, vec![]),
        ));
    }
    let nlags = nlags.min(n - 1);
    
    let acf_vals = acf_raw(x, nlags);
    
    // Critical value for normal distribution
    let z = if alpha <= 0.01 {
        2.576
    } else if alpha <= 0.05 {
        1.96
    } else {
        1.645
    };
    
    let n_f = n as f64;
    let mut lower = Vec::with_capacity(nlags + 1);
    let mut upper = Vec::with_capacity(nlags + 1);
    
    lower.push(1.0);
    upper.push(1.0);
    
    for k in 1..=nlags {
        let mut var_sum = 1.0;
        for j in 1..k {
            var_sum += 2.0 * acf_vals[j] * acf_vals[j];
        }
        let se = (var_sum / n_f).sqrt();
        lower.push(acf_vals[k] - z * se);
        upper.push(acf_vals[k] + z * se);
    }
    
    Ok((
        PyArray1::from_vec_bound(py, acf_vals),
        PyArray1::from_vec_bound(py, lower),
        PyArray1::from_vec_bound(py, upper),
    ))
}

/// Cross-correlation function between two series
///
/// Python signature:
///     ccf(x, y, nlags=40) -> 1D array of length 2*nlags+1
#[pyfunction(signature = (x, y, nlags=40))]
pub fn ccf<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    let n = x.len();
    
    if n == 0 || y.len() != n {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    
    let nlags = nlags.min(n - 1);
    
    // Demean both series
    let mean_x = x.iter().sum::<f64>() / (n as f64);
    let mean_y = y.iter().sum::<f64>() / (n as f64);
    
    // Calculate standard deviations
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        var_x += (x[i] - mean_x).powi(2);
        var_y += (y[i] - mean_y).powi(2);
    }
    let std_x = (var_x / (n as f64)).sqrt();
    let std_y = (var_y / (n as f64)).sqrt();
    
    if std_x <= 0.0 || std_y <= 0.0 {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; 2 * nlags + 1]));
    }
    
    let mut ccf = Vec::with_capacity(2 * nlags + 1);
    
    // Negative lags
    for k in (1..=nlags).rev() {
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += (x[t] - mean_x) * (y[t + k] - mean_y);
        }
        ccf.push(sum / (n as f64 * std_x * std_y));
    }
    
    // Zero lag
    let mut sum = 0.0;
    for t in 0..n {
        sum += (x[t] - mean_x) * (y[t] - mean_y);
    }
    ccf.push(sum / (n as f64 * std_x * std_y));
    
    // Positive lags
    for k in 1..=nlags {
        let mut sum = 0.0;
        for t in k..n {
            sum += (x[t] - mean_x) * (y[t - k] - mean_y);
        }
        ccf.push(sum / (n as f64 * std_x * std_y));
    }
    
    Ok(PyArray1::from_vec_bound(py, ccf))
}

// ======================
// ADDITIONAL FUNCTIONS
// ======================

/// Innovations algorithm for PACF (alternative to Levinson-Durbin)
///
/// Numerically stable for ill-conditioned problems
/// Python signature:
///     pacf_innovations(x, nlags=40) -> 1D array
#[pyfunction(signature = (x, nlags=40))]
pub fn pacf_innovations<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let n = x.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let max_lag = nlags.min(n - 1);
    if max_lag == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![1.0]));
    }

    let r = acf_raw(x, max_lag);
    
    let mut pacf = vec![0.0_f64; max_lag + 1];
    pacf[0] = 1.0;
    
    let mut v = vec![1.0_f64; max_lag + 1];
    let mut theta = vec![vec![0.0; max_lag]; max_lag];
    
    for n_iter in 1..=max_lag {
        // Compute theta coefficients
        for k in 0..n_iter {
            let mut sum = r[n_iter - k];
            for j in 0..k {
                sum -= theta[k][j] * theta[n_iter - 1][j] * v[j];
            }
            theta[n_iter - 1][k] = sum / v[k];
        }
        
        // Update prediction error variance
        let mut var_update = r[0];
        for k in 0..n_iter {
            var_update -= theta[n_iter - 1][k] * theta[n_iter - 1][k] * v[k];
        }
        v[n_iter] = var_update;
        
        // PACF is the last theta coefficient
        pacf[n_iter] = theta[n_iter - 1][n_iter - 1];
        
        if v[n_iter] <= 0.0 || !v[n_iter].is_finite() {
            for i in (n_iter + 1)..=max_lag {
                pacf[i] = f64::NAN;
            }
            break;
        }
    }
    
    Ok(PyArray1::from_vec_bound(py, pacf))
}

/// Burg's algorithm for PACF (maximum entropy method)
///
/// Good for short time series
/// Python signature:
///     pacf_burg(x, nlags=40) -> 1D array
#[pyfunction(signature = (x, nlags=40))]
pub fn pacf_burg<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let n = x.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let max_lag = nlags.min(n - 1);
    if max_lag == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![1.0]));
    }

    // Demean
    let mean = x.iter().sum::<f64>() / (n as f64);
    let x_centered: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    
    let mut pacf = vec![0.0_f64; max_lag + 1];
    pacf[0] = 1.0;
    
    // Initialize forward and backward prediction errors
    let mut ef = x_centered.clone();
    let mut eb = x_centered.clone();
    let mut a = vec![1.0];
    
    for k in 1..=max_lag {
        // Calculate reflection coefficient
        let mut num = 0.0;
        let mut denom = 0.0;
        
        for i in k..n {
            num += ef[i] * eb[i - 1];
            denom += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }
        
        if denom == 0.0 || !denom.is_finite() {
            for i in k..=max_lag {
                pacf[i] = f64::NAN;
            }
            break;
        }
        
        // Burg's reflection coefficient formula
        // Note: Sign convention matches Levinson-Durbin and other PACF methods
        let rc = 2.0 * num / denom;
        pacf[k] = rc;
        
        // Update AR coefficients
        let mut a_new = vec![0.0; k + 1];
        a_new[0] = 1.0;
        for i in 1..k {
            a_new[i] = a[i] + rc * a[k - i];
        }
        a_new[k] = rc;
        a = a_new;
        
        // Update prediction errors
        let mut ef_new = vec![0.0; n];
        let mut eb_new = vec![0.0; n];
        
        for i in k..n {
            ef_new[i] = ef[i] + rc * eb[i - 1];
            eb_new[i] = eb[i - 1] + rc * ef[i];
        }
        
        ef = ef_new;
        eb = eb_new;
    }
    
    Ok(PyArray1::from_vec_bound(py, pacf))
}