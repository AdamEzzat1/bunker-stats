// ======================
// Rolling autocorrelation
// ======================

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Rolling autocorrelation over a sliding window.
///
/// OPTIMIZED: Single-pass calculation of mean and variance per window
/// Python signature:
///     rolling_autocorr(x, lag=1, window=50) -> 1D array (length n-window+1)
#[pyo3::pyfunction(signature = (x, lag=1, window=50))]
pub fn rolling_autocorr<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    lag: usize,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if n == 0 || window == 0 || window > n {
        return Err(PyValueError::new_err(
            "window must be between 1 and len(x)",
        ));
    }
    if lag >= window {
        return Err(PyValueError::new_err(
            "lag must be smaller than window",
        ));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);

    for start in 0..=n - window {
        let slice = &data[start..start + window];

        // OPTIMIZATION: Single pass for mean
        let w_f = window as f64;
        let mean = slice.iter().copied().sum::<f64>() / w_f;

        // OPTIMIZATION: Single pass for variance
        let mut denom = 0.0;
        for &v in slice {
            let d = v - mean;
            denom += d * d;
        }
        
        if denom <= 0.0 {
            out.push(f64::NAN);
            continue;
        }

        // Calculate autocorrelation at specified lag
        let mut numer = 0.0;
        for i in lag..window {
            let a = slice[i] - mean;
            let b = slice[i - lag] - mean;
            numer += a * b;
        }

        out.push(numer / denom);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

// ======================
// NEW CHEAP FUNCTIONS
// ======================

/// Rolling correlation between two series
///
/// Python signature:
///     rolling_correlation(x, y, window=50) -> 1D array
#[pyfunction(signature = (x, y, window=50))]
pub fn rolling_correlation<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_data = x.as_slice()?;
    let y_data = y.as_slice()?;
    let n = x_data.len();

    if n != y_data.len() {
        return Err(PyValueError::new_err("x and y must have same length"));
    }
    
    if window == 0 || window > n {
        return Err(PyValueError::new_err(
            "window must be between 1 and len(x)",
        ));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);

    for start in 0..=n - window {
        let x_slice = &x_data[start..start + window];
        let y_slice = &y_data[start..start + window];

        let w_f = window as f64;
        
        // Calculate means
        let mean_x = x_slice.iter().copied().sum::<f64>() / w_f;
        let mean_y = y_slice.iter().copied().sum::<f64>() / w_f;

        // Calculate covariance and variances
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        
        for i in 0..window {
            let dx = x_slice[i] - mean_x;
            let dy = y_slice[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        
        let denom = (var_x * var_y).sqrt();
        if denom <= 0.0 {
            out.push(f64::NAN);
        } else {
            out.push(cov / denom);
        }
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

/// Rolling autocorrelation at multiple lags
///
/// Returns matrix where each column is rolling autocorr at a different lag
/// Python signature:
///     rolling_autocorr_multi(x, lags=[1,2,3], window=50) -> 2D array
#[pyfunction(signature = (x, lags, window=50))]
pub fn rolling_autocorr_multi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    lags: Vec<usize>,
    window: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 || window > n {
        return Err(PyValueError::new_err(
            "window must be between 1 and len(x)",
        ));
    }
    
    // Validate all lags
    for &lag in &lags {
        if lag >= window {
            return Err(PyValueError::new_err(
                format!("all lags must be smaller than window, got lag={}", lag)
            ));
        }
    }

    let out_len = n - window + 1;
    let n_lags = lags.len();
    let mut out = vec![0.0; out_len * n_lags];

    for start in 0..=n - window {
        let slice = &data[start..start + window];
        let w_f = window as f64;
        let mean = slice.iter().copied().sum::<f64>() / w_f;

        let mut denom = 0.0;
        for &v in slice {
            let d = v - mean;
            denom += d * d;
        }
        
        // Calculate autocorrelation for each lag
        for (lag_idx, &lag) in lags.iter().enumerate() {
            let val = if denom <= 0.0 {
                f64::NAN
            } else {
                let mut numer = 0.0;
                for i in lag..window {
                    let a = slice[i] - mean;
                    let b = slice[i - lag] - mean;
                    numer += a * b;
                }
                numer / denom
            };
            
            // Store ROW-major: from_shape_vec((out_len, n_lags), ..) below reads
            // C-order, so the flat index must be row*n_lags + col. The previous
            // column-major write (start + lag_idx*out_len) scrambled the output
            // whenever more than one lag was requested (single-lag calls happen
            // to coincide, which is why they looked correct).
            out[start * n_lags + lag_idx] = val;
        }
    }

    // Convert to 2D array (shape: [out_len, n_lags])
    use numpy::IntoPyArray;
    let arr = numpy::ndarray::Array2::from_shape_vec((out_len, n_lags), out)
        .map_err(|e| PyValueError::new_err(format!("shape error: {e}")))?;
    Ok(arr.into_pyarray_bound(py))
}
