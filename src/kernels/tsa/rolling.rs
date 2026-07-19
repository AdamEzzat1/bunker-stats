// ======================
// Rolling Statistics (O(1) per step optimizations)
// ======================

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::VecDeque;

// ======================
// O(1) ROLLING MEAN/SUM
// ======================

/// Rolling mean with O(1) updates using cumulative sum
///
/// Python signature:
///     rolling_mean(x, window=20) -> 1D array (length n-window+1)
#[pyfunction(signature = (x, window=20))]
pub fn rolling_mean<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    // Initial window sum
    let mut sum: f64 = data[..window].iter().sum();
    let w_f = window as f64;
    out.push(sum / w_f);
    
    // Rolling updates: O(1) per step
    for i in window..n {
        sum = sum - data[i - window] + data[i];
        out.push(sum / w_f);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

/// Rolling sum with O(1) updates
///
/// Python signature:
///     rolling_sum(x, window=20) -> 1D array
#[pyfunction(signature = (x, window=20))]
pub fn rolling_sum<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    let mut sum: f64 = data[..window].iter().sum();
    out.push(sum);
    
    for i in window..n {
        sum = sum - data[i - window] + data[i];
        out.push(sum);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

// ======================
// WELFORD'S ALGORITHM (O(1) VARIANCE)
// ======================

/// Rolling variance using Welford's online algorithm (O(1) updates)
///
/// Python signature:
///     rolling_var(x, window=20, ddof=1) -> 1D array
#[pyfunction(signature = (x, window=20, ddof=1))]
pub fn rolling_var<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    
    if ddof >= window {
        return Err(PyValueError::new_err(
            "ddof must be less than window",
        ));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    // Initial window using Welford
    let mut mean = 0.0;
    let mut m2 = 0.0;
    
    for (i, &val) in data[..window].iter().enumerate() {
        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    let variance = m2 / (window - ddof) as f64;
    out.push(variance);
    
    // Rolling Welford update (O(1) per step)
    for i in window..n {
        let old_val = data[i - window];
        let new_val = data[i];
        
        // Update mean
        let old_mean = mean;
        mean += (new_val - old_val) / window as f64;
        
        // Update M2 (sum of squared deviations)
        m2 = m2 + (new_val - old_val) * (new_val - mean + old_val - old_mean);
        
        let variance = m2 / (window - ddof) as f64;
        out.push(variance);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

/// Rolling standard deviation using Welford's algorithm (O(1) updates)
///
/// Python signature:
///     rolling_std(x, window=20, ddof=1) -> 1D array
#[pyfunction(signature = (x, window=20, ddof=1))]
pub fn rolling_std<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    
    if ddof >= window {
        return Err(PyValueError::new_err(
            "ddof must be less than window",
        ));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    // Initial window
    let mut mean = 0.0;
    let mut m2 = 0.0;
    
    for (i, &val) in data[..window].iter().enumerate() {
        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    let std = (m2 / (window - ddof) as f64).sqrt();
    out.push(std);
    
    // Rolling updates
    for i in window..n {
        let old_val = data[i - window];
        let new_val = data[i];
        
        let old_mean = mean;
        mean += (new_val - old_val) / window as f64;
        m2 = m2 + (new_val - old_val) * (new_val - mean + old_val - old_mean);
        
        let std = (m2 / (window - ddof) as f64).sqrt();
        out.push(std);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

// ======================
// MONOTONIC DEQUE (O(1) MIN/MAX)
// ======================

/// Rolling minimum using monotonic deque (O(1) amortized)
///
/// Python signature:
///     rolling_min(x, window=20) -> 1D array
#[pyfunction(signature = (x, window=20))]
pub fn rolling_min<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    // Monotonic deque: stores indices of potential minimums
    let mut deque: VecDeque<usize> = VecDeque::new();
    
    // Initialize first window
    for i in 0..window {
        while let Some(&back_idx) = deque.back() {
            if data[back_idx] >= data[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);
    }
    
    out.push(data[*deque.front().unwrap()]);
    
    // Slide window (O(1) amortized)
    for i in window..n {
        // Remove elements outside window
        while let Some(&front_idx) = deque.front() {
            if front_idx <= i - window {
                deque.pop_front();
            } else {
                break;
            }
        }
        
        // Add new element
        while let Some(&back_idx) = deque.back() {
            if data[back_idx] >= data[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);
        
        out.push(data[*deque.front().unwrap()]);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

/// Rolling maximum using monotonic deque (O(1) amortized)
///
/// Python signature:
///     rolling_max(x, window=20) -> 1D array
#[pyfunction(signature = (x, window=20))]
pub fn rolling_max<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    let mut deque: VecDeque<usize> = VecDeque::new();
    
    // Initialize first window
    for i in 0..window {
        while let Some(&back_idx) = deque.back() {
            if data[back_idx] <= data[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);
    }
    
    out.push(data[*deque.front().unwrap()]);
    
    // Slide window
    for i in window..n {
        while let Some(&front_idx) = deque.front() {
            if front_idx <= i - window {
                deque.pop_front();
            } else {
                break;
            }
        }
        
        while let Some(&back_idx) = deque.back() {
            if data[back_idx] <= data[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);
        
        out.push(data[*deque.front().unwrap()]);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

/// Rolling range (max - min) using two monotonic deques
///
/// Python signature:
///     rolling_range(x, window=20) -> 1D array
#[pyfunction(signature = (x, window=20))]
pub fn rolling_range<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    let mut min_deque: VecDeque<usize> = VecDeque::new();
    let mut max_deque: VecDeque<usize> = VecDeque::new();
    
    // Initialize first window
    for i in 0..window {
        while let Some(&back) = min_deque.back() {
            if data[back] >= data[i] {
                min_deque.pop_back();
            } else {
                break;
            }
        }
        min_deque.push_back(i);
        
        while let Some(&back) = max_deque.back() {
            if data[back] <= data[i] {
                max_deque.pop_back();
            } else {
                break;
            }
        }
        max_deque.push_back(i);
    }
    
    out.push(data[*max_deque.front().unwrap()] - data[*min_deque.front().unwrap()]);
    
    // Slide window
    for i in window..n {
        // Update min deque
        while let Some(&front) = min_deque.front() {
            if front <= i - window {
                min_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some(&back) = min_deque.back() {
            if data[back] >= data[i] {
                min_deque.pop_back();
            } else {
                break;
            }
        }
        min_deque.push_back(i);
        
        // Update max deque
        while let Some(&front) = max_deque.front() {
            if front <= i - window {
                max_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some(&back) = max_deque.back() {
            if data[back] <= data[i] {
                max_deque.pop_back();
            } else {
                break;
            }
        }
        max_deque.push_back(i);
        
        out.push(data[*max_deque.front().unwrap()] - data[*min_deque.front().unwrap()]);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

// ======================
// O(1) COUNTING OPERATIONS
// ======================

/// Rolling count of values above threshold (O(1) updates)
///
/// Python signature:
///     rolling_count_above(x, threshold=0.0, window=20) -> 1D array
#[pyfunction(signature = (x, threshold=0.0, window=20))]
pub fn rolling_count_above<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    threshold: f64,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    // Initial count
    let mut count = data[..window].iter().filter(|&&x| x > threshold).count();
    out.push(count as f64);
    
    // Rolling update (O(1))
    for i in window..n {
        if data[i - window] > threshold {
            count -= 1;
        }
        if data[i] > threshold {
            count += 1;
        }
        out.push(count as f64);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

/// Rolling percent above threshold (O(1) updates)
///
/// Python signature:
///     rolling_pct_above(x, threshold=0.0, window=20) -> 1D array
#[pyfunction(signature = (x, threshold=0.0, window=20))]
pub fn rolling_pct_above<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    threshold: f64,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    let w_f = window as f64;
    
    let mut count = data[..window].iter().filter(|&&x| x > threshold).count();
    out.push((count as f64) / w_f);
    
    for i in window..n {
        if data[i - window] > threshold {
            count -= 1;
        }
        if data[i] > threshold {
            count += 1;
        }
        out.push((count as f64) / w_f);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

// ======================
// ADDITIONAL CHEAP FUNCTIONS
// ======================

/// Rolling z-score (combines rolling mean and std)
///
/// Python signature:
///     rolling_zscore(x, window=20) -> 1D array
#[pyfunction(signature = (x, window=20))]
pub fn rolling_zscore<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    // Initial window
    let mut mean = 0.0;
    let mut m2 = 0.0;
    
    for (i, &val) in data[..window].iter().enumerate() {
        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    let std = (m2 / (window - 1) as f64).sqrt();
    let center_idx = window / 2;
    let z = if std > 0.0 {
        (data[center_idx] - mean) / std
    } else {
        f64::NAN
    };
    out.push(z);
    
    // Rolling updates
    for i in window..n {
        let old_val = data[i - window];
        let new_val = data[i];
        
        let old_mean = mean;
        mean += (new_val - old_val) / window as f64;
        m2 = m2 + (new_val - old_val) * (new_val - mean + old_val - old_mean);
        
        let std = (m2 / (window - 1) as f64).sqrt();
        let center_idx = i - window / 2;
        let z = if std > 0.0 {
            (data[center_idx] - mean) / std
        } else {
            f64::NAN
        };
        out.push(z);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

/// Rolling coefficient of variation (std / mean) using Welford
///
/// Python signature:
///     rolling_cv(x, window=20) -> 1D array
#[pyfunction(signature = (x, window=20))]
pub fn rolling_cv<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let n = data.len();

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        // v0.3 edge rule shared by strict rolling kernels: window larger than
        // the series yields an empty result, not an error.
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = Vec::with_capacity(out_len);
    
    // Initial window
    let mut mean = 0.0;
    let mut m2 = 0.0;
    
    for (i, &val) in data[..window].iter().enumerate() {
        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    let cv = if mean.abs() > 1e-10 {
        let std = (m2 / (window - 1) as f64).sqrt();
        std / mean.abs()
    } else {
        f64::NAN
    };
    out.push(cv);
    
    // Rolling updates
    for i in window..n {
        let old_val = data[i - window];
        let new_val = data[i];
        
        let old_mean = mean;
        mean += (new_val - old_val) / window as f64;
        m2 = m2 + (new_val - old_val) * (new_val - mean + old_val - old_mean);
        
        let cv = if mean.abs() > 1e-10 {
            let std = (m2 / (window - 1) as f64).sqrt();
            std / mean.abs()
        } else {
            f64::NAN
        };
        out.push(cv);
    }

    Ok(PyArray1::from_vec_bound(py, out))
}
