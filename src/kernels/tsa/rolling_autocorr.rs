// 8) Rolling autocorrelation
// ======================

/// Rolling autocorrelation over a sliding window.
///
/// Python signature:
///     rolling_autocorr(x, lag=1, window=50) -> 1D array (length n-window+1)
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

        let w_f = window as f64;
        let mean = slice.iter().copied().sum::<f64>() / w_f;

        let mut denom = 0.0;
        for &v in slice {
            let d = v - mean;
            denom += d * d;
        }
        if denom <= 0.0 {
            out.push(f64::NAN);
            continue;
        }

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
