use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::Bound;
use std::f64::consts::PI;

// 9) Periodogram / spectral density
// ======================

fn periodogram_raw(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let x_d = super::demean(x);
    let n_f = n as f64;

    let mut freqs = Vec::new();
    let mut power = Vec::new();

    let k_max = n / 2;
    for k in 0..=k_max {
        let freq = (k as f64) / n_f;
        let mut re = 0.0;
        let mut im = 0.0;
        for t in 0..n {
            let angle = -2.0 * PI * (k as f64) * (t as f64) / n_f;
            let val = x_d[t];
            re += val * angle.cos();
            im += val * angle.sin();
        }
        let p = (re * re + im * im) / (n_f * 2.0 * PI);
        freqs.push(freq);
        power.push(p);
    }

    (freqs, power)
}

/// Raw periodogram (no windowing).
///
/// Python signature:
///     periodogram(x) -> (freqs, power)
#[pyfunction]
pub fn periodogram<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x = x.as_slice()?;
    let (freqs, power) = periodogram_raw(x);
    let f_arr = PyArray1::from_vec_bound(py, freqs);
    let p_arr = PyArray1::from_vec_bound(py, power);
    Ok((f_arr, p_arr))
}
