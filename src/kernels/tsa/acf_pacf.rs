use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::Bound;
use nalgebra::{DMatrix, DVector};

// 5) ACF / PACF
// ======================

pub fn acf_raw(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let x_d = super::demean(x);
    let var = super::variance(&x_d);
    if var <= 0.0 {
        return vec![1.0; max_lag + 1];
    }

    let mut acf = vec![0.0_f64; max_lag + 1];
    acf[0] = 1.0;

    for k in 1..=max_lag {
        let mut num = 0.0;
        for t in k..n {
            num += x_d[t] * x_d[t - k];
        }
        acf[k] = num / (var * (n as f64));
    }

    acf
}

/// Autocorrelation function up to `max_lag`.
///
/// Python signature:
///     acf(x, max_lag=40) -> 1D array
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



/// Partial autocorrelation function (Yule–Walker) up to `max_lag`.
///
/// Python signature:
///     pacf(x, max_lag=40) -> 1D array
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
    let vals = pacf_yw(x, nlags);
    Ok(PyArray1::from_vec_bound(py, vals))
}



fn pacf_yw(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let max_lag = max_lag.min(n - 1);
    if max_lag == 0 {
        return vec![1.0];
    }

    let r = acf_raw(x, max_lag);
    let mut pacf = vec![0.0_f64; max_lag + 1];
    pacf[0] = 1.0;

    for k in 1..=max_lag {
        let mut r_vec = DVector::zeros(k);
        for i in 0..k {
            r_vec[i] = r[(i + 1) as usize];
        }

        let mut r_mat = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let idx = (i as isize - j as isize).abs() as usize;
                r_mat[(i, j)] = r[idx];
            }
        }

        if let Some(phi) = r_mat.lu().solve(&r_vec) {
            pacf[k] = phi[k - 1];
        } else {
            pacf[k] = f64::NAN;
        }
    }

    pacf
}
