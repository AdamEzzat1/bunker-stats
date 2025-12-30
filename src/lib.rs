mod kernels;
mod infer;


use numpy::{
    ndarray::{Array2, ArrayViewD, Axis},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};


use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kernels::rolling::engine::rolling_mean_std_vec;
use kernels::rolling::zscore::zscore_from_mean_std;
use kernels::rolling::axis0::{rolling_mean_axis0_vec, rolling_std_axis0_vec, rolling_mean_std_axis0_vec};
use kernels::rolling::covcorr::rolling_cov_vec;

use kernels::rolling::var::vars_from_stds;
use kernels::quantile::percentile::percentile_slice as percentile_slice_k;
use kernels::quantile::iqr::iqr_slice as iqr_slice_k;
use kernels::quantile::winsor::winsorize_vec as winsorize_vec_k;
use kernels::matrix::cov::cov_matrix_out as cov_matrix_out_k;
use kernels::matrix::corr::corr_matrix_out as corr_matrix_out_k;
use kernels::robust::mad::mad_slice as mad_slice_k;
use kernels::robust::trimmed_mean::trimmed_mean_slice as trimmed_mean_slice_k;



// ======================
// Core slice helpers
// ======================

pub(crate) fn mean_slice(xs: &[f64]) -> f64 {

    if xs.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    for &x in xs {
        sum += x;
    }
    sum / (xs.len() as f64)
}

// Sample variance (ddof=1)
fn var_slice(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n <= 1 {
        return f64::NAN;
    }
    let m = mean_slice(xs);
    let mut acc = 0.0;
    for &x in xs {
        let d = x - m;
        acc += d * d;
    }
    acc / ((n - 1) as f64)
}

fn std_slice(xs: &[f64]) -> f64 {
    var_slice(xs).sqrt()
}

// NaN-aware helpers (skip NaNs, ddof=1 for var/std when >=2 valid values)

fn mean_slice_skipna(xs: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &x in xs {
        if x.is_nan() {
            continue;
        }
        sum += x;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / (count as f64)
    }
}

fn var_slice_skipna(xs: &[f64]) -> f64 {
    let mut values = Vec::with_capacity(xs.len());
    for &x in xs {
        if !x.is_nan() {
            values.push(x);
        }
    }
    let n = values.len();
    if n <= 1 {
        return f64::NAN;
    }
    let m = mean_slice(&values);
    let mut acc = 0.0;
    for &v in &values {
        let d = v - m;
        acc += d * d;
    }
    acc / ((n - 1) as f64)
}

fn std_slice_skipna(xs: &[f64]) -> f64 {
    var_slice_skipna(xs).sqrt()
}


fn mad_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    // Copy once, sort once for the median, then reuse the same buffer for deviations.
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    };

    // Reuse `v` to hold absolute deviations, then sort to get the MAD median.
    for val in &mut v {
        *val = (*val - med).abs();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

// ======================
// Basic stats (1-D)
// ======================

#[pyfunction]
fn mean_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(mean_slice(a.as_slice()?))
}

#[pyfunction]
fn mean_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(mean_slice_skipna(a.as_slice()?))
}

#[pyfunction]
fn var_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(var_slice(a.as_slice()?))
}

#[pyfunction]
fn var_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(var_slice_skipna(a.as_slice()?))
}

#[pyfunction]
fn std_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(std_slice(a.as_slice()?))
}

#[pyfunction]
fn std_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(std_slice_skipna(a.as_slice()?))
}

// --- NaN-aware aliases (Python API compatibility) ---
// Python facade expects *_nan_np names in some versions.
#[pyfunction]
fn mean_nan_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    mean_skipna_np(a)
}
#[pyfunction]
fn var_nan_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    var_skipna_np(a)
}
#[pyfunction]
fn std_nan_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    std_skipna_np(a)
}

#[pyfunction]
fn percentile_np(a: PyReadonlyArray1<f64>, q: f64) -> PyResult<f64> {
    Ok(percentile_slice_k(a.as_slice()?, q))
}

#[pyfunction]
fn iqr_np(a: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    Ok(iqr_slice_k(a.as_slice()?))
}

// Scalar IQR width (kept for convenience; avoids conflicting with tuple iqr_np)
#[pyfunction]
fn iqr_width_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let (q1, q3, iqr) = iqr_slice_k(a.as_slice()?);
    if q1.is_nan() || q3.is_nan() || iqr.is_nan() {
        Ok(f64::NAN)
    } else {
        Ok(iqr)
    }
}

#[pyfunction]
fn mad_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(mad_slice_k(a.as_slice()?))
}

#[pyfunction]
fn trimmed_mean_np(
    a: PyReadonlyArray1<f64>,
    proportion_to_cut: f64,
) -> PyResult<f64> {
    Ok(trimmed_mean_slice_k(a.as_slice()?, proportion_to_cut))
}


#[pyfunction]
fn zscore_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let m = mean_slice(xs);
    let s = std_slice(xs); // sample std (ddof=1)
    if !s.is_finite() || s == 0.0 {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]));
    }
    let out: Vec<f64> = xs.iter().map(|&x| (x - m) / s).collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn zscore_skipna_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let m = mean_slice_skipna(xs);
    let s = std_slice_skipna(xs);
    let out: Vec<f64> = xs
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else if s == 0.0 || s.is_nan() {
                f64::NAN
            } else {
                (v - m) / s
            }
        })
        .collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn skew_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = a.as_slice()?;
    if xs.len() < 3 {
        return Ok(f64::NAN);
    }
    let m = mean_slice(xs);
    let s = std_slice(xs);
    if s == 0.0 || s.is_nan() {
        return Ok(f64::NAN);
    }
    let mut m3 = 0.0f64;
    for &v in xs {
        let z = (v - m) / s;
        m3 += z.powi(3);
    }
    Ok(m3 / (xs.len() as f64))
}

#[pyfunction]
fn kurtosis_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = a.as_slice()?;
    if xs.len() < 4 {
        return Ok(f64::NAN);
    }
    let m = mean_slice(xs);
    let s = std_slice(xs);
    if s == 0.0 || s.is_nan() {
        return Ok(f64::NAN);
    }
    let mut m4 = 0.0f64;
    for &v in xs {
        let z = (v - m) / s;
        m4 += z.powi(4);
    }
    Ok(m4 / (xs.len() as f64) - 3.0)
}

// ======================
// Multi-D mean_axis (1D & 2D + skipna)
// ======================

#[pyfunction(signature = (x, axis, skipna=None))]
fn mean_axis_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
    axis: isize,
    skipna: Option<bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let use_skipna = skipna.unwrap_or(false);
    let a: ArrayViewD<'_, f64> = x.as_array();
    let ndim = a.ndim();

    match ndim {
        1 => {
            if axis != 0 {
                return Err(PyValueError::new_err(
                    "mean_axis_np: for 1D input, axis must be 0",
                ));
            }
            let slice = a
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("mean_axis_np: 1D input must be contiguous"))?;
            let m = if use_skipna {
                mean_slice_skipna(slice)
            } else {
                mean_slice(slice)
            };
            Ok(PyArray1::from_vec_bound(py, vec![m]))
        }
        2 => {
            let axis_u = match axis {
                0 => 0usize,
                1 => 1usize,
                _ => {
                    return Err(PyValueError::new_err(
                        "mean_axis_np: for 2D input, axis must be 0 or 1",
                    ))
                }
            };

            let mut out: Vec<f64> = Vec::new();

            if axis_u == 0 {
                let n_cols = a.len_of(Axis(1));
                for j in 0..n_cols {
                    let col = a.index_axis(Axis(1), j);
                    let v: Vec<f64> = col.iter().copied().collect();
                    out.push(if use_skipna { mean_slice_skipna(&v) } else { mean_slice(&v) });
                }
            } else {
                let n_rows = a.len_of(Axis(0));
                for i in 0..n_rows {
                    let row = a.index_axis(Axis(0), i);
                    let v: Vec<f64> = row.iter().copied().collect();
                    out.push(if use_skipna { mean_slice_skipna(&v) } else { mean_slice(&v) });
                }
            }

            Ok(PyArray1::from_vec_bound(py, out))
        }
        _ => Err(PyValueError::new_err(
            "mean_axis_np currently supports only 1D or 2D arrays",
        )),
    }
}

// ======================
// N-D: mean over last axis (any ndim)
// ======================

#[pyfunction]
fn mean_over_last_axis_dyn_np<'py>(
    py: Python<'py>,
    arr: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let view = arr.as_array();
    let ndim = view.ndim();

    if ndim == 0 {
        let v = *view.iter().next().unwrap_or(&f64::NAN);
        return PyArray1::from_vec_bound(py, vec![v]);
    }

    let shape = view.shape();
    let last_dim = shape[ndim - 1];
    let batch_size: usize = shape[..ndim - 1].iter().product();

    let reshaped = view
        .to_owned()
        .into_shape((batch_size, last_dim))
        .expect("reshape failed in mean_over_last_axis_dyn_np");

    let mut out = Vec::with_capacity(batch_size);
    for row in reshaped.axis_iter(Axis(0)) {
        let sum: f64 = row.iter().copied().sum();
        let len = row.len() as f64;
        out.push(if len > 0.0 { sum / len } else { f64::NAN });
    }

    PyArray1::from_vec_bound(py, out)
}

// ======================
// Rolling stats (1-D) — truncated length (n-window+1) fast path
// ======================


#[pyfunction]
fn rolling_mean_std_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let xs = a.as_slice()?;
    let (means, stds) = rolling_mean_std_vec(xs, window);
    Ok((PyArray1::from_vec_bound(py, means), PyArray1::from_vec_bound(py, stds)))
}

#[pyfunction]
fn rolling_mean_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let (means, _stds) = rolling_mean_std_vec(xs, window);
    Ok(PyArray1::from_vec_bound(py, means))
}

#[pyfunction]
fn rolling_std_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let (_means, stds) = rolling_mean_std_vec(xs, window);
    Ok(PyArray1::from_vec_bound(py, stds))
}

#[pyfunction]
fn rolling_var_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;

// single rolling pass (engine)
let (_means, stds) = rolling_mean_std_vec(xs, window);

// final linear pass (std -> var)
let vars = vars_from_stds(&stds);

Ok(PyArray1::from_vec_bound(py, vars))

}

#[pyfunction]
fn rolling_zscore_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;

    // single rolling pass
    let (means, stds) = rolling_mean_std_vec(xs, window);

    // final linear pass
    let out = zscore_from_mean_std(xs, &means, &stds);

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn rolling_mean_nan_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if window == 0 || n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0f64;
    let mut cnt = 0usize;

    for i in 0..n {
        let x_new = xs[i];
        if !x_new.is_nan() {
            sum += x_new;
            cnt += 1;
        }

        if i >= window {
            let x_old = xs[i - window];
            if !x_old.is_nan() {
                sum -= x_old;
                cnt -= 1;
            }
        }

        if cnt > 0 {
            out[i] = sum / (cnt as f64);
        }
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
fn rolling_std_nan_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if window == 0 || n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut cnt = 0usize;

    for i in 0..n {
        let x_new = xs[i];
        if !x_new.is_nan() {
            sum += x_new;
            sumsq += x_new * x_new;
            cnt += 1;
        }

        if i >= window {
            let x_old = xs[i - window];
            if !x_old.is_nan() {
                sum -= x_old;
                sumsq -= x_old * x_old;
                cnt -= 1;
            }
        }

        if cnt >= 2 {
            let c = cnt as f64;
            let var = (sumsq - (sum * sum) / c) / (c - 1.0);
            out[i] = var.max(0.0).sqrt();
        }
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
fn rolling_zscore_nan_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if window == 0 || n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut cnt = 0usize;

    for i in 0..n {
        let x_new = xs[i];
        if !x_new.is_nan() {
            sum += x_new;
            sumsq += x_new * x_new;
            cnt += 1;
        }

        if i >= window {
            let x_old = xs[i - window];
            if !x_old.is_nan() {
                sum -= x_old;
                sumsq -= x_old * x_old;
                cnt -= 1;
            }
        }

        let x = xs[i];
        if x.is_nan() {
            out[i] = f64::NAN;
            continue;
        }

        if cnt >= 2 {
            let c = cnt as f64;
            let mean = sum / c;
            let var = (sumsq - (sum * sum) / c) / (c - 1.0);
            let std = var.max(0.0).sqrt();
            out[i] = if std > 0.0 && std.is_finite() {
                (x - mean) / std
            } else {
                f64::NAN
            };
        } else {
            out[i] = f64::NAN;
        }
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
fn ewma_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let mut out = Vec::with_capacity(n);
    let mut prev = xs[0];
    out.push(prev);
    let one_minus = 1.0 - alpha;
    for i in 1..n {
        let val = alpha * xs[i] + one_minus * prev;
        out.push(val);
        prev = val;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

// ======================
// Rolling axis=0 (2D) — truncated (n-window+1, p) fast path
// ======================


// Cache-friendly axis-0 rolling mean+std (flat buffers; direct indexing; optional Rayon over columns).


#[pyfunction]
fn rolling_mean_std_axis0_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    window: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let a = x.as_array();
    let (n_rows, n_cols) = a.dim();
    if window == 0 || window > n_rows {
        let empty = PyArray2::zeros_bound(py, (0, n_cols), false);
        return Ok((empty.clone(), empty));
    }
    let flat = a.as_slice().ok_or_else(|| PyValueError::new_err("array must be contiguous"))?;
    let out_rows = n_rows - window + 1;

    let (means, stds) = rolling_mean_std_axis0_vec(flat, n_rows, n_cols, window);
    let means2 = Array2::from_shape_vec((out_rows, n_cols), means).unwrap();
    let stds2 = Array2::from_shape_vec((out_rows, n_cols), stds).unwrap();
    Ok((means2.into_pyarray_bound(py), stds2.into_pyarray_bound(py)))
}

#[pyfunction]
fn rolling_mean_axis0_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = x.as_array();
    let (n_rows, n_cols) = a.dim();
    if window == 0 || window > n_rows {
        return Ok(PyArray2::zeros_bound(py, (0, n_cols), false));
    }
    let flat = a.as_slice().ok_or_else(|| PyValueError::new_err("array must be contiguous"))?;
    let out_rows = n_rows - window + 1;

    let means = rolling_mean_axis0_vec(flat, n_rows, n_cols, window);
    let out2 = Array2::from_shape_vec((out_rows, n_cols), means).unwrap();
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
fn rolling_std_axis0_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = x.as_array();
    let (n_rows, n_cols) = a.dim();
    if window == 0 || window > n_rows {
        return Ok(PyArray2::zeros_bound(py, (0, n_cols), false));
    }
    let flat = a.as_slice().ok_or_else(|| PyValueError::new_err("array must be contiguous"))?;
    let out_rows = n_rows - window + 1;

    let stds = rolling_std_axis0_vec(flat, n_rows, n_cols, window);
    let out2 = Array2::from_shape_vec((out_rows, n_cols), stds).unwrap();
    Ok(out2.into_pyarray_bound(py))
}

// ======================
// Outliers & scaling
// ======================

#[pyfunction]
fn iqr_outliers_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    k: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let xs = a.as_slice()?;
    let (q1, q3, iqr) = iqr_slice_k(xs);
    if iqr.is_nan() {
        return Ok(PyArray1::from_vec_bound(py, vec![false; xs.len()]));
    }
    let low = q1 - k * iqr;
    let high = q3 + k * iqr;
    let mask: Vec<bool> = xs.iter().map(|&x| x < low || x > high).collect();
    Ok(PyArray1::from_vec_bound(py, mask))
}

#[pyfunction]
fn zscore_outliers_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let m = mean_slice(xs);
    let s = std_slice(xs);
    if s == 0.0 || s.is_nan() {
        return Ok(PyArray1::from_vec_bound(py, vec![false; xs.len()]));
    }
    let mask: Vec<bool> = xs.iter().map(|&x| ((x - m) / s).abs() > threshold).collect();
    Ok(PyArray1::from_vec_bound(py, mask))
}

#[pyfunction]
fn minmax_scale_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64)> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok((PyArray1::from_vec_bound(py, vec![]), f64::NAN, f64::NAN));
    }
    let mut mn = xs[0];
    let mut mx = xs[0];
    for &x in xs.iter().skip(1) {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    if mx == mn {
        return Ok((PyArray1::from_vec_bound(py, vec![0.0; xs.len()]), mn, mx));
    }
    let scale = mx - mn;
    let scaled: Vec<f64> = xs.iter().map(|&x| (x - mn) / scale).collect();
    Ok((PyArray1::from_vec_bound(py, scaled), mn, mx))
}

#[pyfunction]
fn robust_scale_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    scale_factor: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64)> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok((PyArray1::from_vec_bound(py, vec![]), f64::NAN, f64::NAN));
    }
    let mad = mad_slice(xs);
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let med = if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) };

    let denom = if mad == 0.0 { 1e-12 } else { mad * scale_factor };
    let scaled: Vec<f64> = xs.iter().map(|&x| (x - med) / denom).collect();
    Ok((PyArray1::from_vec_bound(py, scaled), med, mad))
}

// Quantile-based winsorize (kept)
// NOTE: API accepts quantiles in [0,1] (pytest uses 0.05, 0.95)
// We convert to percentile in [0,100] for percentile_slice_k.
// Quantile-based winsorize (API accepts quantiles in [0,1] like 0.05, 0.95;
// also accepts percent in [0,100] like 5, 95)
#[pyfunction]
fn winsorize_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    lower_q: f64,
    upper_q: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    // Delegate to the kernel (single source of truth)
    let out = winsorize_vec_k(xs, lower_q, upper_q);
    Ok(PyArray1::from_vec_bound(py, out))
}
// Clip-based winsorize (explicit bounds)
#[pyfunction]
fn winsorize_clip_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    lower: f64,
    upper: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    // If caller swapped them, fix it deterministically
    let (lo, hi) = if lower <= upper { (lower, upper) } else { (upper, lower) };

    let out: Vec<f64> = xs
        .iter()
        .map(|&v| if v < lo { lo } else if v > hi { hi } else { v })
        .collect();

    Ok(PyArray1::from_vec_bound(py, out))
}



// ======================
// diff / cum / ecdf / bins / sign helpers
// ======================

#[pyfunction]
#[pyo3(signature = (a, periods=1))]
fn diff_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>, periods: isize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 || periods == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![0.0; n]));
    }
    let p = periods.abs() as usize;
    if p >= n {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]));
    }

    let mut out = vec![f64::NAN; n];
    if periods > 0 {
        for i in p..n {
            out[i] = xs[i] - xs[i - p];
        }
    } else {
        for i in 0..(n - p) {
            out[i] = xs[i] - xs[i + p];
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn pct_change_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    periods: isize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 || periods == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]));
    }

    let p = periods.abs() as usize;
    if p >= n {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]));
    }

    let mut out = vec![f64::NAN; n];
    if periods > 0 {
        for i in p..n {
            let base = xs[i - p];
            out[i] = if base == 0.0 { f64::NAN } else { (xs[i] - base) / base };
        }
    } else {
        for i in 0..(n - p) {
            let base = xs[i + p];
            out[i] = if base == 0.0 { f64::NAN } else { (xs[i] - base) / base };
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn cumsum_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let mut out = Vec::with_capacity(xs.len());
    let mut s = 0.0;
    for &x in xs {
        s += x;
        out.push(s);
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn cummean_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let mut out = Vec::with_capacity(xs.len());
    let mut s = 0.0;
    for (i, &x) in xs.iter().enumerate() {
        s += x;
        out.push(s / ((i + 1) as f64));
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn ecdf_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok((PyArray1::from_vec_bound(py, vec![]), PyArray1::from_vec_bound(py, vec![])));
    }
    let mut vals = xs.to_vec();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    let cdf: Vec<f64> = (0..n).map(|i| (i + 1) as f64 / (n as f64)).collect();
    Ok((PyArray1::from_vec_bound(py, vals), PyArray1::from_vec_bound(py, cdf)))
}

#[pyfunction]
fn quantile_bins_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 || n_bins == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut pairs: Vec<(f64, usize)> = xs.iter().copied().zip(0..n).collect();
    pairs.sort_by(|(v1, _), (v2, _)| v1.partial_cmp(v2).unwrap());

    let mut bins = vec![-1_i64; n];
    let mut start = 0usize;
    for b in 0..n_bins {
        let end = if b == n_bins - 1 { n } else { ((b + 1) * n) / n_bins };
        for i in start..end {
            let (_, idx) = pairs[i];
            bins[idx] = b as i64;
        }
        start = end;
    }

    Ok(PyArray1::from_vec_bound(py, bins))
}

#[pyfunction]
fn sign_mask_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let xs = a.as_slice()?;
    let out: Vec<i8> = xs
        .iter()
        .map(|&x| if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 })
        .collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn demean_with_signs_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i8>>)> {
    let xs = a.as_slice()?;
    let m = mean_slice(xs);
    let mut demeaned = Vec::with_capacity(xs.len());
    let mut signs = Vec::with_capacity(xs.len());
    for &x in xs {
        let d = x - m;
        demeaned.push(d);
        signs.push(if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 });
    }
    Ok((PyArray1::from_vec_bound(py, demeaned), PyArray1::from_vec_bound(py, signs)))
}

// ======================
// Covariance / correlation (non-NaN)
// ======================

fn cov_impl(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n <= 1 {
        return f64::NAN;
    }
    let xs = &xs[..n];
    let ys = &ys[..n];
    let mx = mean_slice(xs);
    let my = mean_slice(ys);
    let mut acc = 0.0;
    for i in 0..n {
        acc += (xs[i] - mx) * (ys[i] - my);
    }
    acc / ((n - 1) as f64)
}

#[pyfunction]
fn cov_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(cov_impl(x.as_slice()?, y.as_slice()?))
}

#[pyfunction]
fn corr_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let c = cov_impl(xs, ys);
    let sx = std_slice(xs);
    let sy = std_slice(ys);
    if sx == 0.0 || sy == 0.0 || sx.is_nan() || sy.is_nan() {
        Ok(f64::NAN)
    } else {
        Ok(c / (sx * sy))
    }
}

#[pyfunction]
fn cov_matrix_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let arr = a.as_array();
    let n_rows = arr.shape()[0];
    let n_cols = arr.shape()[1];
    let x = arr.as_slice().expect("input must be C-contiguous (row-major)");

    if n_rows < 2 || n_cols == 0 {
        let out2 = Array2::<f64>::zeros((n_cols, n_cols));
        return out2.into_pyarray_bound(py);
    }

    // column means
    let mut means = vec![0.0f64; n_cols];
    for r in 0..n_rows {
        let base = r * n_cols;
        for j in 0..n_cols {
            means[j] += x[base + j];
        }
    }
    for j in 0..n_cols {
        means[j] /= n_rows as f64;
    }

    let denom = (n_rows as f64 - 1.0).max(1.0);
    let out = cov_matrix_out_k(x, n_rows, n_cols, &means, denom);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out).unwrap();
    out2.into_pyarray_bound(py)
}


#[pyfunction]
fn corr_matrix_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let arr = a.as_array();
    let n_rows = arr.shape()[0];
    let n_cols = arr.shape()[1];
    let x = arr.as_slice().expect("input must be C-contiguous (row-major)");

    if n_rows < 2 || n_cols == 0 {
        let out2 = Array2::<f64>::zeros((n_cols, n_cols));
        return out2.into_pyarray_bound(py);
    }

    // column means
    let mut means = vec![0.0f64; n_cols];
    for r in 0..n_rows {
        let base = r * n_cols;
        for j in 0..n_cols {
            means[j] += x[base + j];
        }
    }
    for j in 0..n_cols {
        means[j] /= n_rows as f64;
    }

    // column stds (ddof=1)
    let denom = (n_rows as f64 - 1.0).max(1.0);
    let mut stds = vec![0.0f64; n_cols];
    for j in 0..n_cols {
        let mj = means[j];
        let mut acc = 0.0f64;
        for r in 0..n_rows {
            let base = r * n_cols;
            let d = x[base + j] - mj;
            acc += d * d;
        }
        stds[j] = (acc / denom).sqrt();
    }
    let out = corr_matrix_out_k(x, n_rows, n_cols, &means, &stds, denom);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out).unwrap();
    out2.into_pyarray_bound(py)
}


#[pyfunction]
fn rolling_cov_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > xs.len() {
        return Err(PyValueError::new_err("window must be <= len(x)"));
    }

    let out = rolling_cov_vec(xs, ys, window);
    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
fn rolling_corr_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    let n = xs.len();
    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    if window > n {
        return Err(PyValueError::new_err("window must be <= len(x)"));
    }
    let mut out = Vec::with_capacity(n - window + 1);

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..window {
        let xi = xs[i];
        let yi = ys[i];
        sum_x += xi;
        sum_y += yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
        sum_xy += xi * yi;
    }

    for i in (window - 1)..n {
        if i > window - 1 {
            let xi_new = xs[i];
            let yi_new = ys[i];
            let xi_old = xs[i - window];
            let yi_old = ys[i - window];

            sum_x += xi_new - xi_old;
            sum_y += yi_new - yi_old;
            sum_x2 += xi_new * xi_new - xi_old * xi_old;
            sum_y2 += yi_new * yi_new - yi_old * yi_old;
            sum_xy += xi_new * yi_new - xi_old * yi_old;
        }

        let w = window as f64;
        let mx = sum_x / w;
        let my = sum_y / w;
        let var_x = (sum_x2 - w * mx * mx) / ((window - 1) as f64);
        let var_y = (sum_y2 - w * my * my) / ((window - 1) as f64);
        let cov = (sum_xy - w * mx * my) / ((window - 1) as f64);

        let denom = (var_x.max(0.0).sqrt()) * (var_y.max(0.0).sqrt());
        out.push(if denom == 0.0 || denom.is_nan() { f64::NAN } else { cov / denom });
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


// ======================
// Welford (single-pass, NaN-skipping)
// ======================

#[pyfunction]
pub fn welford_np(a: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, usize)> {
    let xs = a.as_slice()?;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut n = 0usize;

    for &x in xs {
        if x.is_nan() {
            continue;
        }
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    if n < 2 {
        return Ok((mean, f64::NAN, n));
    }

    let var = m2 / (n as f64 - 1.0);
    Ok((mean, var, n))
}

// ======================
// NaN-aware covariance/correlation (pairwise deletion, ddof=1)
// ======================

#[pyfunction]
pub fn cov_nan_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }

    let mut mx = 0.0;
    let mut my = 0.0;
    let mut n = 0usize;

    for i in 0..xs.len() {
        let xi = xs[i];
        let yi = ys[i];
        if xi.is_nan() || yi.is_nan() {
            continue;
        }
        n += 1;
        mx += xi;
        my += yi;
    }

    if n < 2 {
        return Ok(f64::NAN);
    }

    mx /= n as f64;
    my /= n as f64;

    let mut cov = 0.0;
    for i in 0..xs.len() {
        let xi = xs[i];
        let yi = ys[i];
        if xi.is_nan() || yi.is_nan() {
            continue;
        }
        cov += (xi - mx) * (yi - my);
    }

    Ok(cov / (n as f64 - 1.0))
}

#[pyfunction]
pub fn corr_nan_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }

    // Means over valid pairs
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut n = 0usize;
    for i in 0..xs.len() {
        let xi = xs[i];
        let yi = ys[i];
        if xi.is_nan() || yi.is_nan() {
            continue;
        }
        n += 1;
        mx += xi;
        my += yi;
    }
    if n < 2 {
        return Ok(f64::NAN);
    }
    mx /= n as f64;
    my /= n as f64;

    // Cov/vars
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..xs.len() {
        let xi = xs[i];
        let yi = ys[i];
        if xi.is_nan() || yi.is_nan() {
            continue;
        }
        let dx = xi - mx;
        let dy = yi - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    let denom = (n as f64) - 1.0;
    let varx = sxx / denom;
    let vary = syy / denom;
    let cov = sxy / denom;

    if varx <= 0.0 || vary <= 0.0 || !varx.is_finite() || !vary.is_finite() || !cov.is_finite() {
        return Ok(f64::NAN);
    }
    Ok(cov / (varx * vary).sqrt())
}

#[pyfunction]
pub fn rolling_cov_nan_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let n = xs.len();

    if ys.len() != n {
        return Err(PyValueError::new_err("length mismatch"));
    }
    if window == 0 || window > n {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut out = Vec::with_capacity(n - window + 1);

    for i in 0..=(n - window) {
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut k = 0usize;

        for j in i..(i + window) {
            let xi = xs[j];
            let yi = ys[j];
            if xi.is_nan() || yi.is_nan() {
                continue;
            }
            k += 1;
            mx += xi;
            my += yi;
        }

        if k < window {
            out.push(f64::NAN);
            continue;
        }

        mx /= k as f64;
        my /= k as f64;

        let mut cov = 0.0;
        for j in i..(i + window) {
            let xi = xs[j];
            let yi = ys[j];
            if xi.is_nan() || yi.is_nan() {
                continue;
            }
            cov += (xi - mx) * (yi - my);
        }

        out.push(cov / (k as f64 - 1.0));
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
pub fn rolling_corr_nan_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let n = xs.len();

    if ys.len() != n {
        return Err(PyValueError::new_err("length mismatch"));
    }
    if window == 0 || window > n {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut out = Vec::with_capacity(n - window + 1);

    for i in 0..=(n - window) {
        // Means over valid pairs in window
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut k = 0usize;
        for j in i..(i + window) {
            let xi = xs[j];
            let yi = ys[j];
            if xi.is_nan() || yi.is_nan() {
                continue;
            }
            k += 1;
            mx += xi;
            my += yi;
        }
        if k < window {
            out.push(f64::NAN);
            continue;
        }
        mx /= k as f64;
        my /= k as f64;

        // Cov/vars
        let mut sxx = 0.0;
        let mut syy = 0.0;
        let mut sxy = 0.0;
        for j in i..(i + window) {
            let xi = xs[j];
            let yi = ys[j];
            if xi.is_nan() || yi.is_nan() {
                continue;
            }
            let dx = xi - mx;
            let dy = yi - my;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }

        let denom = (k as f64) - 1.0;
        let varx = sxx / denom;
        let vary = syy / denom;
        let cov = sxy / denom;

        if varx <= 0.0 || vary <= 0.0 || !varx.is_finite() || !vary.is_finite() || !cov.is_finite() {
            out.push(f64::NAN);
        } else {
            out.push(cov / (varx * vary).sqrt());
        }
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


// ======================
// KDE
// ======================

#[pyfunction(signature = (a, n_points, bandwidth=None))]
fn kde_gaussian_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    n_points: usize,
    bandwidth: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 || n_points == 0 {
        return Ok((PyArray1::from_vec_bound(py, vec![]), PyArray1::from_vec_bound(py, vec![])));
    }

    let mut values = xs.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = values.iter().copied().sum::<f64>() / (values.len() as f64);
    let mut acc = 0.0;
    for &v in &values {
        let d = v - mean;
        acc += d * d;
    }
    let std = (acc / ((values.len().saturating_sub(1)) as f64)).sqrt();

    let bw = match bandwidth {
        Some(b) if b > 0.0 => b,
        _ => {
            if std == 0.0 {
                1e-6
            } else {
                1.06 * std * (n as f64).powf(-1.0 / 5.0)
            }
        }
    };

    let mn = *values.first().unwrap();
    let mx = *values.last().unwrap();

    if mx == mn {
        return Ok((
            PyArray1::from_vec_bound(py, vec![mn; n_points]),
            PyArray1::from_vec_bound(py, vec![0.0; n_points]),
        ));
    }

    let step = (mx - mn) / ((n_points - 1) as f64);
    let grid: Vec<f64> = (0..n_points).map(|i| mn + step * (i as f64)).collect();

    let norm_factor = 1.0 / (bw * (2.0 * std::f64::consts::PI).sqrt());
    let mut dens = Vec::with_capacity(n_points);

    for &x0 in &grid {
        let mut sum = 0.0;
        for &xv in xs {
            let z = (x0 - xv) / bw;
            sum += (-0.5 * z * z).exp();
        }
        dens.push(norm_factor * sum / (n as f64));
    }

    Ok((PyArray1::from_vec_bound(py, grid), PyArray1::from_vec_bound(py, dens)))
}

// ======================
// Padding util
// ======================

#[pyfunction]
fn pad_nan_np<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]))
}

// ======================
// Effect-size naming compatibility
// ======================

#[pyfunction(signature = (x, y, pooled=None))]
fn hedges_g_2samp_np(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    pooled: Option<bool>,
) -> PyResult<f64> {
    // Public API: defaults to pooled variance
    let pooled = pooled.unwrap_or(true);
    infer::effect::hedges_g_2samp_np2(x, y, pooled)
}

#[pyfunction]
fn hedges_g_2samp_raw_np(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    pooled: bool,
) -> PyResult<f64> {
    // Raw API: explicit pooled flag, no defaults
    infer::effect::hedges_g_2samp_np2(x, y, pooled)
}
// ======================
// Module definition
// ======================

#[pymodule]
fn bunker_stats_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // basic stats
    m.add_function(wrap_pyfunction!(mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_nan_np, m)?)?;

    m.add_function(wrap_pyfunction!(zscore_np, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(skew_np, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_np, m)?)?;

    m.add_function(wrap_pyfunction!(percentile_np, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_np, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_width_np, m)?)?;
    m.add_function(wrap_pyfunction!(mad_np, m)?)?;
    m.add_function(wrap_pyfunction!(trimmed_mean_np, m)?)?;

    // multi-D
    m.add_function(wrap_pyfunction!(mean_axis_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_over_last_axis_dyn_np, m)?)?;

    // rolling (truncated)
    m.add_function(wrap_pyfunction!(rolling_mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_var_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_std_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_zscore_np, m)?)?;

    // rolling axis0 (truncated)
    m.add_function(wrap_pyfunction!(rolling_mean_std_axis0_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_axis0_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_axis0_np, m)?)?;

    // rolling (NaN-aware, full-length)
    m.add_function(wrap_pyfunction!(rolling_mean_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_zscore_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(ewma_np, m)?)?;

    // outliers / scaling
    m.add_function(wrap_pyfunction!(iqr_outliers_np, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_outliers_np, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(robust_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(winsorize_np, m)?)?;
    m.add_function(wrap_pyfunction!(winsorize_clip_np, m)?)?;

    // diff / cum / ecdf / bins / signs
    m.add_function(wrap_pyfunction!(diff_np, m)?)?;
    m.add_function(wrap_pyfunction!(pct_change_np, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum_np, m)?)?;
    m.add_function(wrap_pyfunction!(cummean_np, m)?)?;
    m.add_function(wrap_pyfunction!(ecdf_np, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_bins_np, m)?)?;
    m.add_function(wrap_pyfunction!(sign_mask_np, m)?)?;
    m.add_function(wrap_pyfunction!(demean_with_signs_np, m)?)?;

    // covariance / correlation
    m.add_function(wrap_pyfunction!(cov_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_np, m)?)?;
    m.add_function(wrap_pyfunction!(cov_matrix_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_matrix_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_cov_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_corr_np, m)?)?;

    // Welford + NaN-aware covariance/correlation
    m.add_function(wrap_pyfunction!(welford_np, m)?)?;
    m.add_function(wrap_pyfunction!(cov_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_cov_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_corr_nan_np, m)?)?;

    // KDE
    m.add_function(wrap_pyfunction!(kde_gaussian_np, m)?)?;

    // Inference core (keep your existing wiring style)
    m.add_function(wrap_pyfunction!(infer::ttest::t_test_1samp_np, m)?)?;
    m.add_function(wrap_pyfunction!(infer::ttest::t_test_2samp_np, m)?)?;
    m.add_function(wrap_pyfunction!(infer::chi2::chi2_gof_np, m)?)?;
    m.add_function(wrap_pyfunction!(infer::chi2::chi2_independence_np, m)?)?;
    m.add_function(wrap_pyfunction!(infer::effect::mean_diff_ci_np, m)?)?;
    m.add_function(wrap_pyfunction!(infer::effect::cohens_d_2samp_np, m)?)?;
    m.add_function(wrap_pyfunction!(hedges_g_2samp_np, m)?)?;
    m.add_function(wrap_pyfunction!(hedges_g_2samp_raw_np, m)?)?;
    m.add_function(wrap_pyfunction!(infer::mann_whitney::mann_whitney_u_np, m)?)?;
    m.add_function(wrap_pyfunction!(infer::ks::ks_1samp_np, m)?)?;

    // padding
    m.add_function(wrap_pyfunction!(pad_nan_np, m)?)?;

    Ok(())
}