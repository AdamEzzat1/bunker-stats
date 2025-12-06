use numpy::{
    ndarray::{ArrayView2, ArrayViewD, Axis},
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ======================
// Core slice helpers
// ======================

fn mean_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    for &x in xs {
        sum += x;
    }
    sum / (xs.len() as f64)
}

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

// NaN-aware helpers

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

    let m = mean_slice_skipna(&values);
    let mut acc = 0.0;
    for v in &values {
        let d = *v - m;
        acc += d * d;
    }
    acc / ((n - 1) as f64)
}

fn std_slice_skipna(xs: &[f64]) -> f64 {
    var_slice_skipna(xs).sqrt()
}

fn percentile_slice(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    let pos = q * ((n - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let w = pos - (lo as f64);
        (1.0 - w) * v[lo] + w * v[hi]
    }
}

fn iqr_from_sorted(sorted: &[f64]) -> (f64, f64, f64) {
    if sorted.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = sorted.len();
    let q1_pos = 0.25 * ((n - 1) as f64);
    let q3_pos = 0.75 * ((n - 1) as f64);

    let interp = |pos: f64| -> f64 {
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let w = pos - (lo as f64);
            (1.0 - w) * sorted[lo] + w * sorted[hi]
        }
    };

    let q1 = interp(q1_pos);
    let q3 = interp(q3_pos);
    (q1, q3, q3 - q1)
}

fn iqr_slice(xs: &[f64]) -> (f64, f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    iqr_from_sorted(&v)
}

fn mad_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    };
    let mut devs: Vec<f64> = xs.iter().map(|x| (x - med).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = devs.len();
    if m == 0 {
        f64::NAN
    } else if m % 2 == 1 {
        devs[m / 2]
    } else {
        0.5 * (devs[m / 2 - 1] + devs[m / 2])
    }
}

// ======================
// Basic stats (1-D)
// ======================

#[pyfunction]
fn mean_np(a: PyReadonlyArray1<f64>) -> f64 {
    mean_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn mean_skipna_np(a: PyReadonlyArray1<f64>) -> f64 {
    mean_slice_skipna(a.as_slice().unwrap())
}

#[pyfunction]
fn var_np(a: PyReadonlyArray1<f64>) -> f64 {
    var_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn var_skipna_np(a: PyReadonlyArray1<f64>) -> f64 {
    var_slice_skipna(a.as_slice().unwrap())
}

#[pyfunction]
fn std_np(a: PyReadonlyArray1<f64>) -> f64 {
    std_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn std_skipna_np(a: PyReadonlyArray1<f64>) -> f64 {
    std_slice_skipna(a.as_slice().unwrap())
}

#[pyfunction]
fn percentile_np(a: PyReadonlyArray1<f64>, q: f64) -> f64 {
    percentile_slice(a.as_slice().unwrap(), q)
}

#[pyfunction]
fn iqr_np(a: PyReadonlyArray1<f64>) -> (f64, f64, f64) {
    iqr_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn mad_np(a: PyReadonlyArray1<f64>) -> f64 {
    mad_slice(a.as_slice().unwrap())
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
        // 1D: only axis=0 is supported → return length-1 array
        1 => {
            if axis != 0 {
                return Err(PyValueError::new_err(
                    "mean_axis_np: for 1D input, axis must be 0",
                ));
            }
            let slice = a
                .as_slice()
                .ok_or_else(|| {
                    PyValueError::new_err("mean_axis_np: 1D input must be contiguous")
                })?;
            let m = if use_skipna {
                mean_slice_skipna(slice)
            } else {
                mean_slice(slice)
            };
            Ok(PyArray1::from_vec_bound(py, vec![m]))
        }

        // 2D: axis=0 → per-column mean; axis=1 → per-row mean
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
                // axis=0 → mean over rows → one mean per column
                let n_cols = a.len_of(Axis(1));
                for j in 0..n_cols {
                    let col = a.index_axis(Axis(1), j);
                    let v: Vec<f64> = col.iter().copied().collect();
                    let m = if use_skipna {
                        mean_slice_skipna(&v)
                    } else {
                        mean_slice(&v)
                    };
                    out.push(m);
                }
            } else {
                // axis=1 → mean over columns → one mean per row
                let n_rows = a.len_of(Axis(0));
                for i in 0..n_rows {
                    let row = a.index_axis(Axis(0), i);
                    let v: Vec<f64> = row.iter().copied().collect();
                    let m = if use_skipna {
                        mean_slice_skipna(&v)
                    } else {
                        mean_slice(&v)
                    };
                    out.push(m);
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

    // Scalar case → return as length-1 array
    if ndim == 0 {
        let v = *view.iter().next().unwrap_or(&f64::NAN);
        return PyArray1::from_vec_bound(py, vec![v]);
    }

    let shape = view.shape();
    let last_dim = shape[ndim - 1];

    // Flatten all but last axis into a batch dimension
    let batch_size: usize = shape[..ndim - 1].iter().product();

    // Clone to owned array and reshape to (batch_size, last_dim)
    let reshaped = view
        .to_owned()
        .into_shape((batch_size, last_dim))
        .expect("reshape failed in mean_over_last_axis_dyn_np");

    let mut out = Vec::with_capacity(batch_size);

    for row in reshaped.axis_iter(Axis(0)) {
        let sum: f64 = row.iter().copied().sum();
        let len = row.len() as f64;
        let mean = if len > 0.0 { sum / len } else { f64::NAN };
        out.push(mean);
    }

    PyArray1::from_vec_bound(py, out)
}

// ======================
// Rolling stats (1-D)
// ======================

#[pyfunction]
fn rolling_mean_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let n = xs.len();
    if window == 0 || window > n {
        return PyArray1::from_vec_bound(py, Vec::new());
    }
    let mut out = Vec::with_capacity(n - window + 1);
    let mut sum = 0.0;
    for i in 0..window {
        sum += xs[i];
    }
    out.push(sum / (window as f64));
    for i in window..n {
        sum += xs[i] - xs[i - window];
        out.push(sum / (window as f64));
    }
    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn rolling_std_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let n = xs.len();
    if window == 0 || window > n {
        return PyArray1::from_vec_bound(py, Vec::new());
    }

    let mut out = Vec::with_capacity(n - window + 1);
    let mut sum = 0.0;
    let mut sumsq = 0.0;

    for i in 0..window {
        let x = xs[i];
        sum += x;
        sumsq += x * x;
    }

    let denom = (window - 1) as f64;
    let var0 = if window > 1 {
        (sumsq - (sum * sum) / (window as f64)) / denom
    } else {
        0.0
    };
    out.push(var0.max(0.0).sqrt());

    for i in window..n {
        let x_new = xs[i];
        let x_old = xs[i - window];
        sum += x_new - x_old;
        sumsq += x_new * x_new - x_old * x_old;

        let var = if window > 1 {
            (sumsq - (sum * sum) / (window as f64)) / denom
        } else {
            0.0
        };
        out.push(var.max(0.0).sqrt());
    }

    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn rolling_zscore_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let n = xs.len();
    if window == 0 || window > n {
        return PyArray1::from_vec_bound(py, Vec::new());
    }

    let mut out = Vec::with_capacity(n - window + 1);
    let mut sum = 0.0;
    let mut sumsq = 0.0;

    for i in 0..window {
        let x = xs[i];
        sum += x;
        sumsq += x * x;
    }

    let denom = (window - 1) as f64;
    for i in (window - 1)..n {
        if i > window - 1 {
            let x_new = xs[i];
            let x_old = xs[i - window];
            sum += x_new - x_old;
            sumsq += x_new * x_new - x_old * x_old;
        }

        let var = if window > 1 {
            (sumsq - (sum * sum) / (window as f64)) / denom
        } else {
            0.0
        };
        let std = var.max(0.0).sqrt();
        let last = xs[i];
        let z = if std > 0.0 {
            (last - sum / (window as f64)) / std
        } else {
            0.0
        };
        out.push(z);
    }

    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn ewma_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let n = xs.len();
    if n == 0 {
        return PyArray1::from_vec_bound(py, Vec::new());
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
    PyArray1::from_vec_bound(py, out)
}

// ======================
// Outliers & scaling
// ======================

#[pyfunction]
fn iqr_outliers_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    k: f64,
) -> Bound<'py, PyArray1<bool>> {
    let xs = a.as_slice().unwrap();
    let (q1, q3, iqr) = iqr_slice(xs);
    if iqr.is_nan() {
        return PyArray1::from_vec_bound(py, vec![false; xs.len()]);
    }
    let low = q1 - k * iqr;
    let high = q3 + k * iqr;
    let mask: Vec<bool> = xs.iter().map(|&x| x < low || x > high).collect();
    PyArray1::from_vec_bound(py, mask)
}

#[pyfunction]
fn zscore_outliers_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    threshold: f64,
) -> Bound<'py, PyArray1<bool>> {
    let xs = a.as_slice().unwrap();
    if xs.is_empty() {
        return PyArray1::from_vec_bound(py, Vec::new());
    }
    let m = mean_slice(xs);
    let s = std_slice(xs);
    if s == 0.0 || s.is_nan() {
        return PyArray1::from_vec_bound(py, vec![false; xs.len()]);
    }
    let mask: Vec<bool> = xs
        .iter()
        .map(|&x| ((x - m) / s).abs() > threshold)
        .collect();
    PyArray1::from_vec_bound(py, mask)
}

#[pyfunction]
fn minmax_scale_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, f64, f64) {
    let xs = a.as_slice().unwrap();
    if xs.is_empty() {
        return (
            PyArray1::from_vec_bound(py, Vec::new()),
            f64::NAN,
            f64::NAN,
        );
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
        let scaled = vec![0.0; xs.len()];
        return (PyArray1::from_vec_bound(py, scaled), mn, mx);
    }
    let scale = mx - mn;
    let scaled: Vec<f64> = xs.iter().map(|&x| (x - mn) / scale).collect();
    (PyArray1::from_vec_bound(py, scaled), mn, mx)
}

#[pyfunction]
fn robust_scale_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    scale_factor: f64,
) -> (Bound<'py, PyArray1<f64>>, f64, f64) {
    let xs = a.as_slice().unwrap();
    if xs.is_empty() {
        return (
            PyArray1::from_vec_bound(py, Vec::new()),
            f64::NAN,
            f64::NAN,
        );
    }
    let mad = mad_slice(xs);
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    };
    let denom = if mad == 0.0 {
        1e-12
    } else {
        mad * scale_factor
    };
    let scaled: Vec<f64> = xs.iter().map(|&x| (x - med) / denom).collect();
    (PyArray1::from_vec_bound(py, scaled), med, mad)
}

#[pyfunction]
fn winsorize_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    lower_q: f64,
    upper_q: f64,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    if xs.is_empty() {
        return PyArray1::from_vec_bound(py, Vec::new());
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let low = percentile_slice(&v, lower_q);
    let high = percentile_slice(&v, upper_q);

    let out: Vec<f64> = xs
        .iter()
        .map(|&x| {
            if x < low {
                low
            } else if x > high {
                high
            } else {
                x
            }
        })
        .collect();

    PyArray1::from_vec_bound(py, out)
}

// ======================
// diff / cum / ecdf / bins / sign helpers
// ======================

#[pyfunction]
fn diff_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    periods: isize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let n = xs.len();
    if n == 0 || periods == 0 {
        return PyArray1::from_vec_bound(py, vec![0.0; n]);
    }

    let p = periods.abs() as usize;
    if p >= n {
        return PyArray1::from_vec_bound(py, vec![f64::NAN; n]);
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
    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn pct_change_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    periods: isize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let n = xs.len();
    if n == 0 || periods == 0 {
        return PyArray1::from_vec_bound(py, vec![f64::NAN; n]);
    }

    let p = periods.abs() as usize;
    if p >= n {
        return PyArray1::from_vec_bound(py, vec![f64::NAN; n]);
    }

    let mut out = vec![f64::NAN; n];
    if periods > 0 {
        for i in p..n {
            let base = xs[i - p];
            if base == 0.0 {
                out[i] = f64::NAN;
            } else {
                out[i] = (xs[i] - base) / base;
            }
        }
    } else {
        for i in 0..(n - p) {
            let base = xs[i + p];
            if base == 0.0 {
                out[i] = f64::NAN;
            } else {
                out[i] = (xs[i] - base) / base;
            }
        }
    }
    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn cumsum_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let mut out = Vec::with_capacity(xs.len());
    let mut s = 0.0;
    for &x in xs {
        s += x;
        out.push(s);
    }
    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn cummean_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let mut out = Vec::with_capacity(xs.len());
    let mut s = 0.0;
    for (i, &x) in xs.iter().enumerate() {
        s += x;
        out.push(s / ((i + 1) as f64));
    }
    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn ecdf_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let xs = a.as_slice().unwrap();
    if xs.is_empty() {
        return (
            PyArray1::from_vec_bound(py, Vec::new()),
            PyArray1::from_vec_bound(py, Vec::new()),
        );
    }
    let mut vals = xs.to_vec();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    let mut cdf = Vec::with_capacity(n);
    for i in 0..n {
        cdf.push((i + 1) as f64 / (n as f64));
    }
    (
        PyArray1::from_vec_bound(py, vals),
        PyArray1::from_vec_bound(py, cdf),
    )
}

#[pyfunction]
fn quantile_bins_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> Bound<'py, PyArray1<i64>> {
    let xs = a.as_slice().unwrap();
    let n = xs.len();
    if n == 0 || n_bins == 0 {
        return PyArray1::from_vec_bound(py, Vec::new());
    }

    let mut pairs: Vec<(f64, usize)> = xs.iter().cloned().zip(0..n).collect();
    pairs.sort_by(|(v1, _), (v2, _)| v1.partial_cmp(v2).unwrap());

    let mut bins = vec![-1_i64; n];
    let mut start = 0usize;
    for b in 0..n_bins {
        let end = if b == n_bins - 1 {
            n
        } else {
            ((b + 1) * n) / n_bins
        };
        for i in start..end {
            let (_, idx) = pairs[i];
            bins[idx] = b as i64;
        }
        start = end;
    }

    PyArray1::from_vec_bound(py, bins)
}

#[pyfunction]
fn sign_mask_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<i8>> {
    let xs = a.as_slice().unwrap();
    let out: Vec<i8> = xs
        .iter()
        .map(|&x| {
            if x > 0.0 {
                1
            } else if x < 0.0 {
                -1
            } else {
                0
            }
        })
        .collect();
    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn demean_with_signs_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i8>>) {
    let xs = a.as_slice().unwrap();
    let m = mean_slice(xs);
    let mut demeaned = Vec::with_capacity(xs.len());
    let mut signs = Vec::with_capacity(xs.len());
    for &x in xs {
        let d = x - m;
        demeaned.push(d);
        let s = if d > 0.0 {
            1
        } else if d < 0.0 {
            -1
        } else {
            0
        };
        signs.push(s);
    }
    (
        PyArray1::from_vec_bound(py, demeaned),
        PyArray1::from_vec_bound(py, signs),
    )
}

// ======================
// Covariance / correlation
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
fn cov_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> f64 {
    let xs = x.as_slice().unwrap();
    let ys = y.as_slice().unwrap();
    cov_impl(xs, ys)
}

#[pyfunction]
fn corr_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> f64 {
    let xs = x.as_slice().unwrap();
    let ys = y.as_slice().unwrap();
    let c = cov_impl(xs, ys);
    let sx = std_slice(xs);
    let sy = std_slice(ys);
    if sx == 0.0 || sy == 0.0 || sx.is_nan() || sy.is_nan() {
        f64::NAN
    } else {
        c / (sx * sy)
    }
}

#[pyfunction]
fn cov_matrix_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let arr: ArrayView2<'_, f64> = a.as_array();
    let n_rows = arr.nrows();
    let n_cols = arr.ncols();
    let mut out = vec![0.0f64; n_cols * n_cols];

    for i in 0..n_cols {
        let col_i = arr.column(i);
        let mean_i = col_i.iter().copied().sum::<f64>() / (n_rows as f64);

        for j in i..n_cols {
            let col_j = arr.column(j);
            let mean_j = col_j.iter().copied().sum::<f64>() / (n_rows as f64);

            let mut acc = 0.0;
            for k in 0..n_rows {
                let di = col_i[k] - mean_i;
                let dj = col_j[k] - mean_j;
                acc += di * dj;
            }

            let cov = if n_rows > 1 {
                acc / ((n_rows - 1) as f64)
            } else {
                f64::NAN
            };

            out[i * n_cols + j] = cov;
            out[j * n_cols + i] = cov;
        }
    }

    PyArray2::from_vec2_bound(
        py,
        &(0..n_cols)
            .map(|i| out[i * n_cols..(i + 1) * n_cols].to_vec())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[pyfunction]
fn corr_matrix_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let arr: ArrayView2<'_, f64> = a.as_array();
    let n_rows = arr.nrows();
    let n_cols = arr.ncols();
    let mut out = vec![0.0f64; n_cols * n_cols];

    let mut means = Vec::with_capacity(n_cols);
    let mut stds = Vec::with_capacity(n_cols);
    for j in 0..n_cols {
        let col = arr.column(j);
        let v: Vec<f64> = col.iter().copied().collect();
        means.push(mean_slice(&v));
        stds.push(std_slice(&v));
    }

    for i in 0..n_cols {
        for j in i..n_cols {
            let mut acc = 0.0;
            for k in 0..n_rows {
                let xi = arr[[k, i]];
                let xj = arr[[k, j]];
                acc += (xi - means[i]) * (xj - means[j]);
            }
            let cov = if n_rows > 1 {
                acc / ((n_rows - 1) as f64)
            } else {
                f64::NAN
            };
            let denom = stds[i] * stds[j];
            let c = if denom == 0.0 || denom.is_nan() {
                f64::NAN
            } else {
                cov / denom
            };
            out[i * n_cols + j] = c;
            out[j * n_cols + i] = c;
        }
    }

    PyArray2::from_vec2_bound(
        py,
        &(0..n_cols)
            .map(|i| out[i * n_cols..(i + 1) * n_cols].to_vec())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[pyfunction]
fn rolling_cov_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = x.as_slice().unwrap();
    let ys = y.as_slice().unwrap();
    let n = xs.len().min(ys.len());
    if window == 0 || window > n {
        return PyArray1::from_vec_bound(py, Vec::new());
    }

    let xs = &xs[..n];
    let ys = &ys[..n];
    let mut out = Vec::with_capacity(n - window + 1);

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..window {
        let xi = xs[i];
        let yi = ys[i];
        sum_x += xi;
        sum_y += yi;
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
            sum_xy += xi_new * yi_new - xi_old * yi_old;
        }

        let w = window as f64;
        let mx = sum_x / w;
        let my = sum_y / w;
        let cov = (sum_xy - w * mx * my) / ((window - 1) as f64);
        out.push(cov);
    }

    PyArray1::from_vec_bound(py, out)
}

#[pyfunction]
fn rolling_corr_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = x.as_slice().unwrap();
    let ys = y.as_slice().unwrap();
    let n = xs.len().min(ys.len());
    if window == 0 || window > n {
        return PyArray1::from_vec_bound(py, Vec::new());
    }

    let xs = &xs[..n];
    let ys = &ys[..n];
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
        let c = if denom == 0.0 || denom.is_nan() {
            f64::NAN
        } else {
            cov / denom
        };
        out.push(c);
    }

    PyArray1::from_vec_bound(py, out)
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
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let xs = a.as_slice().unwrap();
    let n = xs.len();
    if n == 0 || n_points == 0 {
        return (
            PyArray1::from_vec_bound(py, Vec::new()),
            PyArray1::from_vec_bound(py, Vec::new()),
        );
    }

    let mut values = xs.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut s = 0.0;
    for &v in &values {
        s += v;
    }
    let mean = s / (values.len() as f64);
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
        let grid = vec![mn; n_points];
        let dens = vec![0.0; n_points];
        return (
            PyArray1::from_vec_bound(py, grid),
            PyArray1::from_vec_bound(py, dens),
        );
    }

    let step = (mx - mn) / ((n_points - 1) as f64);
    let mut grid = Vec::with_capacity(n_points);
    for i in 0..n_points {
        grid.push(mn + step * (i as f64));
    }

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

    (
        PyArray1::from_vec_bound(py, grid),
        PyArray1::from_vec_bound(py, dens),
    )
}

// ======================
// Module definition
// ======================

#[pymodule]
fn bunker_stats_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // basic stats
    m.add_function(wrap_pyfunction!(mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_np, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_np, m)?)?;
    m.add_function(wrap_pyfunction!(mad_np, m)?)?;

    // multi-D
    m.add_function(wrap_pyfunction!(mean_axis_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_over_last_axis_dyn_np, m)?)?;

    // rolling
    m.add_function(wrap_pyfunction!(rolling_mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_zscore_np, m)?)?;
    m.add_function(wrap_pyfunction!(ewma_np, m)?)?;

    // outliers / scaling
    m.add_function(wrap_pyfunction!(iqr_outliers_np, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_outliers_np, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(robust_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(winsorize_np, m)?)?;

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

    // KDE
    m.add_function(wrap_pyfunction!(kde_gaussian_np, m)?)?;

    Ok(())
}
