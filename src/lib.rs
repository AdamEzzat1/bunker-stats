// backing up bunker-stats lib.rs file here

use numpy::{
    ndarray::ArrayView2, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::f64::consts::PI;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------- basic helpers ----------

#[inline(always)]
fn mean_slice(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n == 0.0 {
        return f64::NAN;
    }
    xs.iter().copied().sum::<f64>() / n
}

#[inline(always)]
fn var_slice(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n <= 1.0 {
        return f64::NAN;
    }
    let m = mean_slice(xs);
    let var = xs
        .iter()
        .map(|x| {
            let d = x - m;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    var
}

#[inline(always)]
fn std_slice(xs: &[f64]) -> f64 {
    var_slice(xs).sqrt()
}

/// Welford one-pass: returns (mean, variance, n)
fn welford_mean_var(xs: &[f64]) -> (f64, f64, usize) {
    let mut n = 0usize;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;

    for &x in xs {
        n += 1;
        let delta = x - mean;
        mean += delta / (n as f64);
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    if n < 2 {
        (mean, f64::NAN, n)
    } else {
        (mean, m2 / ((n - 1) as f64), n)
    }
}

// ---------- percentiles / quantiles / MAD ----------

/// Quantile when `xs` is already sorted ascending.
fn quantile_from_sorted(xs: &[f64], q: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }

    if q <= 0.0 {
        return xs[0];
    } else if q >= 1.0 {
        return xs[n - 1];
    }

    let pos = q * (n as f64 - 1.0);
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;

    if lower == upper {
        xs[lower]
    } else {
        let w = pos - lower as f64;
        xs[lower] * (1.0 - w) + xs[upper] * w
    }
}

/// Percentile on an *unsorted* vector (sorts in place).
fn percentile_slice(mut xs: Vec<f64>, q: f64) -> f64 {
    xs.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    quantile_from_sorted(&xs, q)
}

/// IQR = Q3 - Q1, returns (q1, q3, iqr)
fn iqr_slice(xs: &[f64]) -> (f64, f64, f64) {
    let mut v = xs.to_vec();
    v.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let q1 = quantile_from_sorted(&v, 0.25);
    let q3 = quantile_from_sorted(&v, 0.75);
    let iqr = q3 - q1;
    (q1, q3, iqr)
}

/// Median
fn median_slice(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    quantile_from_sorted(&v, 0.5)
}

/// MAD = median(|x - median|)
fn mad_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let med = median_slice(xs);
    let mut devs: Vec<f64> = xs.iter().map(|x| (x - med).abs()).collect();
    devs.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    quantile_from_sorted(&devs, 0.5)
}

// ---------- rolling stats ----------

fn rolling_mean(xs: &[f64], w: usize) -> Vec<f64> {
    let n = xs.len();
    if w == 0 || w > n {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - w + 1);
    let mut sum: f64 = xs[..w].iter().copied().sum();
    out.push(sum / w as f64);

    for i in w..n {
        sum += xs[i];
        sum -= xs[i - w];
        out.push(sum / w as f64);
    }
    out
}

/// O(n) rolling variance via prefix sums (instead of O(n * w))
fn rolling_var(xs: &[f64], w: usize) -> Vec<f64> {
    let n = xs.len();
    if w == 0 || w > n {
        return Vec::new();
    }

    let mut prefix_sum = vec![0.0; n + 1];
    let mut prefix_sq = vec![0.0; n + 1];

    for (i, &x) in xs.iter().enumerate() {
        prefix_sum[i + 1] = prefix_sum[i] + x;
        prefix_sq[i + 1] = prefix_sq[i] + x * x;
    }

    let mut out = Vec::with_capacity(n - w + 1);
    let w_f = w as f64;
    let denom = (w - 1) as f64;

    for i in 0..=n - w {
        let j = i + w;
        let sum = prefix_sum[j] - prefix_sum[i];
        let sum_sq = prefix_sq[j] - prefix_sq[i];
        let mean = sum / w_f;
        let var = (sum_sq - w_f * mean * mean) / denom;
        out.push(var);
    }
    out
}

fn ewma(xs: &[f64], alpha: f64) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(xs.len());
    let mut prev = xs[0];
    out.push(prev);
    for &x in &xs[1..] {
        let val = alpha * x + (1.0 - alpha) * prev;
        out.push(val);
        prev = val;
    }
    out
}

// ---------- outliers / masks / scaling ----------

fn sign_mask(xs: &[f64]) -> Vec<i8> {
    xs.iter()
        .map(|&x| {
            if x > 0.0 {
                1
            } else if x < 0.0 {
                -1
            } else {
                0
            }
        })
        .collect()
}

/// IQR rule: x < Q1 - k*IQR or x > Q3 + k*IQR
fn iqr_outliers(xs: &[f64], k: f64) -> Vec<bool> {
    let (q1, q3, iqr) = iqr_slice(xs);
    let low = q1 - k * iqr;
    let high = q3 + k * iqr;
    xs.iter().map(|&x| x < low || x > high).collect()
}

fn zscore_outliers(xs: &[f64], threshold: f64) -> Vec<bool> {
    let m = mean_slice(xs);
    let s = std_slice(xs);
    xs.iter()
        .map(|&x| ((x - m) / s).abs() > threshold)
        .collect()
}

/// min-max scaling
fn minmax_scale(xs: &[f64]) -> (Vec<f64>, f64, f64) {
    if xs.is_empty() {
        return (Vec::new(), f64::NAN, f64::NAN);
    }
    let mut min = xs[0];
    let mut max = xs[0];
    for &x in xs {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    if max == min {
        let scaled = vec![0.0; xs.len()];
        (scaled, min, max)
    } else {
        let scaled = xs.iter().map(|&x| (x - min) / (max - min)).collect();
        (scaled, min, max)
    }
}

/// robust scaling using median & MAD (optionally scaled)
fn robust_scale(xs: &[f64], scale_factor: f64) -> (Vec<f64>, f64, f64) {
    if xs.is_empty() {
        return (Vec::new(), f64::NAN, f64::NAN);
    }
    let med = median_slice(xs);
    let mad = mad_slice(xs);
    let denom = if mad == 0.0 { 1e-12 } else { mad * scale_factor };
    let scaled: Vec<f64> = xs.iter().map(|&x| (x - med) / denom).collect();
    (scaled, med, mad)
}

/// winsorization: clamp values between lower_q, upper_q
fn winsorize(xs: &[f64], lower_q: f64, upper_q: f64) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let mut v = xs.to_vec();
    v.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let low = quantile_from_sorted(&v, lower_q);
    let high = quantile_from_sorted(&v, upper_q);

    xs.iter()
        .map(|&x| {
            if x < low {
                low
            } else if x > high {
                high
            } else {
                x
            }
        })
        .collect()
}

// ---------- quantile bins, diffs, cumulative ----------

fn quantile_bins(xs: &[f64], n_bins: usize) -> Vec<i32> {
    let n = xs.len();
    if n == 0 || n_bins == 0 {
        return Vec::new();
    }

    // sort once
    let mut sorted = xs.to_vec();
    sorted.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // compute boundaries on the sorted data
    let mut boundaries = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        let q = i as f64 / (n_bins as f64);
        boundaries.push(quantile_from_sorted(&sorted, q));
    }

    xs.iter()
        .map(|&x| {
            // find first boundary > x, bin = idx - 1
            match boundaries.binary_search_by(|b| {
                b.partial_cmp(&x)
                    .unwrap_or(std::cmp::Ordering::Less)
            }) {
                Ok(idx) => (idx.saturating_sub(1) as i32).max(0),
                Err(idx) => (idx.saturating_sub(1) as i32).max(0),
            }
        })
        .collect()
}

fn diff_slice(xs: &[f64], periods: usize) -> Vec<f64> {
    let n = xs.len();
    if periods == 0 || n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i < periods {
            out.push(f64::NAN);
        } else {
            out.push(xs[i] - xs[i - periods]);
        }
    }
    out
}

fn pct_change_slice(xs: &[f64], periods: usize) -> Vec<f64> {
    let n = xs.len();
    if periods == 0 || n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i < periods {
            out.push(f64::NAN);
        } else {
            let base = xs[i - periods];
            if base == 0.0 {
                out.push(f64::NAN);
            } else {
                out.push(xs[i] / base - 1.0);
            }
        }
    }
    out
}

fn cumsum_slice(xs: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(xs.len());
    let mut sum = 0.0;
    for &x in xs {
        sum += x;
        out.push(sum);
    }
    out
}

fn cummean_slice(xs: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(xs.len());
    let mut sum = 0.0;
    for (i, &x) in xs.iter().enumerate() {
        sum += x;
        out.push(sum / ((i + 1) as f64));
    }
    out
}

// ---------- ECDF ----------

fn ecdf(xs: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut v = xs.to_vec();
    v.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cdf = Vec::with_capacity(n);
    for i in 0..n {
        cdf.push((i + 1) as f64 / (n as f64));
    }
    (v, cdf)
}

// ---------- covariance / correlation (vector & matrices) ----------

fn cov_pair(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return f64::NAN;
    }
    let mx = mean_slice(xs);
    let my = mean_slice(ys);
    let mut acc = 0.0;
    for i in 0..n {
        acc += (xs[i] - mx) * (ys[i] - my);
    }
    acc / ((n - 1) as f64)
}

fn corr_pair(xs: &[f64], ys: &[f64]) -> f64 {
    let cov = cov_pair(xs, ys);
    let sx = std_slice(xs);
    let sy = std_slice(ys);
    cov / (sx * sy)
}

/// Covariance matrix directly on an ArrayView2, returning Vec<Vec<f64>>.
fn cov_matrix_view(arr: ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    let n_samples = arr.nrows();
    if n_samples == 0 {
        return Vec::new();
    }
    let n_features = arr.ncols();
    let mut out = vec![vec![0.0; n_features]; n_features];

    // column means
    let mut means = vec![0.0; n_features];
    for j in 0..n_features {
        let mut sum = 0.0;
        for i in 0..n_samples {
            sum += arr[(i, j)];
        }
        means[j] = sum / (n_samples as f64);
    }

    let denom = (n_samples - 1) as f64;
    for i in 0..n_features {
        for j in i..n_features {
            let mut acc = 0.0;
            let mi = means[i];
            let mj = means[j];
            for k in 0..n_samples {
                let xi = arr[(k, i)] - mi;
                let xj = arr[(k, j)] - mj;
                acc += xi * xj;
            }
            let c = acc / denom;
            out[i][j] = c;
            out[j][i] = c;
        }
    }
    out
}

/// Correlation matrix directly on an ArrayView2, returning Vec<Vec<f64>>.
fn corr_matrix_view(arr: ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    let n_samples = arr.nrows();
    if n_samples == 0 {
        return Vec::new();
    }
    let n_features = arr.ncols();
    let mut out = vec![vec![0.0; n_features]; n_features];

    // column means & stds
    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];

    for j in 0..n_features {
        let mut sum = 0.0;
        for i in 0..n_samples {
            sum += arr[(i, j)];
        }
        let mean = sum / (n_samples as f64);
        means[j] = mean;

        let mut acc = 0.0;
        for i in 0..n_samples {
            let d = arr[(i, j)] - mean;
            acc += d * d;
        }
        stds[j] = (acc / ((n_samples - 1) as f64)).sqrt();
    }

    let denom = (n_samples - 1) as f64;
    for i in 0..n_features {
        for j in i..n_features {
            let mut acc = 0.0;
            let mi = means[i];
            let mj = means[j];
            let si = stds[i];
            let sj = stds[j];
            for k in 0..n_samples {
                let xi = (arr[(k, i)] - mi) / si;
                let xj = (arr[(k, j)] - mj) / sj;
                acc += xi * xj;
            }
            let c = acc / denom;
            out[i][j] = c;
            out[j][i] = c;
        }
    }
    out
}

// ---------- rolling covariance / correlation ----------

/// O(n) rolling cov via prefix sums (x, y, x*y)
fn rolling_cov(xs: &[f64], ys: &[f64], w: usize) -> Vec<f64> {
    let n = xs.len();
    if w == 0 || w > n || n != ys.len() {
        return Vec::new();
    }

    let mut psx = vec![0.0; n + 1];
    let mut psy = vec![0.0; n + 1];
    let mut psxy = vec![0.0; n + 1];

    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        psx[i + 1] = psx[i] + x;
        psy[i + 1] = psy[i] + y;
        psxy[i + 1] = psxy[i] + x * y;
    }

    let w_f = w as f64;
    let mut out = Vec::with_capacity(n - w + 1);
    let denom = (w - 1) as f64;

    for i in 0..=n - w {
        let j = i + w;
        let sum_x = psx[j] - psx[i];
        let sum_y = psy[j] - psy[i];
        let sum_xy = psxy[j] - psxy[i];

        let mean_x = sum_x / w_f;
        let mean_y = sum_y / w_f;

        let cov = (sum_xy - w_f * mean_x * mean_y) / denom;
        out.push(cov);
    }
    out
}

fn rolling_corr(xs: &[f64], ys: &[f64], w: usize) -> Vec<f64> {
    let covs = rolling_cov(xs, ys, w);
    let vars_x = rolling_var(xs, w);
    let vars_y = rolling_var(ys, w);

    covs.into_iter()
        .zip(vars_x.into_iter().zip(vars_y.into_iter()))
        .map(|(c, (vx, vy))| c / (vx.sqrt() * vy.sqrt()))
        .collect()
}

// ---------- KDE (simple Gaussian) ----------

fn kde_gaussian(xs: &[f64], n_points: usize, bandwidth: Option<f64>) -> (Vec<f64>, Vec<f64>) {
    if xs.is_empty() || n_points == 0 {
        return (Vec::new(), Vec::new());
    }
    let n = xs.len() as f64;
    let s = std_slice(xs);

    let bw = match bandwidth {
        Some(b) if b > 0.0 => b,
        _ => {
            // Silverman's rule of thumb
            1.06 * s * n.powf(-1.0 / 5.0)
        }
    };

    // grid from min to max
    let mut min = xs[0];
    let mut max = xs[0];
    for &x in xs {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    if max == min {
        let grid = vec![min; n_points];
        let dens = vec![0.0; n_points];
        return (grid, dens);
    }

    let step = (max - min) / ((n_points - 1) as f64);
    let mut grid = Vec::with_capacity(n_points);
    let mut dens = Vec::with_capacity(n_points);

    let norm_factor = 1.0 / (bw * (2.0 * PI).sqrt());

    for i in 0..n_points {
        let x0 = min + i as f64 * step;
        grid.push(x0);
        let mut sum = 0.0;
        for &x in xs {
            let z = (x0 - x) / bw;
            sum += (-0.5 * z * z).exp();
        }
        dens.push(norm_factor * sum / n);
    }
    (grid, dens)
}

// ---------- NumPy-exposed functions ----------

#[pyfunction]
fn mean_np(a: PyReadonlyArray1<f64>) -> f64 {
    mean_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn std_np(a: PyReadonlyArray1<f64>) -> f64 {
    std_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn var_np(a: PyReadonlyArray1<f64>) -> f64 {
    var_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn zscore_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let m = mean_slice(xs);
    let s = std_slice(xs);
    let z: Vec<f64> = xs.iter().map(|x| (x - m) / s).collect();
    PyArray1::from_vec_bound(py, z)
}

#[pyfunction]
fn percentile_np(a: PyReadonlyArray1<f64>, q: f64) -> f64 {
    percentile_slice(a.as_slice().unwrap().to_vec(), q)
}

#[pyfunction]
fn iqr_np(a: PyReadonlyArray1<f64>) -> (f64, f64, f64) {
    iqr_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn mad_np(a: PyReadonlyArray1<f64>) -> f64 {
    mad_slice(a.as_slice().unwrap())
}

#[pyfunction]
fn rolling_mean_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = rolling_mean(xs, window);
    PyArray1::from_vec_bound(py, v)
}

/// Faster rolling std, similar to `pandas.Series.rolling(window).std()` (ddof=1).
#[pyfunction]
fn rolling_std_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v: Vec<f64> = rolling_var(xs, window)
        .into_iter()
        .map(|vv| vv.sqrt())
        .collect();
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn rolling_zscore_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let means = rolling_mean(xs, window);
    let vars = rolling_var(xs, window);
    let mut out = Vec::with_capacity(means.len());
    // z-score of last element in each window
    for (i, (m, v)) in means.iter().zip(vars.iter()).enumerate() {
        let idx = i + window - 1;
        let std = v.sqrt();
        let z = (xs[idx] - m) / std;
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
    let v = ewma(xs, alpha);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn welford_np(a: PyReadonlyArray1<f64>) -> (f64, f64, usize) {
    let xs = a.as_slice().unwrap();
    welford_mean_var(xs)
}

#[pyfunction]
fn sign_mask_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<i8>> {
    let xs = a.as_slice().unwrap();
    let v = sign_mask(xs);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn demean_with_signs_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i8>>) {
    let xs = a.as_slice().unwrap();
    let m = mean_slice(xs);
    let mut demeaned = Vec::with_capacity(xs.len());
    for &x in xs {
        demeaned.push(x - m);
    }
    let signs = sign_mask(&demeaned);
    let d_arr = PyArray1::from_vec_bound(py, demeaned);
    let s_arr = PyArray1::from_vec_bound(py, signs);
    (d_arr, s_arr)
}

#[pyfunction]
fn iqr_outliers_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    k: f64,
) -> Bound<'py, PyArray1<bool>> {
    let xs = a.as_slice().unwrap();
    let v = iqr_outliers(xs, k);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn zscore_outliers_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    threshold: f64,
) -> Bound<'py, PyArray1<bool>> {
    let xs = a.as_slice().unwrap();
    let v = zscore_outliers(xs, threshold);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn minmax_scale_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, f64, f64) {
    let xs = a.as_slice().unwrap();
    let (scaled, min, max) = minmax_scale(xs);
    (PyArray1::from_vec_bound(py, scaled), min, max)
}

#[pyfunction]
fn robust_scale_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    scale_factor: f64,
) -> (Bound<'py, PyArray1<f64>>, f64, f64) {
    let xs = a.as_slice().unwrap();
    let (scaled, med, mad) = robust_scale(xs, scale_factor);
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
    let v = winsorize(xs, lower_q, upper_q);
    PyArray1::from_vec_bound(py, v)
}

/// Quantile binning similar to `pandas.qcut`, returns bin indices 0..n_bins-1.
#[pyfunction]
fn quantile_bins_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> Bound<'py, PyArray1<i32>> {
    let xs = a.as_slice().unwrap();
    let v = quantile_bins(xs, n_bins);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn diff_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    periods: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = diff_slice(xs, periods);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn pct_change_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    periods: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = pct_change_slice(xs, periods);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn cumsum_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = cumsum_slice(xs);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn cummean_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = cummean_slice(xs);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn ecdf_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let xs = a.as_slice().unwrap();
    let (vals, cdf) = ecdf(xs);
    (
        PyArray1::from_vec_bound(py, vals),
        PyArray1::from_vec_bound(py, cdf),
    )
}

#[pyfunction]
fn cov_np(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
    let xs = a.as_slice().unwrap();
    let ys = b.as_slice().unwrap();
    cov_pair(xs, ys)
}

#[pyfunction]
fn corr_np(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
    let xs = a.as_slice().unwrap();
    let ys = b.as_slice().unwrap();
    corr_pair(xs, ys)
}

/// Covariance matrix similar to `numpy.cov(x, rowvar=False)`.
#[pyfunction]
fn cov_matrix_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let arr = x.as_array();
    let cov = cov_matrix_view(arr);
    PyArray2::from_vec2_bound(py, &cov)
        .expect("cov_matrix_np: from_vec2_bound failed")
}

#[pyfunction]
fn corr_matrix_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let arr = x.as_array();
    let corr = corr_matrix_view(arr);
    PyArray2::from_vec2_bound(py, &corr)
        .expect("corr_matrix_np: from_vec2_bound failed")
}

#[pyfunction]
fn rolling_cov_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let ys = b.as_slice().unwrap();
    let v = rolling_cov(xs, ys, window);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction]
fn rolling_corr_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let ys = b.as_slice().unwrap();
    let v = rolling_corr(xs, ys, window);
    PyArray1::from_vec_bound(py, v)
}

#[pyfunction(signature = (a, n_points, bandwidth=None))]
fn kde_gaussian_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    n_points: usize,
    bandwidth: Option<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let xs = a.as_slice().unwrap();
    // heavy numeric loop: safely release the GIL while computing
    let (grid, dens) = py.allow_threads(|| kde_gaussian(xs, n_points, bandwidth));

    let grid_arr = PyArray1::from_vec_bound(py, grid);
    let dens_arr = PyArray1::from_vec_bound(py, dens);
    (grid_arr, dens_arr)
}

// ---------- module ----------

#[pymodule]
fn bunker_stats_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_np, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_np, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_np, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_np, m)?)?;
    m.add_function(wrap_pyfunction!(mad_np, m)?)?;

    m.add_function(wrap_pyfunction!(rolling_mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_zscore_np, m)?)?;
    m.add_function(wrap_pyfunction!(ewma_np, m)?)?;

    m.add_function(wrap_pyfunction!(welford_np, m)?)?;
    m.add_function(wrap_pyfunction!(sign_mask_np, m)?)?;
    m.add_function(wrap_pyfunction!(demean_with_signs_np, m)?)?;

    m.add_function(wrap_pyfunction!(iqr_outliers_np, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_outliers_np, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(robust_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(winsorize_np, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_bins_np, m)?)?;

    m.add_function(wrap_pyfunction!(diff_np, m)?)?;
    m.add_function(wrap_pyfunction!(pct_change_np, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum_np, m)?)?;
    m.add_function(wrap_pyfunction!(cummean_np, m)?)?;
    m.add_function(wrap_pyfunction!(ecdf_np, m)?)?;

    m.add_function(wrap_pyfunction!(cov_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_np, m)?)?;
    m.add_function(wrap_pyfunction!(cov_matrix_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_matrix_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_cov_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_corr_np, m)?)?;

    m.add_function(wrap_pyfunction!(kde_gaussian_np, m)?)?;

    // Rayon is wired via the "parallel" feature for future parallel kernels.
    Ok(())
}
