use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2
};
use pyo3::prelude::*;
use std::f64::consts::PI;

// ---------- basic helpers ----------

fn mean_slice(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n == 0.0 {
        return f64::NAN;
    }
    xs.iter().copied().sum::<f64>() / n
}

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

fn percentile_slice(mut xs: Vec<f64>, q: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a
        .partial_cmp(b)
        .unwrap_or(std::cmp::Ordering::Equal));

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

/// IQR = Q3 - Q1, returns (q1, q3, iqr)
fn iqr_slice(xs: &[f64]) -> (f64, f64, f64) {
    let v = xs.to_vec();
    let q1 = percentile_slice(v.clone(), 0.25);
    let q3 = percentile_slice(v, 0.75);
    let iqr = q3 - q1;
    (q1, q3, iqr)
}

/// Median
fn median_slice(xs: &[f64]) -> f64 {
    percentile_slice(xs.to_vec(), 0.5)
}

/// MAD = median(|x - median|)
fn mad_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let med = median_slice(xs);
    let devs: Vec<f64> = xs.iter().map(|x| (x - med).abs()).collect();
    percentile_slice(devs, 0.5)
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

fn rolling_var(xs: &[f64], w: usize) -> Vec<f64> {
    let n = xs.len();
    if w == 0 || w > n {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - w + 1);
    for i in 0..=n - w {
        let window = &xs[i..i + w];
        out.push(var_slice(window));
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
    let v = xs.to_vec();
    let low = percentile_slice(v.clone(), lower_q);
    let high = percentile_slice(v, upper_q);
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

    // compute bin boundaries
    let mut boundaries = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        let q = i as f64 / (n_bins as f64);
        boundaries.push(percentile_slice(xs.to_vec(), q));
    }

    xs.iter()
        .map(|&x| {
            let mut b = 0i32;
            for i in 0..n_bins {
                if x >= boundaries[i] && x <= boundaries[i + 1] {
                    b = i as i32;
                    break;
                }
            }
            b
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
    v.sort_by(|a, b| a
        .partial_cmp(b)
        .unwrap_or(std::cmp::Ordering::Equal));
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

fn cov_matrix(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_samples = x.len();
    if n_samples == 0 {
        return Vec::new();
    }
    let n_features = x[0].len();
    let mut out = vec![vec![0.0; n_features]; n_features];

    // build column vectors
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_features];
    for row in x {
        for (j, &val) in row.iter().enumerate() {
            cols[j].push(val);
        }
    }

    for i in 0..n_features {
        for j in i..n_features {
            let c = cov_pair(&cols[i], &cols[j]);
            out[i][j] = c;
            out[j][i] = c;
        }
    }
    out
}

fn corr_matrix(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_samples = x.len();
    if n_samples == 0 {
        return Vec::new();
    }
    let n_features = x[0].len();
    let mut out = vec![vec![0.0; n_features]; n_features];

    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_features];
    for row in x {
        for (j, &val) in row.iter().enumerate() {
            cols[j].push(val);
        }
    }

    for i in 0..n_features {
        for j in i..n_features {
            let c = corr_pair(&cols[i], &cols[j]);
            out[i][j] = c;
            out[j][i] = c;
        }
    }
    out
}

// ---------- rolling covariance / correlation ----------

fn rolling_cov(xs: &[f64], ys: &[f64], w: usize) -> Vec<f64> {
    let n = xs.len();
    if w == 0 || w > n || n != ys.len() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - w + 1);
    for i in 0..=n - w {
        let xw = &xs[i..i + w];
        let yw = &ys[i..i + w];
        out.push(cov_pair(xw, yw));
    }
    out
}

fn rolling_corr(xs: &[f64], ys: &[f64], w: usize) -> Vec<f64> {
    let n = xs.len();
    if w == 0 || w > n || n != ys.len() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - w + 1);
    for i in 0..=n - w {
        let xw = &xs[i..i + w];
        let yw = &ys[i..i + w];
        out.push(corr_pair(xw, yw));
    }
    out
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
fn zscore_np(py: Python<'_>, a: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let m = mean_slice(xs);
    let s = std_slice(xs);
    let z: Vec<f64> = xs.iter().map(|x| (x - m) / s).collect();
    PyArray1::from_vec(py, z).to_owned()
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
fn rolling_mean_np(py: Python<'_>, a: PyReadonlyArray1<f64>, window: usize) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = rolling_mean(xs, window);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn rolling_std_np(py: Python<'_>, a: PyReadonlyArray1<f64>, window: usize) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v: Vec<f64> = rolling_var(xs, window)
        .into_iter()
        .map(|vv| vv.sqrt())
        .collect();
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn rolling_zscore_np(py: Python<'_>, a: PyReadonlyArray1<f64>, window: usize) -> Py<PyArray1<f64>> {
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
    PyArray1::from_vec(py, out).to_owned()
}

#[pyfunction]
fn ewma_np(py: Python<'_>, a: PyReadonlyArray1<f64>, alpha: f64) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = ewma(xs, alpha);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn welford_np(a: PyReadonlyArray1<f64>) -> (f64, f64, usize) {
    let xs = a.as_slice().unwrap();
    welford_mean_var(xs)
}

#[pyfunction]
fn sign_mask_np(py: Python<'_>, a: PyReadonlyArray1<f64>) -> Py<PyArray1<i8>> {
    let xs = a.as_slice().unwrap();
    let v = sign_mask(xs);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn demean_with_signs_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
) -> (Py<PyArray1<f64>>, Py<PyArray1<i8>>) {
    let xs = a.as_slice().unwrap();
    let m = mean_slice(xs);
    let mut demeaned = Vec::with_capacity(xs.len());
    for &x in xs {
        demeaned.push(x - m);
    }
    let signs = sign_mask(&demeaned);
    let d_arr = PyArray1::from_vec(py, demeaned).to_owned();
    let s_arr = PyArray1::from_vec(py, signs).to_owned();
    (d_arr, s_arr)
}

#[pyfunction]
fn iqr_outliers_np(py: Python<'_>, a: PyReadonlyArray1<f64>, k: f64) -> Py<PyArray1<bool>> {
    let xs = a.as_slice().unwrap();
    let v = iqr_outliers(xs, k);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn zscore_outliers_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
    threshold: f64,
) -> Py<PyArray1<bool>> {
    let xs = a.as_slice().unwrap();
    let v = zscore_outliers(xs, threshold);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn minmax_scale_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
) -> (Py<PyArray1<f64>>, f64, f64) {
    let xs = a.as_slice().unwrap();
    let (scaled, min, max) = minmax_scale(xs);
    (PyArray1::from_vec(py, scaled).to_owned(), min, max)
}

#[pyfunction]
fn robust_scale_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
    scale_factor: f64,
) -> (Py<PyArray1<f64>>, f64, f64) {
    let xs = a.as_slice().unwrap();
    let (scaled, med, mad) = robust_scale(xs, scale_factor);
    (PyArray1::from_vec(py, scaled).to_owned(), med, mad)
}

#[pyfunction]
fn winsorize_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
    lower_q: f64,
    upper_q: f64,
) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = winsorize(xs, lower_q, upper_q);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn quantile_bins_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> Py<PyArray1<i32>> {
    let xs = a.as_slice().unwrap();
    let v = quantile_bins(xs, n_bins);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn diff_np(py: Python<'_>, a: PyReadonlyArray1<f64>, periods: usize) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = diff_slice(xs, periods);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn pct_change_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
    periods: usize,
) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = pct_change_slice(xs, periods);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn cumsum_np(py: Python<'_>, a: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = cumsum_slice(xs);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn cummean_np(py: Python<'_>, a: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let v = cummean_slice(xs);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn ecdf_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let xs = a.as_slice().unwrap();
    let (vals, cdf) = ecdf(xs);
    (
        PyArray1::from_vec(py, vals).to_owned(),
        PyArray1::from_vec(py, cdf).to_owned(),
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

#[pyfunction]
fn cov_matrix_np(py: Python<'_>, x: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>> {
    let arr = x.as_array();
    let n_samples = arr.nrows();
    let n_features = arr.ncols();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut row = Vec::with_capacity(n_features);
        for j in 0..n_features {
            row.push(arr[[i, j]]);
        }
        rows.push(row);
    }
    let cov = cov_matrix(&rows);
    PyArray2::from_vec2(py, &cov).unwrap().to_owned()
}

#[pyfunction]
fn corr_matrix_np(py: Python<'_>, x: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>> {
    let arr = x.as_array();
    let n_samples = arr.nrows();
    let n_features = arr.ncols();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut row = Vec::with_capacity(n_features);
        for j in 0..n_features {
            row.push(arr[[i, j]]);
        }
        rows.push(row);
    }
    let corr = corr_matrix(&rows);
    PyArray2::from_vec2(py, &corr).unwrap().to_owned()
}

#[pyfunction]
fn rolling_cov_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    window: usize,
) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let ys = b.as_slice().unwrap();
    let v = rolling_cov(xs, ys, window);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn rolling_corr_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    window: usize,
) -> Py<PyArray1<f64>> {
    let xs = a.as_slice().unwrap();
    let ys = b.as_slice().unwrap();
    let v = rolling_corr(xs, ys, window);
    PyArray1::from_vec(py, v).to_owned()
}

#[pyfunction]
fn kde_gaussian_np(
    py: Python<'_>,
    a: PyReadonlyArray1<f64>,
    n_points: usize,
    bandwidth: Option<f64>,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let xs = a.as_slice().unwrap();
    let (grid, dens) = kde_gaussian(xs, n_points, bandwidth);
    (
        PyArray1::from_vec(py, grid).to_owned(),
        PyArray1::from_vec(py, dens).to_owned(),
    )
}

// ---------- module ----------

#[pymodule]
fn bunker_stats_rs(_py: Python, m: &PyModule) -> PyResult<()> {
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

    Ok(())
}
