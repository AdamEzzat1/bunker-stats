// src/kernels/matrix/corr.rs
//
// Correlation matrix kernels (flat row-major out).
//
// Design goals:
// - in-place output via &mut [f64]
// - symmetric: compute upper then mirror
// - optional Rayon parallelism behind feature="parallel"
//
// Notes:
// - corr_matrix_out computes covariance once via cov_matrix_out, then normalizes.
// - corr_matrix_skipna_out uses pairwise-complete covariance via cov_matrix_skipna_out.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::kernels::matrix::cov::{cov_matrix_out, cov_matrix_skipna_out};

#[inline]
fn idx(i: usize, j: usize, p: usize) -> usize {
    i * p + j
}

/// Compute correlation from a covariance matrix.
///
/// `cov` is length p*p row-major. `out` is length p*p row-major.
pub fn corr_matrix_from_cov_out(cov: &[f64], p: usize, out: &mut [f64]) {
    debug_assert_eq!(cov.len(), p * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }

    // Std devs from diagonal
    let mut stds = vec![0.0f64; p];
    for i in 0..p {
        let v = cov[idx(i, i, p)];
        stds[i] = if v > 0.0 && v.is_finite() { v.sqrt() } else { f64::NAN };
    }

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            let si = stds[i];
            for j in i..p {
                let sj = stds[j];
                let denom = si * sj;
                row[j] = if denom == 0.0 || denom.is_nan() {
                    f64::NAN
                } else {
                    cov[idx(i, j, p)] / denom
                };
            }
        });

        for i in 0..p {
            // diag: if finite std, corr=1.0 (since cov diag / (std^2) = 1.0)
            out[idx(i, i, p)] = if stds[i].is_nan() || stds[i] == 0.0 { f64::NAN } else { 1.0 };
            for j in (i + 1)..p {
                out[idx(j, i, p)] = out[idx(i, j, p)];
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..p {
            let si = stds[i];
            for j in i..p {
                let sj = stds[j];
                let denom = si * sj;
                let v = if denom == 0.0 || denom.is_nan() {
                    f64::NAN
                } else {
                    cov[idx(i, j, p)] / denom
                };
                out[idx(i, j, p)] = v;
                out[idx(j, i, p)] = v;
            }
        }
        for i in 0..p {
            out[idx(i, i, p)] = if stds[i].is_nan() || stds[i] == 0.0 { f64::NAN } else { 1.0 };
        }
    }
}

/// Fill a correlation matrix (Pearson) into `out`.
///
/// - `x` is row-major data with shape (n, p) flattened as length n*p.
/// - `out` is length p*p, row-major.
///
/// For `n < 2`, fills NaN.
pub fn corr_matrix_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    if n < 2 {
        out.fill(f64::NAN);
        return;
    }

    let mut cov = vec![0.0f64; p * p];
    cov_matrix_out(x, n, p, &mut cov);
    corr_matrix_from_cov_out(&cov, p, out);
}

/// NaN-aware correlation matrix (pairwise complete).
///
/// Uses pairwise-complete covariance and then normalizes per pair.
/// If pairwise count < 2, returns NaN for that entry.
pub fn corr_matrix_skipna_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    if n == 0 {
        out.fill(f64::NAN);
        return;
    }

    let mut cov = vec![0.0f64; p * p];
    cov_matrix_skipna_out(x, n, p, &mut cov);
    corr_matrix_from_cov_out(&cov, p, out);
}

/// Correlation distance matrix: 1 - corr.
///
/// Diagonal is 0 when correlation is defined; otherwise NaN.
pub fn corr_distance_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }

    let mut corr = vec![0.0f64; p * p];
    corr_matrix_out(x, n, p, &mut corr);

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            for j in 0..p {
                let v = corr[idx(i, j, p)];
                row[j] = if v.is_nan() { f64::NAN } else { 1.0 - v };
            }
        });

        for i in 0..p {
            // if corr diag is NaN, keep NaN; else distance 0
            let v = corr[idx(i, i, p)];
            out[idx(i, i, p)] = if v.is_nan() { f64::NAN } else { 0.0 };
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..p {
            for j in 0..p {
                let v = corr[idx(i, j, p)];
                out[idx(i, j, p)] = if v.is_nan() { f64::NAN } else { 1.0 - v };
            }
        }
        for i in 0..p {
            let v = corr[idx(i, i, p)];
            out[idx(i, i, p)] = if v.is_nan() { f64::NAN } else { 0.0 };
        }
    }
}

/// Backward-compatible helper: original interface that accepts precomputed means/stds.
/// Kept only for older internal call sites; new code should prefer corr_matrix_out.
///
/// Returns a flat row-major Vec<f64> of shape (p,p).
pub(crate) fn corr_matrix_out_precomputed(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    means: &[f64],
    stds: &[f64],
    denom: f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; n_cols * n_cols];

    #[cfg(feature = "parallel")]
    {
        let results: Vec<(usize, Vec<f64>)> = (0..n_cols)
            .into_par_iter()
            .map(|i| {
                let mi = means[i];
                let si = stds[i];
                let mut row_upper = vec![0.0f64; n_cols - i];
                for (ofs, j) in (i..n_cols).enumerate() {
                    let mj = means[j];
                    let sj = stds[j];
                    let mut acc = 0.0f64;
                    for r in 0..n_rows {
                        let base = r * n_cols;
                        let xi = x[base + i] - mi;
                        let xj = x[base + j] - mj;
                        acc += xi * xj;
                    }
                    let cov = acc / denom;
                    let corr = if si == 0.0 || sj == 0.0 { f64::NAN } else { cov / (si * sj) };
                    row_upper[ofs] = corr;
                }
                (i, row_upper)
            })
            .collect();

        for (i, row_upper) in results {
            for (ofs, j) in (i..n_cols).enumerate() {
                let v = row_upper[ofs];
                out[i * n_cols + j] = v;
                out[j * n_cols + i] = v;
            }
        }

        return out;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..n_cols {
            let mi = means[i];
            let si = stds[i];
            for j in i..n_cols {
                let mj = means[j];
                let sj = stds[j];
                let mut acc = 0.0f64;
                for r in 0..n_rows {
                    let base = r * n_cols;
                    let xi = x[base + i] - mi;
                    let xj = x[base + j] - mj;
                    acc += xi * xj;
                }
                let cov = acc / denom;
                let corr = if si == 0.0 || sj == 0.0 { f64::NAN } else { cov / (si * sj) };
                out[i * n_cols + j] = corr;
                out[j * n_cols + i] = corr;
            }
        }
        out
    }
}
