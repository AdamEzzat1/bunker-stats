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
// - corr_matrix_skipna_out is pairwise-complete: covariance and both standard
//   deviations of an entry come from the rows where both columns are finite.

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
                    (cov[idx(i, j, p)] / denom).clamp(-1.0, 1.0)
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
                    (cov[idx(i, j, p)] / denom).clamp(-1.0, 1.0)
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

/// Pearson correlation of columns `i` and `j` over the rows where both are
/// finite. NaN with fewer than 2 such rows or when either has no spread there.
///
/// Both columns are shifted by their value in the first shared row before
/// anything is summed. Correlation is shift-invariant, the subtraction of two
/// nearby doubles is exact, and without it the mean of data at an offset of
/// 1e12 is only good to ~1e-4, which costs the result three digits.
fn pairwise_complete_corr(x: &[f64], n: usize, p: usize, i: usize, j: usize) -> f64 {
    let mut origin: Option<(f64, f64)> = None;
    let (mut m, mut sx, mut sy) = (0usize, 0.0f64, 0.0f64);
    for r in 0..n {
        let (a, b) = (x[r * p + i], x[r * p + j]);
        if a.is_finite() && b.is_finite() {
            let (a0, b0) = *origin.get_or_insert((a, b));
            m += 1;
            sx += a - a0;
            sy += b - b0;
        }
    }
    let (a0, b0) = match origin {
        Some(o) if m >= 2 => o,
        _ => return f64::NAN,
    };
    // Second pass on centred values.
    let (mx, my) = (sx / m as f64, sy / m as f64);
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for r in 0..n {
        let (a, b) = (x[r * p + i], x[r * p + j]);
        if a.is_finite() && b.is_finite() {
            let (da, db) = ((a - a0) - mx, (b - b0) - my);
            sxx += da * da;
            syy += db * db;
            sxy += da * db;
        }
    }
    if !(sxx > 0.0 && syy > 0.0) {
        return f64::NAN;
    }
    (sxy / (sxx.sqrt() * syy.sqrt())).clamp(-1.0, 1.0)
}

/// NaN-aware correlation matrix (pairwise complete).
///
/// Each entry uses the rows where both columns are finite, for the
/// covariance *and* for both standard deviations, as `DataFrame.corr()`
/// does. If the pairwise count is below 2 the entry is NaN.
///
/// Up to 0.3.1 this divided the pairwise covariance by standard deviations
/// taken over each column's own rows (the diagonal of the pairwise covariance
/// matrix). That is not the correlation of any one set of rows when two
/// columns are missing in different places, and it can leave [-1, 1]. The
/// shortcut is exact for a pair of gap-free columns, so it is kept for those;
/// every pair involving a column with gaps is computed on its own rows.
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

    let mut gappy = vec![false; p];
    for row in x.chunks_exact(p) {
        for (c, v) in row.iter().enumerate() {
            if !v.is_finite() {
                gappy[c] = true;
            }
        }
    }
    if !gappy.iter().any(|g| *g) {
        return;
    }
    for i in 0..p {
        for j in (i + 1)..p {
            if gappy[i] || gappy[j] {
                let r = pairwise_complete_corr(x, n, p, i, j);
                out[idx(i, j, p)] = r;
                out[idx(j, i, p)] = r;
            }
        }
    }
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

#[cfg(test)]
mod skipna_tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    /// 8 x 3, gaps in different rows of every column. Reference:
    /// `pandas.DataFrame(X).corr()`. Normalising the pairwise covariance by
    /// each column's own standard deviation gave -0.6964 and 0.0923 for the
    /// last two entries.
    #[test]
    fn skipna_correlation_is_pairwise_complete() {
        #[rustfmt::skip]
        let x = [
            1.0, 2.0, 9.0,
            2.0, 1.0, NAN,
            3.0, 5.0, 4.0,
            4.0, 3.0, 1.0,
            NAN, 8.0, 7.0,
            6.0, NAN, 2.0,
            7.0, 9.0, NAN,
            8.0, 4.0, 3.0,
        ];
        let want = [
            1.0, 0.6596916954511481, -0.6714306318216328,
            0.6596916954511481, 1.0, 0.11560614116133207,
            -0.6714306318216328, 0.11560614116133207, 1.0,
        ];
        let mut out = [0.0f64; 9];
        corr_matrix_skipna_out(&x, 8, 3, &mut out);
        for (got, want) in out.iter().zip(want) {
            assert!((got - want).abs() < 1e-13, "{out:?}");
        }
    }

    /// Without gaps the skip-NaN kernel is the strict one.
    #[test]
    fn skipna_equals_strict_on_complete_data() {
        let x: Vec<f64> = (0..60).map(|i| ((i * 37 % 23) as f64).sin() + (i % 3) as f64).collect();
        let (mut a, mut b) = ([0.0f64; 9], [0.0f64; 9]);
        corr_matrix_skipna_out(&x, 20, 3, &mut a);
        corr_matrix_out(&x, 20, 3, &mut b);
        for (u, v) in a.iter().zip(b) {
            assert!((u - v).abs() < 1e-13, "{a:?} vs {b:?}");
        }
    }

    /// A large common offset must not cost precision: the reference is the
    /// same data without the offset (1e12 + k/8 is exact in f64).
    #[test]
    fn skipna_correlation_survives_a_large_offset() {
        let base: Vec<f64> = (0..90)
            .map(|k| (((k * 37 + 11) % 64) as f64) / 8.0 + ((k % 7) as f64) * 0.125)
            .collect();
        let mut near = base.clone();
        let mut far: Vec<f64> = base.iter().map(|v| v + 1e12).collect();
        for r in (0..30).step_by(7) {
            near[r * 3 + 1] = NAN;
            far[r * 3 + 1] = NAN;
        }
        let (mut a, mut b) = ([0.0f64; 9], [0.0f64; 9]);
        corr_matrix_skipna_out(&near, 30, 3, &mut a);
        corr_matrix_skipna_out(&far, 30, 3, &mut b);
        for (u, v) in a.iter().zip(b) {
            assert!((u - v).abs() < 1e-12, "{a:?} vs {b:?}");
        }
    }

    /// Too little overlap, or no spread within the overlap, is undefined.
    #[test]
    fn skipna_undefined_pairs_are_nan() {
        #[rustfmt::skip]
        let x = [
            1.0, NAN, 5.0,
            2.0, NAN, 5.0,
            NAN, 3.0, 5.0,
            NAN, 4.0, 6.0,
            3.0, 9.0, NAN,
        ];
        let mut out = [0.0f64; 9];
        corr_matrix_skipna_out(&x, 5, 3, &mut out);
        assert!(out[1].is_nan() && out[3].is_nan(), "one shared row: {out:?}");
        assert!(out[2].is_nan() && out[6].is_nan(), "constant on the shared rows: {out:?}");
        assert!((out[5] - 1.0).abs() < 1e-13 && out[5] == out[7], "{out:?}");
    }
}
