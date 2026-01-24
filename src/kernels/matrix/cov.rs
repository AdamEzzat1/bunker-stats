// src/kernels/matrix/cov.rs
//
// Covariance / Gram / pairwise distance matrix kernels.
//
// Design goals:
// - hot kernels write into a preallocated flat buffer (row-major) via &mut [f64]
// - symmetric matrices compute upper triangle then mirror
// - optional Rayon parallelism behind feature="parallel"

use numpy::ndarray::ArrayView2;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[inline]
fn idx(i: usize, j: usize, p: usize) -> usize {
    i * p + j
}

/// Fill a sample covariance matrix (ddof=1) into `out`.
///
/// - `x` is row-major data with shape (n, p) flattened as length n*p.
/// - `out` is length p*p, row-major (i*p + j).
///
/// Semantics: matches numpy.cov(rowvar=False, bias=False) for finite data.
pub fn cov_matrix_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    if n < 2 {
        out.fill(f64::NAN);
        return;
    }

    // Column means (one pass)
    let mut means = vec![0.0f64; p];
    for r in 0..n {
        let base = r * p;
        for c in 0..p {
            means[c] += x[base + c];
        }
    }
    let inv_n = 1.0 / (n as f64);
    for c in 0..p {
        means[c] *= inv_n;
    }

    let denom = (n - 1) as f64;

    #[cfg(feature = "parallel")]
    {
        // Parallelize across output rows; each row writes only its own slice.
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    let xi = x[base + i] - means[i];
                    let xj = x[base + j] - means[j];
                    acc += xi * xj;
                }
                row[j] = acc / denom;
            }
        });

        // Mirror upper -> lower.
        for i in 0..p {
            for j in (i + 1)..p {
                out[idx(j, i, p)] = out[idx(i, j, p)];
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..p {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    let xi = x[base + i] - means[i];
                    let xj = x[base + j] - means[j];
                    acc += xi * xj;
                }
                let v = acc / denom;
                out[idx(i, j, p)] = v;
                out[idx(j, i, p)] = v;
            }
        }
    }
}

/// Fill a population covariance matrix (ddof=0) into `out`.
///
/// For `n == 0`, fills NaN. For `n == 1`, covariance is 0 along the diagonal and 0 off-diagonal.
pub fn cov_matrix_bias_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    if n == 0 {
        out.fill(f64::NAN);
        return;
    }

    // Column means
    let mut means = vec![0.0f64; p];
    for r in 0..n {
        let base = r * p;
        for c in 0..p {
            means[c] += x[base + c];
        }
    }
    let inv_n = 1.0 / (n as f64);
    for c in 0..p {
        means[c] *= inv_n;
    }

    let denom = n as f64;

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    let xi = x[base + i] - means[i];
                    let xj = x[base + j] - means[j];
                    acc += xi * xj;
                }
                row[j] = acc / denom;
            }
        });

        for i in 0..p {
            for j in (i + 1)..p {
                out[idx(j, i, p)] = out[idx(i, j, p)];
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..p {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    let xi = x[base + i] - means[i];
                    let xj = x[base + j] - means[j];
                    acc += xi * xj;
                }
                let v = acc / denom;
                out[idx(i, j, p)] = v;
                out[idx(j, i, p)] = v;
            }
        }
    }
}

/// Fill a sample covariance matrix (ddof=1) for *centered* data (means already removed).
///
/// This skips the mean pass entirely.
pub fn cov_matrix_centered_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    if n < 2 {
        out.fill(f64::NAN);
        return;
    }

    let denom = (n - 1) as f64;

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    acc += x[base + i] * x[base + j];
                }
                row[j] = acc / denom;
            }
        });

        for i in 0..p {
            for j in (i + 1)..p {
                out[idx(j, i, p)] = out[idx(i, j, p)];
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..p {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    acc += x[base + i] * x[base + j];
                }
                let v = acc / denom;
                out[idx(i, j, p)] = v;
                out[idx(j, i, p)] = v;
            }
        }
    }
}

/// NaN-aware covariance matrix (pairwise complete).
///
/// For each pair (i,j) we include only rows where both x_i and x_j are finite.
/// We compute covariance using a stable one-pass formula:\n
/// cov(i,j) = (sum_xy - sum_x*sum_y / m) / (m-1)\n
/// where m is the pairwise count. If m < 2, returns NaN.
pub fn cov_matrix_skipna_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    if n == 0 {
        out.fill(f64::NAN);
        return;
    }

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            for j in i..p {
                let mut m: usize = 0;
                let mut sum_x = 0.0f64;
                let mut sum_y = 0.0f64;
                let mut sum_xy = 0.0f64;

                for r in 0..n {
                    let base = r * p;
                    let xi = x[base + i];
                    let yj = x[base + j];
                    if xi.is_finite() && yj.is_finite() {
                        m += 1;
                        sum_x += xi;
                        sum_y += yj;
                        sum_xy += xi * yj;
                    }
                }

                row[j] = if m < 2 {
                    f64::NAN
                } else {
                    let mf = m as f64;
                    (sum_xy - (sum_x * sum_y) / mf) / ((m - 1) as f64)
                };
            }
        });

        for i in 0..p {
            for j in (i + 1)..p {
                out[idx(j, i, p)] = out[idx(i, j, p)];
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..p {
            for j in i..p {
                let mut m: usize = 0;
                let mut sum_x = 0.0f64;
                let mut sum_y = 0.0f64;
                let mut sum_xy = 0.0f64;

                for r in 0..n {
                    let base = r * p;
                    let xi = x[base + i];
                    let yj = x[base + j];
                    if xi.is_finite() && yj.is_finite() {
                        m += 1;
                        sum_x += xi;
                        sum_y += yj;
                        sum_xy += xi * yj;
                    }
                }

                let v = if m < 2 {
                    f64::NAN
                } else {
                    let mf = m as f64;
                    (sum_xy - (sum_x * sum_y) / mf) / ((m - 1) as f64)
                };
                out[idx(i, j, p)] = v;
                out[idx(j, i, p)] = v;
            }
        }
    }
}

/// Gram matrix X^T X (p×p).
///
/// For an empty matrix (n==0), this returns zeros (consistent with numpy dot).
pub fn xtx_matrix_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    out.fill(0.0);
    if n == 0 {
        return;
    }

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    acc += x[base + i] * x[base + j];
                }
                row[j] = acc;
            }
        });

        for i in 0..p {
            for j in (i + 1)..p {
                out[idx(j, i, p)] = out[idx(i, j, p)];
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..p {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    acc += x[base + i] * x[base + j];
                }
                out[idx(i, j, p)] = acc;
                out[idx(j, i, p)] = acc;
            }
        }
    }
}

/// Gram matrix X X^T (n×n).
///
/// WARNING: output is n*n elements and can be huge. For (n==0), returns empty/zeros.
pub fn xxt_matrix_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), n * n);

    if n == 0 {
        return;
    }
    out.fill(0.0);
    if p == 0 {
        return;
    }

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in i..n {
                let mut acc = 0.0f64;
                let base_i = i * p;
                let base_j = j * p;
                for k in 0..p {
                    acc += x[base_i + k] * x[base_j + k];
                }
                row[j] = acc;
            }
        });

        for i in 0..n {
            for j in (i + 1)..n {
                out[j * n + i] = out[i * n + j];
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..n {
            for j in i..n {
                let mut acc = 0.0f64;
                let base_i = i * p;
                let base_j = j * p;
                for k in 0..p {
                    acc += x[base_i + k] * x[base_j + k];
                }
                out[i * n + j] = acc;
                out[j * n + i] = acc;
            }
        }
    }
}

/// Pairwise Euclidean distances between columns (p×p).
pub fn pairwise_euclidean_cols_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    out.fill(0.0);
    if n == 0 {
        return;
    }

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            row[i] = 0.0;
            for j in (i + 1)..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    let d = x[base + i] - x[base + j];
                    acc += d * d;
                }
                row[j] = acc.sqrt();
            }
        });

        for i in 0..p {
            for j in (i + 1)..p {
                out[idx(j, i, p)] = out[idx(i, j, p)];
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..p {
            out[idx(i, i, p)] = 0.0;
            for j in (i + 1)..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    let d = x[base + i] - x[base + j];
                    acc += d * d;
                }
                let v = acc.sqrt();
                out[idx(i, j, p)] = v;
                out[idx(j, i, p)] = v;
            }
        }
    }
}

/// Pairwise cosine similarity between columns (p×p).
///
/// If a column has zero norm, similarities involving it are NaN.
pub fn pairwise_cosine_cols_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }

    // Precompute column norms
    let mut norms = vec![0.0f64; p];
    for r in 0..n {
        let base = r * p;
        for c in 0..p {
            let v = x[base + c];
            norms[c] += v * v;
        }
    }
    for c in 0..p {
        norms[c] = norms[c].sqrt();
    }

    // Fill upper triangle (cosine *distance* = 1 - cosine_similarity)
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(p)
            .enumerate()
            .for_each(|(i, row)| {
                let ni = norms[i];
                for j in i..p {
                    let nj = norms[j];

                    // Diagonal: distance to self is 0 when defined
                    if i == j {
                        row[j] = if ni == 0.0 { f64::NAN } else { 0.0 };
                        continue;
                    }

                    let denom = ni * nj;
                    if denom == 0.0 || denom.is_nan() {
                        row[j] = f64::NAN;
                        continue;
                    }

                    let mut dot = 0.0f64;
                    for r in 0..n {
                        let base = r * p;
                        dot += x[base + i] * x[base + j];
                    }

                    let cos_sim = dot / denom;
                    row[j] = 1.0 - cos_sim;
                }
            });

        // Mirror upper -> lower
        for i in 0..p {
            for j in (i + 1)..p {
                out[j * p + i] = out[i * p + j];
            }
        }
        return;
    }

    // Sequential path
    for i in 0..p {
        let ni = norms[i];

        // diagonal
        out[i * p + i] = if ni == 0.0 { f64::NAN } else { 0.0 };

        for j in (i + 1)..p {
            let nj = norms[j];
            let denom = ni * nj;

            let v = if denom == 0.0 || denom.is_nan() {
                f64::NAN
            } else {
                let mut dot = 0.0f64;
                for r in 0..n {
                    let base = r * p;
                    dot += x[base + i] * x[base + j];
                }
                1.0 - (dot / denom)
            };

            out[i * p + j] = v;
            out[j * p + i] = v;
        }
    }
}


/// Column stats pack: sums, sumsqs, counts (finite-only).
///
/// - `sums` and `sumsqs` must be length p
/// - `counts` must be length p
pub fn col_sums_sumsq_count_out(
    x: &[f64],
    n: usize,
    p: usize,
    sums: &mut [f64],
    sumsqs: &mut [f64],
    counts: &mut [usize],
) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(sums.len(), p);
    debug_assert_eq!(sumsqs.len(), p);
    debug_assert_eq!(counts.len(), p);

    sums.fill(0.0);
    sumsqs.fill(0.0);
    counts.fill(0);

    for r in 0..n {
        let base = r * p;
        for c in 0..p {
            let v = x[base + c];
            if v.is_finite() {
                sums[c] += v;
                sumsqs[c] += v * v;
                counts[c] += 1;
            }
        }
    }
}

/// Backward-compatible helper for internal callers that operate on `ndarray` views.
///
/// Returns `Vec<Vec<f64>>` to preserve older call sites.
/// Prefer `cov_matrix_out` + a flat output buffer for performance-sensitive paths.
pub fn cov_matrix_view(arr: ArrayView2<f64>) -> Vec<Vec<f64>> {
    let (n_rows, n_cols) = arr.dim();

    if n_rows < 2 || n_cols == 0 {
        return vec![vec![f64::NAN; n_cols]; n_cols];
    }

    // Prefer a zero-copy contiguous slice when possible; otherwise materialize.
    let owned;
    let x: &[f64] = match arr.as_slice() {
        Some(s) => s,
        None => {
            owned = arr.to_owned();
            owned.as_slice().expect("owned ndarray must be contiguous")
        }
    };

    let mut flat = vec![0.0f64; n_cols * n_cols];
    cov_matrix_out(x, n_rows, n_cols, &mut flat);

    let mut out = vec![vec![0.0f64; n_cols]; n_cols];
    for i in 0..n_cols {
        out[i].copy_from_slice(&flat[i * n_cols..(i + 1) * n_cols]);
    }
    out
}
