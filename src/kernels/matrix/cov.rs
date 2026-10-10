// src/kernels/matrix/cov.rs
//
// Covariance / Gram / pairwise distance matrix kernels.
//
// Design goals:
// - hot kernels write into a preallocated flat buffer (row-major) via &mut [f64]
// - symmetric matrices compute upper triangle then mirror
// - optional Rayon parallelism behind feature="parallel"
// - the dense kernels (everything except the NaN-skipping covariance) are
//   cache-blocked and bit-reproducible: the same input gives the same bytes on
//   every CPU and for every thread count

use numpy::ndarray::ArrayView2;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[inline]
fn idx(i: usize, j: usize, p: usize) -> usize {
    i * p + j
}

/// Independent accumulators per dot product. A compile-time constant, never
/// chosen from the running CPU, so the order of additions is fixed everywhere.
const LANES: usize = 8;

/// Target size (in f64s) of the column-major scratch block: 512 KiB, small
/// enough that every column pair of a block is computed on cached data.
const BLOCK_ELEMS: usize = 1 << 16;

/// Floor on rows per block, so very wide inputs still get useful lane work.
const MIN_BLOCK_ROWS: usize = 64;

/// Per-block work (rows * p * p) below which Rayon dispatch costs more than it
/// saves; with full blocks this means matrices narrower than 64 columns stay
/// serial.
#[cfg(feature = "parallel")]
const PAR_MIN_WORK: usize = 1 << 22;

/// Dot product with `LANES` fixed accumulators.
///
/// Rust never reorders floating-point reductions, so a single-accumulator loop
/// cannot be vectorized. Splitting the sum across fixed lanes lets LLVM use
/// SIMD while the addition order stays a pure function of the slice length.
#[inline]
fn dot_lanes(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    let mut acc = [0.0f64; LANES];
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (u, v) in ca.zip(cb) {
        for k in 0..LANES {
            acc[k] += u[k] * v[k];
        }
    }

    let mut s = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
    for (u, v) in ra.iter().zip(rb) {
        s += u * v;
    }
    s
}

/// Squared Euclidean distance with `LANES` fixed accumulators (see `dot_lanes`).
#[inline]
fn sq_dist_lanes(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    let mut acc = [0.0f64; LANES];
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (u, v) in ca.zip(cb) {
        for k in 0..LANES {
            let d = u[k] - v[k];
            acc[k] += d * d;
        }
    }

    let mut s = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
    for (u, v) in ra.iter().zip(rb) {
        let d = u - v;
        s += d * d;
    }
    s
}

/// Column means in one row-major pass.
fn column_means(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut means = vec![0.0f64; p];
    for row in x.chunks_exact(p) {
        for c in 0..p {
            means[c] += row[c];
        }
    }
    let inv_n = 1.0 / (n as f64);
    for m in means.iter_mut() {
        *m *= inv_n;
    }
    means
}

/// Fill the upper triangle of `out` (diagonal included) with
/// `sum over rows of pair(column_i, column_j)`, columns optionally centered.
///
/// Rows are processed in blocks: each block is transposed (and centered) into
/// a small column-major scratch, every column pair is reduced on that block
/// with `pair`, and the per-block results are added to `out` in block order.
/// Extra memory is one block (about `BLOCK_ELEMS` f64s), independent of `n`.
///
/// Threads split the work by output row and never share an accumulator, so the
/// result does not depend on the thread count.
///
/// Requires `n >= 1` and `p >= 1`. The lower triangle is left zeroed; callers
/// finish with `finish_symmetric`.
fn blocked_pair_sums_out<F>(
    x: &[f64],
    n: usize,
    p: usize,
    means: Option<&[f64]>,
    out: &mut [f64],
    pair: F,
) where
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    debug_assert!(n >= 1 && p >= 1);

    let block_rows = ((BLOCK_ELEMS / p).max(MIN_BLOCK_ROWS) / LANES * LANES).min(n);
    let mut scratch = vec![0.0f64; block_rows * p];
    out.fill(0.0);

    for rows in x.chunks(block_rows * p) {
        let m = rows.len() / p;
        let w = &mut scratch[..m * p];
        match means {
            Some(mu) => {
                for (r, row) in rows.chunks_exact(p).enumerate() {
                    for c in 0..p {
                        w[c * m + r] = row[c] - mu[c];
                    }
                }
            }
            None => {
                for (r, row) in rows.chunks_exact(p).enumerate() {
                    for c in 0..p {
                        w[c * m + r] = row[c];
                    }
                }
            }
        }
        let w = &*w;
        let pair = &pair;

        let add_block = |(i, out_row): (usize, &mut [f64])| {
            let a = &w[i * m..(i + 1) * m];
            for j in i..p {
                out_row[j] += pair(a, &w[j * m..(j + 1) * m]);
            }
        };

        #[cfg(feature = "parallel")]
        if m * p * p >= PAR_MIN_WORK {
            // Coarse tasks: one output row is only a few microseconds of work.
            out.par_chunks_mut(p).enumerate().with_min_len((p / 32).max(1)).for_each(add_block);
            continue;
        }
        out.chunks_mut(p).enumerate().for_each(add_block);
    }
}

/// Map every upper-triangle entry through `finish` and mirror it below the diagonal.
fn finish_symmetric(out: &mut [f64], p: usize, finish: impl Fn(f64) -> f64) {
    for i in 0..p {
        for j in i..p {
            let v = finish(out[idx(i, j, p)]);
            out[idx(i, j, p)] = v;
            out[idx(j, i, p)] = v;
        }
    }
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

    let means = column_means(x, n, p);
    let denom = (n - 1) as f64;
    blocked_pair_sums_out(x, n, p, Some(&means), out, dot_lanes);
    finish_symmetric(out, p, |v| v / denom);
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

    let means = column_means(x, n, p);
    let denom = n as f64;
    blocked_pair_sums_out(x, n, p, Some(&means), out, dot_lanes);
    finish_symmetric(out, p, |v| v / denom);
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
    blocked_pair_sums_out(x, n, p, None, out, dot_lanes);
    finish_symmetric(out, p, |v| v / denom);
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

    // Per-column translation offsets (first finite value in each column).
    // Covariance is invariant under a constant shift of each column, and
    // shifting keeps the one-pass `sum_xy - sum_x*sum_y/m` formula from
    // catastrophically cancelling on large-offset data.
    let mut offs = vec![0.0f64; p];
    for c in 0..p {
        for r in 0..n {
            let v = x[r * p + c];
            if v.is_finite() {
                offs[c] = v;
                break;
            }
        }
    }
    let offs = &offs;

    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            let oi = offs[i];
            for j in i..p {
                let oj = offs[j];
                let mut m: usize = 0;
                let mut sum_x = 0.0f64;
                let mut sum_y = 0.0f64;
                let mut sum_xy = 0.0f64;

                for r in 0..n {
                    let base = r * p;
                    let xi = x[base + i];
                    let yj = x[base + j];
                    if xi.is_finite() && yj.is_finite() {
                        let xi = xi - oi;
                        let yj = yj - oj;
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
            let oi = offs[i];
            for j in i..p {
                let oj = offs[j];
                let mut m: usize = 0;
                let mut sum_x = 0.0f64;
                let mut sum_y = 0.0f64;
                let mut sum_xy = 0.0f64;

                for r in 0..n {
                    let base = r * p;
                    let xi = x[base + i];
                    let yj = x[base + j];
                    if xi.is_finite() && yj.is_finite() {
                        let xi = xi - oi;
                        let yj = yj - oj;
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
    if n == 0 {
        out.fill(0.0);
        return;
    }

    blocked_pair_sums_out(x, n, p, None, out, dot_lanes);
    finish_symmetric(out, p, |v| v);
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

    // Rows are already contiguous in row-major data, so no scratch is needed:
    // walk the columns in blocks narrow enough that all n row segments of a
    // block stay cached, adding each block's dot products in block order.
    // Segments shorter than a page make the row-to-row stride defeat the
    // prefetcher, hence the larger floor than the column kernels use.
    const MIN_BLOCK_COLS: usize = 512;
    let block_cols = ((BLOCK_ELEMS / n).max(MIN_BLOCK_COLS) / LANES * LANES).min(p);

    for c0 in (0..p).step_by(block_cols) {
        let c1 = (c0 + block_cols).min(p);

        let add_block = |(i, row): (usize, &mut [f64])| {
            let xi = &x[i * p + c0..i * p + c1];
            for j in i..n {
                row[j] += dot_lanes(xi, &x[j * p + c0..j * p + c1]);
            }
        };

        #[cfg(feature = "parallel")]
        if n * n * (c1 - c0) >= PAR_MIN_WORK {
            out.par_chunks_mut(n).enumerate().with_min_len((n / 32).max(1)).for_each(add_block);
            continue;
        }
        out.chunks_mut(n).enumerate().for_each(add_block);
    }

    finish_symmetric(out, n, |v| v);
}

/// Pairwise Euclidean distances between columns (p×p).
pub fn pairwise_euclidean_cols_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    if n == 0 {
        out.fill(0.0);
        return;
    }

    // Differences are squared directly (no Gram trick), so nearly identical
    // columns keep full relative accuracy.
    blocked_pair_sums_out(x, n, p, None, out, sq_dist_lanes);
    finish_symmetric(out, p, f64::sqrt);
    for i in 0..p {
        out[idx(i, i, p)] = 0.0;
    }
}

/// Pairwise cosine *distance* (1 - cosine similarity) between columns (p×p).
///
/// If a column has zero norm, distances involving it are NaN.
pub fn pairwise_cosine_cols_out(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(out.len(), p * p);

    if p == 0 {
        return;
    }
    if n == 0 {
        out.fill(f64::NAN);
        return;
    }

    // Upper triangle of X^T X; its diagonal holds the squared column norms.
    blocked_pair_sums_out(x, n, p, None, out, dot_lanes);
    let norms: Vec<f64> = (0..p).map(|i| out[idx(i, i, p)].sqrt()).collect();

    for i in 0..p {
        let ni = norms[i];
        out[idx(i, i, p)] = if ni == 0.0 { f64::NAN } else { 0.0 };

        for j in (i + 1)..p {
            let denom = ni * norms[j];
            let v = if denom == 0.0 || denom.is_nan() {
                f64::NAN
            } else {
                1.0 - out[idx(i, j, p)] / denom
            };
            out[idx(i, j, p)] = v;
            out[idx(j, i, p)] = v;
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The textbook triple loop the blocked kernel replaced: one running sum
    /// per column pair. Kept as the oracle for the blocked implementation.
    fn naive_cross(x: &[f64], n: usize, p: usize, center: bool, denom: f64) -> Vec<f64> {
        let means = if center { column_means(x, n, p) } else { vec![0.0; p] };
        let mut out = vec![0.0f64; p * p];
        for i in 0..p {
            for j in i..p {
                let mut acc = 0.0f64;
                for r in 0..n {
                    acc += (x[r * p + i] - means[i]) * (x[r * p + j] - means[j]);
                }
                out[idx(i, j, p)] = acc / denom;
                out[idx(j, i, p)] = acc / denom;
            }
        }
        out
    }

    /// Deterministic pseudo-random data in [-1, 1) + `offset`.
    fn lcg_data(len: usize, offset: f64) -> Vec<f64> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 52) as f64 - 1.0) + offset
            })
            .collect()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Entries agree to `tol` in correlation units (error / sqrt(var_i var_j)),
    /// which keeps near-zero covariances from dominating a relative test.
    fn assert_close(got: &[f64], want: &[f64], p: usize, tol: f64) {
        for i in 0..p {
            for j in 0..p {
                let (g, w) = (got[idx(i, j, p)], want[idx(i, j, p)]);
                if w.is_nan() {
                    assert!(g.is_nan(), "({i},{j}): expected NaN, got {g}");
                    continue;
                }
                let scale = (want[idx(i, i, p)] * want[idx(j, j, p)]).sqrt();
                let bound = tol * scale.max(f64::MIN_POSITIVE);
                assert!((g - w).abs() <= bound, "({i},{j}): {g} vs {w} (bound {bound:e})");
            }
        }
    }

    #[test]
    fn dot_lanes_matches_plain_dot_on_every_tail_length() {
        for len in 0..40 {
            let a = lcg_data(len, 0.0);
            let b = lcg_data(len, 0.5);
            let plain: f64 = a.iter().zip(&b).map(|(u, v)| u * v).sum();
            let norm: f64 = a.iter().zip(&b).map(|(u, v)| (u * v).abs()).sum();
            assert!((dot_lanes(&a, &b) - plain).abs() <= 1e-15 * norm.max(1.0), "len {len}");
        }
    }

    #[test]
    fn degenerate_shapes_keep_their_documented_fill() {
        let mut out = vec![7.0; 4];
        cov_matrix_out(&[1.0, 2.0], 1, 2, &mut out);
        assert!(out.iter().all(|v| v.is_nan()), "sample cov with n < 2 is NaN");

        let mut out = vec![7.0; 4];
        cov_matrix_centered_out(&[1.0, 2.0], 1, 2, &mut out);
        assert!(out.iter().all(|v| v.is_nan()));

        let mut out = vec![7.0; 4];
        cov_matrix_bias_out(&[], 0, 2, &mut out);
        assert!(out.iter().all(|v| v.is_nan()), "population cov with n == 0 is NaN");

        let mut out = vec![7.0; 4];
        cov_matrix_bias_out(&[1.0, 2.0], 1, 2, &mut out);
        assert_eq!(out, vec![0.0; 4], "population cov of one row is exactly 0");

        let mut out: Vec<f64> = vec![];
        cov_matrix_out(&[], 5, 0, &mut out); // p == 0 must not panic
    }

    #[test]
    fn constant_column_has_exactly_zero_covariance() {
        let (n, p) = (1000, 3);
        let mut x = lcg_data(n * p, 0.0);
        for r in 0..n {
            x[r * p + 1] = 7.5;
        }
        let mut out = vec![0.0; p * p];
        cov_matrix_out(&x, n, p, &mut out);
        for k in 0..p {
            assert_eq!(out[idx(1, k, p)], 0.0);
            assert_eq!(out[idx(k, 1, p)], 0.0);
        }
    }

    #[test]
    fn nan_poisons_exactly_the_rows_and_columns_it_touches() {
        let (n, p) = (300, 4);
        let mut x = lcg_data(n * p, 0.0);
        x[17 * p + 2] = f64::NAN;
        let mut out = vec![0.0; p * p];
        cov_matrix_out(&x, n, p, &mut out);
        for i in 0..p {
            for j in 0..p {
                assert_eq!(out[idx(i, j, p)].is_nan(), i == 2 || j == 2, "({i},{j})");
            }
        }
    }

    #[test]
    fn multi_block_shapes_match_the_naive_oracle() {
        // (n, p) chosen to cross block boundaries with ragged last blocks and
        // lane tails: tall-narrow, mid, and wide (MIN_BLOCK_ROWS-sized blocks).
        for &(n, p) in &[(20_003usize, 3usize), (9_001, 40), (333, 700)] {
            let x = lcg_data(n * p, 0.0);
            let mut out = vec![0.0; p * p];

            cov_matrix_out(&x, n, p, &mut out);
            assert_close(&out, &naive_cross(&x, n, p, true, (n - 1) as f64), p, 1e-12);

            cov_matrix_bias_out(&x, n, p, &mut out);
            assert_close(&out, &naive_cross(&x, n, p, true, n as f64), p, 1e-12);

            cov_matrix_centered_out(&x, n, p, &mut out);
            assert_close(&out, &naive_cross(&x, n, p, false, (n - 1) as f64), p, 1e-12);
        }
    }

    #[test]
    fn large_offset_data_stays_accurate() {
        // Mean 1e8, spread 1: the column mean is only good to ~1e-8 relative,
        // so the covariance must still agree with the unshifted data to 1e-7.
        let (n, p) = (50_000, 4);
        let mut got = vec![0.0; p * p];
        let mut want = vec![0.0; p * p];
        cov_matrix_out(&lcg_data(n * p, 1.0e8), n, p, &mut got);
        cov_matrix_out(&lcg_data(n * p, 0.0), n, p, &mut want);
        assert_close(&got, &want, p, 1e-7);
    }

    #[test]
    fn repeated_calls_are_bit_identical() {
        let (n, p) = (4_000, 120);
        let x = lcg_data(n * p, 0.0);
        let mut a = vec![0.0; p * p];
        let mut b = vec![1.0; p * p];
        cov_matrix_out(&x, n, p, &mut a);
        cov_matrix_out(&x, n, p, &mut b);
        assert_eq!(bits(&a), bits(&b));
    }

    /// Big enough per block to take the Rayon path, then compared bit-for-bit
    /// against the same call confined to pools of other sizes.
    #[cfg(feature = "parallel")]
    #[test]
    fn result_does_not_depend_on_thread_count() {
        let (n, p) = (4_000, 120);
        assert!(BLOCK_ELEMS / p * p * p >= PAR_MIN_WORK, "shape must exercise the parallel path");
        let x = lcg_data(n * p, 0.0);

        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut out = vec![0.0; p * p];
            pool.install(|| cov_matrix_out(&x, n, p, &mut out));
            bits(&out)
        };

        let one = run(1);
        assert_eq!(one, run(2));
        assert_eq!(one, run(7));
    }

    /// One running sum per pair, for the Gram and distance kernels.
    /// `kind`: 0 = X^T X, 1 = Euclidean distance, 2 = cosine distance.
    fn naive_cols(x: &[f64], n: usize, p: usize, kind: u8) -> Vec<f64> {
        let col = |c: usize| (0..n).map(move |r| x[r * p + c]);
        let dot = |i: usize, j: usize| col(i).zip(col(j)).fold(0.0, |a, (u, v)| a + u * v);
        let mut out = vec![0.0f64; p * p];
        for i in 0..p {
            for j in 0..p {
                out[idx(i, j, p)] = match kind {
                    0 => dot(i, j),
                    1 => col(i).zip(col(j)).fold(0.0, |a, (u, v)| a + (u - v) * (u - v)).sqrt(),
                    _ if dot(i, i) == 0.0 || dot(j, j) == 0.0 => f64::NAN,
                    _ if i == j => 0.0,
                    _ => 1.0 - dot(i, j) / (dot(i, i).sqrt() * dot(j, j).sqrt()),
                };
            }
        }
        out
    }

    /// |got - want| <= rel * |want| + abs, NaN matching NaN.
    fn assert_near(got: &[f64], want: &[f64], rel: f64, abs: f64) {
        assert_eq!(got.len(), want.len());
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "[{k}]: expected NaN, got {g}");
            } else {
                assert!((g - w).abs() <= rel * w.abs() + abs, "[{k}]: {g} vs {w}");
            }
        }
    }

    #[test]
    fn gram_and_distance_kernels_match_the_naive_oracle_across_blocks() {
        for &(n, p) in &[(20_003usize, 3usize), (5_001, 24), (500, 300)] {
            let x = lcg_data(n * p, 0.25);
            let mut out = vec![0.0; p * p];

            xtx_matrix_out(&x, n, p, &mut out);
            assert_close(&out, &naive_cols(&x, n, p, 0), p, 1e-12);

            pairwise_euclidean_cols_out(&x, n, p, &mut out);
            assert_near(&out, &naive_cols(&x, n, p, 1), 1e-12, 0.0);

            pairwise_cosine_cols_out(&x, n, p, &mut out);
            assert_near(&out, &naive_cols(&x, n, p, 2), 0.0, 1e-12);
        }
    }

    #[test]
    fn xxt_matches_xtx_of_the_transpose() {
        for &(n, p) in &[(7usize, 1usize), (60, 37), (25, 5_003)] {
            let x = lcg_data(n * p, 0.25);
            let mut xt = vec![0.0; n * p];
            for r in 0..n {
                for c in 0..p {
                    xt[c * n + r] = x[r * p + c];
                }
            }
            let mut got = vec![0.0; n * n];
            xxt_matrix_out(&x, n, p, &mut got);
            assert_close(&got, &naive_cols(&xt, p, n, 0), n, 1e-12);
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(got[i * n + j].to_bits(), got[j * n + i].to_bits());
                }
            }
        }
    }

    #[test]
    fn gram_and_distance_degenerate_shapes() {
        let mut out = vec![7.0; 4];
        xtx_matrix_out(&[], 0, 2, &mut out);
        assert_eq!(out, vec![0.0; 4], "X^T X of zero rows is zeros");

        let mut out = vec![7.0; 4];
        pairwise_euclidean_cols_out(&[], 0, 2, &mut out);
        assert_eq!(out, vec![0.0; 4]);

        let mut out = vec![7.0; 9];
        xxt_matrix_out(&[], 3, 0, &mut out);
        assert_eq!(out, vec![0.0; 9], "X X^T with zero columns is zeros");

        let mut none: Vec<f64> = vec![];
        xtx_matrix_out(&[], 4, 0, &mut none);
        xxt_matrix_out(&[], 0, 4, &mut none);
        pairwise_euclidean_cols_out(&[], 4, 0, &mut none);
        pairwise_cosine_cols_out(&[], 4, 0, &mut none);
    }

    #[test]
    fn identical_columns_are_at_distance_exactly_zero() {
        let (n, p) = (10_000, 3);
        let mut x = lcg_data(n * p, 1.0e6);
        for r in 0..n {
            x[r * p + 2] = x[r * p];
        }
        let mut out = vec![1.0; p * p];
        pairwise_euclidean_cols_out(&x, n, p, &mut out);
        assert_eq!(out[idx(0, 2, p)], 0.0);
        assert_eq!(out[idx(2, 0, p)], 0.0);
        assert!(out[idx(0, 1, p)] > 0.0);
        for i in 0..p {
            assert_eq!(out[idx(i, i, p)], 0.0);
        }
    }

    #[test]
    fn cosine_zero_norm_column_is_nan_everywhere_it_appears() {
        let (n, p) = (500, 4);
        let mut x = lcg_data(n * p, 0.25);
        for r in 0..n {
            x[r * p + 1] = 0.0;
        }
        let mut out = vec![0.0; p * p];
        pairwise_cosine_cols_out(&x, n, p, &mut out);
        for i in 0..p {
            for j in 0..p {
                let v = out[idx(i, j, p)];
                assert_eq!(v.is_nan(), i == 1 || j == 1, "({i},{j})");
                if i == j && i != 1 {
                    assert_eq!(v, 0.0);
                }
            }
        }
    }

    /// Every dense kernel, bit-for-bit across pool sizes.
    #[cfg(feature = "parallel")]
    #[test]
    fn gram_and_distance_results_do_not_depend_on_thread_count() {
        let (n, p) = (4_000, 120);
        let x = lcg_data(n * p, 0.25);
        let (rows, cols) = (300, 257);

        let kernels: [(&str, usize, &(dyn Fn(&mut [f64]) + Sync)); 4] = [
            ("xtx", p * p, &|out| xtx_matrix_out(&x, n, p, out)),
            ("euclidean", p * p, &|out| pairwise_euclidean_cols_out(&x, n, p, out)),
            ("cosine", p * p, &|out| pairwise_cosine_cols_out(&x, n, p, out)),
            ("xxt", rows * rows, &|out| xxt_matrix_out(&x[..rows * cols], rows, cols, out)),
        ];

        for (name, len, kernel) in kernels {
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let mut out = vec![0.0; len];
                pool.install(|| kernel(&mut out));
                bits(&out)
            };
            let one = run(1);
            assert_eq!(one, run(2), "{name}");
            assert_eq!(one, run(7), "{name}");
        }
    }

    proptest! {
        /// Any small finite matrix: symmetric, non-negative diagonal, and in
        /// agreement with the naive oracle for all three dense kernels.
        #[test]
        fn prop_blocked_matches_naive(
            n in 2usize..90,
            p in 1usize..9,
            vals in proptest::collection::vec(-1e6f64..1e6, 90 * 9),
        ) {
            let x = &vals[..n * p];
            let mut out = vec![0.0f64; p * p];

            cov_matrix_out(x, n, p, &mut out);
            for i in 0..p {
                prop_assert!(out[idx(i, i, p)] >= 0.0);
                for j in 0..p {
                    prop_assert_eq!(out[idx(i, j, p)].to_bits(), out[idx(j, i, p)].to_bits());
                }
            }
            assert_close(&out, &naive_cross(x, n, p, true, (n - 1) as f64), p, 1e-11);

            cov_matrix_bias_out(x, n, p, &mut out);
            assert_close(&out, &naive_cross(x, n, p, true, n as f64), p, 1e-11);

            cov_matrix_centered_out(x, n, p, &mut out);
            assert_close(&out, &naive_cross(x, n, p, false, (n - 1) as f64), p, 1e-11);
        }
    }
}
