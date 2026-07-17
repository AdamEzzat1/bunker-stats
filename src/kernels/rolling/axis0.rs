/// Axis-0 rolling kernels (pure Rust helpers)

pub(crate) fn rolling_mean_axis0_vec(x: &[f64], n_rows: usize, n_cols: usize, window: usize) -> Vec<f64> {
    if window == 0 || window > n_rows || n_cols == 0 {
        return Vec::new();
    }
    let out_rows = n_rows - window + 1;
    let mut out = vec![0.0f64; out_rows * n_cols];

    // running column sums
    let mut sum = vec![0.0f64; n_cols];

    // init first window
    for r in 0..window {
        let base = r * n_cols;
        for jb in (0..n_cols).step_by(64) {
            let j_end = (jb + 64).min(n_cols);
            for j in jb..j_end {
                sum[j] += x[base + j];
            }
        }
    }

    // first output row
    for jb in (0..n_cols).step_by(64) {
        let j_end = (jb + 64).min(n_cols);
        for j in jb..j_end {
            out[j] = sum[j] / window as f64;
        }
    }

    // slide
    for out_r in 1..out_rows {
        let r_new = out_r + window - 1;
        let r_old = out_r - 1;
        let base_new = r_new * n_cols;
        let base_old = r_old * n_cols;

        for jb in (0..n_cols).step_by(64) {
            let j_end = (jb + 64).min(n_cols);
            for j in jb..j_end {
                sum[j] += x[base_new + j] - x[base_old + j];
                out[out_r * n_cols + j] = sum[j] / window as f64;
            }
        }
    }
    out
}
pub(crate) fn rolling_std_axis0_vec(x: &[f64], n_rows: usize, n_cols: usize, window: usize) -> Vec<f64> {
    if window == 0 || window > n_rows || n_cols == 0 {
        return Vec::new();
    }
    let out_rows = n_rows - window + 1;
    let mut out = vec![0.0f64; out_rows * n_cols];

    let mut sum = vec![0.0f64; n_cols];
    let mut sumsq = vec![0.0f64; n_cols];

    // Per-column finite offset. Variance is translation-invariant, so shifting
    // each column by a value near its magnitude keeps `sumsq - sum^2/w` from
    // catastrophically cancelling on large-offset data. Must be finite so a NaN
    // in row 0 does not poison every window of that column.
    let off = column_offsets(x, n_cols);

    // init first window
    for r in 0..window {
        let base = r * n_cols;
        for jb in (0..n_cols).step_by(64) {
            let j_end = (jb + 64).min(n_cols);
            for j in jb..j_end {
                let v = x[base + j] - off[j];
                sum[j] += v;
                sumsq[j] += v * v;
            }
        }
    }

    // first output row
    for jb in (0..n_cols).step_by(64) {
        let j_end = (jb + 64).min(n_cols);
        for j in jb..j_end {
            out[j] = std_from_moments(sum[j], sumsq[j], window);
        }
    }

    for out_r in 1..out_rows {
        let r_new = out_r + window - 1;
        let r_old = out_r - 1;
        let base_new = r_new * n_cols;
        let base_old = r_old * n_cols;
        for jb in (0..n_cols).step_by(64) {
            let j_end = (jb + 64).min(n_cols);
            for j in jb..j_end {
                let vn = x[base_new + j] - off[j];
                let vo = x[base_old + j] - off[j];
                sum[j] += vn - vo;
                sumsq[j] += vn * vn - vo * vo;

                out[out_r * n_cols + j] = std_from_moments(sum[j], sumsq[j], window);
            }
        }
    }
    out
}

/// Finite per-column offset taken from row 0 (0.0 where row 0 is non-finite).
#[inline]
fn column_offsets(x: &[f64], n_cols: usize) -> Vec<f64> {
    let mut off = vec![0.0f64; n_cols];
    for j in 0..n_cols {
        let v0 = x[j];
        off[j] = if v0.is_finite() { v0 } else { 0.0 };
    }
    off
}

/// Sample std (ddof=1) from window sum/sumsq of offset-shifted values. Returns
/// NaN for window<2 or when the window contains a NaN (never masks NaN as 0).
#[inline]
fn std_from_moments(sum: f64, sumsq: f64, window: usize) -> f64 {
    if window < 2 {
        return f64::NAN;
    }
    let var = (sumsq - (sum * sum) / window as f64) / ((window - 1) as f64);
    if var.is_nan() {
        f64::NAN
    } else {
        var.max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Column-wise rolling std over large-offset data. Each column is three
    /// consecutive integers + 1e8; sample std of any 3 consecutive integers is
    /// exactly 1.0 regardless of offset. The old naive `sumsq - sum^2/w` lost all
    /// precision at 1e8 and the `.max(0.0)` masked the garbage.
    #[test]
    fn test_std_axis0_large_offset() {
        // 4 rows x 2 cols, row-major: col c row r = 1e8 + r + 1 (+ 10*c to differ)
        let n_rows = 4;
        let n_cols = 2;
        let mut x = vec![0.0; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                x[r * n_cols + c] = 1e8 + (r as f64 + 1.0) + 10.0 * c as f64;
            }
        }
        let (means, stds) = rolling_mean_std_axis0_vec(&x, n_rows, n_cols, 3);
        // out_rows = 2, each std entry must be 1.0
        for s in &stds {
            assert!((s - 1.0).abs() < 1e-9, "std={s}, expected 1.0");
        }
        // mean of rows 1..=3 for col 0 = 1e8 + 2
        assert!((means[0] - (1e8 + 2.0)).abs() < 1e-6);
    }

    /// window=1 std must be NaN (pandas semantics), not a masked 0.
    #[test]
    fn test_std_axis0_window_1_is_nan() {
        let x = vec![1.0, 2.0, 3.0];
        let stds = rolling_std_axis0_vec(&x, 3, 1, 1);
        assert!(stds.iter().all(|s| s.is_nan()));
    }
}
pub(crate) fn rolling_mean_std_axis0_vec(x: &[f64], n_rows: usize, n_cols: usize, window: usize) -> (Vec<f64>, Vec<f64>) {
    if window == 0 || window > n_rows || n_cols == 0 {
        return (Vec::new(), Vec::new());
    }
    let out_rows = n_rows - window + 1;
    let mut means_out = vec![0.0f64; out_rows * n_cols];
    let mut stds_out = vec![0.0f64; out_rows * n_cols];

    let mut sum = vec![0.0f64; n_cols];
    let mut sumsq = vec![0.0f64; n_cols];

    // Per-column finite offset (see rolling_std_axis0_vec).
    let off = column_offsets(x, n_cols);

    // init first window
    for r in 0..window {
        let base = r * n_cols;
        for jb in (0..n_cols).step_by(64) {
            let j_end = (jb + 64).min(n_cols);
            for j in jb..j_end {
                let v = x[base + j] - off[j];
                sum[j] += v;
                sumsq[j] += v * v;
            }
        }
    }

    for jb in (0..n_cols).step_by(64) {
        let j_end = (jb + 64).min(n_cols);
        for j in jb..j_end {
            means_out[j] = sum[j] / window as f64 + off[j];
            stds_out[j] = std_from_moments(sum[j], sumsq[j], window);
        }
    }

    for out_r in 1..out_rows {
        let r_new = out_r + window - 1;
        let r_old = out_r - 1;
        let base_new = r_new * n_cols;
        let base_old = r_old * n_cols;
        for jb in (0..n_cols).step_by(64) {
            let j_end = (jb + 64).min(n_cols);
            for j in jb..j_end {
                let vn = x[base_new + j] - off[j];
                let vo = x[base_old + j] - off[j];
                sum[j] += vn - vo;
                sumsq[j] += vn * vn - vo * vo;

                means_out[out_r * n_cols + j] = sum[j] / window as f64 + off[j];
                stds_out[out_r * n_cols + j] = std_from_moments(sum[j], sumsq[j], window);
            }
        }
    }
    (means_out, stds_out)
}