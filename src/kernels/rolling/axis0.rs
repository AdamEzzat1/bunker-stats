/// Axis-0 rolling kernels (pure Rust helpers)
///
/// Each column is gathered into a contiguous scratch and run through the same
/// drift-free window driver as the 1-D kernels.

use super::engine::{for_each_window, strict_std};
use super::moments::{Plain, RollingMoments};

/// Run `emit(out_row, column, moments)` for every window of every column.
fn for_each_column_window(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    window: usize,
    mut emit: impl FnMut(usize, usize, &RollingMoments<Plain>),
) {
    let mut col = vec![0.0f64; n_rows];
    for j in 0..n_cols {
        for (r, v) in col.iter_mut().enumerate() {
            *v = x[r * n_cols + j];
        }
        for_each_window::<Plain>(&col, window, |k, m| emit(k, j, m));
    }
}

pub(crate) fn rolling_mean_axis0_vec(x: &[f64], n_rows: usize, n_cols: usize, window: usize) -> Vec<f64> {
    if window == 0 || window > n_rows || n_cols == 0 {
        return Vec::new();
    }
    let out_rows = n_rows - window + 1;
    let mut out = vec![0.0f64; out_rows * n_cols];
    for_each_column_window(x, n_rows, n_cols, window, |k, j, m| {
        out[k * n_cols + j] = m.mean(false);
    });
    out
}

pub(crate) fn rolling_std_axis0_vec(x: &[f64], n_rows: usize, n_cols: usize, window: usize) -> Vec<f64> {
    if window == 0 || window > n_rows || n_cols == 0 {
        return Vec::new();
    }
    let out_rows = n_rows - window + 1;
    let mut out = vec![0.0f64; out_rows * n_cols];
    for_each_column_window(x, n_rows, n_cols, window, |k, j, m| {
        out[k * n_cols + j] = strict_std(m);
    });
    out
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
    for_each_column_window(x, n_rows, n_cols, window, |k, j, m| {
        means_out[k * n_cols + j] = m.mean(false);
        stds_out[k * n_cols + j] = strict_std(m);
    });
    (means_out, stds_out)
}
