//! Fused multi-statistic rolling kernel (2D, axis=0).

use super::bounds::output_length;
use super::config::RollingConfig;
use super::masks::StatsMask;
use super::multi::rolling_multi_into;

/// Compute rolling statistics along axis 0 (column-wise).
///
/// Input data is in row-major order: slice of length (n_rows * n_cols).
/// Each column is processed independently.
///
/// # Arguments
/// - `data`: input array (row-major, length = n_rows * n_cols)
/// - `n_rows`: number of rows
/// - `n_cols`: number of columns
/// - `config`: rolling window configuration
/// - `mask`: which statistics to compute
/// - Output slices (all in row-major order):
///   - Each has length = out_rows * n_cols
///   - out_rows depends on alignment
///
/// # Performance
/// - Processes each column independently (no parallelization by default)
/// - Can add column-level parallelism threshold in future
pub fn rolling_multi_axis0_into(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    config: &RollingConfig,
    mask: StatsMask,
    out_mean: &mut [f64],
    out_std: &mut [f64],
    out_var: &mut [f64],
    out_count: &mut [f64],
    out_min: &mut [f64],
    out_max: &mut [f64],
) {
    let window = config.window;
    if window == 0 || window > n_rows || n_cols == 0 {
        return;
    }
    
    let out_rows = output_length(n_rows, window, config.alignment);
    
    // Validate output lengths
    let expected_len = out_rows * n_cols;
    if mask.has_mean() {
        assert_eq!(out_mean.len(), expected_len, "out_mean length mismatch");
    }
    if mask.has_std() {
        assert_eq!(out_std.len(), expected_len, "out_std length mismatch");
    }
    if mask.has_var() {
        assert_eq!(out_var.len(), expected_len, "out_var length mismatch");
    }
    if mask.has_count() {
        assert_eq!(out_count.len(), expected_len, "out_count length mismatch");
    }
    if mask.has_min() {
        assert_eq!(out_min.len(), expected_len, "out_min length mismatch");
    }
    if mask.has_max() {
        assert_eq!(out_max.len(), expected_len, "out_max length mismatch");
    }
    
    // Extract each column and process independently
    let mut col_buffer = vec![0.0; n_rows];
    let mut col_out_mean = vec![0.0; out_rows];
    let mut col_out_std = vec![0.0; out_rows];
    let mut col_out_var = vec![0.0; out_rows];
    let mut col_out_count = vec![0.0; out_rows];
    let mut col_out_min = vec![0.0; out_rows];
    let mut col_out_max = vec![0.0; out_rows];
    
    for col_idx in 0..n_cols {
        // Clear buffers to prevent state leakage between columns
        col_buffer.fill(0.0);
        if mask.has_mean() { col_out_mean.fill(0.0); }
        if mask.has_std() { col_out_std.fill(0.0); }
        if mask.has_var() { col_out_var.fill(0.0); }
        if mask.has_count() { col_out_count.fill(0.0); }
        if mask.has_min() { col_out_min.fill(0.0); }
        if mask.has_max() { col_out_max.fill(0.0); }
        
        // Extract column
        for row in 0..n_rows {
            col_buffer[row] = data[row * n_cols + col_idx];
        }
        
        // Process column
        rolling_multi_into(
            &col_buffer,
            config,
            mask,
            if mask.has_mean() { &mut col_out_mean } else { &mut [] },
            if mask.has_std() { &mut col_out_std } else { &mut [] },
            if mask.has_var() { &mut col_out_var } else { &mut [] },
            if mask.has_count() { &mut col_out_count } else { &mut [] },
            if mask.has_min() { &mut col_out_min } else { &mut [] },
            if mask.has_max() { &mut col_out_max } else { &mut [] },
        );
        
        // Write results back (row-major)
        for out_row in 0..out_rows {
            let out_idx = out_row * n_cols + col_idx;
            
            if mask.has_mean() {
                out_mean[out_idx] = col_out_mean[out_row];
            }
            if mask.has_std() {
                out_std[out_idx] = col_out_std[out_row];
            }
            if mask.has_var() {
                out_var[out_idx] = col_out_var[out_row];
            }
            if mask.has_count() {
                out_count[out_idx] = col_out_count[out_row];
            }
            if mask.has_min() {
                out_min[out_idx] = col_out_min[out_row];
            }
            if mask.has_max() {
                out_max[out_idx] = col_out_max[out_row];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::{Alignment, NanPolicy, RollingConfig};
    
    #[test]
    fn test_axis0_basic() {
        // 2x3 matrix (2 rows, 3 cols)
        // Row 0: [1, 2, 3]
        // Row 1: [4, 5, 6]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        
        let config = RollingConfig::trailing(2);
        let mask = StatsMask::MEAN;
        
        // Output will be 1x3 (1 row, 3 cols)
        let mut out_mean = vec![0.0; 3];
        
        rolling_multi_axis0_into(
            &data,
            2, 3,
            &config,
            mask,
            &mut out_mean,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        
        // Col 0: [1, 4] -> mean = 2.5
        assert!((out_mean[0] - 2.5).abs() < 1e-10);
        // Col 1: [2, 5] -> mean = 3.5
        assert!((out_mean[1] - 3.5).abs() < 1e-10);
        // Col 2: [3, 6] -> mean = 4.5
        assert!((out_mean[2] - 4.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_axis0_centered() {
        // 5x2 matrix (5 rows, 2 cols)
        let data = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
            5.0, 50.0,
        ];
        
        let config = RollingConfig::centered(3);
        let mask = StatsMask::MEAN;
        
        // Output will be 5x2 (same shape)
        let mut out_mean = vec![0.0; 10];
        
        rolling_multi_axis0_into(
            &data,
            5, 2,
            &config,
            mask,
            &mut out_mean,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        
        // Row-major layout: [row0_col0, row0_col1, row1_col0, row1_col1, ...]
        // Position 0, col 0: window [1, 2] -> 1.5
        assert!((out_mean[0] - 1.5).abs() < 1e-10);
        // Position 0, col 1: window [10, 20] -> 15.0
        assert!((out_mean[1] - 15.0).abs() < 1e-10);
    }
}