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

    // init first window
    for r in 0..window {
        let base = r * n_cols;
        for jb in (0..n_cols).step_by(64) {
            let j_end = (jb + 64).min(n_cols);
            for j in jb..j_end {
                let v = x[base + j];
                sum[j] += v;
                sumsq[j] += v * v;
            }
        }
    }

    // first output row
    let denom = (window as f64 - 1.0).max(1.0);
    for jb in (0..n_cols).step_by(64) {
        let j_end = (jb + 64).min(n_cols);
        for j in jb..j_end {
            let _mean = sum[j] / window as f64;
            let var = (sumsq[j] - (sum[j] * sum[j]) / window as f64) / denom;
            out[j] = var.max(0.0).sqrt();
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
                let vn = x[base_new + j];
                let vo = x[base_old + j];
                sum[j] += vn - vo;
                sumsq[j] += vn * vn - vo * vo;

                let _mean = sum[j] / window as f64;
                let var = (sumsq[j] - (sum[j] * sum[j]) / window as f64) / denom;
                out[out_r * n_cols + j] = var.max(0.0).sqrt();
            }
        }
    }
    out
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

    // init first window
    for r in 0..window {
        let base = r * n_cols;
        for jb in (0..n_cols).step_by(64) {
            let j_end = (jb + 64).min(n_cols);
            for j in jb..j_end {
                let v = x[base + j];
                sum[j] += v;
                sumsq[j] += v * v;
            }
        }
    }

    let denom = (window as f64 - 1.0).max(1.0);
    for jb in (0..n_cols).step_by(64) {
        let j_end = (jb + 64).min(n_cols);
        for j in jb..j_end {
            let mean = sum[j] / window as f64;
            let var = (sumsq[j] - (sum[j] * sum[j]) / window as f64) / denom;
            means_out[j] = mean;
            stds_out[j] = var.max(0.0).sqrt();
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
                let vn = x[base_new + j];
                let vo = x[base_old + j];
                sum[j] += vn - vo;
                sumsq[j] += vn * vn - vo * vo;

                let mean = sum[j] / window as f64;
                let var = (sumsq[j] - (sum[j] * sum[j]) / window as f64) / denom;
                means_out[out_r * n_cols + j] = mean;
                stds_out[out_r * n_cols + j] = var.max(0.0).sqrt();
            }
        }
    }
    (means_out, stds_out)
}