// src/kernels/matrix/corr.rs
//
// Correlation matrix kernel (fills full symmetric matrix).

pub(crate) fn corr_matrix_out(
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
        use rayon::prelude::*;
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
