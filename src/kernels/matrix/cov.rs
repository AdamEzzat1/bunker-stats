use numpy::ndarray::ArrayView2;

/// Covariance matrix similar to numpy.cov(x, rowvar=False)
pub fn cov_matrix_view(arr: ArrayView2<f64>) -> Vec<Vec<f64>> {
    let (n_rows, n_cols) = arr.dim();
    if n_rows < 2 || n_cols == 0 {
        return vec![vec![f64::NAN; n_cols]; n_cols];
    }

    // column means
    let mut means = vec![0.0; n_cols];
    for j in 0..n_cols {
        let mut s = 0.0;
        for i in 0..n_rows {
            s += arr[(i, j)];
        }
        means[j] = s / (n_rows as f64);
    }

    // cov
    let denom = (n_rows as f64) - 1.0;
    let mut out = vec![vec![0.0; n_cols]; n_cols];
    for a in 0..n_cols {
        for b in a..n_cols {
            let mut s = 0.0;
            for i in 0..n_rows {
                let da = arr[(i, a)] - means[a];
                let db = arr[(i, b)] - means[b];
                s += da * db;
            }
            let v = s / denom;
            out[a][b] = v;
            out[b][a] = v;
        }
    }
    out
}
