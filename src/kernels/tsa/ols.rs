// src/kernels/tsa/ols.rs
//
// Least squares for the small regressions inside the unit-root and
// serial-correlation tests (a handful of columns, up to millions of rows).
//
// The tests need two things from a regression: the coefficients, and selected
// diagonal entries of (X'X)^-1 for standard errors. Forming X'X and solving it
// squares the condition number of X, which is large here by construction: a
// constant next to a trend 1..n, or next to the level of a series that sits
// far from zero. Householder QR works on X itself, so it loses about half as
// many digits.

/// Result of `ols_qr`.
pub(crate) struct OlsFit {
    /// Coefficients, one per column of X.
    pub beta: Vec<f64>,
    /// `y - X * beta`, one per row.
    pub residuals: Vec<f64>,
    /// Sum of squared residuals.
    pub rss: f64,
    /// Row-major k*k inverse of the triangular factor of the column-scaled X.
    r_inv: Vec<f64>,
    /// Column 2-norms X was scaled by.
    scale: Vec<f64>,
}

impl OlsFit {
    /// `(X'X)^-1[j, j]`: multiply by the residual variance for Var(beta_j).
    ///
    /// With X = QR, (X'X)^-1 = R^-1 R^-T, so the diagonal entry is the squared
    /// norm of row j of R^-1. No inverse of X'X is ever formed.
    pub fn xtx_inv_diag(&self, j: usize) -> f64 {
        let k = self.beta.len();
        let row = &self.r_inv[j * k..(j + 1) * k];
        let norm2: f64 = row[j..].iter().map(|v| v * v).sum();
        norm2 / (self.scale[j] * self.scale[j])
    }
}

/// Solve `min ||y - X beta||` by Householder QR.
///
/// `x` is row-major with `m` rows and `k` columns; `y` has `m` entries.
/// Columns are scaled to unit norm first, which removes conditioning that is
/// only due to units (a trend column of size n beside a constant of size 1).
///
/// Returns `None` when `m < k`, when any input is non-finite, or when X is
/// rank-deficient to working precision (for example a constant series, whose
/// lagged level is a multiple of the intercept column).
pub(crate) fn ols_qr(x: &[f64], m: usize, k: usize, y: &[f64]) -> Option<OlsFit> {
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(y.len(), m);
    if k == 0 || m < k {
        return None;
    }

    // Column-major working copy, scaled to unit column norms.
    let mut a = vec![0.0f64; m * k];
    let mut scale = vec![0.0f64; k];
    for (r, row) in x.chunks_exact(k).enumerate() {
        for c in 0..k {
            a[c * m + r] = row[c];
        }
    }
    for c in 0..k {
        let col = &mut a[c * m..(c + 1) * m];
        let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            return None;
        }
        for v in col.iter_mut() {
            *v /= norm;
        }
        scale[c] = norm;
    }
    let mut qty = y.to_vec();

    // Unit columns keep every |R[j][j]| <= 1, so this is a relative tolerance.
    let rank_tol = (m as f64) * f64::EPSILON;

    let mut r = vec![0.0f64; k * k]; // row-major upper triangle
    for j in 0..k {
        // Householder vector for column j, rows j..m.
        let (done, rest) = a.split_at_mut((j + 1) * m);
        let col = &mut done[j * m..];
        let norm = col[j..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(norm > rank_tol) {
            return None;
        }
        let alpha = if col[j] > 0.0 { -norm } else { norm };
        col[j] -= alpha;
        let v = &col[j..];
        let vtv: f64 = v.iter().map(|t| t * t).sum();

        // Reflect the remaining columns and y:  z -= 2 v (v.z) / (v.v)
        for other in rest.chunks_exact_mut(m) {
            let z = &mut other[j..];
            let f = 2.0 * v.iter().zip(z.iter()).map(|(p, q)| p * q).sum::<f64>() / vtv;
            for (zi, vi) in z.iter_mut().zip(v) {
                *zi -= f * vi;
            }
        }
        let z = &mut qty[j..];
        let f = 2.0 * v.iter().zip(z.iter()).map(|(p, q)| p * q).sum::<f64>() / vtv;
        for (zi, vi) in z.iter_mut().zip(v) {
            *zi -= f * vi;
        }

        r[j * k + j] = alpha;
        for c in (j + 1)..k {
            r[j * k + c] = a[c * m + j];
        }
    }

    // R^-1 by back substitution, one column at a time (k is small).
    let mut r_inv = vec![0.0f64; k * k];
    for c in 0..k {
        r_inv[c * k + c] = 1.0 / r[c * k + c];
        for i in (0..c).rev() {
            let s: f64 = ((i + 1)..=c).map(|l| r[i * k + l] * r_inv[l * k + c]).sum();
            r_inv[i * k + c] = -s / r[i * k + i];
        }
    }

    // beta (scaled) = R^-1 (Q'y)[..k], then undo the column scaling.
    let mut beta = vec![0.0f64; k];
    for i in 0..k {
        let b: f64 = (i..k).map(|c| r_inv[i * k + c] * qty[c]).sum();
        beta[i] = b / scale[i];
    }
    if beta.iter().any(|b| !b.is_finite()) {
        return None;
    }

    let mut residuals = Vec::with_capacity(m);
    let mut rss = 0.0f64;
    for (row, &yi) in x.chunks_exact(k).zip(y) {
        let fitted: f64 = row.iter().zip(&beta).map(|(xv, b)| xv * b).sum();
        let e = yi - fitted;
        rss += e * e;
        residuals.push(e);
    }
    if !rss.is_finite() {
        return None;
    }

    Some(OlsFit { beta, residuals, rss, r_inv, scale })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn lcg(len: usize) -> Vec<f64> {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    /// Textbook route this module replaces: form X'X, eliminate, invert.
    fn normal_equations(x: &[f64], m: usize, k: usize, y: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let mut g = vec![0.0f64; k * 2 * k]; // [X'X | I], then Gauss-Jordan
        let mut xty = vec![0.0f64; k];
        for (row, &yi) in x.chunks_exact(k).zip(y) {
            for i in 0..k {
                xty[i] += row[i] * yi;
                for j in 0..k {
                    g[i * 2 * k + j] += row[i] * row[j];
                }
            }
        }
        for i in 0..k {
            g[i * 2 * k + k + i] = 1.0;
        }
        for c in 0..k {
            let piv = (c..k).max_by(|&p, &q| g[p * 2 * k + c].abs().total_cmp(&g[q * 2 * k + c].abs()))?;
            if g[piv * 2 * k + c] == 0.0 {
                return None;
            }
            for j in 0..2 * k {
                g.swap(c * 2 * k + j, piv * 2 * k + j);
            }
            let d = g[c * 2 * k + c];
            for j in 0..2 * k {
                g[c * 2 * k + j] /= d;
            }
            for r in 0..m.min(k) {
                if r != c {
                    let f = g[r * 2 * k + c];
                    for j in 0..2 * k {
                        g[r * 2 * k + j] -= f * g[c * 2 * k + j];
                    }
                }
            }
        }
        let beta = (0..k).map(|i| (0..k).map(|j| g[i * 2 * k + k + j] * xty[j]).sum()).collect();
        let diag = (0..k).map(|i| g[i * 2 * k + k + i]).collect();
        Some((beta, diag))
    }

    #[test]
    fn recovers_exact_coefficients_and_matches_the_explicit_inverse() {
        let (m, k) = (200, 4);
        let x = lcg(m * k);
        let truth = [1.5, -2.0, 0.25, 3.0];
        let y: Vec<f64> = x.chunks_exact(k).map(|r| r.iter().zip(&truth).map(|(a, b)| a * b).sum()).collect();

        let fit = ols_qr(&x, m, k, &y).unwrap();
        for (b, t) in fit.beta.iter().zip(&truth) {
            assert!((b - t).abs() < 1e-12, "{b} vs {t}");
        }
        assert!(fit.rss < 1e-22);
        assert_eq!(fit.residuals.len(), m);

        // Well-conditioned here, so the normal equations are a fair reference.
        let (_, diag) = normal_equations(&x, m, k, &y).unwrap();
        for j in 0..k {
            let got = fit.xtx_inv_diag(j);
            assert!((got - diag[j]).abs() < 1e-10 * diag[j], "diag {j}: {got} vs {}", diag[j]);
        }
    }

    /// The shape the unit-root tests produce: intercept, trend, and a level
    /// that sits far from zero. The fit is exact by construction, so any error
    /// is the solver's.
    #[test]
    fn stays_accurate_where_the_normal_equations_lose_their_digits() {
        let m = 2_000;
        let noise = lcg(m);
        let mut x = Vec::with_capacity(m * 3);
        for t in 0..m {
            x.extend_from_slice(&[1.0, (t + 1) as f64, 1.0e7 + 0.01 * (t as f64) + noise[t]]);
        }
        let truth = [3.0, -0.5, 2.0];
        let y: Vec<f64> = x.chunks_exact(3).map(|r| r.iter().zip(&truth).map(|(a, b)| a * b).sum()).collect();

        // Judge the coefficient the tests actually use: the one on the level.
        // (The intercept absorbs level-coefficient error times 1e7, so its own
        // relative error says more about the problem than about the solver.)
        let fit = ols_qr(&x, m, 3, &y).unwrap();
        let qr_err = ((fit.beta[2] - truth[2]) / truth[2]).abs();
        assert!(qr_err < 1e-9, "QR relative error {qr_err:e}");

        // Not asserted against a fixed bound (it is allowed to be bad); it must
        // just not beat QR, which is the reason this module exists.
        if let Some((ne_beta, _)) = normal_equations(&x, m, 3, &y) {
            let ne_err = ((ne_beta[2] - truth[2]) / truth[2]).abs();
            assert!(ne_err.is_nan() || ne_err >= qr_err, "normal equations {ne_err:e} vs QR {qr_err:e}");
        }
    }

    #[test]
    fn column_scale_does_not_change_the_fit() {
        let (m, k) = (300, 3);
        let x = lcg(m * k);
        let y = lcg(m);
        let base = ols_qr(&x, m, k, &y).unwrap();

        let factors = [1.0e9, 1.0e-6, 42.0];
        let scaled: Vec<f64> = x.chunks_exact(k).flat_map(|r| r.iter().zip(&factors).map(|(v, f)| v * f)).collect();
        let fit = ols_qr(&scaled, m, k, &y).unwrap();

        for j in 0..k {
            assert!((fit.beta[j] * factors[j] - base.beta[j]).abs() < 1e-11 * base.beta[j].abs().max(1.0));
            let want = base.xtx_inv_diag(j) / (factors[j] * factors[j]);
            assert!((fit.xtx_inv_diag(j) - want).abs() < 1e-10 * want);
        }
        assert!((fit.rss - base.rss).abs() < 1e-10 * base.rss);
    }

    #[test]
    fn rank_deficient_and_degenerate_designs_are_rejected() {
        let m = 50;
        // Third column = 2 * first: exactly collinear.
        let x: Vec<f64> = lcg(m * 2).chunks_exact(2).flat_map(|r| [r[0], r[1], 2.0 * r[0]]).collect();
        assert!(ols_qr(&x, m, 3, &lcg(m)).is_none());

        // A zero column, more columns than rows, and non-finite input.
        let x: Vec<f64> = lcg(m).iter().flat_map(|&v| [v, 0.0]).collect();
        assert!(ols_qr(&x, m, 2, &lcg(m)).is_none());
        assert!(ols_qr(&lcg(6), 2, 3, &lcg(2)).is_none());
        let mut x = lcg(m * 2);
        x[7] = f64::NAN;
        assert!(ols_qr(&x, m, 2, &lcg(m)).is_none());
    }

    proptest! {
        /// Residuals of a least-squares fit are orthogonal to every column.
        #[test]
        fn prop_residuals_are_orthogonal_to_the_columns(
            m in 6usize..60,
            k in 1usize..5,
            xs in proptest::collection::vec(-100.0f64..100.0, 60 * 5),
            ys in proptest::collection::vec(-100.0f64..100.0, 60),
        ) {
            let x = &xs[..m * k];
            let y = &ys[..m];
            if let Some(fit) = ols_qr(x, m, k, y) {
                let ynorm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
                for c in 0..k {
                    let dot: f64 = x.chunks_exact(k).zip(&fit.residuals).map(|(r, e)| r[c] * e).sum();
                    let cnorm = x.chunks_exact(k).map(|r| r[c] * r[c]).sum::<f64>().sqrt();
                    prop_assert!(dot.abs() <= 1e-9 * cnorm * ynorm.max(1.0), "column {}: {}", c, dot);
                }
                let rss: f64 = fit.residuals.iter().map(|e| e * e).sum();
                prop_assert!((rss - fit.rss).abs() <= 1e-12 * rss.max(1.0));
            }
        }
    }
}
