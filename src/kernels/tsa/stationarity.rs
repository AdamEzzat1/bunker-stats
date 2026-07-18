use numpy::{PyReadonlyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ======================
// OPTIMIZED STATIONARITY TESTS
// ======================

/// Augmented Dickey–Fuller test (simplified: DF with intercept only; no extra lags by default).
///
/// OPTIMIZED: Single-pass calculation of mean and sums
/// Python signature:
///     adf_test(x, regression="c", max_lag=None) -> (statistic, pvalue)
// ======================
// DICKEY-FULLER CRITICAL VALUES
// ======================

/// Approximate p-value for a Dickey-Fuller t-statistic via the MacKinnon
/// (1994) regression surface — the same method as statsmodels' `mackinnonp`
/// (N=1): p = Φ(c₀ + c₁τ + c₂τ² [+ c₃τ³]), with a small-p polynomial below τ*
/// and a large-p polynomial above it, clamped to exactly 0/1 outside the
/// tabulated range. Replaces a coarse critical-value interpolation whose tail
/// clamp (min p = 5e-4) and two-sided-Normal "nc" branch both diverged from
/// statsmodels.
fn df_pvalue(stat: f64, _n: usize, regression: &str) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};

    // (tau_min, tau_max, tau_star, small-p coeffs, large-p coeffs), constants
    // from statsmodels.tsa.adfvalues (scaling already applied).
    let (tau_min, tau_max, tau_star, smallp, largep): (f64, f64, f64, [f64; 3], [f64; 4]) =
        match regression {
            "nc" => (
                -19.04, 1.51, -1.04,
                [0.6344, 1.2378, 0.032496],
                [0.4797, 0.93557, -0.06999, 0.033066],
            ),
            "ct" => (
                -16.18, 0.70, -2.21,
                [3.2512, 1.6047, 0.049588],
                [2.5261, 0.61654, -0.37956, -0.060285],
            ),
            // "c" and any fallback
            _ => (
                -18.83, 2.74, -1.61,
                [2.1659, 1.4412, 0.038269],
                [1.7339, 0.93202, -0.12745, -0.010368],
            ),
        };

    if stat > tau_max {
        return 1.0;
    }
    if stat < tau_min {
        return 0.0;
    }
    let z = if stat <= tau_star {
        smallp[0] + smallp[1] * stat + smallp[2] * stat * stat
    } else {
        largep[0] + largep[1] * stat + largep[2] * stat * stat + largep[3] * stat * stat * stat
    };
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.cdf(z)
}
/// Normalize a Dickey-Fuller deterministic specification.
/// "c" = constant, "ct" = constant+trend, "n"/"nc" = none.
fn normalize_reg(reg: &str) -> PyResult<&'static str> {
    match reg {
        "c" | "" => Ok("c"),
        "ct" => Ok("ct"),
        "n" | "nc" => Ok("nc"),
        other => Err(PyValueError::new_err(format!(
            "regression must be one of 'c', 'ct', 'n' (got '{other}')"
        ))),
    }
}

/// OLS for the (augmented) Dickey-Fuller regression:
///   Δy_t = [deterministic] + β·y_{t-1} + Σ_{i=1}^p γ_i·Δy_{t-i} + ε_t
/// where deterministic is {} (nc), {1} (c), or {1, t} (ct). The test statistic
/// is the t-ratio on β. Returns
///   (t_stat, se_β, β, residuals, m_obs, k_params, regression_stderr_s)
/// or None if the design is rank-deficient / too small.
fn df_ols(x: &[f64], reg: &str, p: usize) -> Option<(f64, f64, f64, Vec<f64>, usize, usize, f64)> {
    use nalgebra::{DMatrix, DVector};
    let n = x.len();
    let det = match reg {
        "ct" => 2usize,
        "c" => 1,
        _ => 0, // "nc"
    };
    if n <= p + 2 {
        return None;
    }
    let start = p + 1; // first t with a full lag set and y_{t-1}
    let m = n - start; // regression rows
    let k = det + 1 + p; // params: deterministic + y_{t-1} + p lags
    if m <= k {
        return None;
    }
    let ylag_idx = det; // column index of the y_{t-1} coefficient

    let mut y = DVector::zeros(m);
    let mut xd = vec![0.0f64; m * k]; // row-major design
    for (row, t) in (start..n).enumerate() {
        y[row] = x[t] - x[t - 1];
        let base = row * k;
        let mut col = 0;
        if det >= 1 {
            xd[base + col] = 1.0; // constant
            col += 1;
        }
        if det == 2 {
            xd[base + col] = t as f64; // linear trend
            col += 1;
        }
        xd[base + col] = x[t - 1]; // y_{t-1}
        col += 1;
        for i in 1..=p {
            xd[base + col] = x[t - i] - x[t - i - 1]; // Δy_{t-i}
            col += 1;
        }
    }

    let xm = DMatrix::from_row_slice(m, k, &xd);
    let xtx = xm.transpose() * &xm;
    let xty = xm.transpose() * &y;
    let beta = xtx.clone().lu().solve(&xty)?;
    let resid_v = &y - &xm * &beta;
    let rss: f64 = resid_v.iter().map(|e| e * e).sum();
    let dof = m as f64 - k as f64;
    if dof <= 0.0 {
        return None;
    }
    let sigma2 = rss / dof;
    let xtx_inv = xtx.try_inverse()?;
    let var_b = sigma2 * xtx_inv[(ylag_idx, ylag_idx)];
    if !(var_b > 0.0) {
        return None;
    }
    let se = var_b.sqrt();
    let b = beta[ylag_idx];
    let residuals: Vec<f64> = resid_v.iter().copied().collect();
    Some((b / se, se, b, residuals, m, k, sigma2.sqrt()))
}

/// Augmented Dickey-Fuller test.
///
/// `regression`: "c" (constant, default), "ct" (constant+trend), "n"/"nc" (none).
/// `max_lag`: number of augmenting lagged differences of Δy (0 = plain DF).
/// Both arguments are now honored (previously ignored). Returns (t_stat, p_value)
/// with p-values from the Dickey-Fuller distribution matching the specification.
#[pyfunction(signature = (x, regression="c", max_lag=None))]
pub fn adf_test(x: PyReadonlyArray1<f64>, regression: &str, max_lag: Option<usize>) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();
    let reg = normalize_reg(regression)?;
    let p = max_lag.unwrap_or(0);

    match df_ols(x, reg, p) {
        Some((t_stat, _, _, _, _, _, _)) => Ok((t_stat, df_pvalue(t_stat, n, reg))),
        None => Ok((f64::NAN, f64::NAN)),
    }
}

/// Calculate KPSS p-value using critical value tables
/// Critical values from Kwiatkowski et al. (1992) Table 1
fn kpss_pvalue(stat: f64, regression: &str) -> f64 {
    // Critical values: (critical_value, p_value)
    let critical_values = match regression {
        "c" => {
            // Level stationarity critical values
            vec![
                (0.347, 0.10),
                (0.463, 0.05),
                (0.574, 0.025),
                (0.739, 0.01),
            ]
        }
        "ct" => {
            // Trend stationarity critical values
            vec![
                (0.119, 0.10),
                (0.146, 0.05),
                (0.176, 0.025),
                (0.216, 0.01),
            ]
        }
        _ => {
            vec![(0.463, 0.05)]
        }
    };

    // Find p-value by interpolation
    if stat < critical_values[0].0 {
        return 0.10;
    }

    for i in 0..(critical_values.len() - 1) {
        let (cv_low, p_low) = critical_values[i];
        let (cv_high, p_high) = critical_values[i + 1];
        
        if stat >= cv_low && stat < cv_high {
            // Linear interpolation
            let weight = (stat - cv_low) / (cv_high - cv_low);
            return p_low - weight * (p_low - p_high);
        }
    }

    0.01
}

/// Newey-West long-run variance estimator with Bartlett weights
/// Used by KPSS test for HAC-consistent variance estimation
/// Data-dependent bandwidth for the KPSS long-run variance (Hobijn, Franses &
/// Ooms), identical to statsmodels' `_kpss_autolag` (`nlags="auto"`, the
/// statsmodels default that the parity tests pin). The previous default —
/// Schwert's 12·(n/100)^¼ — over-smooths white-noise-like residuals (lag 23 vs
/// ~9 at n=1200), shifting the KPSS statistic by >15%.
fn kpss_autolag(resid: &[f64]) -> usize {
    let n = resid.len();
    if n < 2 {
        return 0;
    }
    let nf = n as f64;
    let covlags = (nf.powf(2.0 / 9.0) as usize).min(n - 1);
    let mut s0: f64 = resid.iter().map(|e| e * e).sum::<f64>() / nf;
    let mut s1 = 0.0;
    for i in 1..=covlags {
        let mut prod = 0.0;
        for t in i..n {
            prod += resid[t] * resid[t - i];
        }
        prod /= nf / 2.0;
        s0 += prod;
        s1 += (i as f64) * prod;
    }
    if s0 <= 0.0 || !s0.is_finite() {
        return 0;
    }
    let s_hat = s1 / s0;
    let pwr = 1.0 / 3.0;
    let gamma_hat = 1.1447 * (s_hat * s_hat).powf(pwr);
    let autolags = gamma_hat * nf.powf(pwr);
    if !autolags.is_finite() || autolags < 0.0 {
        return 0;
    }
    (autolags as usize).min(n - 1)
}

fn long_run_variance(resid: &[f64], max_lag: Option<usize>) -> f64 {
    let n = resid.len();
    if n < 2 {
        return f64::NAN;
    }

    // Automatic bandwidth selection (statsmodels-compatible rule of thumb)
    let l = match max_lag {
        Some(v) => v.min(n - 1),
        None => {
            // Schwert (1989) rule: 12 * (n/100)^(1/4)
            // This matches statsmodels' implementation
            let nf = n as f64;
            let bw = (12.0 * (nf / 100.0).powf(1.0 / 4.0)).ceil() as usize;
            bw.max(1).min(n - 1)
        }
    };

    // Compute gamma_0 (variance)
    let mut gamma0 = 0.0;
    for &e in resid {
        gamma0 += e * e;
    }
    gamma0 /= n as f64;

    // LRV = gamma_0 + 2 * sum_{k=1}^L w_k * gamma_k
    // where w_k = 1 - k/(L+1) (Bartlett weights)
    let mut lrv = gamma0;

    for k in 1..=l {
        // Compute autocovariance at lag k
        let mut gamma_k = 0.0;
        for t in k..n {
            gamma_k += resid[t] * resid[t - k];
        }
        gamma_k /= n as f64;

        // Apply Bartlett kernel weight
        let w = 1.0 - (k as f64) / ((l + 1) as f64);
        lrv += 2.0 * w * gamma_k;
    }

    lrv
}

/// KPSS test for stationarity.
///
/// OPTIMIZED: Single-pass regression calculation
/// Python signature:
///     kpss_test(x, regression="c", max_lag=None) -> (statistic, pvalue)
#[pyfunction(signature = (x, regression="c", max_lag=None))]
pub fn kpss_test(
    x: PyReadonlyArray1<f64>,
    regression: &str,
    max_lag: Option<usize>,
) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();

    if n < 3 {
        return Ok((f64::NAN, f64::NAN));
    }

    let (resid, _) = match regression {
        "c" => {
            // Level stationarity: y_t = μ + e_t
            // OPTIMIZATION: Single-pass mean
            let mean = x.iter().sum::<f64>() / (n as f64);
            (x.iter().map(|v| v - mean).collect::<Vec<f64>>(), 1)
        }
        "ct" => {
            // Trend stationarity: y_t = μ + β t + e_t
            // OPTIMIZATION: Build XtX and XtY in single pass
            let mut xtx = [[0.0_f64; 2]; 2];
            let mut xty = [0.0_f64; 2];
            
            for (i, &v) in x.iter().enumerate() {
                let t = (i + 1) as f64;
                xtx[0][0] += 1.0;
                xtx[0][1] += t;
                xtx[1][0] += t;
                xtx[1][1] += t * t;
                xty[0] += v;
                xty[1] += v * t;
            }
            
            let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
            if det.abs() < 1e-12 {
                return Ok((f64::NAN, f64::NAN));
            }
            
            let inv00 = xtx[1][1] / det;
            let inv01 = -xtx[0][1] / det;
            let inv10 = -xtx[1][0] / det;
            let inv11 = xtx[0][0] / det;
            
            let mu = inv00 * xty[0] + inv01 * xty[1];
            let beta = inv10 * xty[0] + inv11 * xty[1];

            let resid: Vec<f64> = x.iter().enumerate()
                .map(|(i, &v)| {
                    let t = (i + 1) as f64;
                    v - (mu + beta * t)
                })
                .collect();
            
            (resid, 2)
        }
        _ => {
            return Err(PyValueError::new_err(
                "regression must be 'c' or 'ct'",
            ));
        }
    };

    // Cumulative sum of residuals
    let mut s = Vec::with_capacity(n);
    let mut cum = 0.0;
    for &e in &resid {
        cum += e;
        s.push(cum);
    }

    // Bandwidth: user-supplied, else the Hobijn data-dependent rule
    // (statsmodels `nlags="auto"`, its default).
    let nlags = match max_lag {
        Some(v) => v.min(n - 1),
        None => kpss_autolag(&resid),
    };
    let lrv = long_run_variance(&resid, Some(nlags));

    if !lrv.is_finite() || lrv <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    // KPSS statistic
    let n_f = n as f64;
    let eta: f64 = s.iter().map(|v| v * v).sum();
    let stat = eta / (n_f * n_f * lrv);

    let p_val = kpss_pvalue(stat, regression);

    Ok((stat, p_val))
}

// ======================
// DEBUG VERSIONS FOR DIAGNOSTICS
// ======================

/// DEBUG: Newey-West long-run variance estimator with extensive logging
fn long_run_variance_debug(resid: &[f64], max_lag: Option<usize>) -> f64 {
    let n = resid.len();
    if n < 2 {
        eprintln!("[LRV DEBUG] n={} < 2, returning NaN", n);
        return f64::NAN;
    }

    eprintln!("\n=== LONG RUN VARIANCE DEBUG ===");
    eprintln!("n = {}", n);
    eprintln!("max_lag input = {:?}", max_lag);

    // Automatic bandwidth selection (statsmodels-compatible rule of thumb)
    let l = match max_lag {
        Some(v) => {
            let capped = v.min(n - 1);
            eprintln!("Using user-specified lag: {} (capped to {})", v, capped);
            capped
        }
        None => {
            // Schwert (1989) rule: 12 * (n/100)^(1/4)
            let nf = n as f64;
            let bw = (12.0 * (nf / 100.0).powf(1.0 / 4.0)).ceil() as usize;
            let capped = bw.max(1).min(n - 1);
            eprintln!("Auto bandwidth (Schwert): raw={}, capped={}", bw, capped);
            capped
        }
    };
    eprintln!("Final bandwidth L = {}", l);

    // Compute gamma_0 (variance)
    let mut gamma0 = 0.0;
    for &e in resid {
        gamma0 += e * e;
    }
    gamma0 /= n as f64;
    eprintln!("gamma_0 (variance) = {:.12}", gamma0);

    // Show first few residuals
    eprintln!("First 5 residuals: {:?}", &resid[..5.min(n)]);
    
    // LRV = gamma_0 + 2 * sum_{k=1}^L w_k * gamma_k
    let mut lrv = gamma0;
    
    eprintln!("\nAutocovariance terms:");
    eprintln!("  k | gamma_k        | weight         | contribution");
    eprintln!("----+----------------+----------------+----------------");

    for k in 1..=l {
        // Compute autocovariance at lag k
        let mut gamma_k = 0.0;
        for t in k..n {
            gamma_k += resid[t] * resid[t - k];
        }
        gamma_k /= n as f64;

        // Apply Bartlett kernel weight
        let w = 1.0 - (k as f64) / ((l + 1) as f64);
        let contrib = 2.0 * w * gamma_k;
        lrv += contrib;
        
        if k <= 5 || k == l {
            eprintln!("{:3} | {:.12} | {:.12} | {:.12}", k, gamma_k, w, contrib);
        } else if k == 6 {
            eprintln!("... (showing first 5 and last)");
        }
    }
    
    eprintln!("\nFinal LRV = {:.12}", lrv);
    eprintln!("=================================\n");

    lrv
}

/// DEBUG: KPSS test with extensive logging for diagnostics
///
/// Python signature:
///     kpss_test_debug(x, regression="c", max_lag=None) -> (statistic, pvalue)
#[pyfunction(signature = (x, regression="c", max_lag=None))]
pub fn kpss_test_debug(
    x: PyReadonlyArray1<f64>,
    regression: &str,
    max_lag: Option<usize>,
) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();

    eprintln!("\n╔════════════════════════════════════════╗");
    eprintln!("║      KPSS TEST DEBUG MODE              ║");
    eprintln!("╚════════════════════════════════════════╝");
    eprintln!("n = {}", n);
    eprintln!("regression = {}", regression);
    eprintln!("max_lag = {:?}", max_lag);

    if n < 3 {
        eprintln!("ERROR: n < 3, returning NaN");
        return Ok((f64::NAN, f64::NAN));
    }

    // Show data statistics
    let data_mean = x.iter().sum::<f64>() / (n as f64);
    let data_var = x.iter().map(|v| (v - data_mean).powi(2)).sum::<f64>() / (n as f64);
    eprintln!("\nInput data statistics:");
    eprintln!("  Mean: {:.12}", data_mean);
    eprintln!("  Variance: {:.12}", data_var);
    eprintln!("  First 5 values: {:?}", &x[..5.min(n)]);

    let (resid, _) = match regression {
        "c" => {
            eprintln!("\nDemeaning (regression='c'):");
            let mean = x.iter().sum::<f64>() / (n as f64);
            eprintln!("  Computed mean: {:.12}", mean);
            let resid: Vec<f64> = x.iter().map(|v| v - mean).collect();
            eprintln!("  First 5 residuals: {:?}", &resid[..5.min(n)]);
            
            let resid_mean = resid.iter().sum::<f64>() / (n as f64);
            eprintln!("  Residual mean (should be ~0): {:.2e}", resid_mean);
            
            (resid, 1)
        }
        "ct" => {
            eprintln!("\nDetrending (regression='ct'):");
            let mut xtx = [[0.0_f64; 2]; 2];
            let mut xty = [0.0_f64; 2];
            
            for (i, &v) in x.iter().enumerate() {
                let t = (i + 1) as f64;
                xtx[0][0] += 1.0;
                xtx[0][1] += t;
                xtx[1][0] += t;
                xtx[1][1] += t * t;
                xty[0] += v;
                xty[1] += v * t;
            }
            
            let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
            eprintln!("  Matrix determinant: {:.12}", det);
            
            if det.abs() < 1e-12 {
                eprintln!("ERROR: Singular matrix");
                return Ok((f64::NAN, f64::NAN));
            }
            
            let inv00 = xtx[1][1] / det;
            let inv01 = -xtx[0][1] / det;
            let inv10 = -xtx[1][0] / det;
            let inv11 = xtx[0][0] / det;
            
            let mu = inv00 * xty[0] + inv01 * xty[1];
            let beta = inv10 * xty[0] + inv11 * xty[1];
            
            eprintln!("  Intercept (μ): {:.12}", mu);
            eprintln!("  Slope (β): {:.12}", beta);

            let resid: Vec<f64> = x.iter().enumerate()
                .map(|(i, &v)| {
                    let t = (i + 1) as f64;
                    v - (mu + beta * t)
                })
                .collect();
            
            eprintln!("  First 5 residuals: {:?}", &resid[..5.min(n)]);
            
            (resid, 2)
        }
        _ => {
            return Err(PyValueError::new_err("regression must be 'c' or 'ct'"));
        }
    };

    // Cumulative sum
    eprintln!("\nComputing cumulative sum:");
    let mut s = Vec::with_capacity(n);
    let mut cum = 0.0;
    for &e in &resid {
        cum += e;
        s.push(cum);
    }
    eprintln!("  First 5 cumsum: {:?}", &s[..5.min(n)]);
    eprintln!("  Last 5 cumsum: {:?}", &s[n.saturating_sub(5)..n]);
    
    let sum_s_squared: f64 = s.iter().map(|v| v * v).sum();
    eprintln!("  Sum of squared cumsum (eta): {:.12}", sum_s_squared);

    // Long-run variance
    let lrv = long_run_variance_debug(&resid, max_lag);
    
    if !lrv.is_finite() || lrv <= 0.0 {
        eprintln!("ERROR: Invalid LRV = {}", lrv);
        return Ok((f64::NAN, f64::NAN));
    }

    // KPSS statistic
    let n_f = n as f64;
    let eta: f64 = s.iter().map(|v| v * v).sum();
    eprintln!("\nKPSS statistic calculation:");
    eprintln!("  eta (sum S_t^2): {:.12}", eta);
    eprintln!("  n: {}", n);
    eprintln!("  LRV: {:.12}", lrv);
    eprintln!("  n^2 * LRV: {:.12}", n_f * n_f * lrv);
    
    let stat = eta / (n_f * n_f * lrv);
    eprintln!("  KPSS statistic: {:.12}", stat);

    let p_val = kpss_pvalue(stat, regression);
    eprintln!("  p-value: {:.6}", p_val);
    
    eprintln!("\n╔════════════════════════════════════════╗");
    eprintln!("║      END KPSS DEBUG                    ║");
    eprintln!("╚════════════════════════════════════════╝\n");

    Ok((stat, p_val))
}

/// Phillips–Perron (PP) test (simplified).
///
/// OPTIMIZED: Single-pass calculation
/// Python signature:
///     pp_test(x, regression="c") -> (statistic, pvalue)
#[pyfunction(signature = (x, regression="c"))]
pub fn pp_test(x: PyReadonlyArray1<f64>, regression: &str) -> PyResult<(f64, f64)> {
    let x = x.as_slice()?;
    let n = x.len();
    let reg = normalize_reg(regression)?;

    // Base DF regression (no augmenting lags): Δy_t = [det] + β·y_{t-1} + u_t.
    let (t_stat, se_b, _beta, resid, m, _k, s) = match df_ols(x, reg, 0) {
        Some(v) => v,
        None => return Ok((f64::NAN, f64::NAN)),
    };
    let m_f = m as f64;

    // Short-run vs long-run residual variance.
    let gamma0 = resid.iter().map(|e| e * e).sum::<f64>() / m_f; // ≈ RSS/T
    let lam2 = long_run_variance(&resid, None); // Newey-West HAC (Bartlett)

    if !(gamma0 > 0.0) || !(lam2 > 0.0) || !(s > 0.0) {
        return Ok((f64::NAN, f64::NAN));
    }
    let lam = lam2.sqrt();

    // Phillips-Perron Z_t (Hamilton 1994, Prop. 17.6): correct the OLS t-ratio
    // for serial correlation in the residuals using the long-run variance,
    //   Z_t = sqrt(γ0/λ²)·t − (λ² − γ0)·T·SE(β) / (2·λ·s),
    // and evaluate against the Dickey-Fuller distribution (NOT the Normal — the
    // previous implementation used neither the HAC correction nor the DF CVs).
    let z_t = (gamma0 / lam2).sqrt() * t_stat - (lam2 - gamma0) * m_f * se_b / (2.0 * lam * s);
    let p_val = df_pvalue(z_t, n, reg);

    Ok((z_t, p_val))
}

// ======================
// NEW CHEAP FUNCTIONS
// ======================

/// Variance ratio test for random walk hypothesis
///
/// Tests if a series follows a random walk by comparing variance of
/// q-period returns to variance of 1-period returns
/// 
/// Python signature:
///     variance_ratio_test(x, lags=2) -> (vr, z_score, pvalue)
#[pyfunction(signature = (x, lags=2))]
pub fn variance_ratio_test(x: PyReadonlyArray1<f64>, lags: usize) -> PyResult<(f64, f64, f64)> {
    // `x` is the INCREMENT (return) series — the repo's own tests pass
    // np.diff(levels). The previous implementation differenced again (treating
    // returns as levels), so a random walk's returns came out as MA(1) with
    // VR(2) ≈ 0.5 instead of ≈ 1.
    let r = x.as_slice()?;
    let n = r.len();
    let q = lags;

    if q < 2 || n < q + 2 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let nf = n as f64;
    let qf = q as f64;
    let mu = r.iter().sum::<f64>() / nf;

    // Unbiased 1-period variance.
    let var1: f64 = r.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / (nf - 1.0);
    if var1 <= 0.0 || !var1.is_finite() {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    // q-period variance from OVERLAPPING q-sums (Lo–MacKinlay 1988) with the
    // bias-adjusted denominator m = q(n−q+1)(1−q/n). Overlapping is required
    // for consistency with the asymptotic variance θ below; the old code mixed
    // a non-overlapping estimator with the overlapping θ, mis-sizing z.
    let mut window: f64 = r[..q].iter().sum();
    let target = qf * mu;
    let mut dev = window - target;
    let mut acc = dev * dev;
    for t in q..n {
        window += r[t] - r[t - q];
        dev = window - target;
        acc += dev * dev;
    }
    // m already contains the factor q, so varq is the PER-PERIOD variance of
    // the q-sums and VR is varq/var1 directly (no further /q).
    let m = qf * (nf - qf + 1.0) * (1.0 - qf / nf);
    let varq = acc / m;

    let vr = varq / var1;

    // Homoscedastic asymptotic variance of VR(q) under the random-walk null.
    let theta = 2.0 * (2.0 * qf - 1.0) * (qf - 1.0) / (3.0 * qf * nf);
    let z = (vr - 1.0) / theta.sqrt();

    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_val = 2.0 * (1.0 - normal.cdf(z.abs()));

    Ok((vr, z, p_val))
}
/// Zivot-Andrews test for unit root with structural break
///
/// Tests for unit root allowing one structural break in both level and trend
/// Uses Model C (both intercept and trend shift)
/// 
/// Python signature:
///     zivot_andrews_test(x, max_lag=None) -> (min_stat, breakpoint, pvalue)
#[pyfunction(signature = (x, max_lag=None))]
pub fn zivot_andrews_test(x: PyReadonlyArray1<f64>, max_lag: Option<usize>) -> PyResult<(f64, usize, f64)> {
    use nalgebra::{DMatrix, DVector};
    
    let x = x.as_slice()?;
    let n = x.len();
    
    if n < 20 {
        return Ok((f64::NAN, 0, f64::NAN));
    }
    
    // Determine lag length (simplified - use sqrt(n) rule)
    let p = max_lag.unwrap_or_else(|| {
        let p_max = ((n as f64).sqrt()).floor() as usize;
        p_max.min(12).max(1)
    });
    
    if n <= p + 5 {
        return Ok((f64::NAN, 0, f64::NAN));
    }
    
    // Trim 15% from each end (standard ZA practice)
    let trim_pct = 0.15;
    let start = ((n as f64) * trim_pct).floor() as usize;
    let end = n - start;
    
    if end <= start + p + 2 {
        return Ok((f64::NAN, 0, f64::NAN));
    }
    
    let mut min_stat = f64::INFINITY;
    let mut min_break = start;
    
    // Try each potential breakpoint
    for tau in start..end {
        // Construct regression variables
        // Model C: Δy_t = μ + β·t + θ·DU_t + γ·DT_t + α·y_{t-1} + Σ φ_i·Δy_{t-i} + ε_t
        
        let m = n - p - 1;  // Sample size for regression (after lagging)
        
        // Build design matrix X and dependent variable y.
        // Columns: intercept, trend, DU, DT, y_{t-1} (5 base columns) plus p
        // lagged differences of Δy written at index 4+i for i=1..=p. That is
        // 5 + p columns; the previous `4 + p` under-counted by one and caused an
        // out-of-bounds Vec write at `t*n_regressors + 4 + p` on every call
        // (a guaranteed SIGABRT under panic="abort").
        let n_regressors = 5 + p;
        let mut x_mat = vec![0.0; m * n_regressors];
        let mut y_vec = vec![0.0; m];
        
        for t in 0..m {
            let actual_t = p + 1 + t;  // Actual time index in original series
            
            // Dependent variable: Δy_t
            y_vec[t] = x[actual_t] - x[actual_t - 1];
            
            // Regressor 1: Intercept
            x_mat[t * n_regressors + 0] = 1.0;
            
            // Regressor 2: Time trend
            x_mat[t * n_regressors + 1] = actual_t as f64;
            
            // Regressor 3: Level shift dummy DU_t (1 if t > τ, else 0)
            x_mat[t * n_regressors + 2] = if actual_t > tau { 1.0 } else { 0.0 };
            
            // Regressor 4: Trend shift dummy DT_t ((t - τ) if t > τ, else 0)
            x_mat[t * n_regressors + 3] = if actual_t > tau {
                (actual_t - tau) as f64
            } else {
                0.0
            };
            
            // Regressor 5: Lagged level y_{t-1}
            x_mat[t * n_regressors + 4] = x[actual_t - 1];
            
            // Regressors 6+: Lagged differences Δy_{t-i} for i=1..p
            for i in 1..=p {
                x_mat[t * n_regressors + 4 + i] = x[actual_t - i] - x[actual_t - i - 1];
            }
        }
        
        // OLS regression: solve (X'X) β = X'y
        // x_mat is built row-major (t*n_regressors + col). from_vec reads
        // COLUMN-major and would scramble every regressor, silently producing
        // garbage coefficients (breakpoints/t-stats were noise).
        let x_matrix = DMatrix::from_row_slice(m, n_regressors, &x_mat);
        let y_vector = DVector::from_vec(y_vec);
        
        let xtx = x_matrix.transpose() * &x_matrix;
        let xty = x_matrix.transpose() * &y_vector;
        
        // Solve for coefficients
		let beta = match xtx.clone().lu().solve(&xty) {
			Some(b) => b,
			None => continue,
		};       
        // Calculate residuals and RSS
        let y_hat = &x_matrix * &beta;
        let residuals = &y_vector - &y_hat;
        let rss: f64 = residuals.iter().map(|&e| e * e).sum();
        
        // Degrees of freedom
        let df = m as f64 - n_regressors as f64;
        if df <= 0.0 {
            continue;
        }
        
        let sigma2 = rss / df;
        
        // Compute (X'X)^-1 for variance-covariance matrix
        let xtx_inv = match xtx.lu().try_inverse() {
            Some(inv) => inv,
            None => continue,  // Skip if singular
        };
        
        // Standard error of α (coefficient on y_{t-1}, which is regressor index 4)
        let var_alpha = sigma2 * xtx_inv[(4, 4)];
        if var_alpha <= 0.0 || !var_alpha.is_finite() {
            continue;
        }
        
        let se_alpha = var_alpha.sqrt();
        
        // t-statistic for H0: α = 0 (unit root)
        let alpha = beta[4];
        let t_stat = alpha / se_alpha;
        
        // Keep track of minimum (most negative) t-statistic
        if t_stat < min_stat {
            min_stat = t_stat;
            min_break = tau;
        }
    }
    
    // P-value from approximate Zivot-Andrews critical values
    // Model C (both intercept and trend) critical values at n=100:
    // 1%: -5.57, 5%: -5.08, 10%: -4.82
    // These are approximate and should be adjusted for sample size
    let p_val = if min_stat < -5.57 {
        0.01
    } else if min_stat < -5.08 {
        0.05
    } else if min_stat < -4.82 {
        0.10
    } else {
        0.15  // Greater than 10%
    };
    
    Ok((min_stat, min_break, p_val))
}
/// Test for trend stationarity (linear detrending + KPSS)
///
/// Combines linear detrending with KPSS test
/// Python signature:
///     trend_stationarity_test(x) -> (stat, pvalue, is_stationary)
#[pyfunction]
pub fn trend_stationarity_test(x: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, bool)> {
    // Just use KPSS with trend
    let (stat, pval) = kpss_test(x, "ct", None)?;
    let is_stationary = pval > 0.05;  // Fail to reject null = stationary
    Ok((stat, pval, is_stationary))
}

/// Difference the series and test stationarity
///
/// Returns (is_i0, is_i1, adf_level, adf_diff1)
/// Python signature:
///     integration_order_test(x) -> (is_i0, is_i1, adf_level, adf_diff1)
#[pyfunction]
pub fn integration_order_test(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
) -> PyResult<(bool, bool, f64, f64)> {
    let x_slice = x.as_slice()?;
    let n = x_slice.len();

    if n < 3 {
        return Ok((false, false, f64::NAN, f64::NAN));
    }

    // Test on levels - recreate array from slice to avoid move.
    // `py` is a real GIL token injected by PyO3 (invisible in the Python
    // signature), replacing the previous unsound `assume_gil_acquired()`.
    use numpy::PyArray1;
    let x_arr = PyArray1::from_slice_bound(py, x_slice);
    let x_readonly = x_arr.readonly();
    
    let (stat_level, pval_level) = adf_test(x_readonly, "c", None)?;
    let is_i0 = pval_level < 0.05;  // Reject null = stationary
    
    // First difference
    let mut dx = Vec::with_capacity(n - 1);
    for t in 1..n {
        dx.push(x_slice[t] - x_slice[t - 1]);
    }
    
    // Test on first difference - convert Vec to PyReadonlyArray1
    let dx_array = PyArray1::from_vec_bound(py, dx);
    let dx_readonly = dx_array.readonly();
    
    let (stat_diff, pval_diff) = adf_test(dx_readonly, "c", None)?;
    let is_i1 = !is_i0 && pval_diff < 0.05;
    
    Ok((is_i0, is_i1, stat_level, stat_diff))
}

/// Seasonal differencing test
///
/// Apply seasonal difference and test stationarity
/// Python signature:
///     seasonal_diff_test(x, period=12) -> (stat, pvalue, is_stationary)
#[pyfunction(signature = (x, period=12))]
pub fn seasonal_diff_test(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<(f64, f64, bool)> {
    let x_slice = x.as_slice()?;
    let n = x_slice.len();

    if n <= period {
        return Ok((f64::NAN, f64::NAN, false));
    }

    // Seasonal difference
    let mut dx = Vec::with_capacity(n - period);
    for t in period..n {
        dx.push(x_slice[t] - x_slice[t - period]);
    }

    // Test differenced series - convert Vec to PyReadonlyArray1.
    // `py` is a real injected GIL token (invisible in the Python signature).
    use numpy::PyArray1;
    let dx_array = PyArray1::from_vec_bound(py, dx);
    let dx_readonly = dx_array.readonly();
    
    let (stat, pval) = adf_test(dx_readonly, "c", None)?;
    let is_stationary = pval < 0.05;
    
    Ok((stat, pval, is_stationary))
}

/// Test if series has unit root at multiple lags (for seasonal unit roots)
///
/// Tests for unit roots at lags 1, period, and 2*period
/// Python signature:
///     seasonal_unit_root_test(x, period=12) -> Vec<(lag, stat, pvalue)>
#[pyfunction(signature = (x, period=12))]
pub fn seasonal_unit_root_test(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Vec<(usize, f64, f64)>> {
    let x_slice = x.as_slice()?;
    let n = x_slice.len();

    if n < period * 2 {
        return Ok(vec![]);
    }

    let mut results = Vec::new();

    // Test at lag 1 (regular ADF) - recreate PyReadonlyArray.
    // `py` is a real injected GIL token (invisible in the Python signature),
    // replacing the previous unsound `assume_gil_acquired()`.
    use numpy::PyArray1;
    let x_arr = PyArray1::from_slice_bound(py, x_slice);
    let x_readonly = x_arr.readonly();

    let (stat1, pval1) = adf_test(x_readonly, "c", None)?;
    results.push((1, stat1, pval1));

    // Test at seasonal lag
    if n > period {
        let x_arr2 = PyArray1::from_slice_bound(py, x_slice);
        let x_readonly2 = x_arr2.readonly();
        let (stat_s, pval_s, _is_stat) = seasonal_diff_test(py, x_readonly2, period)?;
        results.push((period, stat_s, pval_s));
    }

    // Test at 2*seasonal lag
    if n > period * 2 {
        let x_arr3 = PyArray1::from_slice_bound(py, x_slice);
        let x_readonly3 = x_arr3.readonly();
        let (stat_2s, pval_2s, _is_stat) = seasonal_diff_test(py, x_readonly3, period * 2)?;
        results.push((period * 2, stat_2s, pval_2s));
    }

    Ok(results)
}