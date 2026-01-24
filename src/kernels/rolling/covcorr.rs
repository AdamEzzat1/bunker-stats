/// Rolling covariance/correlation over two 1D slices.
///
/// This file intentionally contains **hot kernels** only (no PyO3, no Python).
///
/// - `rolling_cov_skipna` / `rolling_corr_skipna` are **O(n)** sliding-window
///   implementations that skip NaN pairs ("pairwise complete"), writing into a
///   caller-provided output slice (no allocation inside the kernel).
/// - The legacy `rolling_cov_vec` / `rolling_corr_vec` helpers are kept for
///   backwards-compatibility with older call sites; they are "strict" (NaNs
///   propagate) and allocate an output `Vec`.

#[derive(Clone, Copy, Debug, Default)]
struct RollingPairState {
    count: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

#[inline(always)]
fn add_pair(s: &mut RollingPairState, x: f64, y: f64) {
    if x.is_nan() || y.is_nan() {
        return;
    }
    s.count += 1;
    s.sum_x += x;
    s.sum_y += y;
    s.sum_xx += x * x;
    s.sum_yy += y * y;
    s.sum_xy += x * y;
}

#[inline(always)]
fn remove_pair(s: &mut RollingPairState, x: f64, y: f64) {
    if x.is_nan() || y.is_nan() {
        return;
    }
    s.count -= 1;
    s.sum_x -= x;
    s.sum_y -= y;
    s.sum_xx -= x * x;
    s.sum_yy -= y * y;
    s.sum_xy -= x * y;
}

#[inline(always)]
fn cov_from_state(s: &RollingPairState) -> f64 {
    if s.count < 2 {
        return f64::NAN;
    }
    let c = s.count as f64;
    // sample covariance: divide by (c - 1)
    (s.sum_xy - (s.sum_x * s.sum_y) / c) / (c - 1.0)
}

#[inline(always)]
fn corr_from_state(s: &RollingPairState) -> f64 {
    if s.count < 2 {
        return f64::NAN;
    }
    let c = s.count as f64;
    let cov = (s.sum_xy - (s.sum_x * s.sum_y) / c) / (c - 1.0);

    let var_x = (s.sum_xx - (s.sum_x * s.sum_x) / c) / (c - 1.0);
    let var_y = (s.sum_yy - (s.sum_y * s.sum_y) / c) / (c - 1.0);

    // Guard against zero/negative variance from all-equal values or numeric drift.
    if !(var_x > 0.0) || !(var_y > 0.0) {
        return f64::NAN;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 || denom.is_nan() {
        return f64::NAN;
    }
    cov / denom
}

/// O(n) rolling sample covariance with pairwise-NaN skipping.
///
/// Requirements (enforced by caller / debug_asserts):
/// - `x.len() == y.len()` (or caller passes consistent slices)
/// - `window > 0` and `window <= n`
/// - `out.len() == n - window + 1`
pub fn rolling_cov_skipna(x: &[f64], y: &[f64], window: usize, out: &mut [f64]) {
    let n = x.len().min(y.len());
    debug_assert!(window > 0 && window <= n);
    debug_assert!(out.len() == n.saturating_sub(window) + 1);
    if window == 0 || window > n {
        return;
    }

    let x = &x[..n];
    let y = &y[..n];

    let mut s = RollingPairState::default();
    for i in 0..window {
        add_pair(&mut s, x[i], y[i]);
    }
    out[0] = cov_from_state(&s);

    for i in window..n {
        remove_pair(&mut s, x[i - window], y[i - window]);
        add_pair(&mut s, x[i], y[i]);
        out[i - window + 1] = cov_from_state(&s);
    }
}

/// O(n) rolling sample correlation with pairwise-NaN skipping.
///
/// Requirements (enforced by caller / debug_asserts):
/// - `x.len() == y.len()` (or caller passes consistent slices)
/// - `window > 0` and `window <= n`
/// - `out.len() == n - window + 1`
pub fn rolling_corr_skipna(x: &[f64], y: &[f64], window: usize, out: &mut [f64]) {
    let n = x.len().min(y.len());
    debug_assert!(window > 0 && window <= n);
    debug_assert!(out.len() == n.saturating_sub(window) + 1);
    if window == 0 || window > n {
        return;
    }

    let x = &x[..n];
    let y = &y[..n];

    let mut s = RollingPairState::default();
    for i in 0..window {
        add_pair(&mut s, x[i], y[i]);
    }
    out[0] = corr_from_state(&s);

    for i in window..n {
        remove_pair(&mut s, x[i - window], y[i - window]);
        add_pair(&mut s, x[i], y[i]);
        out[i - window + 1] = corr_from_state(&s);
    }
}
pub(crate) fn rolling_cov_vec(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    if window == 0 || window > n {
        return Vec::new();
    }
    let x = &x[..n];
    let y = &y[..n];

    let mut out = Vec::with_capacity(n - window + 1);

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;

    for i in 0..window {
        let xi = x[i];
        let yi = y[i];
        sum_x += xi;
        sum_y += yi;
        sum_xy += xi * yi;
    }

    for i in (window - 1)..n {
        if i >= window {
            let xi_new = x[i];
            let yi_new = y[i];
            let xi_old = x[i - window];
            let yi_old = y[i - window];
            sum_x += xi_new - xi_old;
            sum_y += yi_new - yi_old;
            sum_xy += xi_new * yi_new - xi_old * yi_old;
        }

        let w = window as f64;
        let mx = sum_x / w;
        let my = sum_y / w;
        let cov = (sum_xy - w * mx * my) / ((window - 1) as f64);
        out.push(cov);
    }

    out
}

#[allow(dead_code)]
pub(crate) fn rolling_corr_vec(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    if window == 0 || window > n {
        return Vec::new();
    }
    let x = &x[..n];
    let y = &y[..n];

    let mut out = Vec::with_capacity(n - window + 1);

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;
    let mut sum_xy = 0.0f64;

    for i in 0..window {
        let xi = x[i];
        let yi = y[i];
        sum_x += xi;
        sum_y += yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
        sum_xy += xi * yi;
    }

    for i in (window - 1)..n {
        if i >= window {
            let xi_new = x[i];
            let yi_new = y[i];
            let xi_old = x[i - window];
            let yi_old = y[i - window];
            sum_x += xi_new - xi_old;
            sum_y += yi_new - yi_old;
            sum_x2 += xi_new * xi_new - xi_old * xi_old;
            sum_y2 += yi_new * yi_new - yi_old * yi_old;
            sum_xy += xi_new * yi_new - xi_old * yi_old;
        }

        let w = window as f64;
        let mx = sum_x / w;
        let my = sum_y / w;

        let cov = (sum_xy - w * mx * my) / ((window - 1) as f64);
        let vx = (sum_x2 - w * mx * mx) / ((window - 1) as f64);
        let vy = (sum_y2 - w * my * my) / ((window - 1) as f64);

        let denom = (vx * vy).sqrt();
        out.push(cov / denom);
    }

    out
}