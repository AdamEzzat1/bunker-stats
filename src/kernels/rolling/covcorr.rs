/// Rolling covariance/correlation over two 1D slices.
///
/// This file intentionally contains **hot kernels** only (no PyO3, no Python).
///
/// All four kernels slide a `RollingPairMoments` over the data: O(1) per step,
/// re-summed from the window whenever rounding error has built up, so a window
/// is never affected by pairs that have already left it (see `moments.rs`).
///
/// - `rolling_cov_skipna` / `rolling_corr_skipna` ignore pairs in which either
///   value is NaN or infinite ("pairwise complete"), writing into a
///   caller-provided output slice.
/// - `rolling_cov_vec` / `rolling_corr_vec` are strict: a window containing any
///   such pair gives NaN. They allocate an output `Vec`.

use super::moments::RollingPairMoments;

/// Slide a window over (x, y), calling `emit(k, moments)` for the k-th window.
/// Requires `0 < window <= x.len() == y.len()`.
#[inline]
pub(crate) fn for_each_pair_window(
    x: &[f64],
    y: &[f64],
    window: usize,
    mut emit: impl FnMut(usize, &RollingPairMoments),
) {
    let n = x.len();
    let mut m = RollingPairMoments::new(&x[..window], &y[..window]);
    emit(0, &m);
    for i in window..n {
        m.remove(x[i - window], y[i - window]);
        m.add(x[i], y[i]);
        if m.needs_refresh() {
            m.refresh(&x[i + 1 - window..=i], &y[i + 1 - window..=i]);
        }
        emit(i + 1 - window, &m);
    }
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
    for_each_pair_window(&x[..n], &y[..n], window, |k, m| out[k] = m.cov());
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
    for_each_pair_window(&x[..n], &y[..n], window, |k, m| out[k] = m.corr());
}

pub(crate) fn rolling_cov_vec(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    if window == 0 || window > n {
        return Vec::new();
    }
    let mut out = vec![f64::NAN; n - window + 1];
    for_each_pair_window(&x[..n], &y[..n], window, |k, m| {
        if m.bad_count() == 0 {
            out[k] = m.cov();
        }
    });
    out
}

#[allow(dead_code)]
pub(crate) fn rolling_corr_vec(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    if window == 0 || window > n {
        return Vec::new();
    }
    let mut out = vec![f64::NAN; n - window + 1];
    for_each_pair_window(&x[..n], &y[..n], window, |k, m| {
        if m.bad_count() == 0 {
            out[k] = m.corr();
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rolling cov/corr over large-offset series. Both series are
    /// 1e8 + [1,2,3,4]; sample covariance of 3 consecutive equal integer runs is
    /// 1.0 and correlation is 1.0, regardless of the 1e8 offset. The old
    /// cross-moment `sum_xy - sum_x*sum_y/c` cancelled catastrophically here.
    #[test]
    fn test_rolling_cov_corr_large_offset() {
        let x: Vec<f64> = (1..=4).map(|k| 1e8 + k as f64).collect();
        let y = x.clone();
        let mut cov = vec![0.0; 2];
        let mut corr = vec![0.0; 2];
        rolling_cov_skipna(&x, &y, 3, &mut cov);
        rolling_corr_skipna(&x, &y, 3, &mut corr);
        for c in &cov {
            assert!((c - 1.0).abs() < 1e-9, "cov={c}, expected 1.0");
        }
        for c in &corr {
            assert!((c - 1.0).abs() < 1e-9, "corr={c}, expected 1.0");
        }
    }
}