//! Fused multi-statistic rolling kernel (1D).

use super::bounds::output_length;
use super::config::{Alignment, NanPolicy, RollingConfig};
use super::masks::StatsMask;

/// Internal state for compensated summation (Kahan).
#[derive(Debug, Clone, Copy, Default)]
struct KahanState {
    sum: f64,
    compensation: f64,
}

impl KahanState {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        let y = x - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }
    
    #[inline(always)]
    fn value(&self) -> f64 {
        self.sum
    }
}

/// Internal rolling state for one window.
#[derive(Debug, Clone, Default)]
struct RollingState {
    sum: KahanState,
    sumsq: KahanState,
    count: usize,
    valid_count: usize,  // Non-NaN count
    min: f64,
    max: f64,
    // Translation offset (the first finite value added). Variance is
    // shift-invariant, so accumulating sum/sumsq on `x - offset` avoids
    // catastrophic cancellation in `sumsq - n*mean^2` for large-offset data.
    // min/max add the offset back on read; variance/std need no adjustment.
    offset: f64,
    // Raw (unshifted) Kahan sum, used only for the mean. Deriving the mean from
    // the shifted sum as `sum/n + offset` loses precision when |offset| is huge
    // relative to the true mean (e.g. [1e10, 1, -1e10, 1, 1] -> 0.6); the raw
    // sum stays small under Kahan compensation and gives the exact mean.
    sum_raw: KahanState,
}

impl RollingState {
    fn new() -> Self {
        Self {
            sum: KahanState::default(),
            sumsq: KahanState::default(),
            count: 0,
            valid_count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            offset: 0.0,
            sum_raw: KahanState::default(),
        }
    }
    
    #[inline]
    fn reset(&mut self) {
        *self = Self::new();
    }
    
    /// Add a value to the window.
    #[inline]
    fn add(&mut self, x: f64) {
        self.count += 1;
        if x.is_nan() {
            return;
        }
        // Anchor the offset on the first finite value in this window.
        if self.valid_count == 0 {
            self.offset = x;
        }
        let xs = x - self.offset;
        self.valid_count += 1;
        self.sum.add(xs);
        self.sumsq.add(xs * xs);
        self.sum_raw.add(x);
        if xs < self.min {
            self.min = xs;
        }
        if xs > self.max {
            self.max = xs;
        }
    }
    
    /// Compute mean (or NaN if insufficient data).
    #[inline]
    fn mean(&self, min_periods: usize, nan_policy: NanPolicy) -> f64 {
        if !self.is_valid(min_periods, nan_policy) {
            return f64::NAN;
        }
        // Raw sum keeps the mean exact regardless of the offset magnitude.
        self.sum_raw.value() / self.valid_count as f64
    }
    
    /// Compute sample variance (ddof=1).
    #[inline]
    fn variance(&self, min_periods: usize, nan_policy: NanPolicy) -> f64 {
        if !self.is_valid(min_periods, nan_policy) {
            return f64::NAN;
        }
        if self.valid_count < 2 {
            return f64::NAN;
        }
        
        let n = self.valid_count as f64;
        let mean_sq = self.sum.value() / n;
        let var = (self.sumsq.value() - n * mean_sq * mean_sq) / (n - 1.0);
        var.max(0.0)  // Clamp negative due to FP error
    }
    
    /// Compute sample standard deviation.
    #[inline]
    fn std(&self, min_periods: usize, nan_policy: NanPolicy) -> f64 {
        self.variance(min_periods, nan_policy).sqrt()
    }
    
    /// Get valid count.
    #[inline]
    fn count_value(&self) -> f64 {
        self.valid_count as f64
    }
    
    /// Get min (or NaN if insufficient data).
    #[inline]
    fn min_value(&self, min_periods: usize, nan_policy: NanPolicy) -> f64 {
        if !self.is_valid(min_periods, nan_policy) {
            f64::NAN
        } else {
            self.min + self.offset
        }
    }
    
    /// Get max (or NaN if insufficient data).
    #[inline]
    fn max_value(&self, min_periods: usize, nan_policy: NanPolicy) -> f64 {
        if !self.is_valid(min_periods, nan_policy) {
            f64::NAN
        } else {
            self.max + self.offset
        }
    }
    
    /// Check if window has sufficient valid data.
    #[inline]
    fn is_valid(&self, min_periods: usize, nan_policy: NanPolicy) -> bool {
        match nan_policy {
            NanPolicy::Propagate => {
                // Any NaN in window -> invalid
                self.valid_count == self.count && self.valid_count >= min_periods
            }
            NanPolicy::Ignore | NanPolicy::RequireMinPeriods => {
                // Check valid count only
                self.valid_count >= min_periods
            }
        }
    }
}

/// Compute rolling statistics into pre-allocated output slices.
///
/// This is the core fused kernel for 1D arrays.
///
/// # Arguments
/// - `xs`: input array (length n)
/// - `config`: rolling window configuration
/// - `mask`: which statistics to compute
/// - Output slices (all must have correct length or be empty):
///   - `out_mean`, `out_std`, `out_var`: length = output_length(n, config)
///   - `out_count`, `out_min`, `out_max`: length = output_length(n, config)
///
/// # Panics
/// - If output slices have incorrect length
/// - If window validation fails
///
/// # Performance
/// - Single pass through input data
/// - Compensated summation (Kahan) for numerical stability
/// - No allocations (caller provides output buffers)
pub fn rolling_multi_into(
    xs: &[f64],
    config: &RollingConfig,
    mask: StatsMask,
    out_mean: &mut [f64],
    out_std: &mut [f64],
    out_var: &mut [f64],
    out_count: &mut [f64],
    out_min: &mut [f64],
    out_max: &mut [f64],
) {
    let n = xs.len();
    let window = config.window;
    let min_periods = config.effective_min_periods();
    
    // Validate
    if window == 0 || window > n {
        return;
    }
    
    let out_len = output_length(n, window, config.alignment);
    
    // Early return for empty output
    if out_len == 0 {
        return;
    }
    
    // Validate output lengths
    if mask.has_mean() {
        assert_eq!(out_mean.len(), out_len, "out_mean length mismatch");
    }
    if mask.has_std() {
        assert_eq!(out_std.len(), out_len, "out_std length mismatch");
    }
    if mask.has_var() {
        assert_eq!(out_var.len(), out_len, "out_var length mismatch");
    }
    if mask.has_count() {
        assert_eq!(out_count.len(), out_len, "out_count length mismatch");
    }
    if mask.has_min() {
        assert_eq!(out_min.len(), out_len, "out_min length mismatch");
    }
    if mask.has_max() {
        assert_eq!(out_max.len(), out_len, "out_max length mismatch");
    }
    
    match config.alignment {
        Alignment::Trailing => {
            rolling_multi_trailing(
                xs, window, min_periods, config.nan_policy, mask,
                out_mean, out_std, out_var, out_count, out_min, out_max,
            );
        }
        Alignment::Centered => {
            rolling_multi_centered(
                xs, window, config, mask,
                out_mean, out_std, out_var, out_count, out_min, out_max,
            );
        }
    }
}

/// Trailing window implementation (optimized sliding window).
fn rolling_multi_trailing(
    xs: &[f64],
    window: usize,
    min_periods: usize,
    nan_policy: NanPolicy,
    mask: StatsMask,
    out_mean: &mut [f64],
    out_std: &mut [f64],
    out_var: &mut [f64],
    out_count: &mut [f64],
    out_min: &mut [f64],
    out_max: &mut [f64],
) {
    // For Propagate policy with trailing, we can optimize:
    // - Check for NaNs once per window
    // - Use fast path if no NaNs
    
    if nan_policy == NanPolicy::Propagate {
        rolling_multi_trailing_propagate(
            xs, window, min_periods, mask,
            out_mean, out_std, out_var, out_count, out_min, out_max,
        );
    } else {
        rolling_multi_trailing_skipna(
            xs, window, min_periods, nan_policy, mask,
            out_mean, out_std, out_var, out_count, out_min, out_max,
        );
    }
}

/// Trailing window with NaN propagation (optimized).
fn rolling_multi_trailing_propagate(
    xs: &[f64],
    window: usize,
    _min_periods: usize,
    mask: StatsMask,
    out_mean: &mut [f64],
    out_std: &mut [f64],
    out_var: &mut [f64],
    out_count: &mut [f64],
    out_min: &mut [f64],
    out_max: &mut [f64],
) {
    let n = xs.len();
    let out_len = n - window + 1;
    
    let mut sum = KahanState::default();
    let mut sumsq = KahanState::default();
    let mut sum_raw = KahanState::default();
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut has_nan = false;

    // Per-window translation offset for numerical stability of the variance
    // (see write_stats). NaN windows output all-NaN, so the offset is only
    // consulted for finite windows where xs[start] is finite.
    let off0 = if xs[0].is_finite() { xs[0] } else { 0.0 };

    // Initialize first window
    for &x in &xs[..window] {
        if x.is_nan() {
            has_nan = true;
        }
        let xv = x - off0;
        sum.add(xv);
        sumsq.add(xv * xv);
        sum_raw.add(x);
        if xv < min_val {
            min_val = xv;
        }
        if xv > max_val {
            max_val = xv;
        }
    }

    // First output
    write_stats(
        0, &sum, &sumsq, &sum_raw, window, min_val, max_val, has_nan, off0, mask,
        out_mean, out_std, out_var, out_count, out_min, out_max,
    );

    // Slide window (recalculate everything to avoid NaN contamination)
    for k in 1..out_len {
        // Recalculate sum, sumsq, min, max, and NaN check for current window
        sum = KahanState::default();
        sumsq = KahanState::default();
        sum_raw = KahanState::default();
        min_val = f64::INFINITY;
        max_val = f64::NEG_INFINITY;
        has_nan = false;

        let offk = if xs[k].is_finite() { xs[k] } else { 0.0 };
        for &x in &xs[k..k + window] {
            if x.is_nan() {
                has_nan = true;
            }
            let xv = x - offk;
            sum.add(xv);
            sumsq.add(xv * xv);
            sum_raw.add(x);
            if xv < min_val {
                min_val = xv;
            }
            if xv > max_val {
                max_val = xv;
            }
        }

        write_stats(
            k, &sum, &sumsq, &sum_raw, window, min_val, max_val, has_nan, offk, mask,
            out_mean, out_std, out_var, out_count, out_min, out_max,
        );
    }
}

/// Trailing window with NaN skipping.
fn rolling_multi_trailing_skipna(
    xs: &[f64],
    window: usize,
    min_periods: usize,
    nan_policy: NanPolicy,
    mask: StatsMask,
    out_mean: &mut [f64],
    out_std: &mut [f64],
    out_var: &mut [f64],
    out_count: &mut [f64],
    out_min: &mut [f64],
    out_max: &mut [f64],
) {
    let n = xs.len();
    let out_len = n - window + 1;
    
    for k in 0..out_len {
        let window_slice = &xs[k..k + window];
        
        let mut state = RollingState::new();
        for &x in window_slice {
            state.add(x);
        }
        
        // Write outputs
        if mask.has_mean() {
            out_mean[k] = state.mean(min_periods, nan_policy);
        }
        if mask.has_std() {
            out_std[k] = state.std(min_periods, nan_policy);
        }
        if mask.has_var() {
            out_var[k] = state.variance(min_periods, nan_policy);
        }
        if mask.has_count() {
            out_count[k] = state.count_value();
        }
        if mask.has_min() {
            out_min[k] = state.min_value(min_periods, nan_policy);
        }
        if mask.has_max() {
            out_max[k] = state.max_value(min_periods, nan_policy);
        }
    }
}

/// Centered window implementation (recompute each window).
fn rolling_multi_centered(
    xs: &[f64],
    window: usize,
    config: &RollingConfig,
    mask: StatsMask,
    out_mean: &mut [f64],
    out_std: &mut [f64],
    out_var: &mut [f64],
    out_count: &mut [f64],
    out_min: &mut [f64],
    out_max: &mut [f64],
) {
    let n = xs.len();
    let half = window / 2;
    let min_periods = config.effective_min_periods();
    let nan_policy = config.nan_policy;
    
    for k in 0..n {
        let start = k.saturating_sub(half);
        let end = (k + half + 1).min(n);
        let window_slice = &xs[start..end];
        
        let mut state = RollingState::new();
        for &x in window_slice {
            state.add(x);
        }
        
        // Adjust min_periods for centered windows:
        // - When min_periods was None (default), accept partial windows at edges
        // - When explicitly set, enforce it strictly
        let actual_window_size = window_slice.len();
        let effective_min = if config.min_periods.is_none() {
            // Default behavior: accept partial windows for centered alignment
            actual_window_size
        } else {
            // Explicit min_periods: enforce the requirement
            min_periods
        };
        
        // Write outputs
        if mask.has_mean() {
            out_mean[k] = state.mean(effective_min, nan_policy);
        }
        if mask.has_std() {
            out_std[k] = state.std(effective_min, nan_policy);
        }
        if mask.has_var() {
            out_var[k] = state.variance(effective_min, nan_policy);
        }
        if mask.has_count() {
            out_count[k] = state.count_value();
        }
        if mask.has_min() {
            out_min[k] = state.min_value(effective_min, nan_policy);
        }
        if mask.has_max() {
            out_max[k] = state.max_value(effective_min, nan_policy);
        }
    }
}

/// Helper to write statistics for propagate mode.
#[inline]
fn write_stats(
    k: usize,
    sum: &KahanState,
    sumsq: &KahanState,
    sum_raw: &KahanState,
    window: usize,
    min_val: f64,
    max_val: f64,
    has_nan: bool,
    offset: f64,
    mask: StatsMask,
    out_mean: &mut [f64],
    out_std: &mut [f64],
    out_var: &mut [f64],
    out_count: &mut [f64],
    out_min: &mut [f64],
    out_max: &mut [f64],
) {
    if has_nan {
        // Propagate NaN
        if mask.has_mean() {
            out_mean[k] = f64::NAN;
        }
        if mask.has_std() {
            out_std[k] = f64::NAN;
        }
        if mask.has_var() {
            out_var[k] = f64::NAN;
        }
        if mask.has_count() {
            out_count[k] = f64::NAN;
        }
        if mask.has_min() {
            out_min[k] = f64::NAN;
        }
        if mask.has_max() {
            out_max[k] = f64::NAN;
        }
    } else {
        let n = window as f64;
        // sum/sumsq/min_val/max_val were accumulated on values shifted by
        // `offset` (variance is shift-invariant). The shifted mean feeds the
        // variance formula; add the offset back for the reported mean/min/max.
        let mean_shifted = sum.value() / n;

        if mask.has_mean() {
            // Mean from the raw sum (exact at any offset); the shifted mean is
            // only used for the variance formula below.
            out_mean[k] = sum_raw.value() / n;
        }
        if mask.has_var() || mask.has_std() {
            let var = (sumsq.value() - n * mean_shifted * mean_shifted) / (n - 1.0);
            let var = var.max(0.0);
            if mask.has_var() {
                out_var[k] = var;
            }
            if mask.has_std() {
                out_std[k] = var.sqrt();
            }
        }
        if mask.has_count() {
            out_count[k] = n;
        }
        if mask.has_min() {
            out_min[k] = min_val + offset;
        }
        if mask.has_max() {
            out_max[k] = max_val + offset;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trailing_basic() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = RollingConfig::trailing(3);
        let mask = StatsMask::MEAN | StatsMask::STD;
        
        let mut out_mean = vec![0.0; 3];
        let mut out_std = vec![0.0; 3];
        
        rolling_multi_into(
            &xs,
            &config,
            mask,
            &mut out_mean,
            &mut out_std,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        
        // Window [1,2,3]: mean=2.0
        assert!((out_mean[0] - 2.0).abs() < 1e-10);
        // Window [2,3,4]: mean=3.0
        assert!((out_mean[1] - 3.0).abs() < 1e-10);
        // Window [3,4,5]: mean=4.0
        assert!((out_mean[2] - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_nan_propagate() {
        let xs = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let config = RollingConfig::trailing(3);
        let mask = StatsMask::MEAN;
        
        let mut out_mean = vec![0.0; 3];
        
        rolling_multi_into(
            &xs,
            &config,
            mask,
            &mut out_mean,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        
        // Window [1, NaN, 3]: should be NaN
        assert!(out_mean[0].is_nan());
        // Window [NaN, 3, 4]: should be NaN
        assert!(out_mean[1].is_nan());
        // Window [3, 4, 5]: should be 4.0
        assert!((out_mean[2] - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_centered() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = RollingConfig::centered(3);
        let mask = StatsMask::MEAN;
        
        let mut out_mean = vec![0.0; 5];
        
        rolling_multi_into(
            &xs,
            &config,
            mask,
            &mut out_mean,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        
        // Position 0: window [1, 2] (truncated), mean = 1.5
        assert!((out_mean[0] - 1.5).abs() < 1e-10);
        // Position 1: window [1, 2, 3], mean = 2.0
        assert!((out_mean[1] - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_nan_ignore() {
        let xs = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let config = RollingConfig::new(
            3,
            Some(2),
            Alignment::Trailing,
            NanPolicy::Ignore
        ).unwrap();
        let mask = StatsMask::MEAN;
        
        let mut out_mean = vec![0.0; 3];
        
        rolling_multi_into(
            &xs,
            &config,
            mask,
            &mut out_mean,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        
        // Window [1, 2, NaN]: valid=[1, 2], mean=1.5
        assert!((out_mean[0] - 1.5).abs() < 1e-10);
        // Window [2, NaN, 4]: valid=[2, 4], mean=3.0
        assert!((out_mean[1] - 3.0).abs() < 1e-10);
        // Window [NaN, 4, 5]: valid=[4, 5], mean=4.5
        assert!((out_mean[2] - 4.5).abs() < 1e-10);
    }
}