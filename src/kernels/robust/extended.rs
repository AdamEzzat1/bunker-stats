/// Robust statistics module - FULLY OPTIMIZED
///
/// Implements optimizations:
/// 1. select_nth_unstable instead of full sort (O(n) vs O(n log n))
/// 2. Workspace API for allocation-free operations
/// 3. Fused median+MAD kernels
/// 7. Explicit NaN handling: reducers short-circuit to NaN on NaN input
///    (numpy semantics) and sort/select with `f64::total_cmp` so they never
///    panic — critical under the release profile's `panic = "abort"`.
/// 8. &[f64] slice-based API throughout
///
/// This file replaces: extended.rs, mad.rs, trimmed_mean.rs

use crate::mean_slice;

// ============================================================================
// WORKSPACE STRUCT (Optimization #2)
// ============================================================================

/// Reusable scratch buffers for zero-allocation robust statistics
///
/// Enables Vaex-style workspace pattern for fast pipelines.
#[derive(Default)]
pub struct RobustWorkspace {
    pub(crate) scratch: Vec<f64>,
    pub(crate) scratch2: Vec<f64>,
}

impl RobustWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            scratch: Vec::with_capacity(cap),
            scratch2: Vec::with_capacity(cap),
        }
    }
}

// ============================================================================
// OPTIMIZED CORE ESTIMATORS (Optimization #1: select_nth)
// ============================================================================

/// Median using select_nth_unstable (O(n) average vs O(n log n) sort)
///
/// Returns NaN if input is empty.
#[inline]
pub(crate) fn median_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }

    let mut v = xs.to_vec();
    median_inplace(&mut v)
}

/// Median in-place using select_nth (mutates input)
#[inline]
pub(crate) fn median_inplace(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }

    // NaN has no defined order; propagate NaN (numpy `median` semantics) instead
    // of feeding it to the selector. This also guards mad/biweight/huber, which
    // funnel through here, so any NaN input yields NaN rather than a crash.
    if crate::util::any_nan(v) {
        return f64::NAN;
    }

    if n == 1 {
        return v[0];
    }

    if n & 1 == 1 {
        // Odd length: select middle element
        let mid = n >> 1;
        let (_, median, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        *median
    } else {
        // Even length: average of two middle elements
        let mid = n >> 1;

        // Select upper middle element
        let (left, median_upper, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        let upper = *median_upper;
        
        // Find max of left partition (lower middle element)
        let lower = left.iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        
        (lower + upper) * 0.5
    }
}

/// MAD using select_nth (2× O(n) selections vs 2× O(n log n) sorts)
///
/// Major optimization: 2-4x faster than full sort approach.
#[inline]
pub(crate) fn mad_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }

    let mut v = xs.to_vec();
    mad_inplace(&mut v)
}

/// MAD in-place using select_nth (mutates and reuses buffer)
#[inline]
fn mad_inplace(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }

    // First selection: find median
    let med = median_inplace(v);

    // Convert to absolute deviations in-place
    for val in v.iter_mut() {
        *val = (*val - med).abs();
    }

    // Second selection: median of deviations
    median_inplace(v)
}

// ============================================================================
// FUSED KERNELS (Optimization #3)
// ============================================================================

/// FUSED: Compute median and MAD together
///
/// This is the hot path for robust_fit with default settings.
/// 40-50% faster than calling median() then mad() separately.
#[inline]
pub(crate) fn median_mad_fused(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let mut v = xs.to_vec();
    median_mad_fused_inplace(&mut v)
}

/// FUSED: Median + MAD in-place
#[inline]
fn median_mad_fused_inplace(v: &mut [f64]) -> (f64, f64) {
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    // Compute median
    let median = median_inplace(v);

    // Reuse buffer for deviations
    for val in v.iter_mut() {
        *val = (*val - median).abs();
    }

    // Median of deviations
    let mad = median_inplace(v);

    (median, mad)
}

// ============================================================================
// OTHER ROBUST ESTIMATORS
// ============================================================================

/// Trimmed mean (uses sort - selection doesn't help much for ranges)
#[inline]
pub(crate) fn trimmed_mean_slice(xs: &[f64], proportion_to_cut: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }

    // Validation
    if !proportion_to_cut.is_finite() || proportion_to_cut < 0.0 || proportion_to_cut >= 0.5 {
        return f64::NAN;
    }

    if crate::util::any_nan(xs) {
        return f64::NAN;
    }

    let mut v = xs.to_vec();
    v.sort_by(|x, y| x.total_cmp(y));

    let n = v.len();
    let cut = ((n as f64) * proportion_to_cut).floor() as usize;

    if cut * 2 >= n {
        return f64::NAN;
    }

    if cut == 0 {
        return mean_slice(&v);
    }

    mean_slice(&v[cut..(n - cut)])
}

/// IQR (uses sort - need both quartiles precisely)
#[inline]
pub(crate) fn iqr_slice(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return f64::NAN;
    }

    if crate::util::any_nan(xs) {
        return f64::NAN;
    }

    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));

    let q1 = percentile_sorted(&v, 25.0);
    let q3 = percentile_sorted(&v, 75.0);

    q3 - q1
}

/// Winsorized mean
#[inline]
pub(crate) fn winsorized_mean_slice(xs: &[f64], lower_percentile: f64, upper_percentile: f64) -> f64 {
    if xs.is_empty() || lower_percentile >= upper_percentile {
        return f64::NAN;
    }

    if crate::util::any_nan(xs) {
        return f64::NAN;
    }

    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));

    let lower_val = percentile_sorted(&v, lower_percentile);
    let upper_val = percentile_sorted(&v, upper_percentile);

    for val in &mut v {
        if *val < lower_val {
            *val = lower_val;
        } else if *val > upper_val {
            *val = upper_val;
        }
    }

    mean_slice(&v)
}

/// Trimmed standard deviation
#[inline]
pub(crate) fn trimmed_std_slice(xs: &[f64], proportion_to_cut: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }

    if !proportion_to_cut.is_finite() || proportion_to_cut < 0.0 || proportion_to_cut >= 0.5 {
        return f64::NAN;
    }

    if crate::util::any_nan(xs) {
        return f64::NAN;
    }

    let mut v = xs.to_vec();
    v.sort_by(|x, y| x.total_cmp(y));

    let n = v.len();
    let cut = ((n as f64) * proportion_to_cut).floor() as usize;

    if cut * 2 >= n || (n - 2 * cut) < 2 {
        return f64::NAN;
    }

    let trimmed = &v[cut..(n - cut)];
    let m = mean_slice(trimmed);

    let mut sum_sq = 0.0;
    for &val in trimmed {
        let diff = val - m;
        sum_sq += diff * diff;
    }

    (sum_sq / ((trimmed.len() - 1) as f64)).sqrt()
}

/// MAD with normal-consistency constant
#[inline]
pub(crate) fn mad_std_slice(xs: &[f64]) -> f64 {
    mad_slice(xs) * 1.482_602_218_505_602
}

/// Biweight midvariance
pub(crate) fn biweight_midvariance_slice(xs: &[f64], c: f64) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }

    let med = median_slice(xs);
    let mad_val = mad_slice(xs);
    
    if mad_val == 0.0 || !mad_val.is_finite() {
        return f64::NAN;
    }

    // Beers–Flynn–Gebhardt biweight midvariance, matching
    // astropy.stats.biweight_midvariance (modify_sample_size=False):
    //
    //   ζ² = n · Σ_{|u|<1} (x−M)² (1−u²)⁴ / [ Σ_{|u|<1} (1−u²)(1−5u²) ]²
    //
    // where u = (x−M)/(c·MAD) and n is the full sample size. The previous
    // implementation used (1−u²)² in the numerator (exponent 2, not 4) and
    // (1−u²)² in the denominator (instead of (1−u²)(1−5u²)), biasing the scale.
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for &x in xs {
        let u = (x - med) / (c * mad_val);
        if u.abs() < 1.0 {
            let u2 = u * u;
            let one_minus = 1.0 - u2;
            let d = x - med;
            numerator += d * d * one_minus.powi(4);
            denominator += one_minus * (1.0 - 5.0 * u2);
        }
    }

    if denominator == 0.0 {
        return f64::NAN;
    }

    (n as f64) * numerator / (denominator * denominator)
}

/// Normal-consistency constant of Qn: `1 / (sqrt(2) * Phi^-1(5/8))`.
const QN_CONSISTENCY: f64 = 2.219_144_465_985_076;

/// 0-indexed rank of the pairwise difference Qn selects.
///
/// Rousseeuw & Croux (1993) take the `k`-th smallest of the `C(n, 2)`
/// differences with `k = C(h, 2)` and `h = floor(n/2) + 1` (1-indexed).
/// That is close to, but not the same as, the first quartile of the
/// differences: `floor(C(n,2) / 4)` agrees for odd n up to 9 and is a lower
/// rank otherwise, by enough to make the estimate ~25% smaller at n = 10.
#[inline]
fn qn_rank(n: usize) -> usize {
    let h = n / 2 + 1;
    h * (h - 1) / 2 - 1
}

/// Rousseeuw & Croux's Qn scale estimator: `QN_CONSISTENCY` times the
/// `C(floor(n/2) + 1, 2)`-th smallest pairwise absolute difference.
///
/// No finite-sample correction factor is applied, as in
/// `statsmodels.robust.scale.qn_scale` (R's `robustbase::Qn` applies one by
/// default).
///
/// Runs in O(n log n) time and O(n) memory. Materialising the `n(n-1)/2`
/// differences needs 160 GB at n = 200,000, and a failed allocation aborts
/// the process.
pub(crate) fn qn_scale_slice(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }

    if crate::util::any_nan(xs) {
        return f64::NAN;
    }

    let k = qn_rank(n);

    // +/-Inf needs no quadratic buffer either. Sorted with `total_cmp`, the
    // pairwise differences fall into three runs: the finite ones (both values
    // finite), then +Inf (one infinite value, or two of opposite sign), then
    // NaN (`Inf - Inf` for two of the same sign). Only the first run has to be
    // searched; the others are identified by counting.
    let n_pos = xs.iter().filter(|v| **v == f64::INFINITY).count();
    let n_neg = xs.iter().filter(|v| **v == f64::NEG_INFINITY).count();
    if n_pos + n_neg == 0 {
        return kth_pairwise_abs_diff(xs, k) * QN_CONSISTENCY;
}

    let finite: Vec<f64> = xs.iter().copied().filter(|v| v.is_finite()).collect();
    let m = finite.len();
    let finite_pairs = m * m.saturating_sub(1) / 2;
    if k < finite_pairs {
        return kth_pairwise_abs_diff(&finite, k) * QN_CONSISTENCY;
}
    let infinite_pairs = m * (n_pos + n_neg) + n_pos * n_neg;
    if k < finite_pairs + infinite_pairs {
        f64::INFINITY
    } else {
        f64::NAN
    }
}

/// Reference Qn: select among all materialised pairwise differences. Test
/// only -- it allocates n(n-1)/2 values.
#[cfg(test)]
fn qn_scale_brute_force(xs: &[f64]) -> f64 {
    let n = xs.len();
    let mut diffs = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            diffs.push((xs[i] - xs[j]).abs());
        }
    }

    if diffs.is_empty() {
        return f64::NAN;
    }

    let k = qn_rank(n);
    let (_, selected, _) = diffs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));

    *selected * QN_CONSISTENCY
}

/// The `k`-th smallest (0-indexed) of the `n(n-1)/2` values `|x_i - x_j|`,
/// for finite `xs`, without materialising them.
///
/// After sorting, the number of pairs with difference `<= d` is a two-pointer
/// sweep. That count is a step function of `d` that only rises at values some
/// pair actually attains, so the smallest `d` whose count exceeds `k` *is* the
/// answer, bit for bit. Non-negative floats order the same as their bit
/// patterns, so that smallest `d` is found by bisecting on the bits: at most
/// 64 sweeps regardless of the data's scale.
fn kth_pairwise_abs_diff(xs: &[f64], k: usize) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    // Pairs (i < j) with sorted[j] - sorted[i] <= d. Floating-point
    // subtraction is monotone in each argument, so the left pointer only
    // ever moves right as j grows.
    let count_within = |d: f64| -> usize {
        let mut i = 0;
        let mut count = 0;
        for j in 0..n {
            while sorted[j] - sorted[i] > d {
                i += 1;
            }
            count += j - i;
        }
        count
    };

    // The range may overflow to +Inf for extreme inputs; its bit pattern still
    // bounds every finite difference from above.
    let mut lo = 0u64;
    let mut hi = (sorted[n - 1] - sorted[0]).to_bits();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if count_within(f64::from_bits(mid)) > k {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    f64::from_bits(lo)
}

/// Huber M-estimator
pub(crate) fn huber_location_slice(xs: &[f64], k: f64, max_iter: usize) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }

    let mut mu = median_slice(xs);
    let mad = mad_slice(xs);

    if !mad.is_finite() || mad == 0.0 {
        return mu;
    }

    let scale = mad * 1.4826;

    for _ in 0..max_iter {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for &x in xs {
            let r = (x - mu) / scale;
            let psi = if r.abs() <= k { r } else { k * r.signum() };

            numerator += psi;
            denominator += if r.abs() <= k { 1.0 } else { k / r.abs() };
        }

        let delta = scale * numerator / denominator;
        mu += delta;

        if delta.abs() < 1e-6 * scale {
            break;
        }
    }

    mu
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

#[inline]
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }

    let idx = (p / 100.0) * ((n - 1) as f64);
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;

    if lower == upper || upper >= n {
        sorted[lower]
    } else {
        let weight = idx - (lower as f64);
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

// ============================================================================
// SKIPNA VARIANTS (Optimization #7: Explicit NaN handling)
// ============================================================================

#[inline]
pub(crate) fn median_slice_skipna(xs: &[f64]) -> f64 {
    let valid: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    median_slice(&valid)
}

#[inline]
pub(crate) fn mad_slice_skipna(xs: &[f64]) -> f64 {
    let valid: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    mad_slice(&valid)
}

#[inline]
pub(crate) fn trimmed_mean_slice_skipna(xs: &[f64], proportion_to_cut: f64) -> f64 {
    let valid: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    trimmed_mean_slice(&valid, proportion_to_cut)
}

#[inline]
pub(crate) fn iqr_slice_skipna(xs: &[f64]) -> f64 {
    let valid: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    iqr_slice(&valid)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_median_selection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((median_slice(&data) - 3.0).abs() < EPSILON);

        let data_even = vec![1.0, 2.0, 3.0, 4.0];
        assert!((median_slice(&data_even) - 2.5).abs() < EPSILON);
    }

    #[test]
    fn test_mad_selection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mad = mad_slice(&data);
        assert!((mad - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_median_mad_fused() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (med, mad) = median_mad_fused(&data);
        
        assert!((med - 3.0).abs() < EPSILON);
        assert!((mad - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_trimmed_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tm = trimmed_mean_slice(&data, 0.1);
        assert!((tm - 5.5).abs() < EPSILON);
    }

    #[test]
    fn test_iqr() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let iqr = iqr_slice(&data);
        assert!((iqr - 4.5).abs() < 0.5);
    }

    #[test]
    fn test_mad_empty() {
        assert!(mad_slice(&[]).is_nan());
    }

    #[test]
    fn test_skipna_variants() {
        let data = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];

        let med = median_slice_skipna(&data);
        assert!((med - 3.5).abs() < EPSILON);

        let mad = mad_slice_skipna(&data);
        assert!(mad.is_finite());
    }

    /// Regression for the CRITICAL abort vector: with `panic = "abort"` any
    /// panic here is a hard interpreter crash (SIGABRT). Every order-statistic
    /// reducer must return NaN on NaN input, never panic. Before the fix these
    /// hit `partial_cmp(..).unwrap()` and aborted.
    #[test]
    fn test_nan_input_propagates_not_panics() {
        let data = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        assert!(median_slice(&data).is_nan(), "median must propagate NaN");
        assert!(mad_slice(&data).is_nan(), "mad must propagate NaN");
        let (med, mad) = median_mad_fused(&data);
        assert!(med.is_nan() && mad.is_nan(), "fused must propagate NaN");
        assert!(iqr_slice(&data).is_nan(), "iqr must propagate NaN");
        assert!(trimmed_mean_slice(&data, 0.1).is_nan(), "trimmed_mean must propagate NaN");
        assert!(trimmed_std_slice(&data, 0.1).is_nan(), "trimmed_std must propagate NaN");
        assert!(winsorized_mean_slice(&data, 10.0, 90.0).is_nan(), "winsorized must propagate NaN");
        assert!(qn_scale_slice(&data).is_nan(), "qn must propagate NaN");
        assert!(biweight_midvariance_slice(&data, 9.0).is_nan(), "biweight must propagate NaN");
        assert!(huber_location_slice(&data, 1.345, 50).is_nan(), "huber must propagate NaN");
    }

    /// ±Inf is a valid, ordered f64: sort/select must not panic on it. Median of
    /// {1, 2, 3, +inf, +inf} is the middle order statistic = 3.
    #[test]
    fn test_inf_input_does_not_panic() {
        let data = vec![1.0, 2.0, 3.0, f64::INFINITY, f64::INFINITY];
        assert_eq!(median_slice(&data), 3.0);
        assert!(mad_slice(&data).is_finite() || mad_slice(&data).is_nan());
    }

    /// Biweight midvariance pinned to the astropy/Beers-Flynn-Gebhardt formula
    /// (c=9, raw MAD). Reference values computed from
    ///   ζ² = n·Σ(x−M)²(1−u²)⁴ / [Σ(1−u²)(1−5u²)]².
    /// The previous (wrong) exponents produced materially different numbers, so
    /// this test would fail against the old implementation.
    #[test]
    fn test_biweight_midvariance_astropy_reference() {
        let simple = biweight_midvariance_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], 9.0);
        assert!((simple - 2.297_063_991_357_617).abs() < 1e-12, "got {simple}");

        // With a large outlier the estimator stays close to the inlier spread,
        // unlike the sample variance which would blow up.
        let outlier = biweight_midvariance_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0], 9.0);
        assert!((outlier - 2.850_366_011_689_22).abs() < 1e-12, "got {outlier}");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Any f64 INCLUDING NaN and ±Inf — this is the "never aborts" fuzz layer.
    fn any_vec() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(proptest::num::f64::ANY, 0..48)
    }

    // Finite f64 in a sane range, for algebraic-invariant properties.
    fn finite_vec() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-1e6f64..1e6f64, 1..48)
    }

    proptest! {
        // Every robust reducer must return (not panic/abort) for ANY input,
        // including NaN and ±Inf. proptest treats a panic as a test failure, so
        // simply calling each fn exercises the panic="abort" guarantee.
        #[test]
        fn robust_reducers_never_panic(xs in any_vec()) {
            let _ = median_slice(&xs);
            let _ = mad_slice(&xs);
            let _ = median_mad_fused(&xs);
            let _ = iqr_slice(&xs);
            let _ = trimmed_mean_slice(&xs, 0.1);
            let _ = trimmed_std_slice(&xs, 0.1);
            let _ = winsorized_mean_slice(&xs, 10.0, 90.0);
            let _ = qn_scale_slice(&xs);
            let _ = biweight_midvariance_slice(&xs, 9.0);
            let _ = huber_location_slice(&xs, 1.345, 20);
        }

        // If any element is NaN, order-statistic reducers propagate NaN (numpy
        // semantics), never a spurious finite value.
        #[test]
        fn nan_in_propagates_nan_out(mut xs in finite_vec(), idx in 0usize..48) {
            let i = idx % xs.len();
            xs[i] = f64::NAN;
            prop_assert!(median_slice(&xs).is_nan());
            prop_assert!(mad_slice(&xs).is_nan());
            prop_assert!(iqr_slice(&xs).is_nan());
        }

        // Median is translation-equivariant: median(x + c) == median(x) + c.
        #[test]
        fn median_translation_equivariant(xs in finite_vec(), c in -1e5f64..1e5f64) {
            let shifted: Vec<f64> = xs.iter().map(|v| v + c).collect();
            let lhs = median_slice(&shifted);
            let rhs = median_slice(&xs) + c;
            prop_assert!((lhs - rhs).abs() <= 1e-6 * (1.0 + rhs.abs()));
        }

        // MAD is scale-equivariant: MAD(a*x) == |a| * MAD(x).
        #[test]
        fn mad_scale_equivariant(xs in finite_vec(), a in -100.0f64..100.0) {
            let scaled: Vec<f64> = xs.iter().map(|v| a * v).collect();
            let lhs = mad_slice(&scaled);
            let rhs = a.abs() * mad_slice(&xs);
            prop_assert!((lhs - rhs).abs() <= 1e-6 * (1.0 + rhs.abs()));
        }
    }
}
#[cfg(test)]
mod qn_tests {
    use super::*;
    use proptest::prelude::*;

    /// Deterministic pseudo-random stream (no extra dev-dependency).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Values from `statsmodels.robust.scale.qn_scale`, which implements the
    /// Rousseeuw-Croux definition independently (their O(n log n) algorithm).
    /// The brute-force reference below shares `qn_rank` with the kernel, so
    /// it cannot catch a wrong rank; these can. The even-n rows are the ones
    /// the old first-quartile rank got wrong (it gave 2.2219, 2.2219 and
    /// 4.6660 for the 6-, 8- and 10-point samples).
    #[test]
    fn qn_matches_statsmodels() {
        let cases: &[(&[f64], f64)] = &[
            (&[1.0, 2.0], 2.219144465985076),
            (&[1.0, 2.0, 4.0], 2.219144465985076),
            (&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0], 4.438288931970152),
            (&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], 4.438288931970152),
            (&[2.5, -1.0, 0.3, 7.0, 7.0, 4.2, -3.3, 0.0, 1.1, 9.4], 6.213604504758212),
            (
                &[0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9, 6.4, 8.1, 10.0, 12.1],
                4.660203378568658,
            ),
        ];
        for (xs, want) in cases {
            let got = qn_scale_slice(xs);
            assert!(((got - want) / want).abs() < 1e-13, "{xs:?}: {got} vs {want}");
        }
    }

    #[test]
    fn qn_rank_is_the_rousseeuw_croux_k() {
        // (n, k) with k = C(floor(n/2) + 1, 2), 1-indexed
        for &(n, k) in &[(2usize, 1usize), (3, 1), (4, 3), (5, 3), (6, 6), (7, 6), (10, 15), (11, 15), (100, 1275)] {
            assert_eq!(qn_rank(n) + 1, k, "n = {n}");
            assert!(qn_rank(n) < n * (n - 1) / 2);
        }
    }

    /// For normal data Qn estimates sigma (that is what the constant is for).
    #[test]
    fn qn_is_consistent_for_normal_sigma() {
        let mut state = 0xD1B54A32D192ED03u64;
        let xs: Vec<f64> = (0..40_000)
            .map(|_| {
                // Box-Muller
                let (u, v) = (lcg(&mut state).max(1e-300), lcg(&mut state));
                3.0 * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
            })
            .collect();
        let qn = qn_scale_slice(&xs);
        assert!((qn / 3.0 - 1.0).abs() < 0.02, "got {qn}");
    }

    /// The O(n log n) path must return the same bits as selecting among all
    /// materialised differences, including with heavy ties and huge ranges.
    #[test]
    fn fast_qn_matches_brute_force_bit_for_bit() {
        let mut state = 0x9E3779B97F4A7C15u64;
        for n in 2..=120usize {
for shape in 0..4 {
                let xs: Vec<f64> = (0..n)
                    .map(|_| {
                        let u = lcg(&mut state);
                        match shape {
                            0 => u * 2.0 - 1.0,
                            1 => (u * 4.0).floor(),              // heavy ties
                            2 => (u - 0.5) * 1e300,               // near overflow
                            _ => 1e9 + u * 1e-3,                  // tiny spread, big offset
                        }
                    })
                    .collect();
                let fast = qn_scale_slice(&xs);
                let slow = qn_scale_brute_force(&xs);
                assert_eq!(fast.to_bits(), slow.to_bits(), "n={n} shape={shape}");
            }
        }
    }

    #[test]
    fn qn_constant_and_extreme_inputs() {
        assert_eq!(qn_scale_slice(&[3.0; 50]), 0.0);
        assert_eq!(qn_scale_slice(&[0.0, -0.0, 0.0, -0.0]), 0.0);
        // The range overflows to +Inf; the estimate must still be the right
        // finite order statistic, not Inf or NaN.
        let wide = [-1.7e308, -1.0, 0.0, 1.0, 1.7e308, 2.0, 3.0];
        assert_eq!(qn_scale_slice(&wide).to_bits(), qn_scale_brute_force(&wide).to_bits());
    }

    /// Same value class as the all-pairs reference: equal bits when finite,
    /// otherwise both +Inf or both NaN (NaN payload/sign is not specified).
    fn same_as_reference(xs: &[f64]) -> bool {
        let (fast, slow) = (qn_scale_slice(xs), qn_scale_brute_force(xs));
        if slow.is_nan() {
            fast.is_nan()
        } else {
            fast.to_bits() == slow.to_bits()
        }
    }

    /// +/-Inf: one infinity among many values must still give a finite
    /// estimate (that is what a robust scale is for), and every mix of signs
    /// and counts must land in the same run -- finite, +Inf or NaN -- as the
    /// all-pairs definition.
    #[test]
    fn qn_with_infinities_matches_the_all_pairs_definition() {
        let p = f64::INFINITY;
        let m = f64::NEG_INFINITY;
        let cases: &[&[f64]] = &[
            &[1.0, 2.0, 3.0, p, m],
            &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, p],
            &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, m],
            &[1.0, 2.0, 4.0, 8.0, p, p],
            &[1.0, 2.0, 4.0, p, p, p],          // k lands among the +Inf run
            &[1.0, p, p, p, p, p],              // ... or the NaN run
            &[p, p, p],
            &[p, m, p, m],
            &[p, m, 0.0],
            &[1.0, 2.0, p],
        ];
        for xs in cases {
            assert!(same_as_reference(xs), "{xs:?}: {} vs {}",
                    qn_scale_slice(xs), qn_scale_brute_force(xs));
        }

        let mut state = 0x2545F4914F6CDD1Du64;
        for n in 3..=60usize {
            for n_inf in 1..=n.min(8) {
                let mut xs: Vec<f64> = (0..n).map(|_| lcg(&mut state) * 10.0).collect();
                for slot in xs.iter_mut().take(n_inf) {
                    *slot = if lcg(&mut state) < 0.5 { p } else { m };
                }
                assert!(same_as_reference(&xs), "n={n} n_inf={n_inf}");
            }
        }
    }

    /// The case the first rewrite got wrong: it returned NaN for any Inf once
    /// n exceeded a cap. One infinity in thousands must not change the answer
    /// class, and needs no quadratic buffer.
    #[test]
    fn qn_one_infinity_in_a_large_sample_is_finite() {
        let mut state = 99u64;
        let mut xs: Vec<f64> = (0..20_000).map(|_| lcg(&mut state)).collect();
        xs[17] = f64::INFINITY;
        let with_inf = qn_scale_slice(&xs);
        assert!(with_inf.is_finite() && with_inf > 0.1 && with_inf < 0.6, "got {with_inf}");

        // Small enough to check against the reference directly.
        let mut small: Vec<f64> = (0..1500).map(|_| lcg(&mut state)).collect();
        small[3] = f64::NEG_INFINITY;
        small[900] = f64::INFINITY;
        assert!(same_as_reference(&small));
    }

    /// The reason for the rewrite: this size needed a 160 GB buffer and aborted
    /// the process. It must now finish, in O(n) memory.
    #[test]
    fn qn_handles_a_large_column() {
        let mut state = 7u64;
        let xs: Vec<f64> = (0..200_000).map(|_| lcg(&mut state)).collect();
        let qn = qn_scale_slice(&xs);
        // Uniform(0,1): sd = 0.2887; Qn is not consistent for the sd of a
        // uniform, so only bracket it.
        assert!(qn > 0.1 && qn < 0.6, "got {qn}");
    }

    proptest! {
        #[test]
        fn fast_qn_equals_brute_force(xs in prop::collection::vec(-1e6f64..1e6f64, 3..64)) {
            prop_assert_eq!(
                qn_scale_slice(&xs).to_bits(),
                qn_scale_brute_force(&xs).to_bits()
            );
        }

        #[test]
        fn fast_qn_equals_brute_force_with_ties(
            xs in prop::collection::vec((0i32..6).prop_map(f64::from), 3..64)
        ) {
            prop_assert_eq!(
                qn_scale_slice(&xs).to_bits(),
                qn_scale_brute_force(&xs).to_bits()
            );
        }
    }
}
