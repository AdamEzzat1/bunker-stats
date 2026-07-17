// src/kernels/quantile/select.rs
//
// Selection helpers used by percentile/quantile kernels.

/// Select the k-th smallest element (0-indexed) from `v` in-place using Rust's
/// `select_nth_unstable_by`. This matches the prior inline logic.
pub(crate) fn select_nth_f64(v: &mut [f64], k: usize) -> f64 {
    // total_cmp is a total order over f64, so selection never panics even if a
    // caller passes NaN (callers that need numpy NaN-propagation guard upstream).
    let (_, m, _) = v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *m
}
