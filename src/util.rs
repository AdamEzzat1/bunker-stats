//! Small shared numeric helpers.

/// Returns `true` if any element is NaN.
///
/// Order-statistic kernels (median, MAD, IQR, quantiles, …) sort or select on
/// the data. A NaN has no defined order, so the kernels short-circuit to a NaN
/// result — matching numpy's `median`/`percentile` propagation semantics —
/// rather than sorting with a NaN-tolerant comparator, which would silently
/// return a wrong (non-NaN) value. This helper makes that guard uniform.
#[inline]
pub(crate) fn any_nan(xs: &[f64]) -> bool {
    xs.iter().any(|x| x.is_nan())
}
