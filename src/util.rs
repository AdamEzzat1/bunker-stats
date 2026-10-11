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

/// Check that `count` values of `T` can be allocated right now, and raise
/// `MemoryError` if not.
///
/// A caller-chosen count (`n_resamples=10**12`) otherwise reaches `vec![..; n]`
/// or `collect()`, whose allocation failure is not an error the caller can
/// catch: with `panic = "abort"` it kills the interpreter. Reserving and
/// dropping an empty vector asks the allocator the same question up front
/// (and catches the `capacity overflow` panic for counts beyond `isize::MAX`
/// bytes) without touching the memory.
pub(crate) fn ensure_allocatable<T>(count: usize, what: &str) -> pyo3::PyResult<()> {
    let mut probe: Vec<T> = Vec::new();
    probe.try_reserve_exact(count).map_err(|_| {
        pyo3::exceptions::PyMemoryError::new_err(format!(
            "{what}={count} needs {} bytes for its results, which cannot be allocated",
            (count as u128) * (std::mem::size_of::<T>() as u128)
        ))
    })
}

#[cfg(test)]
mod alloc_tests {
    use super::ensure_allocatable;

    #[test]
    fn ordinary_counts_pass_and_absurd_ones_fail() {
        assert!(ensure_allocatable::<f64>(0, "n").is_ok());
        assert!(ensure_allocatable::<f64>(1_000_000, "n").is_ok());
        // beyond isize::MAX bytes: a capacity overflow, not an allocation
        assert!(ensure_allocatable::<f64>(usize::MAX, "n").is_err());
        assert!(ensure_allocatable::<f64>(usize::MAX / 8, "n").is_err());
    }
}
