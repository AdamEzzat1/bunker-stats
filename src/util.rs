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

/// Whether `count` values of `T` can be allocated right now.
///
/// Reserving and dropping an empty vector asks the allocator the question
/// without touching the memory, and reports the `capacity overflow` case
/// (more than `isize::MAX` bytes) as a failure instead of a panic.
pub(crate) fn can_allocate<T>(count: usize) -> bool {
    let mut probe: Vec<T> = Vec::new();
    probe.try_reserve_exact(count).is_ok()
}

/// Raise `MemoryError` unless `count` values of `T` can be allocated.
///
/// A caller-chosen count (`n_resamples=10**12`) otherwise reaches `vec![..; n]`
/// or `collect()`, whose allocation failure is not an error the caller can
/// catch: with `panic = "abort"` it kills the interpreter.
pub(crate) fn ensure_allocatable<T>(count: usize, what: &str) -> pyo3::PyResult<()> {
    if can_allocate::<T>(count) {
        return Ok(());
    }
    Err(pyo3::exceptions::PyMemoryError::new_err(format!(
        "{what}={count} needs {} bytes for its results, which cannot be allocated",
        (count as u128) * (std::mem::size_of::<T>() as u128)
    )))
}

#[cfg(test)]
mod alloc_tests {
    // Only the pure half is tested here. Building a Python exception needs
    // libpython, which `cargo test --lib` does not link on Linux or macOS
    // (the crate is an extension module); the MemoryError itself is covered
    // from Python in tests/test_known_issues.py.
    use super::can_allocate;

    #[test]
    fn ordinary_counts_pass_and_absurd_ones_fail() {
        assert!(can_allocate::<f64>(0));
        assert!(can_allocate::<f64>(1_000_000));
        // beyond isize::MAX bytes: a capacity overflow, not an allocation
        assert!(!can_allocate::<f64>(usize::MAX));
        assert!(!can_allocate::<f64>(usize::MAX / 8));
    }
}
