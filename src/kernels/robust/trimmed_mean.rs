use crate::mean_slice;

/// Trimmed mean: mean after removing proportion_to_cut from each tail
///
/// For example, with proportion_to_cut=0.1, removes 10% from bottom and 10% from top,
/// then computes the mean of the middle 80%.
///
/// Returns NaN if input is empty or if trimming would remove all data.
///
/// Optimized with inline hint for better optimization.
#[inline]
pub(crate) fn trimmed_mean_slice(xs: &[f64], proportion_to_cut: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }

    // Contract: invalid/over-trimming returns NaN.
    // Tests expect proportion_to_cut >= 0.5 to yield NaN.
    if !proportion_to_cut.is_finite() || proportion_to_cut < 0.0 || proportion_to_cut >= 0.5 {
        return f64::NAN;
    }

    // NaN in input -> NaN out (numpy semantics) and avoids a NaN reaching the
    // comparator (which under panic="abort" would crash the interpreter).
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

    mean_slice(&v[cut..(n - cut)])
}
