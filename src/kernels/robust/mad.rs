/// Median Absolute Deviation (MAD): median(|x - median(x)|).
/// Behavior matches the legacy lib.rs helper.
pub(crate) fn mad_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    // Copy once, sort once for the median, then reuse the same buffer for deviations.
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    };

    // Reuse `v` to hold absolute deviations, then sort to get the MAD median.
    for val in &mut v {
        *val = (*val - med).abs();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
