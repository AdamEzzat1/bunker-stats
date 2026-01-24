/// Median Absolute Deviation (MAD): median(|x - median(x)|).
/// 
/// Optimized implementation with:
/// - Bit operations for even/odd checks (n & 1 vs n % 2)
/// - Bit shift for division by 2 (n >> 1 vs n / 2)
/// - Multiplication instead of division (x * 0.5 vs x / 2.0)
/// - Inline hint for better optimization
#[inline]
pub(crate) fn mad_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    // Copy once, sort once for the median, then reuse the same buffer for deviations.
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    // Optimized: use bitwise AND for even/odd check, bit shift for division
    let med = if n & 1 == 1 {
        v[n >> 1]
    } else {
        let mid = n >> 1;
        (v[mid - 1] + v[mid]) * 0.5
    };

    // Reuse `v` to hold absolute deviations, then sort to get the MAD median.
    for val in &mut v {
        *val = (*val - med).abs();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if n & 1 == 1 {
        v[n >> 1]
    } else {
        let mid = n >> 1;
        (v[mid - 1] + v[mid]) * 0.5
    }
}
