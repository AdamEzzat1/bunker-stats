use crate::kernels::quantile::percentile::percentile_slice;

/// Interpret q as either:
/// - quantile in [0, 1]  (e.g., 0.05, 0.95)
/// - percentile in [0, 100] (e.g., 5.0, 95.0)
#[inline]
fn normalize_q(q: f64) -> f64 {
    if q.is_nan() {
        return f64::NAN;
    }
    // If caller passes quantile-style [0,1], convert to [0,100]
    if (0.0..=1.0).contains(&q) {
        q * 100.0
    } else {
        q
    }
}

pub(crate) fn winsor_bounds(xs: &[f64], lower_q: f64, upper_q: f64) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let lq = normalize_q(lower_q);
    let uq = normalize_q(upper_q);

    let low = percentile_slice(xs, lq);
    let high = percentile_slice(xs, uq);

    (low, high)
}

pub(crate) fn winsorize_vec(xs: &[f64], lower_q: f64, upper_q: f64) -> Vec<f64> {
    if xs.is_empty() {
        return vec![];
    }
    let (low, high) = winsor_bounds(xs, lower_q, upper_q);

    xs.iter()
        .map(|&x| if x < low { low } else if x > high { high } else { x })
        .collect()
}
