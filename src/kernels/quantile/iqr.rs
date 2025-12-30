// src/kernels/quantile/iqr.rs

pub(crate) fn iqr_from_sorted(sorted: &[f64]) -> (f64, f64, f64) {
    if sorted.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = sorted.len();
    let q1_pos = 0.25 * ((n - 1) as f64);
    let q3_pos = 0.75 * ((n - 1) as f64);

    let interp = |pos: f64| -> f64 {
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let w = pos - (lo as f64);
            (1.0 - w) * sorted[lo] + w * sorted[hi]
        }
    };

    let q1 = interp(q1_pos);
    let q3 = interp(q3_pos);
    (q1, q3, q3 - q1)
}

pub(crate) fn iqr_slice(xs: &[f64]) -> (f64, f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    iqr_from_sorted(&v)
}
