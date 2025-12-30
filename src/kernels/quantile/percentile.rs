// src/kernels/quantile/percentile.rs

pub(crate) fn percentile_slice(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    if q.is_nan() {
        return f64::NAN;
    }
    // Match NumPy percentile semantics: q in [0, 100]
    let q01 = (q.max(0.0).min(100.0)) / 100.0;

    // Keep NaN behavior aligned with np.percentile (NaN in input -> NaN out).
    if xs.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }

    let mut v = xs.to_vec();
    let n = v.len();
    if n == 1 {
        return v[0];
    }

    let pos = q01 * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let w = pos - lo as f64;

    // Quickselect (expected O(n)) instead of full sort (O(n log n))
    let lo_val = crate::kernels::quantile::select::select_nth_f64(&mut v, lo);
    if lo == hi {
        return lo_val;
    }
    let hi_val = crate::kernels::quantile::select::select_nth_f64(&mut v, hi);

    lo_val * (1.0 - w) + hi_val * w
}
