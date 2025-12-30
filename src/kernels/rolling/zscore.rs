/// Final linear pass converting rolling mean/std into z-scores.
///
/// Behavior matches legacy `rolling_zscore_np`:
/// - if std == 0 → z = 0.0
/// - if std is NaN/inf → z = NaN
pub(crate) fn zscore_from_mean_std(
    x: &[f64],
    means: &[f64],
    stds: &[f64],
) -> Vec<f64> {
    let n = means.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let s = stds[i];
        let z = if !s.is_finite() {
            f64::NAN
        } else if s > 0.0 {
            (x[i] - means[i]) / s
        } else {
            0.0
        };
        out.push(z);
    }

    out
}
