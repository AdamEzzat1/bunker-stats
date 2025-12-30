// src/kernels/rolling/var.rs

/// Final linear pass converting rolling standard deviations into variances.
///
/// This is intentionally a thin post-pass used by `rolling_var_np`.
/// It preserves NaNs/Infs naturally through multiplication.
pub(crate) fn vars_from_stds(stds: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(stds.len());
    for &s in stds {
        out.push(s * s);
    }
    out
}
