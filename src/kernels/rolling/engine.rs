pub(crate) fn rolling_mean_std_vec(xs: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len();
    if window == 0 || window > n {
        return (Vec::new(), Vec::new());
    }

    // Kahan-compensated rolling sums (more accurate than naive sum/sumsq, minimal overhead)
    #[inline(always)]
    fn kahan_add(sum: &mut f64, c: &mut f64, x: f64) {
        let y = x - *c;
        let t = *sum + y;
        *c = (t - *sum) - y;
        *sum = t;
    }

    let out_len = n - window + 1;
    let mut means = Vec::with_capacity(out_len);
    let mut stds = Vec::with_capacity(out_len);

    // Variance is translation-invariant, so we accumulate on values shifted by a
    // finite constant `off`. Without this, `sumsq - sum^2/w` subtracts two
    // O(magnitude^2) quantities whose difference is only O(spread^2): for
    // large-offset data (e.g. ~1e8) that is catastrophic cancellation, and the
    // old `.max(0.0)` then silently reported a *wrong zero* variance. Shifting by
    // a value near the data magnitude keeps the accumulators O(spread^2).
    // `off` must be finite so a NaN at xs[0] does not poison every window.
    let off = if xs[0].is_finite() { xs[0] } else { 0.0 };

    let mut sum = 0.0f64;
    let mut sum_c = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut sumsq_c = 0.0f64;

    for &x in &xs[..window] {
        let xs_ = x - off;
        kahan_add(&mut sum, &mut sum_c, xs_);
        kahan_add(&mut sumsq, &mut sumsq_c, xs_ * xs_);
    }

    let mut push_stats = |sum: f64, sumsq: f64| {
        // `sum`/`sumsq` are over shifted values; add `off` back for the mean.
        let mean = sum / window as f64 + off;
        // Sample variance (ddof=1). Clamp only genuine tiny-negative FP noise;
        // preserve NaN (e.g. a NaN in the window) rather than masking it to 0.
        let var = (sumsq - (sum * sum) / window as f64) / ((window - 1) as f64);
        let std = if var.is_nan() { f64::NAN } else { var.max(0.0).sqrt() };
        means.push(mean);
        stds.push(std);
    };

    push_stats(sum, sumsq);

    for i in window..n {
        let x_new = xs[i] - off;
        let x_old = xs[i - window] - off;
        kahan_add(&mut sum, &mut sum_c, x_new);
        kahan_add(&mut sum, &mut sum_c, -x_old);

        kahan_add(&mut sumsq, &mut sumsq_c, x_new * x_new);
        kahan_add(&mut sumsq, &mut sumsq_c, -(x_old * x_old));

        push_stats(sum, sumsq);
    }

    (means, stds)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for catastrophic cancellation. Three consecutive integers have
    /// sample std (ddof=1) of exactly 1.0 regardless of their offset, so windows
    /// over base-1e8 data must still report 1.0. The old naive
    /// `sumsq - sum^2/w` lost all precision at this magnitude and the `.max(0.0)`
    /// masked the resulting garbage as a wrong value.
    #[test]
    fn test_rolling_std_large_offset_no_cancellation() {
        let base = 1e8;
        let xs: Vec<f64> = (1..=6).map(|k| base + k as f64).collect();
        let (means, stds) = rolling_mean_std_vec(&xs, 3);
        assert_eq!(stds.len(), 4);
        for (i, s) in stds.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-9, "window {i}: std={s}, expected 1.0");
        }
        // Means must also be recovered exactly (offset added back).
        assert!((means[0] - (base + 2.0)).abs() < 1e-6);
    }

    /// A constant window has zero variance — and must not produce a spurious
    /// tiny-negative-then-masked value or NaN.
    #[test]
    fn test_rolling_std_constant_window_is_zero() {
        let xs = vec![5.0, 5.0, 5.0, 5.0];
        let (_means, stds) = rolling_mean_std_vec(&xs, 3);
        for s in &stds {
            assert_eq!(*s, 0.0);
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // The sliding accumulator must agree with a fresh two-pass recompute per
        // window to ~1e-9 relative — for ANY finite data and ANY offset. This
        // catches both accumulator drift and large-offset cancellation. numpy's
        // rolling std uses exactly such a per-window computation.
        #[test]
        fn rolling_std_matches_two_pass(
            data in prop::collection::vec(-1e3f64..1e3f64, 2..40),
            window in 2usize..12,
            offset in prop::sample::select(vec![0.0f64, 1e3, 1e6, 1e8]),
        ) {
            prop_assume!(window <= data.len());
            let xs: Vec<f64> = data.iter().map(|v| v + offset).collect();
            let (_means, stds) = rolling_mean_std_vec(&xs, window);
            prop_assert_eq!(stds.len(), xs.len() - window + 1);
            for (i, s) in stds.iter().enumerate() {
                let mean = xs[i..i + window].iter().sum::<f64>() / window as f64;
                let ss: f64 = xs[i..i + window].iter().map(|v| (v - mean) * (v - mean)).sum();
                let expected = (ss / (window - 1) as f64).sqrt();
                prop_assert!(
                    (s - expected).abs() <= 1e-9 * (1.0 + expected.abs()),
                    "window {i}: got {s}, expected {expected} (offset {offset})"
                );
            }
        }
    }
}

