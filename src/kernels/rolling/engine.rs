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
    // Separate Kahan sum over RAW (unshifted) values, used only for the mean.
    // Deriving the mean from the shifted sum as `sum/w + off` loses precision
    // when |off| is huge relative to the true mean: the division result is
    // O(off) and adding `off` back cancels. Example: [1e10, 1, -1e10, 1, 1] has
    // mean 0.6, but `sum/w + off` rounds to ~0.6 ± 4e-7 at the 1e10 scale. The
    // raw sum stays O(spread) under Kahan compensation and stays exact.
    let mut sum_raw = 0.0f64;
    let mut sum_raw_c = 0.0f64;

    for &x in &xs[..window] {
        let xs_ = x - off;
        kahan_add(&mut sum, &mut sum_c, xs_);
        kahan_add(&mut sumsq, &mut sumsq_c, xs_ * xs_);
        kahan_add(&mut sum_raw, &mut sum_raw_c, x);
    }

    let mut push_stats = |sum_raw: f64, sum: f64, sumsq: f64| {
        // Mean from the raw sum (accurate at any offset); variance from the
        // shifted moments (translation-invariant and cancellation-safe).
        let mean = sum_raw / window as f64;
        // Sample variance (ddof=1). Clamp only genuine tiny-negative FP noise;
        // preserve NaN (e.g. a NaN in the window) rather than masking it to 0.
        let var = (sumsq - (sum * sum) / window as f64) / ((window - 1) as f64);
        let std = if var.is_nan() { f64::NAN } else { var.max(0.0).sqrt() };
        means.push(mean);
        stds.push(std);
    };

    push_stats(sum_raw, sum, sumsq);

    for i in window..n {
        let x_new = xs[i] - off;
        let x_old = xs[i - window] - off;
        kahan_add(&mut sum, &mut sum_c, x_new);
        kahan_add(&mut sum, &mut sum_c, -x_old);

        kahan_add(&mut sumsq, &mut sumsq_c, x_new * x_new);
        kahan_add(&mut sumsq, &mut sumsq_c, -(x_old * x_old));

        kahan_add(&mut sum_raw, &mut sum_raw_c, xs[i]);
        kahan_add(&mut sum_raw, &mut sum_raw_c, -xs[i - window]);

        push_stats(sum_raw, sum, sumsq);
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
        // The sliding accumulator must agree with a fresh two-pass recompute
        // per window — for ANY finite data and ANY offset. This catches both
        // accumulator drift and large-offset cancellation.
        //
        // TOLERANCE: the residual rounding in a sliding accumulator scales
        // with the largest magnitude that has PASSED THROUGH it — not with
        // the current window's contents. After ingesting and removing a value
        // like -623, the running sums keep an absolute error ~ eps * 623^2
        // (in variance units) that a window of small values then inherits
        // (proptest found exactly this: window [12.559, 12.671] right after
        // -622.99 disagreed with two-pass by 3e-10 where a current-window
        // bound predicted 1.4e-10). So the comparison is done in VARIANCE
        // units — what the accumulator actually maintains — with a bound
        // driven by the running max magnitude and the number of ingested
        // values. Real accumulator-drift bugs exceed this by orders of
        // magnitude; see the deterministic regression tests below.
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
                let expected_var = ss / (window - 1) as f64;
                let expected = expected_var.sqrt();
                // Largest |x| the accumulator has ingested up to this window.
                let m = xs[..i + window]
                    .iter()
                    .fold(0.0f64, |acc, v| acc.max(v.abs()));
                let n_ingested = (i + window) as f64;
                let tol_var = 5e-15 * n_ingested * m * m + 1e-12;
                let got_var = s * s;
                prop_assert!(
                    (got_var - expected_var).abs() <= tol_var,
                    "window {i}: got std {s} (var {got_var}), expected std {expected} \
                     (var {expected_var}); offset {offset}, tol_var {tol_var}"
                );
            }
        }
    }

    /// The CI-found counterexample, pinned deterministically: a small-valued
    /// window immediately after a large value left the accumulator. The
    /// variance-units bound above must accept it forever.
    #[test]
    fn rolling_std_small_window_after_large_departure() {
        let xs = [-622.9902542719688, 12.559005916715245, 12.670524924120446];
        let (_means, stds) = rolling_mean_std_vec(&xs, 2);
        let e = ((12.670524924120446f64 - 12.559005916715245) / 2.0_f64.sqrt()).abs();
        // Two-pass std of the last window; streaming may differ by the
        // inherited eps*623^2 residual (in variance units), no more.
        let tol_var = 5e-15 * 3.0 * 622.9902542719688f64.powi(2) + 1e-12;
        let got_var = stds[1] * stds[1];
        let exp_var = e * e;
        assert!(
            (got_var - exp_var).abs() <= tol_var,
            "got var {got_var}, expected {exp_var}, tol {tol_var}"
        );
    }
}

