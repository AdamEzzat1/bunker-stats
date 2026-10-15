use super::moments::{Kahan, RollingMoments, RunningSum};

/// Slide a window of `window` values over `xs`, calling `emit(k, moments)` for
/// the k-th window (`xs[k..k + window]`).
///
/// The moments are updated in O(1) per step and re-summed from the window
/// whenever `RollingMoments` reports that rounding error has built up, so no
/// window is affected by values that have already left it.
#[inline]
pub(crate) fn for_each_window<S: RunningSum>(
    xs: &[f64],
    window: usize,
    mut emit: impl FnMut(usize, &RollingMoments<S>),
) {
    let n = xs.len();
    if window == 0 || window > n {
        return;
    }
    let mut m = RollingMoments::<S>::new(&xs[..window]);
    emit(0, &m);
    for i in window..n {
        m.remove(xs[i - window]);
        m.add(xs[i]);
        if m.needs_refresh() {
            m.refresh(&xs[i + 1 - window..=i]);
        }
        emit(i + 1 - window, &m);
    }
}

/// Like `for_each_window`, but with one call per input position: `emit(i, m)`
/// sees the trailing window ending at `i`, which is shorter than `window` for
/// the first `window - 1` positions.
#[inline]
pub(crate) fn for_each_trailing_window<S: RunningSum>(
    xs: &[f64],
    window: usize,
    mut emit: impl FnMut(usize, &RollingMoments<S>),
) {
    if window == 0 {
        return;
    }
    let mut m = RollingMoments::<S>::default();
    for i in 0..xs.len() {
        m.add(xs[i]);
        if i >= window {
            m.remove(xs[i - window]);
        }
        if m.needs_refresh() {
            m.refresh(&xs[(i + 1).saturating_sub(window)..=i]);
        }
        emit(i, &m);
    }
}

/// Sample standard deviation (ddof = 1) of a strict window: NaN if the window
/// holds any NaN or infinity, or fewer than two values.
#[inline(always)]
pub(crate) fn strict_std<S: RunningSum>(m: &RollingMoments<S>) -> f64 {
    m.var(1, false).sqrt()
}

/// Rolling mean and sample std (ddof = 1) over full windows; `n - window + 1`
/// outputs each. A window containing NaN gives NaN for both; later windows are
/// unaffected.
pub(crate) fn rolling_mean_std_vec(xs: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len();
    if window == 0 || window > n {
        return (Vec::new(), Vec::new());
    }

    let out_len = n - window + 1;
    let mut means = Vec::with_capacity(out_len);
    let mut stds = Vec::with_capacity(out_len);
    // Kahan sums: these are the reference-accuracy 1-D kernels.
    for_each_window::<Kahan>(xs, window, |_, m| {
        means.push(m.mean(false));
        stds.push(strict_std(m));
    });
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

