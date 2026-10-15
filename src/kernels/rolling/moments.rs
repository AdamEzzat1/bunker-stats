// src/kernels/rolling/moments.rs
//
// Sliding-window moments that cannot drift.
//
// The O(1)-per-step rolling kernels keep running sums and update them by
// adding the value that enters the window and subtracting the one that leaves.
// Three things go wrong with that if nothing else is done:
//
// 1. Rounding error accumulates and never leaves. After a value of 1e12 has
//    passed through, the sums carry an error of ~1e24 * eps, which is larger
//    than the variance of ordinary data: every later window is garbage.
// 2. A NaN or infinity poisons the sums for good (NaN - NaN is NaN), so every
//    later window reports NaN even though it no longer contains one.
// 3. Moments taken about a fixed origin cancel catastrophically once the data
//    has wandered far from it.
//
// `RollingMoments` and `RollingPairMoments` fix all three:
//
// * Non-finite values are counted, never summed, so they leave the window
//   cleanly.
// * Sums are compensated (Kahan) and taken about an offset near the window
//   mean.
// * They track the largest deviation summed since the sums were last built and
//   how many updates have happened. Together those bound the rounding error
//   the sums can hold. When that bound is no longer small next to the quantity
//   the caller will read (variance, covariance, mean), `needs_refresh` says so
//   and the caller hands over the current window to be re-summed from scratch
//   around a fresh offset. On well-behaved data this happens about once per
//   hundred thousand steps; after an outlier leaves the window it happens once,
//   immediately.
//
// Everything is deterministic: the decision to refresh depends only on the
// data seen so far. The check uses no division or square root.

/// The sums are rebuilt once their rounding-error bound exceeds this fraction
/// of the statistic they feed. The bound is a worst case for plain summation
/// while the sums are compensated, so real errors are far smaller.
const REL_TOL: f64 = 1e-9;
const EPS: f64 = f64::EPSILON;

/// A running sum. The refresh logic is the same for both kinds; they differ
/// only in how much rounding error builds up between refreshes.
pub(crate) trait RunningSum: Copy + Default {
    fn add(&mut self, x: f64);
    fn value(&self) -> f64;
}

/// Kahan-compensated sum: about twice the cost of `Plain`, and accurate to the
/// last few bits between refreshes.
#[derive(Clone, Copy, Default)]
pub(crate) struct Kahan {
    sum: f64,
    comp: f64,
}

impl RunningSum for Kahan {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        let y = x - self.comp;
        let t = self.sum + y;
        self.comp = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.sum
    }
}

/// Ordinary sum: fastest, with error bounded by `REL_TOL` through refreshes
/// (in practice several orders of magnitude below it).
#[derive(Clone, Copy, Default)]
pub(crate) struct Plain(f64);

impl RunningSum for Plain {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        self.0 += x;
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.0
    }
}

/// Mean and sum of squared deviations of the values in a sliding window.
#[derive(Clone, Default)]
pub(crate) struct RollingMoments<S: RunningSum = Kahan> {
    off: f64,
    sum: S,   // sum of (x - off) over the finite values
    sumsq: S, // sum of (x - off)^2 over the finite values
    /// Largest (x - off)^2 summed since the last refresh, in or out of the window.
    peak_sq: f64,
    /// Updates since the last refresh.
    steps: f64,
    finite: usize,
    nan: usize,
    pos_inf: usize,
    neg_inf: usize,
}

impl<S: RunningSum> RollingMoments<S> {
    /// Moments of `window`, summed from scratch.
    pub fn new(window: &[f64]) -> Self {
        let mut m = Self::default();
        m.refresh(window);
        m
    }

    /// Re-derive everything from the current window, about a fresh offset.
    pub fn refresh(&mut self, window: &[f64]) {
        *self = Self::default();
        let mut raw = Kahan::default();
        for &x in window {
            if x.is_finite() {
                raw.add(x);
                self.finite += 1;
            } else {
                self.count_non_finite(x, true);
            }
        }
        if self.finite > 0 {
            let off = raw.value() / self.finite as f64;
            // A window of huge values can overflow the raw sum; any finite
            // offset is valid, a nearby one is merely better conditioned.
            self.off = if off.is_finite() { off } else { 0.0 };
        }
        for &x in window {
            if x.is_finite() {
                let d = x - self.off;
                let d2 = d * d;
                self.sum.add(d);
                self.sumsq.add(d2);
                self.peak_sq = self.peak_sq.max(d2);
            }
        }
    }

    #[cold]
    fn count_non_finite(&mut self, x: f64, entering: bool) {
        let slot = if x.is_nan() {
            &mut self.nan
        } else if x > 0.0 {
            &mut self.pos_inf
        } else {
            &mut self.neg_inf
        };
        if entering {
            *slot += 1;
        } else {
            *slot -= 1;
        }
    }

    /// A value enters the window.
    #[inline(always)]
    pub fn add(&mut self, x: f64) {
        if x.is_finite() {
            let d = x - self.off;
            let d2 = d * d;
            self.sum.add(d);
            self.sumsq.add(d2);
            self.peak_sq = self.peak_sq.max(d2);
            self.steps += 1.0;
            self.finite += 1;
        } else {
            self.count_non_finite(x, true);
        }
    }

    /// A value leaves the window. Must be one that was added.
    #[inline(always)]
    pub fn remove(&mut self, x: f64) {
        if x.is_finite() {
            let d = x - self.off;
            self.sum.add(-d);
            self.sumsq.add(-(d * d));
            self.steps += 1.0;
            self.finite -= 1;
        } else {
            self.count_non_finite(x, false);
        }
    }

    /// True when the sums are no longer trustworthy for the statistics below;
    /// the caller should then call `refresh` with the current window.
    ///
    /// With c finite values, none larger than sqrt(peak_sq) in deviation, each
    /// update of `sumsq` rounds by at most eps * c * peak_sq, so after `steps`
    /// updates it is off by at most `bound_q` below. Everything is kept
    /// multiplied through by c so the test needs no division or square root.
    #[inline(always)]
    pub fn needs_refresh(&self) -> bool {
        if self.steps == 0.0 {
            return false;
        }
        let c = self.finite as f64;
        let s = self.sum.value();
        if self.finite >= 2 {
            // Error in c * (sum of squared deviations) is at most
            // c * 3 * bound_q, plus the one-off cancellation of `q - s^2/c`
            // once the offset has gone stale.
            //
            // Passing this also bounds the error of the mean by REL_TOL times
            // the spread: the mean's bound squared is k^2 * peak_sq with
            // k = eps * steps * c, and this test caps k * peak_sq at
            // REL_TOL * c_ss / (3c) and steps below REL_TOL / eps.
            let q = self.sumsq.value();
            let c_ss = q * c - s * s;
            let bound_q = EPS * self.steps * c * self.peak_sq;
            return !(c * (3.0 * bound_q + EPS * q) <= REL_TOL * c_ss);
        }
        if self.finite == 0 {
            // Nothing left to be wrong about; drop any residue.
            return true;
        }
        // A single value: its "sum" should be that value to within a few ulps.
        let k = EPS * self.steps;
        let total = self.off + s;
        !(k * k * self.peak_sq <= 64.0 * EPS * EPS * total * total)
    }

    /// Number of finite values in the window.
    #[inline(always)]
    pub fn finite_count(&self) -> usize {
        self.finite
    }

    /// Number of NaNs in the window.
    #[inline(always)]
    pub fn nan_count(&self) -> usize {
        self.nan
    }

    /// True if the window holds an infinity.
    #[inline(always)]
    pub fn has_inf(&self) -> bool {
        self.pos_inf + self.neg_inf > 0
    }

    /// Mean of the finite values (NaN if there are none). Ignores NaN and
    /// infinities entirely; see `mean` for IEEE semantics.
    #[inline(always)]
    pub fn finite_mean(&self) -> f64 {
        if self.finite == 0 {
            f64::NAN
        } else {
            self.off + self.sum.value() / self.finite as f64
        }
    }

    /// Mean with IEEE semantics for infinities: +inf or -inf if only one sign
    /// is present, NaN if both are. NaNs make it NaN unless `skip_nan`.
    #[inline(always)]
    pub fn mean(&self, skip_nan: bool) -> f64 {
        if self.nan + self.pos_inf + self.neg_inf == 0 {
            return self.finite_mean();
        }
        if self.nan > 0 && !skip_nan {
            return f64::NAN;
        }
        match (self.pos_inf > 0, self.neg_inf > 0) {
            (true, true) => f64::NAN,
            (true, false) => f64::INFINITY,
            (false, true) => f64::NEG_INFINITY,
            (false, false) => self.finite_mean(),
        }
    }

    /// Sum of squared deviations of the finite values, clamped at zero.
    #[inline(always)]
    pub fn finite_ss(&self) -> f64 {
        if self.finite == 0 {
            return f64::NAN;
        }
        let s = self.sum.value();
        (self.sumsq.value() - s * s / self.finite as f64).max(0.0)
    }

    /// Variance with `ddof` delta degrees of freedom. NaN if the window holds
    /// an infinity, a NaN (unless `skip_nan`), or `ddof` or fewer usable values.
    #[inline(always)]
    pub fn var(&self, ddof: usize, skip_nan: bool) -> f64 {
        if self.has_inf() || (self.nan > 0 && !skip_nan) || self.finite <= ddof {
            return f64::NAN;
        }
        self.finite_ss() / (self.finite - ddof) as f64
    }
}

/// Co-moments of the (x, y) pairs in a sliding window. A pair counts only if
/// both values are finite; other pairs are tallied in `bad_count`.
#[derive(Clone, Default)]
pub(crate) struct RollingPairMoments {
    off_x: f64,
    off_y: f64,
    sx: Plain,
    sy: Plain,
    sxx: Plain,
    syy: Plain,
    sxy: Plain,
    peak_xx: f64,
    peak_yy: f64,
    steps: f64,
    good: usize,
    bad: usize,
}

impl RollingPairMoments {
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let mut m = Self::default();
        m.refresh(xs, ys);
        m
    }

    /// Re-derive everything from the current window, about fresh offsets.
    pub fn refresh(&mut self, xs: &[f64], ys: &[f64]) {
        debug_assert_eq!(xs.len(), ys.len());
        *self = Self::default();
        let (mut rx, mut ry) = (Kahan::default(), Kahan::default());
        for (&x, &y) in xs.iter().zip(ys) {
            if x.is_finite() && y.is_finite() {
                rx.add(x);
                ry.add(y);
                self.good += 1;
            } else {
                self.bad += 1;
            }
        }
        if self.good > 0 {
            let (ox, oy) = (rx.value() / self.good as f64, ry.value() / self.good as f64);
            self.off_x = if ox.is_finite() { ox } else { 0.0 };
            self.off_y = if oy.is_finite() { oy } else { 0.0 };
        }
        for (&x, &y) in xs.iter().zip(ys) {
            if x.is_finite() && y.is_finite() {
                self.accumulate::<false>(x - self.off_x, y - self.off_y);
            }
        }
        self.steps = 0.0;
    }

    #[inline(always)]
    fn accumulate<const LEAVING: bool>(&mut self, dx: f64, dy: f64) {
        let (dxx, dyy, dxy) = (dx * dx, dy * dy, dx * dy);
        if LEAVING {
            self.sx.add(-dx);
            self.sy.add(-dy);
            self.sxx.add(-dxx);
            self.syy.add(-dyy);
            self.sxy.add(-dxy);
        } else {
            self.sx.add(dx);
            self.sy.add(dy);
            self.sxx.add(dxx);
            self.syy.add(dyy);
            self.sxy.add(dxy);
            self.peak_xx = self.peak_xx.max(dxx);
            self.peak_yy = self.peak_yy.max(dyy);
        }
        self.steps += 1.0;
    }

    /// A pair enters the window.
    #[inline(always)]
    pub fn add(&mut self, x: f64, y: f64) {
        if x.is_finite() && y.is_finite() {
            self.accumulate::<false>(x - self.off_x, y - self.off_y);
            self.good += 1;
        } else {
            self.bad += 1;
        }
    }

    /// A pair leaves the window. Must be one that was added.
    #[inline(always)]
    pub fn remove(&mut self, x: f64, y: f64) {
        if x.is_finite() && y.is_finite() {
            self.accumulate::<true>(x - self.off_x, y - self.off_y);
            self.good -= 1;
        } else {
            self.bad -= 1;
        }
    }

    /// Centered sums: (S_xx, S_yy, S_xy), each already minus the mean terms.
    #[inline(always)]
    fn centered(&self) -> (f64, f64, f64) {
        let c = self.good as f64;
        let (sx, sy) = (self.sx.value(), self.sy.value());
        (
            self.sxx.value() - sx * sx / c,
            self.syy.value() - sy * sy / c,
            self.sxy.value() - sx * sy / c,
        )
    }

    /// True when the sums should be re-derived; see `RollingMoments::needs_refresh`.
    #[inline(always)]
    pub fn needs_refresh(&self) -> bool {
        if self.steps == 0.0 {
            return false;
        }
        if self.good < 2 {
            // No second moment is read from fewer than two pairs; just make
            // sure residue from earlier windows does not outlive them.
            return self.good == 0;
        }
        let c = self.good as f64;
        let (sx, sy) = (self.sx.value(), self.sy.value());
        let (qxx, qyy) = (self.sxx.value(), self.syy.value());
        let (c_xx, c_yy) = (qxx * c - sx * sx, qyy * c - sy * sy);
        let k = EPS * self.steps * c;

        if !(c * (3.0 * k * self.peak_xx + EPS * qxx) <= REL_TOL * c_xx)
            || !(c * (3.0 * k * self.peak_yy + EPS * qyy) <= REL_TOL * c_yy)
        {
            return true;
        }
        // Cross term, judged in correlation units (so a covariance that is
        // genuinely near zero does not force a refresh on every step) and in
        // squared form: (3 * k * sqrt(peak_xx * peak_yy) * c)^2 against
        // (tol * sqrt(c_xx * c_yy))^2.
        let lhs = 9.0 * k * k * c * c * self.peak_xx * self.peak_yy;
        !(lhs <= REL_TOL * REL_TOL * c_xx * c_yy)
    }

    /// Pairs with both values finite.
    #[inline(always)]
    pub fn good_count(&self) -> usize {
        self.good
    }

    /// Pairs with a NaN or infinity in either value.
    #[inline(always)]
    pub fn bad_count(&self) -> usize {
        self.bad
    }

    /// Sample covariance (ddof = 1) of the good pairs; NaN with fewer than two.
    #[inline(always)]
    pub fn cov(&self) -> f64 {
        if self.good < 2 {
            return f64::NAN;
        }
        self.centered().2 / (self.good - 1) as f64
    }

    /// Sample variance of x over the good pairs; NaN with fewer than two.
    #[inline(always)]
    pub fn var_x(&self) -> f64 {
        if self.good < 2 {
            return f64::NAN;
        }
        self.centered().0.max(0.0) / (self.good - 1) as f64
    }

    /// Pearson correlation of the good pairs, clamped to [-1, 1]; NaN with
    /// fewer than two pairs or when either side has no variance.
    #[inline(always)]
    pub fn corr(&self) -> f64 {
        if self.good < 2 {
            return f64::NAN;
        }
        let (cxx, cyy, cxy) = self.centered();
        if !(cxx > 0.0) || !(cyy > 0.0) {
            return f64::NAN;
        }
        let denom = (cxx * cyy).sqrt();
        if denom == 0.0 || denom.is_nan() {
            return f64::NAN;
        }
        (cxy / denom).clamp(-1.0, 1.0)
    }

    /// Least-squares slope and intercept of y on x over the good pairs, or
    /// `None` with fewer than two pairs or no variance in x.
    #[inline(always)]
    pub fn linreg(&self) -> Option<(f64, f64)> {
        if self.good < 2 {
            return None;
        }
        let c = self.good as f64;
        let (cxx, _, cxy) = self.centered();
        if !(cxx > 0.0) || !cxx.is_finite() {
            return None;
        }
        let slope = cxy / cxx;
        let mean_x = self.off_x + self.sx.value() / c;
        let mean_y = self.off_y + self.sy.value() / c;
        Some((slope, mean_y - slope * mean_x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn lcg(len: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    /// Direct two-pass statistics of one window: (mean, ss) over finite values.
    fn direct(win: &[f64]) -> (f64, f64, usize) {
        let v: Vec<f64> = win.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return (f64::NAN, f64::NAN, 0);
        }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let m = m + v.iter().map(|x| x - m).sum::<f64>() / v.len() as f64;
        (m, v.iter().map(|x| (x - m) * (x - m)).sum(), v.len())
    }

    fn direct_pair(xs: &[f64], ys: &[f64]) -> (f64, f64, f64, usize) {
        let p: Vec<(f64, f64)> =
            xs.iter().zip(ys).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).collect();
        if p.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN, 0);
        }
        let c = p.len() as f64;
        let (mx, my) = (p.iter().map(|q| q.0).sum::<f64>() / c, p.iter().map(|q| q.1).sum::<f64>() / c);
        (
            p.iter().map(|q| (q.0 - mx) * (q.0 - mx)).sum(),
            p.iter().map(|q| (q.1 - my) * (q.1 - my)).sum(),
            p.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum(),
            p.len(),
        )
    }

    /// Slide over `xs`, checking every window against `direct`. Returns the
    /// number of refreshes.
    fn check_series(xs: &[f64], w: usize, tol: f64) -> usize {
        let mut m: RollingMoments = RollingMoments::new(&xs[..w]);
        let mut refreshes = 0;
        for i in (w - 1)..xs.len() {
            if i >= w {
                m.remove(xs[i - w]);
                m.add(xs[i]);
                if m.needs_refresh() {
                    m.refresh(&xs[i + 1 - w..=i]);
                    refreshes += 1;
                }
            }
            let win = &xs[i + 1 - w..=i];
            let (mean, ss, count) = direct(win);
            assert_eq!(m.finite_count(), count, "count at {i}");
            if count == 0 {
                assert!(m.finite_mean().is_nan());
                continue;
            }
            let spread = (ss / count as f64).sqrt();
            let mean_tol = tol * spread + 1e-13 * mean.abs();
            assert!((m.finite_mean() - mean).abs() <= mean_tol, "mean at {i}: {} vs {mean}", m.finite_mean());
            assert!((m.finite_ss() - ss).abs() <= tol * ss + 1e-290, "ss at {i}: {} vs {ss}", m.finite_ss());
        }
        refreshes
    }

    #[test]
    fn ordinary_data_stays_accurate_and_rarely_refreshes() {
        let xs = lcg(200_000, 1);
        let refreshes = check_series(&xs, 50, 1e-9);
        assert!(refreshes <= 2, "{refreshes} refreshes on benign data");
    }

    #[test]
    fn a_burst_of_huge_values_leaves_no_trace() {
        let mut xs = lcg(20_000, 2);
        for v in &mut xs[1000..1100] {
            *v *= 1.0e12;
        }
        xs[5000] = -3.0e15;
        check_series(&xs, 50, 1e-9);
    }

    #[test]
    fn levels_far_from_zero_and_trends_stay_accurate() {
        let noise = lcg(50_000, 3);
        let offset: Vec<f64> = noise.iter().map(|v| v + 1.0e9).collect();
        check_series(&offset, 30, 1e-6); // data itself only resolves ~1e-7 here

        let trend: Vec<f64> = noise.iter().enumerate().map(|(i, v)| v + 0.5 * i as f64).collect();
        let refreshes = check_series(&trend, 100, 1e-9);
        assert!(refreshes < trend.len() / 50, "{refreshes} refreshes on a linear trend");

        let mut walk = noise.clone();
        for i in 1..walk.len() {
            walk[i] += walk[i - 1];
        }
        check_series(&walk, 20, 1e-9);
    }

    #[test]
    fn constant_and_stepped_data_do_not_refresh_forever() {
        let mut xs = vec![7.5; 10_000];
        xs[5000..].iter_mut().for_each(|v| *v = -1234.25);
        let refreshes = check_series(&xs, 64, 1e-9);
        assert!(refreshes <= 80, "{refreshes} refreshes on stepped constant data");
    }

    #[test]
    fn non_finite_values_leave_the_window_cleanly() {
        let mut xs = lcg(400, 4);
        xs[50] = f64::NAN;
        xs[120] = f64::INFINITY;
        xs[200] = f64::NEG_INFINITY;
        xs[201] = f64::INFINITY;
        let w = 10;
        let mut m: RollingMoments = RollingMoments::new(&xs[..w]);
        for i in (w - 1)..xs.len() {
            if i >= w {
                m.remove(xs[i - w]);
                m.add(xs[i]);
                if m.needs_refresh() {
                    m.refresh(&xs[i + 1 - w..=i]);
                }
            }
            let win = &xs[i + 1 - w..=i];
            let naive: f64 = win.iter().sum::<f64>() / w as f64;
            // Strict mean has exactly numpy's per-window value class.
            assert_eq!(m.mean(false).is_nan(), naive.is_nan(), "nan-ness at {i}");
            if naive.is_infinite() {
                assert_eq!(m.mean(false), naive, "at {i}");
            }
            let clean = win.iter().all(|v| v.is_finite());
            assert_eq!(m.var(1, false).is_nan(), !clean, "var nan-ness at {i}");
            if win.iter().any(|v| v.is_nan()) && !win.iter().any(|v| v.is_infinite()) {
                assert!(m.var(1, true).is_finite(), "skip_nan var at {i}");
                assert!(m.mean(true).is_finite());
            }
        }
        check_series(&xs, w, 1e-9);
    }

    #[test]
    fn pair_moments_survive_bursts_and_bad_pairs() {
        let n = 30_000;
        let mut xs = lcg(n, 5);
        let mut ys: Vec<f64> = lcg(n, 6).iter().zip(&xs).map(|(e, x)| 0.5 * x + e).collect();
        for i in 1000..1100 {
            xs[i] *= 1.0e12;
            ys[i] = -1.0e12;
        }
        xs[4000] = f64::NAN;
        ys[4005] = f64::INFINITY;
        for i in 20_000..n {
            xs[i] += 1.0e7; // level shift: stale offsets
            ys[i] -= 3.0e7;
        }
        let w = 40;
        let mut m = RollingPairMoments::new(&xs[..w], &ys[..w]);
        for i in (w - 1)..n {
            if i >= w {
                m.remove(xs[i - w], ys[i - w]);
                m.add(xs[i], ys[i]);
                if m.needs_refresh() {
                    m.refresh(&xs[i + 1 - w..=i], &ys[i + 1 - w..=i]);
                }
            }
            let (cxx, cyy, cxy, good) = direct_pair(&xs[i + 1 - w..=i], &ys[i + 1 - w..=i]);
            assert_eq!(m.good_count(), good);
            assert_eq!(m.good_count() + m.bad_count(), w);
            if good < 2 {
                continue;
            }
            let scale = (cxx * cyy).sqrt();
            if scale == 0.0 {
                // One side is constant inside the window: no correlation.
                assert!(m.corr().is_nan(), "corr at {i}");
                continue;
            }
            assert!((m.cov() * (good - 1) as f64 - cxy).abs() <= 1e-7 * scale, "cov at {i}");
            assert!((m.var_x() * (good - 1) as f64 - cxx).abs() <= 1e-7 * cxx, "var_x at {i}");
            assert!((m.corr() - cxy / scale).abs() <= 1e-7, "corr at {i}: {} vs {}", m.corr(), cxy / scale);
        }
    }

    proptest! {
        /// Arbitrary data with wild magnitudes and holes: every window agrees
        /// with a direct computation.
        #[test]
        fn prop_every_window_matches_a_direct_computation(
            w in 2usize..12,
            vals in proptest::collection::vec(-1.0f64..1.0, 80),
            exps in proptest::collection::vec(-6i32..13, 80),
            holes in proptest::collection::vec(0u8..25, 80),
        ) {
            let xs: Vec<f64> = vals.iter().zip(&exps).zip(&holes).map(|((v, e), h)| match h {
                0 => f64::NAN,
                1 => f64::INFINITY,
                _ => v * 10f64.powi(*e),
            }).collect();
            let mut m: RollingMoments = RollingMoments::new(&xs[..w]);
            for i in (w - 1)..xs.len() {
                if i >= w {
                    m.remove(xs[i - w]);
                    m.add(xs[i]);
                    if m.needs_refresh() {
                        m.refresh(&xs[i + 1 - w..=i]);
                    }
                }
                let (mean, ss, count) = direct(&xs[i + 1 - w..=i]);
                prop_assert_eq!(m.finite_count(), count);
                if count > 0 {
                    let spread = (ss / count as f64).sqrt();
                    prop_assert!((m.finite_mean() - mean).abs() <= 1e-8 * spread + 1e-13 * mean.abs(),
                        "mean at {}: {} vs {}", i, m.finite_mean(), mean);
                    prop_assert!((m.finite_ss() - ss).abs() <= 1e-8 * ss, "ss at {}: {} vs {}", i, m.finite_ss(), ss);
                }
            }
        }
    }
}
