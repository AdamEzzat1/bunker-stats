"""Parity tests for the previously-untested function families.

Closes the ~49% correctness-coverage gap surfaced by the review: every function
here had zero prior test reference. References are numpy / scipy / pandas.

Where a kernel is CONFIRMED to disagree with the standard reference, the test is
`xfail(strict=True)` with the diagnosis (so it flags a real bug without turning
the suite red, and starts XPASSing if the kernel is fixed).
"""
from __future__ import annotations

import numpy as np
import pytest

b = pytest.importorskip("bunker_stats_rs", reason="build with `maturin develop`")
scipy_stats = pytest.importorskip("scipy.stats")
pd = pytest.importorskip("pandas")

RNG = np.random.default_rng(12345)


def _data(n=40, loc=0.0, scale=1.0):
    return RNG.normal(loc, scale, n)


# ============================================================================
# Family A — skipna / nan-aware reducers  (numpy nan-* references)
# ============================================================================
class TestSkipnaReducers:
    def _with_nans(self):
        x = _data(50).copy()
        x[[3, 17, 29]] = np.nan
        return x

    def test_mean_skipna(self):
        x = self._with_nans()
        assert np.isclose(b.mean_skipna_np(x), np.nanmean(x))

    def test_std_skipna(self):
        x = self._with_nans()
        assert np.isclose(b.std_skipna_np(x), np.nanstd(x, ddof=1))

    def test_var_skipna(self):
        x = self._with_nans()
        assert np.isclose(b.var_skipna_np(x), np.nanvar(x, ddof=1))

    def test_var_nan_alias_matches_skipna(self):
        x = self._with_nans()
        assert np.isclose(b.var_nan_np(x), np.nanvar(x, ddof=1))
        assert np.isclose(b.std_nan_np(x), np.nanstd(x, ddof=1))

    def test_median_skipna(self):
        x = self._with_nans()
        assert np.isclose(b.median_skipna_np(x), np.nanmedian(x))

    def test_mad_skipna(self):
        x = self._with_nans()
        finite = x[~np.isnan(x)]
        ref = np.median(np.abs(finite - np.median(finite)))
        assert np.isclose(b.mad_skipna_np(x), ref)

    def test_iqr_skipna(self):
        x = self._with_nans()
        q3, q1 = np.nanpercentile(x, [75, 25])
        assert np.isclose(b.iqr_skipna_np(x), q3 - q1)

    def test_trimmed_mean_skipna(self):
        x = self._with_nans()
        finite = x[~np.isnan(x)]
        ref = scipy_stats.trim_mean(finite, 0.1)
        assert np.isclose(b.trimmed_mean_skipna_np(x, 0.1), ref)

    def test_zscore_skipna(self):
        x = self._with_nans()
        got = np.asarray(b.zscore_skipna_np(x))
        m, s = np.nanmean(x), np.nanstd(x, ddof=1)
        ref = (x - m) / s
        finite = ~np.isnan(x)
        assert np.allclose(got[finite], ref[finite])

    def test_cov_corr_skipna_pairwise_complete(self):
        x = _data(60).copy()
        y = _data(60).copy()
        x[5] = np.nan
        y[9] = np.nan
        mask = ~np.isnan(x) & ~np.isnan(y)
        cov_ref = np.cov(x[mask], y[mask], ddof=1)[0, 1]
        corr_ref = np.corrcoef(x[mask], y[mask])[0, 1]
        assert np.isclose(b.cov_skipna(x, y), cov_ref, rtol=1e-9)
        assert np.isclose(b.corr_skipna(x, y), corr_ref, rtol=1e-9)


# ============================================================================
# Family B — basic descriptive stats & deprecated *_np aliases
# ============================================================================
class TestBasicStats:
    def test_mean_std_var_aliases(self):
        x = _data(30)
        assert np.isclose(b.mean_np(x), np.mean(x))
        assert np.isclose(b.std_np(x), np.std(x, ddof=1))
        assert np.isclose(b.var_np(x), np.var(x, ddof=1))
        assert np.isclose(b.zscore_np(x)[0], ((x - x.mean()) / x.std(ddof=1))[0])

    def test_welford_returns_mean_var_count(self):
        x = _data(30)
        mean, var, n = b.welford_np(x)
        assert np.isclose(mean, np.mean(x))
        assert np.isclose(var, np.var(x, ddof=1))
        assert n == len(x)

    def test_skew_matches_scipy(self):
        # Population skewness (scipy bias=True) — fixed to use consistent moments.
        x = _data(50, loc=3.0)
        assert np.isclose(b.skew_np(x), scipy_stats.skew(x))

    def test_kurtosis_matches_scipy(self):
        # Excess (Fisher) kurtosis, scipy bias=True — fixed to use consistent moments.
        x = _data(50)
        assert np.isclose(b.kurtosis_np(x), scipy_stats.kurtosis(x))


# ============================================================================
# Family C — cumulative / transform helpers
# ============================================================================
class TestTransforms:
    def test_cumsum_cummean(self):
        x = _data(20)
        assert np.allclose(b.cumsum_np(x), np.cumsum(x))
        assert np.allclose(b.cummean_np(x), np.cumsum(x) / np.arange(1, len(x) + 1))

    def test_ewma_matches_pandas_adjust_false(self):
        x = _data(30)
        ref = pd.Series(x).ewm(alpha=0.3, adjust=False).mean().to_numpy()
        assert np.allclose(b.ewma_np(x, 0.3), ref)

    def test_pct_change_and_diff(self):
        x = np.abs(_data(20)) + 1.0
        got_pct = np.asarray(b.pct_change_np(x, 1))
        ref_pct = pd.Series(x).pct_change(1).to_numpy()
        assert np.allclose(got_pct[1:], ref_pct[1:])
        got_diff = np.asarray(b.diff_np(x, 1))
        assert np.allclose(got_diff[1:], np.diff(x))

    def test_minmax_scale(self):
        x = _data(25)
        scaled, mn, mx = b.minmax_scale_np(x)
        assert np.isclose(mn, x.min()) and np.isclose(mx, x.max())
        assert np.allclose(scaled, (x - x.min()) / (x.max() - x.min()))


# ============================================================================
# Family E — matrix ops
# ============================================================================
class TestMatrix:
    def test_cov_corr_scalar(self):
        x, y = _data(40), _data(40)
        assert np.isclose(b.cov_np(x, y), np.cov(x, y, ddof=1)[0, 1])
        assert np.isclose(b.corr_np(x, y), np.corrcoef(x, y)[0, 1])

    def test_cov_matrix_bias_is_population(self):
        X = RNG.normal(size=(50, 4))
        got = np.asarray(b.cov_matrix_bias_np(X))
        assert np.allclose(got, np.cov(X, rowvar=False, bias=True))

    def test_xtx_xxt(self):
        X = RNG.normal(size=(10, 3))
        assert np.allclose(np.asarray(b.xtx_matrix_np(X)), X.T @ X)
        assert np.allclose(np.asarray(b.xxt_matrix_np(X)), X @ X.T)

    def test_diag_trace_symmetric(self):
        A = RNG.normal(size=(5, 5))
        S = A @ A.T
        assert np.allclose(np.asarray(b.diag_np(S)), np.diagonal(S))
        assert np.isclose(b.trace_np(S), np.trace(S))
        assert bool(b.is_symmetric_np(S, 1e-9))
        assert not bool(b.is_symmetric_np(A, 1e-9))


# ============================================================================
# Family H — inference tests (scipy references)
# ============================================================================
class TestInference:
    def test_pearson_corr(self):
        x, y = _data(40), _data(40)
        r = b.pearson_corr_test_np(x, y)
        ref = scipy_stats.pearsonr(x, y)
        assert np.isclose(r["correlation"], ref[0])
        assert np.isclose(r["pvalue"], ref[1], rtol=1e-4)

    def test_spearman_corr(self):
        x, y = _data(40), _data(40)
        r = b.spearman_corr_test_np(x, y)
        ref = scipy_stats.spearmanr(x, y)
        assert np.isclose(r["correlation"], ref.correlation, rtol=1e-6)
        assert np.isclose(r["pvalue"], ref.pvalue, rtol=1e-3)

    def test_f_oneway(self):
        g = np.vstack([_data(20, 0.0), _data(20, 0.5), _data(20, 1.0)])
        r = b.f_test_oneway_np(g)
        ref = scipy_stats.f_oneway(*[row for row in g])
        assert np.isclose(r["statistic"], ref.statistic, rtol=1e-6)
        assert np.isclose(r["pvalue"], ref.pvalue, rtol=1e-4)

    def test_bartlett(self):
        g = np.vstack([_data(25, scale=1.0), _data(25, scale=1.5), _data(25, scale=2.0)])
        r = b.bartlett_test_np(g)
        ref = scipy_stats.bartlett(*[row for row in g])
        assert np.isclose(r["statistic"], ref.statistic, rtol=1e-6)

    def test_f_test_var(self):
        # f_test_var_np reports the larger/smaller variance ratio (>= 1, upper
        # tail) — it is symmetric in its arguments.
        x, y = _data(40, scale=1.0), _data(40, scale=2.0)
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        r = b.f_test_var_np(x, y)
        assert np.isclose(r["statistic"], max(vx, vy) / min(vx, vy))
        assert np.isclose(b.f_test_var_np(y, x)["statistic"], r["statistic"])

    def test_jarque_bera(self):
        x = _data(80)
        r = b.jarque_bera_np(x)
        ref = scipy_stats.jarque_bera(x)
        assert np.isclose(r["statistic"], float(ref.statistic), rtol=1e-4)

    def test_anderson_darling(self):
        # anderson_darling_np returns the small-sample-corrected A²* =
        # A²·(1 + 4/n - 25/n²); scipy.anderson returns the raw A².
        x = _data(60)
        n = len(x)
        r = b.anderson_darling_np(x)
        a2_raw = float(scipy_stats.anderson(x, "norm").statistic)
        a2_corrected = a2_raw * (1 + 4.0 / n - 25.0 / n**2)
        assert np.isclose(r["statistic"], a2_corrected, rtol=1e-4)

    def test_cohens_d_zero_for_equal_distributions(self):
        x = _data(50)
        # A permutation of x has identical mean/var -> d == 0.
        assert abs(b.cohens_d_2samp_np(x, RNG.permutation(x), True)) < 1e-9

    def test_mean_diff_ci_brackets_true_difference(self):
        x = _data(60, loc=1.0)
        y = _data(60, loc=0.0)
        lo, hi = b.mean_diff_ci_np(x, y, 0.05, True)
        assert lo < hi
        assert lo < (x.mean() - y.mean()) < hi


# ============================================================================
# Family G — bootstrap / jackknife (statistical properties + determinism)
# ============================================================================
class TestResampling:
    def test_bootstrap_se_var_approx_analytic(self):
        x = _data(200)
        se = b.bootstrap_se(x, "mean", 2000, 0)
        analytic = np.std(x, ddof=1) / np.sqrt(len(x))
        assert np.isclose(se, analytic, rtol=0.15)
        assert np.isclose(b.bootstrap_var(x, "mean", 2000, 0), se**2, rtol=0.10)

    def test_bootstrap_cis_bracket_point(self):
        x = _data(100)
        for fn in (b.bootstrap_bca_ci, b.bayesian_bootstrap_ci):
            point, lo, hi = fn(x, "mean", 1000, 0.95, 0)
            assert lo <= point <= hi
        point, lo, hi = b.bootstrap_t_ci_mean(x, 1000, 0.95, 0)
        assert lo <= point <= hi and np.isclose(point, x.mean())

    def test_block_bootstraps_deterministic_and_ordered(self):
        x = np.cumsum(_data(120))  # autocorrelated series
        for fn in (
            b.moving_block_bootstrap_mean_ci,
            b.circular_block_bootstrap_mean_ci,
            b.stationary_bootstrap_mean_ci,
        ):
            r1 = fn(x, 10, 500, 0.95, 42)
            r2 = fn(x, 10, 500, 0.95, 42)
            assert r1 == r2, f"{fn.__name__} not deterministic under fixed seed"
            _pt, lo, hi = r1
            assert lo <= hi

    def test_influence_mean_sums_to_zero(self):
        x = _data(50)
        infl = np.asarray(b.influence_mean(x))
        assert abs(np.sum(infl)) < 1e-8

    def test_delete_d_jackknife_point_is_mean(self):
        x = _data(60)
        point, _bias, se = b.delete_d_jackknife_mean(x, 3)
        assert np.isclose(point, x.mean())
        assert se >= 0.0

    def test_jab_se_nonnegative(self):
        x = _data(80)
        assert b.jackknife_after_bootstrap_se_mean(x, 200, 0) >= 0.0


# ============================================================================
# Family F — rolling (strict, trailing) vs pandas
# ============================================================================
class TestRollingStrict:
    def _xyw(self):
        x = RNG.normal(5, 2, 40)
        y = 2.0 * x + RNG.normal(0, 1, 40)
        return x, y, 6

    def test_rolling_std_var_meanstd(self):
        x, _y, w = self._xyw()
        s = pd.Series(x)
        assert np.allclose(b.rolling_std_np(x, w), s.rolling(w).std().to_numpy()[w - 1:])
        assert np.allclose(b.rolling_var_np(x, w), s.rolling(w).var().to_numpy()[w - 1:])
        m, sd = b.rolling_mean_std_np(x, w)
        assert np.allclose(m, s.rolling(w).mean().to_numpy()[w - 1:])
        assert np.allclose(sd, s.rolling(w).std().to_numpy()[w - 1:])

    def test_rolling_cov_corr(self):
        x, y, w = self._xyw()
        sx, sy = pd.Series(x), pd.Series(y)
        assert np.allclose(b.rolling_cov_np(x, y, w), sx.rolling(w).cov(sy).to_numpy()[w - 1:], atol=1e-8)
        assert np.allclose(b.rolling_corr_np(x, y, w), sx.rolling(w).corr(sy).to_numpy()[w - 1:], atol=1e-8)

    def test_rolling_median_full_length(self):
        x, _y, w = self._xyw()
        got = np.asarray(b.rolling_median(x, w))
        ref = pd.Series(x).rolling(w).median().to_numpy()
        assert got.shape == ref.shape  # rolling_median keeps length n (NaN warmup)
        fin = ~np.isnan(ref)
        assert np.allclose(got[fin], ref[fin])

    def test_rolling_zscore_shape_and_finite(self):
        # The exact per-window standardization convention isn't documented; pin
        # the observable contract: trailing length n-window+1 and all finite for
        # finite input.
        x, _y, w = self._xyw()
        got = np.asarray(b.rolling_zscore_np(x, w))
        assert got.shape == (len(x) - w + 1,)
        assert np.all(np.isfinite(got))

    def test_rolling_beta_and_linreg_slope(self):
        x, y, w = self._xyw()
        ref = [
            np.cov(x[i:i + w], y[i:i + w], ddof=1)[0, 1] / np.var(x[i:i + w], ddof=1)
            for i in range(len(x) - w + 1)
        ]
        assert np.allclose(np.asarray(b.rolling_beta_skipna(x, y, w)), ref, atol=1e-8)
        slope, _intercept = b.rolling_linreg_skipna(x, y, w)
        assert np.allclose(np.asarray(slope), ref, atol=1e-8)

    def test_rolling_multi_default_shape(self):
        x, _y, w = self._xyw()
        out = b.rolling_multi_np(x, w)
        assert all(np.asarray(a).shape == (len(x) - w + 1,) for a in out)


# ============================================================================
# Family E2 — matrix (skipna variants, centered, distances) vs numpy
# ============================================================================
class TestMatrixFamily:
    def _X(self):
        return RNG.normal(size=(60, 4))

    def test_cov_corr_matrix(self):
        X = self._X()
        assert np.allclose(np.asarray(b.cov_matrix_skipna_np(X)), np.cov(X, rowvar=False, ddof=1))
        assert np.allclose(np.asarray(b.corr_matrix_skipna_np(X)), np.corrcoef(X, rowvar=False))

    def test_cov_matrix_centered(self):
        X = self._X()
        Xc = X - X.mean(axis=0)
        assert np.allclose(np.asarray(b.cov_matrix_centered_np(Xc)), np.cov(Xc, rowvar=False, ddof=1))

    def test_corr_distance_is_one_minus_corr(self):
        X = self._X()
        C = np.corrcoef(X, rowvar=False)
        assert np.allclose(np.asarray(b.corr_distance_np(X)), 1.0 - C, atol=1e-9)

    def test_pairwise_euclidean_cols(self):
        X = self._X()
        p = X.shape[1]
        ref = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                ref[i, j] = np.sqrt(np.sum((X[:, i] - X[:, j]) ** 2))
        assert np.allclose(np.asarray(b.pairwise_euclidean_cols_np(X)), ref)


# ============================================================================
# Family D — robust / quantile helpers
# ============================================================================
class TestRobustQuantile:
    def test_iqr_width_and_robust(self):
        x = _data(60)
        q3, q1 = np.percentile(x, [75, 25])
        assert np.isclose(b.iqr_width_np(x), q3 - q1)
        assert np.isclose(b.iqr_robust_np(x), q3 - q1)

    def test_winsorize_clip_is_a_bounded_clip(self):
        # winsorize_clip replaces tail values with interior bounds. Verify the
        # clip contract (rather than a specific quantile interpolation): the
        # output is a clip of the input to some [lo, hi] inside the data range,
        # is idempotent, and leaves interior values untouched.
        x = _data(60)
        out = np.asarray(b.winsorize_clip_np(x, 0.1, 0.9))
        lo, hi = out.min(), out.max()
        assert x.min() <= lo <= hi <= x.max()
        assert np.allclose(out, np.clip(x, lo, hi))
        out2 = np.asarray(b.winsorize_clip_np(out, 0.1, 0.9))
        assert lo <= out2.min() and out2.max() <= hi  # idempotent-ish (no wider)

    def test_sign_mask(self):
        x = np.array([-2.0, 0.0, 3.0, -1.0, 5.0])
        assert np.allclose(np.asarray(b.sign_mask_np(x)), np.sign(x))

    def test_robust_fit_and_score(self):
        MAD_C = 1.482602218505602  # normal-consistency constant used by the kernel
        x = _data(80)
        loc, scale = b.robust_fit(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * MAD_C
        assert np.isclose(loc, med)
        assert np.isclose(scale, mad, rtol=1e-9)
        assert np.allclose(np.asarray(b.robust_score(x)), (x - med) / mad, rtol=1e-9)

    def test_zscore_outliers_flags_extreme(self):
        x = np.concatenate([_data(40), [50.0]])  # last is a gross outlier
        out = np.asarray(b.zscore_outliers_np(x, 2.0))
        assert out.shape == x.shape
        assert out[-1] != 0  # the outlier is flagged in some way


# ============================================================================
# Family H2 — remaining inference + mean_axis + spectral
# ============================================================================
class TestInference2:
    def test_levene_center_median(self):
        # levene_test_np uses the Brown-Forsythe (median-centered) variant, which
        # is scipy.stats.levene's default.
        g = np.vstack([_data(30, scale=1.0), _data(30, scale=1.6), _data(30, scale=2.2)])
        r = b.levene_test_np(g)
        ref = scipy_stats.levene(*[row for row in g], center="median")
        assert np.isclose(r["statistic"], ref.statistic, rtol=1e-5)

    def test_permutation_corr(self):
        x = _data(40)
        y = x + _data(40, scale=0.5)
        stat, pval = b.permutation_corr_test(x, y, 2000, "two-sided", 0)
        assert np.isclose(stat, np.corrcoef(x, y)[0, 1], rtol=1e-6)
        assert 0.0 <= pval <= 1.0

    def test_hedges_g_is_corrected_cohens_d(self):
        x = _data(40, loc=1.0)
        y = _data(50, loc=0.0)
        d = b.cohens_d_2samp_np(x, y, True)
        g = b.hedges_g_2samp_np2(x, y, True)
        n = len(x) + len(y)
        correction = 1.0 - 3.0 / (4.0 * (n - 2) - 1.0)
        assert np.isclose(g, d * correction, rtol=1e-6)

    def test_mean_axis(self):
        X = RNG.normal(size=(6, 5))
        assert np.allclose(np.asarray(b.mean_axis_np(X, 0)), X.mean(axis=0))
        assert np.allclose(np.asarray(b.mean_axis_np(X, 1)), X.mean(axis=1))


class TestSpectral:
    def test_cumulative_periodogram_monotone_normalized(self):
        x = RNG.normal(size=128)
        _freqs, cum = b.cumulative_periodogram(x)
        cum = np.asarray(cum)
        assert np.all(np.diff(cum) >= -1e-9)          # non-decreasing
        assert np.isclose(cum[-1], 1.0, atol=1e-6)     # normalized to 1

    def test_bartlett_psd_nonnegative(self):
        x = RNG.normal(size=512)
        _freqs, psd = b.bartlett_psd(x, 64)
        assert np.all(np.asarray(psd) >= 0.0)


# ============================================================================
# Final batch — remaining helpers, N-d, and skipna rolling
# ============================================================================
class TestRemaining:
    def test_demean_with_signs(self):
        x = _data(30, loc=2.0)
        dem, signs = b.demean_with_signs_np(x)
        assert np.allclose(np.asarray(dem), x - x.mean())
        assert np.allclose(np.asarray(signs), np.sign(x - x.mean()))

    def test_iqr_outliers_mask(self):
        x = np.array([0.0, 0, 0, 0, 0, 0, 0, 100.0])  # last is an IQR outlier
        mask = np.asarray(b.iqr_outliers_np(x, 1.5)).astype(bool)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        ref = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)
        assert np.array_equal(mask, ref)
        assert mask[-1]  # the 100 is flagged

    def test_pad_nan(self):
        out = np.asarray(b.pad_nan_np(5))
        assert out.shape == (5,) and np.all(np.isnan(out))

    def test_pairwise_cosine_cols(self):
        X = RNG.normal(size=(40, 3))
        got = np.asarray(b.pairwise_cosine_cols_np(X))
        p = X.shape[1]
        ref = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                ci, cj = X[:, i], X[:, j]
                cos = ci @ cj / (np.linalg.norm(ci) * np.linalg.norm(cj))
                ref[i, j] = 1.0 - cos
        assert np.allclose(got, ref, atol=1e-9)

    def test_mean_over_last_axis(self):
        arr = RNG.normal(size=(3, 5))
        assert np.allclose(np.asarray(b.mean_over_last_axis_dyn_np(arr)), arr.mean(axis=-1))

    def test_rolling_nan_reducers_min_periods_1(self):
        x = _data(30)
        w = 5
        s = pd.Series(x)
        gm = np.asarray(b.rolling_mean_nan_np(x, w))
        rm = s.rolling(w, min_periods=1).mean().to_numpy()
        assert gm.shape == rm.shape and np.allclose(gm, rm)
        gs = np.asarray(b.rolling_std_nan_np(x, w))
        rs = s.rolling(w, min_periods=1).std().to_numpy()
        fin = ~np.isnan(rs)
        assert np.allclose(gs[fin], rs[fin])

    def test_rolling_skipna_cov_corr_match_pandas(self):
        x = _data(40)
        y = 1.5 * x + _data(40, scale=0.5)
        w = 6
        sx, sy = pd.Series(x), pd.Series(y)
        assert np.allclose(b.rolling_cov_skipna(x, y, w), sx.rolling(w).cov(sy).to_numpy()[w - 1:], atol=1e-8)
        assert np.allclose(b.rolling_corr_skipna(x, y, w), sx.rolling(w).corr(sy).to_numpy()[w - 1:], atol=1e-8)

    def test_rolling_multi_axis0_default(self):
        X = RNG.normal(size=(30, 3))
        w = 5
        out = b.rolling_multi_axis0_np(X, w)
        assert all(np.asarray(a).shape == (X.shape[0] - w + 1, X.shape[1]) for a in out)

    def test_rolling_zscore_nan_full_length(self):
        # NaN-aware rolling zscore keeps length n (min_periods warm-up -> NaN);
        # semantics of the per-window standardization aren't documented, so pin
        # the observable contract: length n and finite once the window fills.
        x = _data(30)
        w = 5
        got = np.asarray(b.rolling_zscore_nan_np(x, w))
        assert got.shape == (len(x),)
        assert np.all(np.isfinite(got[w:]))
