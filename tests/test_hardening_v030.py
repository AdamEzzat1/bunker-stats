"""Regression tests for the 0.3.0 hardening pass.

Tier 0 — numeric bug fixes:
  * catastrophic-cancellation offset shifts in scalar/matrix/rolling cov & corr
  * previously unregistered kernels (rolling_min/max/range/cv/count_above/
    pct_above, kde_gaussian) are importable and correct
  * welch_psd / bartlett_psd density scaling matches scipy.signal.welch
  * ks_1samp one-sided p-values are exact (Birnbaum-Tingey), matching scipy
  * exp_cdf keeps precision for tiny arguments (expm1 path)

Tier 1 — API consistency (edge/window/unit/validation rules).
Tier 2 — determinism (random_state=None == seed 0, bit-exact).
"""
from __future__ import annotations

import numpy as np
import pytest

b = pytest.importorskip("bunker_stats_rs", reason="build with `maturin develop`")
bs = pytest.importorskip("bunker_stats")
scipy_stats = pytest.importorskip("scipy.stats")
scipy_signal = pytest.importorskip("scipy.signal")
pd = pytest.importorskip("pandas")

RTOL = 1e-6
OFFSETS = [1e8, 1e12]


def _corr_data(offset: float, n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = offset + rng.normal(scale=1.0, size=n)
    y = offset + 0.5 * (x - offset) + rng.normal(scale=0.5, size=n)
    return x, y


# ======================================================================
# Tier 0a: catastrophic cancellation
# ======================================================================

class TestScalarCovCorrLargeOffset:
    @pytest.mark.parametrize("offset", OFFSETS)
    def test_cov_matches_numpy(self, offset):
        x, y = _corr_data(offset)
        got = b.cov_np(x, y)
        want = np.cov(x, y, ddof=1)[0, 1]
        np.testing.assert_allclose(got, want, rtol=RTOL)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_corr_matches_numpy(self, offset):
        x, y = _corr_data(offset)
        got = b.corr_np(x, y)
        want = np.corrcoef(x, y)[0, 1]
        np.testing.assert_allclose(got, want, rtol=RTOL)
        assert -1.0 <= got <= 1.0

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_cov_corr_skipna_match_pairwise_numpy(self, offset):
        x, y = _corr_data(offset)
        x = x.copy()
        y = y.copy()
        x[::17] = np.nan
        y[::23] = np.nan
        mask = ~(np.isnan(x) | np.isnan(y))
        want_cov = np.cov(x[mask], y[mask], ddof=1)[0, 1]
        want_corr = np.corrcoef(x[mask], y[mask])[0, 1]
        np.testing.assert_allclose(b.cov_skipna(x, y), want_cov, rtol=RTOL)
        np.testing.assert_allclose(b.corr_skipna(x, y), want_corr, rtol=RTOL)

    def test_perfect_correlation_is_clamped(self):
        x = 1e8 + np.arange(50, dtype=np.float64)
        r = b.corr_np(x, 2.0 * x)
        assert r <= 1.0
        np.testing.assert_allclose(r, 1.0, rtol=1e-12)


def _pairwise_cov_ref(X: np.ndarray) -> np.ndarray:
    """Two-pass pairwise-complete covariance (exact reference)."""
    p = X.shape[1]
    out = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(p):
            m = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if m.sum() >= 2:
                out[i, j] = np.cov(X[m, i], X[m, j], ddof=1)[0, 1]
    return out


class TestMatrixSkipnaLargeOffset:
    # NOTE: pandas' own rolling/pairwise one-pass kernels catastrophically
    # cancel at these offsets, so the references below are exact two-pass
    # numpy computations rather than pandas.
    @pytest.mark.parametrize("offset", OFFSETS)
    def test_cov_matrix_skipna_matches_two_pass(self, offset):
        rng = np.random.default_rng(11)
        X = offset + rng.normal(size=(120, 4))
        X[::13, 0] = np.nan
        X[::19, 2] = np.nan
        got = np.asarray(b.cov_matrix_skipna_np(X))
        want = _pairwise_cov_ref(X)
        np.testing.assert_allclose(got, want, rtol=RTOL)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_corr_matrix_skipna_matches_two_pass(self, offset):
        rng = np.random.default_rng(12)
        base = rng.normal(size=(150, 3))
        X = offset + base
        X[::11, 1] = np.nan
        got = np.asarray(b.corr_matrix_skipna_np(X))
        # Kernel convention: pairwise-complete cov_ij normalized by the
        # pairwise-complete variances from the covariance matrix diagonal,
        # corr_ij = cov_ij / sqrt(cov_ii * cov_jj).
        cov = _pairwise_cov_ref(X)
        d = np.sqrt(np.diag(cov))
        want = cov / np.outer(d, d)
        # atol covers near-zero correlations at 1e12 where the comparison is
        # limited by input quantization, not by the kernel
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=1e-8)
        assert np.all(np.abs(got[np.isfinite(got)]) <= 1.0)


class TestRollingNanLargeOffset:
    # Exact two-pass references (pandas' own one-pass rolling kernels lose
    # precision at these offsets).
    @pytest.mark.parametrize("offset", OFFSETS)
    def test_rolling_std_nan_matches_two_pass(self, offset):
        rng = np.random.default_rng(3)
        x = offset + rng.normal(size=300)
        x[::29] = np.nan
        got = np.asarray(b.rolling_std_nan_np(x, 20))
        want = np.array([
            np.nanstd(x[max(0, i - 19):i + 1], ddof=1)
            if np.sum(~np.isnan(x[max(0, i - 19):i + 1])) >= 2 else np.nan
            for i in range(len(x))
        ])
        np.testing.assert_allclose(got, want, rtol=RTOL, equal_nan=True)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_rolling_zscore_nan_matches_two_pass(self, offset):
        rng = np.random.default_rng(4)
        x = offset + rng.normal(size=300)
        x[::31] = np.nan
        got = np.asarray(b.rolling_zscore_nan_np(x, 25))
        want = np.empty(len(x))
        for i in range(len(x)):
            win = x[max(0, i - 24):i + 1]
            valid = win[~np.isnan(win)]
            if np.isnan(x[i]) or len(valid) < 2 or np.std(valid, ddof=1) == 0.0:
                want[i] = np.nan
            else:
                want[i] = (x[i] - np.mean(valid)) / np.std(valid, ddof=1)
        # At 1e12 the reference's own `x - np.mean(...)` numerator carries
        # ~ulp(1e12) of rounding, so the achievable agreement is ~1e-3 absolute
        # in z-units; the kernel's shifted accumulation is the more accurate side.
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-3, equal_nan=True)


class TestRollingStrictLargeOffset:
    @pytest.mark.parametrize("offset", OFFSETS)
    def test_rolling_cov_matches_two_pass(self, offset):
        x, y = _corr_data(offset, n=250, seed=8)
        w = 30
        got = np.asarray(b.rolling_cov_np(x, y, w))
        want = np.array([
            np.cov(x[i:i + w], y[i:i + w], ddof=1)[0, 1]
            for i in range(len(x) - w + 1)
        ])
        np.testing.assert_allclose(got, want, rtol=RTOL)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_rolling_corr_matches_two_pass(self, offset):
        x, y = _corr_data(offset, n=250, seed=9)
        w = 30
        got = np.asarray(b.rolling_corr_np(x, y, w))
        want = np.array([
            np.corrcoef(x[i:i + w], y[i:i + w])[0, 1]
            for i in range(len(x) - w + 1)
        ])
        np.testing.assert_allclose(got, want, rtol=RTOL)
        assert np.all(np.abs(got) <= 1.0)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_rolling_cov_corr_skipna_match_two_pass(self, offset):
        x, y = _corr_data(offset, n=250, seed=10)
        w = 25
        got_cov = np.asarray(b.rolling_cov_skipna(x, y, w))
        got_corr = np.asarray(b.rolling_corr_skipna(x, y, w))
        want_cov = np.array([
            np.cov(x[i:i + w], y[i:i + w], ddof=1)[0, 1]
            for i in range(len(x) - w + 1)
        ])
        want_corr = np.array([
            np.corrcoef(x[i:i + w], y[i:i + w])[0, 1]
            for i in range(len(x) - w + 1)
        ])
        np.testing.assert_allclose(got_cov, want_cov, rtol=RTOL)
        np.testing.assert_allclose(got_corr, want_corr, rtol=RTOL)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_rolling_linreg_skipna_recovers_slope_and_fit(self, offset):
        rng = np.random.default_rng(13)
        x = offset + rng.normal(size=120)
        y = 3.0 * x + 5.0 + rng.normal(scale=1e-3, size=120)
        slope, intercept = b.rolling_linreg_skipna(x, y, 40)
        slope = np.asarray(slope)
        intercept = np.asarray(intercept)
        # noise/spread gives per-window slope-estimation scatter of O(1e-4)
        np.testing.assert_allclose(slope, 3.0, rtol=1e-3)
        # The intercept itself is an O(offset)-scale cancellation of two huge
        # terms, so check the fitted line instead: prediction at x = offset
        # must reproduce y = 3*offset + 5 to full relative precision.
        pred = slope * offset + intercept
        np.testing.assert_allclose(pred, 3.0 * offset + 5.0, rtol=1e-9)


# ======================================================================
# Tier 0b: phantom API now registered
# ======================================================================

class TestPhantomApiRegistered:
    def test_every_facade_all_name_resolves_and_is_callable(self):
        missing = []
        for name in bs.__all__:
            obj = getattr(bs, name, None)
            if obj is None or not callable(obj):
                missing.append(name)
        assert not missing, f"facade names not callable: {missing}"

    def test_rolling_min_max_range_match_pandas(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=100)
        w = 9
        np.testing.assert_allclose(
            np.asarray(b.rolling_min(x, window=w)),
            pd.Series(x).rolling(w).min().to_numpy()[w - 1:],
        )
        np.testing.assert_allclose(
            np.asarray(b.rolling_max(x, window=w)),
            pd.Series(x).rolling(w).max().to_numpy()[w - 1:],
        )
        np.testing.assert_allclose(
            np.asarray(b.rolling_range(x, window=w)),
            (pd.Series(x).rolling(w).max() - pd.Series(x).rolling(w).min()).to_numpy()[w - 1:],
        )

    def test_rolling_count_and_pct_above(self):
        x = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0])
        got_count = np.asarray(b.rolling_count_above(x, threshold=0.0, window=3))
        want = np.array([2.0, 1.0, 2.0, 1.0, 2.0])
        np.testing.assert_allclose(got_count, want)
        got_pct = np.asarray(b.rolling_pct_above(x, threshold=0.0, window=3))
        np.testing.assert_allclose(got_pct, want / 3.0)

    def test_rolling_cv_matches_definition(self):
        rng = np.random.default_rng(22)
        x = 10.0 + rng.normal(size=60)
        w = 12
        got = np.asarray(b.rolling_cv(x, window=w))
        s = pd.Series(x)
        want = (s.rolling(w).std() / s.rolling(w).mean().abs()).to_numpy()[w - 1:]
        np.testing.assert_allclose(got, want, rtol=1e-9)

    def test_kde_gaussian_matches_scipy(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=400)
        grid, dens = b.kde_gaussian_np(x, 101)
        grid = np.asarray(grid)
        dens = np.asarray(dens)
        want = scipy_stats.gaussian_kde(x)(grid)
        np.testing.assert_allclose(dens, want, rtol=1e-8)

    def test_kde_gaussian_facade_alias(self):
        x = np.random.default_rng(24).normal(size=50)
        grid, dens = bs.kde_gaussian(x, 31)
        assert len(np.asarray(grid)) == 31
        assert np.all(np.asarray(dens) >= 0.0)


# ======================================================================
# Tier 0c: welch/bartlett PSD scaling
# ======================================================================

class TestWelchScaling:
    @pytest.mark.parametrize("nperseg", [64, 128, 255])
    def test_welch_matches_scipy(self, nperseg):
        rng = np.random.default_rng(31)
        x = rng.normal(size=1024)
        freqs, psd = b.welch_psd(x, nperseg=nperseg)
        f_ref, p_ref = scipy_signal.welch(x, nperseg=nperseg)
        np.testing.assert_allclose(np.asarray(freqs), f_ref, atol=1e-15)
        np.testing.assert_allclose(np.asarray(psd), p_ref, rtol=1e-10, atol=1e-15)

    def test_bartlett_matches_scipy_zero_overlap(self):
        rng = np.random.default_rng(32)
        x = rng.normal(size=1024)
        freqs, psd = b.bartlett_psd(x, nperseg=128)
        f_ref, p_ref = scipy_signal.welch(x, nperseg=128, noverlap=0)
        np.testing.assert_allclose(np.asarray(freqs), f_ref, atol=1e-15)
        np.testing.assert_allclose(np.asarray(psd), p_ref, rtol=1e-10, atol=1e-15)


# ======================================================================
# Tier 0d: exact one-sided KS p-values
# ======================================================================

class TestKsOneSidedExact:
    @pytest.mark.parametrize("n", [20, 50, 100])
    @pytest.mark.parametrize("alt", ["greater", "less"])
    def test_matches_scipy(self, n, alt):
        rng = np.random.default_rng(41 + n)
        x = rng.normal(loc=0.15, size=n)
        got = b.ks_1samp_np(x, "norm", [0.0, 1.0], alt)
        want = scipy_stats.ks_1samp(x, scipy_stats.norm(0.0, 1.0).cdf, alternative=alt)
        # statistic tolerance is bounded by tiny normal-CDF implementation
        # differences (statrs vs scipy erf), not by the KS logic itself
        np.testing.assert_allclose(got["statistic"], want.statistic, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(got["pvalue"], want.pvalue, rtol=1e-8)


# ======================================================================
# Tier 0e: exp_cdf precision at tiny x
# ======================================================================

class TestExpCdfTinyArgument:
    def test_tiny_x(self):
        x = np.array([1e-18, 1e-15, 1e-12, 1e-9])
        got = np.asarray(b.exp_cdf(x, lam=1.0))
        want = scipy_stats.expon.cdf(x)
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert got[0] == pytest.approx(1e-18, rel=1e-12)
        assert got[0] > 0.0

    def test_regular_range_unchanged(self):
        x = np.linspace(0.0, 20.0, 200)
        got = np.asarray(b.exp_cdf(x, lam=0.7))
        want = scipy_stats.expon.cdf(x, scale=1.0 / 0.7)
        np.testing.assert_allclose(got, want, rtol=1e-13)
