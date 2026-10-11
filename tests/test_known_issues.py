"""Regression tests for the "known issues" sweep.

* ``TestAbsurdArguments``   - a count or lag far beyond the data aborted the
                              interpreter (or never returned).
* ``TestQuantileBins``      - the O(n) rewrite assigns the same bins.
* ``TestConstantSamples``   - t-tests on constant data divided rounding error
                              by rounding error, and compared means with an
                              absolute 1e-15 tolerance.
* ``TestNormalTails``       - Mann-Whitney and the proportion tests carried
                              ~1e-11 relative error from the normal tail.
* ``TestRollingCount``      - ``count`` was NaN under ``propagate`` and a
                              valid count everywhere else.
* ``TestTrailingMinPeriods``- ``rolling_report`` left the first window - 1
                              rows NaN whatever ``min_periods`` said.
* ``TestJackknifeMean``     - the estimate was not the mean, to ~1e-11.

Crash tests run in a subprocess that must import this checkout. References
are scipy / pandas / numpy; small quantities are compared with ``close`` (no
absolute tolerance).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

import bunker_stats as bs

PACKAGE_DIR = str(pathlib.Path(bs.__file__).resolve().parent)
BIG = 2 ** 64 - 1


def close(value: float, reference: float, rel: float) -> bool:
    """Relative comparison with no absolute tolerance."""
    return value == pytest.approx(reference, rel=rel, abs=0)


def run_isolated(body: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter that must import *this* package."""
    root = str(pathlib.Path(PACKAGE_DIR).parent)
    code = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, {root!r})\n"
        "import numpy as np, bunker_stats as bs\n"
        "import bunker_stats.resampling as bsr\n"
        f"assert str(pathlib.Path(bs.__file__).resolve().parent) == {PACKAGE_DIR!r}, bs.__file__\n"
        "rs = bs._rs\n"
        "x = np.random.default_rng(0).normal(size=200)\n"
        "y = np.random.default_rng(1).normal(size=200)\n"
        + body
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
    )


def outcome(call: str) -> str:
    """'ok', 'raised <Error>', or a description of how the process died."""
    done = run_isolated(
        "try:\n"
        f"    {call}\n"
        "    print('ok')\n"
        "except BaseException as exc:\n"
        "    print('raised', type(exc).__name__)\n"
    )
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        return f"process died (rc={done.returncode}): {done.stderr.strip()[-200:]}"
    return lines[-1]


# ======================================================================
# Absurd counts and lags
# ======================================================================

RESAMPLERS = [
    "rs.bootstrap_mean(x, n_resamples=N)",
    "rs.bootstrap_mean_ci(x, n_resamples=N)",
    "rs.bootstrap_ci(x, n_resamples=N)",
    "rs.bootstrap_corr(x, y, n_resamples=N)",
    "rs.bootstrap_se(x, n_resamples=N)",
    "rs.bootstrap_var(x, n_resamples=N)",
    "rs.bootstrap_t_ci_mean(x, n_resamples=N)",
    "rs.bootstrap_bca_ci(x, n_resamples=N)",
    "rs.bayesian_bootstrap_ci(x, n_resamples=N)",
    "rs.moving_block_bootstrap_mean_ci(x, 5, n_resamples=N)",
    "rs.circular_block_bootstrap_mean_ci(x, 5, n_resamples=N)",
    "rs.stationary_bootstrap_mean_ci(x, 5, n_resamples=N)",
    "rs.jackknife_after_bootstrap_se_mean(x, n_resamples=N)",
    "bsr.BootstrapConfig(n_resamples=N).run(x)",
]


class TestAbsurdArguments:
    @pytest.mark.parametrize("call", RESAMPLERS)
    @pytest.mark.parametrize("n", [10 ** 12, BIG])
    def test_resample_counts_raise_memory_error(self, call, n):
        """n_resamples=10**12 asks for an 8 TB result buffer. The failed
        allocation used to abort the process; it is now a MemoryError."""
        assert outcome(call.replace("N", str(n))) == "raised MemoryError"

    @pytest.mark.parametrize("call", RESAMPLERS)
    def test_ordinary_resample_counts_still_work(self, call):
        assert outcome(call.replace("N", "200")) == "ok"

    def test_notebook_report_passes_the_error_through(self):
        pd = pytest.importorskip("pandas")  # noqa: F841
        call = (
            "import pandas as pd; from bunker_stats import notebook as nb; "
            f"nb.bootstrap_ci_report(pd.DataFrame({{'x': x}}), n_resamples={10 ** 12})"
        )
        assert outcome(call) == "raised MemoryError"

    @pytest.mark.parametrize("n", [10 ** 12, BIG])
    def test_pad_nan(self, n):
        assert outcome(f"rs.pad_nan_np({n})") == "raised MemoryError"
        assert np.isnan(bs._rs.pad_nan_np(4)).all() and bs._rs.pad_nan_np(4).shape == (4,)

    @pytest.mark.parametrize(
        "call",
        [
            f"assert all(np.isnan(v) for v in bs.adf_test(x, max_lag={BIG}))",
            f"s, b, p = bs.zivot_andrews_test(x, max_lag={BIG}); assert np.isnan(s) and np.isnan(p)",
            f"assert all(np.isnan(v) for v in bs.variance_ratio_test(x, lags={BIG}))",
            f"assert all(np.isnan(v) for v in bs.bg_test(x, max_lag={BIG}))",
            f"assert all(np.isnan(v) for v in bs.bg_test(x, max_lag={2 ** 63}))",
            f"assert all(np.isnan(v) for v in bs.adf_test(x, max_lag={2 ** 63}))",
        ],
    )
    def test_lags_beyond_the_series_return_nan(self, call):
        """`lag + k` wrapped for a lag near 2**64, the "too few rows" checks
        passed, and the kernel indexed out of bounds or never returned."""
        assert outcome(call) == "ok"

    def test_lag_just_inside_and_outside_the_series(self):
        x = np.random.default_rng(0).normal(size=60)
        for lag in (60, 61, 10 ** 6):
            assert all(np.isnan(v) for v in bs.adf_test(x, max_lag=lag))
            assert all(np.isnan(v) for v in bs.bg_test(x, max_lag=lag))
            assert all(np.isnan(v) for v in bs.variance_ratio_test(x, lags=lag))
        assert all(np.isfinite(v) for v in bs.adf_test(x, max_lag=4))
        assert all(np.isfinite(v) for v in bs.bg_test(x, max_lag=4))
        assert all(np.isfinite(v) for v in bs.variance_ratio_test(x, lags=4))

    def test_absurd_bin_count_returns(self):
        assert outcome(f"bs.quantile_bins(x, {BIG})") == "raised ValueError"
        assert outcome(f"assert bs.quantile_bins(x, {2 ** 62}).shape == (200,)") == "ok"


# ======================================================================
# quantile_bins
# ======================================================================

def _bins_by_looping_over_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """The definition: bin b takes sorted positions [b*n//n_bins, (b+1)*n//n_bins)."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    bins = np.full(n, -1, dtype=np.int64)
    start = 0
    for b in range(n_bins):
        end = n if b == n_bins - 1 else ((b + 1) * n) // n_bins
        bins[order[start:end]] = b
        start = end
    return bins


class TestQuantileBins:
    @pytest.mark.parametrize("n", [1, 2, 7, 50, 333])
    @pytest.mark.parametrize("n_bins", [1, 2, 3, 4, 10, 49, 50, 51, 400])
    def test_same_bins_as_the_definition(self, n, n_bins):
        x = np.random.default_rng(n * 1000 + n_bins).permutation(n).astype(float)
        np.testing.assert_array_equal(
            np.asarray(bs.quantile_bins(x, n_bins)), _bins_by_looping_over_bins(x, n_bins)
        )

    def test_quartiles_are_balanced(self):
        bins = np.asarray(bs.quantile_bins(np.arange(100.0), 4))
        assert np.bincount(bins).tolist() == [25, 25, 25, 25]
        assert bins[0] == 0 and bins[-1] == 3


# ======================================================================
# t-tests on constant samples
# ======================================================================

class TestConstantSamples:
    def test_mean_of_a_constant_is_not_exact_in_floating_point(self):
        """Why this needs handling: the premise of the tests below."""
        assert np.full(3, 0.1).sum() / 3 != 0.1

    @pytest.mark.parametrize("equal_var", [True, False])
    def test_same_constant_different_sizes_is_no_difference(self, equal_var):
        res = bs.t_test_2samp(np.full(3, 0.1), np.full(7, 0.1), equal_var=equal_var)
        assert res["statistic"] == 0.0 and res["pvalue"] == 1.0

    @pytest.mark.parametrize("equal_var", [True, False])
    @pytest.mark.parametrize("scale", [1e-20, 1e-200, 1.0, 1e150])
    def test_different_constants_are_separated_at_any_scale(self, equal_var, scale):
        """1e-20 vs 2e-20 differ by less than the old absolute 1e-15 cutoff,
        so two perfectly separated samples reported t = 0, p = 1."""
        a, b = np.full(5, 1.0 * scale), np.full(5, 2.0 * scale)
        res = bs.t_test_2samp(a, b, equal_var=equal_var)
        assert res["statistic"] == -np.inf and res["pvalue"] == 0.0
        assert bs.t_test_2samp(b, a, equal_var=equal_var)["statistic"] == np.inf
        one_sided = bs.t_test_2samp(a, b, equal_var=equal_var, alternative="greater")
        assert one_sided["pvalue"] == 1.0

    def test_one_sample_constant(self):
        at = bs.t_test_1samp(np.full(3, 0.1), 0.1)
        assert at["statistic"] == 0.0 and at["pvalue"] == 1.0 and at["mean"] == 0.1
        off = bs.t_test_1samp(np.full(3, 0.1), 0.2)
        assert off["statistic"] == -np.inf and off["pvalue"] == 0.0

    def test_paired_constant_difference(self):
        a = np.array([1.0, 2.0, 4.0, 8.0])
        same = bs.t_test_paired(a, a.copy())
        assert same["statistic"] == 0.0 and same["pvalue"] == 1.0
        shifted = bs.t_test_paired(a + 0.5, a)       # every difference is exactly 0.5
        assert shifted["statistic"] == np.inf and shifted["pvalue"] == 0.0

    def test_ordinary_samples_are_unchanged(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=30), rng.normal(0.3, 2.0, size=25)
        for equal_var in (True, False):
            got = bs.t_test_2samp(x, y, equal_var=equal_var)
            ref = stats.ttest_ind(x, y, equal_var=equal_var)
            assert close(got["statistic"], ref.statistic, 1e-12)
            assert close(got["pvalue"], ref.pvalue, 1e-10)
        got, ref = bs.t_test_1samp(x, 0.1), stats.ttest_1samp(x, 0.1)
        assert close(got["statistic"], ref.statistic, 1e-12) and close(got["pvalue"], ref.pvalue, 1e-10)
        got, ref = bs.t_test_paired(x[:25], y), stats.ttest_rel(x[:25], y)
        assert close(got["statistic"], ref.statistic, 1e-12) and close(got["pvalue"], ref.pvalue, 1e-10)


# ======================================================================
# Normal tails
# ======================================================================

class TestNormalTails:
    """1e-13 on purpose: the statrs tail these used was good to ~1e-11."""

    @pytest.mark.parametrize("shift", [0.0, 0.4, 1.5, 4.0])
    @pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
    def test_mann_whitney_matches_scipy(self, shift, alternative):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(int(shift * 10))
        x, y = rng.normal(size=40), rng.normal(shift, 1.0, size=45)
        got = bs.mann_whitney_u(x, y, alternative=alternative)
        ref = stats.mannwhitneyu(
            x, y, alternative=alternative, method="asymptotic", use_continuity=True
        )
        assert close(got["pvalue"], ref.pvalue, 1e-13), (got["pvalue"], ref.pvalue)

    @pytest.mark.parametrize("count", [30, 50, 61, 80, 97])
    @pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
    def test_proportion_tests_match_the_normal_tail(self, count, alternative):
        stats = pytest.importorskip("scipy.stats")

        def tail(z):
            return {
                "two-sided": 2 * stats.norm.sf(abs(z)),
                "less": stats.norm.cdf(z),
                "greater": stats.norm.sf(z),
            }[alternative]

        one = bs.proportion_ztest(count, 100, 0.5, alternative=alternative)
        assert close(one["pvalue"], min(tail(one["statistic"]), 1.0), 1e-13)
        two = bs.two_proportions_ztest(count, 100, 45, 110, alternative=alternative)
        assert close(two["pvalue"], min(tail(two["statistic"]), 1.0), 1e-13)


# ======================================================================
# Rolling count
# ======================================================================

class TestRollingCount:
    X = np.array([1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0])

    @pytest.mark.parametrize("policy", ["propagate", "ignore", "require_min_periods"])
    @pytest.mark.parametrize("window", [1, 3, 4])
    def test_trailing_count_is_the_number_of_valid_values(self, policy, window):
        pd = pytest.importorskip("pandas")
        got = np.asarray(bs.Rolling(self.X, window=window, nan_policy=policy).count())
        ref = pd.Series(self.X).notna().rolling(window).sum().to_numpy()[window - 1:]
        np.testing.assert_array_equal(got, ref)

    @pytest.mark.parametrize("policy", ["propagate", "ignore"])
    def test_centered_count(self, policy):
        pd = pytest.importorskip("pandas")
        got = np.asarray(
            bs.Rolling(self.X, window=3, nan_policy=policy, alignment="centered").count()
        )
        ref = pd.Series(self.X).notna().rolling(3, center=True, min_periods=1).sum().to_numpy()
        np.testing.assert_array_equal(got, ref)

    def test_count_does_not_depend_on_the_policy(self):
        for alignment in ("trailing", "centered"):
            counts = [
                np.asarray(bs.Rolling(self.X, window=3, nan_policy=p, alignment=alignment).count())
                for p in ("propagate", "ignore", "require_min_periods")
            ]
            np.testing.assert_array_equal(counts[0], counts[1])
            np.testing.assert_array_equal(counts[0], counts[2])
            assert not np.isnan(counts[0]).any()

    def test_other_statistics_still_propagate(self):
        mean = np.asarray(bs.Rolling(self.X, window=3).mean())
        np.testing.assert_array_equal(np.isnan(mean), [True, True, False, True, True, True])


# ======================================================================
# rolling_report: trailing windows and min_periods
# ======================================================================

class TestTrailingMinPeriods:
    STATS = ["mean", "std", "var", "min", "max"]

    @pytest.fixture
    def frame(self):
        pd = pytest.importorskip("pandas")
        x = np.random.default_rng(0).normal(size=40)
        x[[4, 5, 17, 30]] = np.nan
        return pd.DataFrame({"x": x})

    @pytest.mark.parametrize("window, min_periods", [(5, 1), (5, 2), (5, 4), (7, 3), (2, 1), (12, 6)])
    def test_matches_pandas_on_every_row(self, frame, window, min_periods):
        from bunker_stats import notebook as nb

        got = nb.rolling_report(
            frame, "x", window, stats=self.STATS + ["count"],
            min_periods=min_periods, nan_policy="ignore",
        )
        roll = frame["x"].rolling(window, min_periods=min_periods)
        for stat in self.STATS:
            ours = got[f"x_roll{window}_{stat}"].to_numpy()
            ref = getattr(roll, stat)().to_numpy()
            np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref), err_msg=stat)
            np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-13, equal_nan=True, err_msg=stat)
        valid = frame["x"].notna().rolling(window, min_periods=1).sum().to_numpy()
        np.testing.assert_array_equal(got[f"x_roll{window}_count"].to_numpy(), valid)

    def test_leading_rows_are_filled(self, frame):
        from bunker_stats import notebook as nb

        got = nb.rolling_report(frame, "x", 5, stats=["mean"], min_periods=2, nan_policy="ignore")
        lead = got["x_roll5_mean"].to_numpy()[:4]
        np.testing.assert_array_equal(np.isnan(lead), [True, False, False, False])
        assert lead[1] == pytest.approx(frame["x"].iloc[:2].mean())

    def test_propagate_keeps_a_window_with_a_nan_empty(self):
        pd = pytest.importorskip("pandas")
        from bunker_stats import notebook as nb

        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        got = nb.rolling_report(
            pd.DataFrame({"x": x}), "x", 5, stats=["mean", "count"], min_periods=2
        )
        mean = got["x_roll5_mean"].to_numpy()
        # row 0: one value (< min_periods); row 1: mean of two; rows 2-6 hold the NaN
        np.testing.assert_array_equal(np.isnan(mean), [True, False, True, True, True, True, True, False])
        assert mean[1] == 1.5 and mean[7] == 6.0
        np.testing.assert_array_equal(got["x_roll5_count"].to_numpy(), [1, 2, 2, 3, 4, 4, 4, 5])

    def test_default_min_periods_is_unchanged(self, frame):
        from bunker_stats import notebook as nb

        clean = frame.fillna(0.0)
        got = nb.rolling_report(clean, "x", 5, stats=["mean"])["x_roll5_mean"].to_numpy()
        ref = clean["x"].rolling(5).mean().to_numpy()
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        assert np.isnan(got[:4]).all()


# ======================================================================
# jackknife_mean
# ======================================================================

class TestJackknifeMean:
    def test_the_hypothesis_counterexample(self):
        """`n*theta - (n-1)*mean(loo)` returned 9.000000000014559 here and
        reported the difference as a bias."""
        x = np.array([0.0, 0.0, 0.0, 999999.0, -999945.0, 0.0])
        estimate, bias, se = bs.jackknife_mean(x)
        assert estimate == 9.0 and bias == 0.0
        assert close(se, np.std(x, ddof=1) / np.sqrt(len(x)), 1e-12)

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("offset", [0.0, 1e9, -1e12])
    def test_estimate_is_the_mean_and_se_is_the_standard_error(self, seed, offset):
        x = offset + np.random.default_rng(seed).normal(size=50)
        estimate, bias, se = bs.jackknife_mean(x)
        assert bias == 0.0
        assert close(estimate, float(np.mean(x)), 1e-14) if offset else abs(estimate - np.mean(x)) < 1e-15
        assert close(se, np.std(x, ddof=1) / np.sqrt(len(x)), 1e-6)
