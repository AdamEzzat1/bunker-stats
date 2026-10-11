"""Regression tests for kernel defects that gave wrong numbers or killed the process.

* ``TestKsTwoSided``        - ``ks_1samp`` returned p ~ 1.0 for n from about
                              550 to 10,000 (a scale-bookkeeping bug in the
                              matrix power), could not report p below ~1e-16,
                              and took minutes for large n with a large D.
* ``TestAndersonDarling``   - the statistic depended on the sign of an outlier
                              and was negative for n <= 3.
* ``TestRollingEdgeCases``  - zeros for a centered window longer than the data,
                              inexact min/max at large offsets, 0.0 instead of
                              NaN for the std of a one-point window.
* ``TestPearsonScale``      - correlation was inf / 0 for data scaled by
                              1e-100 / 1e100.
* ``TestNoProcessAbort``    - ordinary bad inputs aborted the interpreter.

Conventions (see ``test_notebook_regressions.py`` for why): comparisons of
small quantities use ``close`` with no absolute tolerance; references are
independent of the kernel; crash tests run in a subprocess that must import
this checkout.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest

import bunker_stats as bs

PACKAGE_DIR = str(pathlib.Path(bs.__file__).resolve().parent)


def close(value: float, reference: float, rel: float) -> bool:
    """Relative comparison with no absolute tolerance."""
    return value == pytest.approx(reference, rel=rel, abs=0)


@pytest.fixture(scope="module")
def scipy_stats():
    return pytest.importorskip("scipy.stats")


@pytest.fixture(scope="module")
def pd():
    return pytest.importorskip("pandas")


def run_isolated(body: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter that must import *this* package."""
    root = str(pathlib.Path(PACKAGE_DIR).parent)
    code = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, {root!r})\n"
        "import numpy as np, bunker_stats as bs\n"
        f"assert str(pathlib.Path(bs.__file__).resolve().parent) == {PACKAGE_DIR!r}, bs.__file__\n"
        + body
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
    )


# ======================================================================
# Kolmogorov-Smirnov, two-sided
# ======================================================================

def _uniform_sample_with_statistic(n: int, d: float) -> np.ndarray:
    """A sample whose two-sided KS statistic against U(0, 1) is ``d``.

    The uniform plotting positions compressed towards 0 by a factor c < 1:
    the empirical CDF then runs ahead of F by most at the last point, where
    the gap is 1 - c (1 - 1/(2n)). Solving for c gives D+ = d, and D- is only
    c / (2n). Valid for any 1/(2n) <= d < 1.
    """
    assert 0.5 / n <= d < 1.0
    c = (1.0 - d) / (1.0 - 0.5 / n)
    return c * (np.arange(1, n + 1) - 0.5) / n


# Exact two-sided survival function P(D_n >= d): Durbin's matrix formula
# evaluated with 60-digit arithmetic (mpmath). Independent of the kernel and of
# scipy, whose kstwo is itself an approximation (~1e-7) for n >= 400.
KS_EXACT = [
    # (n, d, sf)            -- matrix path (p above the tail switch)
    (5, 0.3, 0.66400000000000004),
    (20, 0.2, 0.35272017362341598),
    (100, 0.1, 0.25269275700639007),
    (400, 0.05, 0.26121260292565059),
    (500, 0.04, 0.39025101632640605),
    (550, 0.04, 0.33349655643955383),   # n >= 550: the matrix power rescales;
    (600, 0.04, 0.2846796163446239),    # these all used to come back as ~1.0
    (1000, 0.03, 0.32269024641329994),
    (1000, 0.045, 0.033783028682297522),
    (3000, 0.02, 0.1788981376320468),
    #                       -- tail path (p <= 0.01): 2 * P(D+ >= d)
    (200, 0.25, 1.6840667400683395e-11),
    (100, 0.2, 0.00055519273280280968),
    (60, 0.21, 0.0084503171147745367),
    (30, 0.3, 0.0068522433629125412),
]


class TestKsTwoSided:
    @pytest.mark.parametrize("n, d, exact", KS_EXACT)
    def test_matches_exact_reference(self, n, d, exact):
        x = _uniform_sample_with_statistic(n, d)
        out = bs.ks_1samp(x, "uniform", [0.0, 1.0])
        assert close(out["statistic"], d, 1e-9)
        # The statistic is reproduced to ~1e-12; the p-value's sensitivity to
        # it is what bounds this tolerance, not the p-value algorithm.
        assert close(out["pvalue"], exact, 2e-7)

    @pytest.mark.parametrize("n", [550, 600, 1000, 3000, 10000])
    @pytest.mark.parametrize("shift", [0.0, 0.05, 0.15])
    def test_large_n_no_longer_returns_one(self, n, shift, scipy_stats):
        """The original symptom: p ~ 1.0 whatever the data, for n >= 550."""
        x = np.random.default_rng(n).standard_normal(n) + shift
        ref = scipy_stats.kstest(x, "norm").pvalue
        ours = bs.ks_1samp(x, "norm", [0.0, 1.0])["pvalue"]
        # scipy's own value is an approximation at these sizes
        assert close(ours, ref, 2e-6)

    def test_shifted_sample_is_rejected(self):
        """A plain-language version of the bug: 1,000 points from N(0.15, 1)
        tested against N(0, 1) reported p = 1.0."""
        x = np.random.default_rng(0).standard_normal(1000) + 0.15
        assert bs.ks_1samp(x, "norm", [0.0, 1.0])["pvalue"] < 0.05

    @pytest.mark.parametrize("n, d", [(2, 0.6), (3, 0.7), (5, 0.85), (10, 0.95), (100, 0.995)])
    def test_closed_form_for_large_d(self, n, d):
        """For d >= 1/2 both tails cannot be exceeded at once, so
        P(D >= d) = 2 P(D+ >= d); and for d > 1 - 1/n only one term of the
        one-sided sum survives, giving P(D+ >= d) = (1 - d)^n."""
        assert n * (1.0 - d) < 1.0
        out = bs.ks_1samp(_uniform_sample_with_statistic(n, d), "uniform", [0.0, 1.0])
        assert close(out["statistic"], d, 1e-12)
        assert close(out["pvalue"], 2.0 * (1.0 - d) ** n, 1e-9)

    def test_far_tail_is_not_cut_off_at_1e_minus_16(self, scipy_stats):
        """`1 - cdf` cannot go below ~1e-16; this p-value is ~5e-47."""
        x = np.random.default_rng(0).uniform(0.0, 0.5, 200)
        ref = scipy_stats.kstest(x, "uniform").pvalue
        assert 0.0 < ref < 1e-40
        assert close(bs.ks_1samp(x, "uniform", [0.0, 1.0])["pvalue"], ref, 1e-9)

    def test_large_statistic_is_fast(self):
        """n = 4,000 with D ~ 0.5 took about 146 s: the matrix has 2*floor(nD)+1
        rows and the cost is cubic in that."""
        x = np.random.default_rng(1).uniform(0.0, 0.5, 4000)
        start = time.perf_counter()
        out = bs.ks_1samp(x, "uniform", [0.0, 1.0])
        assert time.perf_counter() - start < 5.0
        assert 0.0 <= out["pvalue"] < 1e-300

    def test_pvalue_is_one_below_the_smallest_possible_statistic(self):
        """D_n >= 1/(2n) always; the perfectly spaced sample attains it."""
        for n in (1, 2, 7, 100):
            x = (np.arange(1, n + 1) - 0.5) / n
            out = bs.ks_1samp(x, "uniform", [0.0, 1.0])
            assert close(out["statistic"], 0.5 / n, 1e-9)
            assert out["pvalue"] == 1.0

    @pytest.mark.parametrize("n", [30, 550, 2000])
    def test_pvalue_decreases_as_the_statistic_grows(self, n):
        """Across the three evaluation paths (matrix, tail, d >= 1/2)."""
        ds = np.linspace(0.6 / n, 0.9, 80)
        ps = [
            bs.ks_1samp(_uniform_sample_with_statistic(n, d), "uniform", [0.0, 1.0])["pvalue"]
            for d in ds
        ]
        assert all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in ps)
        assert ps[0] > 0.99 and ps[-1] < 1e-10
        assert all(later <= earlier * (1 + 1e-6) for earlier, later in zip(ps, ps[1:]))

    def test_asymptotic_branch_above_ten_thousand(self, scipy_stats):
        """Moderate p-values for n > 10,000 use the Stephens series, which is
        within about 0.5% of scipy; the tail still uses the exact sum."""
        x = np.random.default_rng(2).standard_normal(20001) + 0.01
        ref = scipy_stats.kstest(x, "norm").pvalue
        assert 0.01 < ref < 0.5
        assert close(bs.ks_1samp(x, "norm", [0.0, 1.0])["pvalue"], ref, 5e-3)
        tail = np.random.default_rng(3).standard_normal(20001) + 0.1
        ref = scipy_stats.kstest(tail, "norm").pvalue
        assert 0.0 < ref < 1e-20
        assert close(bs.ks_1samp(tail, "norm", [0.0, 1.0])["pvalue"], ref, 1e-9)

    @pytest.mark.parametrize("alternative", ["less", "greater"])
    def test_one_sided_is_unchanged(self, alternative, scipy_stats):
        x = np.random.default_rng(4).standard_normal(300) + 0.1
        ref = scipy_stats.kstest(x, "norm", alternative=alternative).pvalue
        assert close(bs.ks_1samp(x, "norm", [0.0, 1.0], alternative)["pvalue"], ref, 1e-9)


# ======================================================================
# Anderson-Darling
# ======================================================================

class TestAndersonDarling:
    @staticmethod
    def _stephens(n: int) -> float:
        return 1.0 + 4.0 / n - 25.0 / n**2

    @pytest.mark.parametrize("outlier", [6.0, 9.0, 12.0, 30.0])
    def test_statistic_does_not_depend_on_the_sign_of_an_outlier(self, outlier):
        """`1 - cdf(z)` is exactly 0 from z ~ 8.3; a clamp then stood in
        ln(1e-300) for the log-tail, so +12 and -12 gave 5.22 and 3.96."""
        x = np.r_[np.random.default_rng(1).standard_normal(500), outlier]
        up = bs.anderson_darling(x)["statistic"]
        down = bs.anderson_darling(-x)["statistic"]
        assert close(up, down, 1e-10)

    @pytest.mark.parametrize("outlier", [0.0, 12.0, -12.0])
    def test_matches_scipy_times_the_stephens_factor(self, outlier, scipy_stats):
        x = np.r_[np.random.default_rng(2).standard_normal(300), outlier]
        ref = scipy_stats.anderson(x, "norm").statistic * self._stephens(len(x))
        assert close(bs.anderson_darling(x)["statistic"], ref, 1e-9)

    def test_normality_summary_agrees(self, scipy_stats):
        x = np.r_[np.random.default_rng(3).standard_normal(200), 11.0]
        ref = scipy_stats.anderson(x, "norm").statistic * self._stephens(len(x))
        assert close(bs.normality_summary(x)["ad_statistic"], ref, 1e-9)
        assert close(bs.normality_summary(-x)["ad_statistic"], ref, 1e-9)

    @pytest.mark.parametrize("n", [2, 3])
    def test_tiny_samples_are_refused_not_negative(self, n):
        """The factor 1 + 4/n - 25/n^2 is negative for n <= 3, so the reported
        statistic was negative (-0.08 at n = 3)."""
        assert self._stephens(n) < 0
        with pytest.raises(ValueError, match="requires n >= 4"):
            bs.anderson_darling(np.arange(1.0, n + 1) ** 2)

    def test_smallest_allowed_sample_is_non_negative(self):
        assert bs.anderson_darling(np.array([1.0, 2.0, 4.0, 9.0]))["statistic"] >= 0.0

    def test_normality_report_shows_nan_for_tiny_samples(self, pd):
        from bunker_stats import notebook as nb

        out = nb.normality_report(pd.DataFrame({"x": [1.0, 2.0, 4.0]}))
        assert np.isnan(out.loc["x", "ad_statistic"])


# ======================================================================
# Rolling edge cases
# ======================================================================

class TestRollingEdgeCases:
    @pytest.mark.parametrize("n, window", [(3, 4), (3, 7), (1, 2), (5, 6)])
    def test_centered_window_longer_than_the_data(self, n, window, pd):
        """Returned all zeros: the kernel bailed out and left its
        zero-initialised output buffers as the result."""
        x = np.random.default_rng(n * 10 + window).normal(size=n) + 5.0
        roller = bs.Rolling(x, window=window, alignment="centered", min_periods=1)
        ref = pd.Series(x).rolling(window, center=True, min_periods=1)
        np.testing.assert_allclose(np.asarray(roller.mean()), ref.mean(), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(roller.min()), ref.min(), rtol=0)
        np.testing.assert_allclose(np.asarray(roller.max()), ref.max(), rtol=0)

    @pytest.mark.parametrize("alignment", ["trailing", "centered"])
    @pytest.mark.parametrize("offset", [1e16, -1e16, 1e300])
    def test_min_and_max_are_exact_at_any_offset(self, alignment, offset, pd):
        """min/max were stored as `x - offset` and returned as `+ offset`,
        which is not exact: [1e16, 1, 3, 2, 5] gave a window minimum of 0."""
        x = np.array([offset, 1.0, 3.0, 2.0, 5.0, -4.0, 0.5])
        kwargs = {"min_periods": 3} if alignment == "centered" else {}
        roller = bs.Rolling(x, window=3, alignment=alignment, **kwargs)
        ref = pd.Series(x).rolling(3, center=(alignment == "centered"))
        ours_min, ours_max = np.asarray(roller.min()), np.asarray(roller.max())
        ref_min, ref_max = ref.min().dropna().to_numpy(), ref.max().dropna().to_numpy()
        assert ours_min[~np.isnan(ours_min)].tolist() == ref_min.tolist()
        assert ours_max[~np.isnan(ours_max)].tolist() == ref_max.tolist()

    @pytest.mark.parametrize("stat", ["std", "var"])
    def test_one_point_window_has_no_sample_variance(self, stat, pd):
        """Trailing window=1 returned 0.0: 0/0 is NaN, but a clamp to >= 0
        turned it into 0."""
        x = np.array([1.0, 4.0, 2.0, 8.0])
        ours = np.asarray(getattr(bs.Rolling(x, window=1), stat)())
        ref = getattr(pd.Series(x).rolling(1), stat)().to_numpy()
        assert np.isnan(ref).all() and np.isnan(ours).all()

    def test_std_is_unchanged_for_ordinary_windows(self, pd):
        x = np.random.default_rng(5).normal(size=200)
        for window in (2, 5, 20):
            ours = np.asarray(bs.Rolling(x, window=window).std())
            ref = pd.Series(x).rolling(window).std().dropna().to_numpy()
            np.testing.assert_allclose(ours, ref, rtol=1e-9)

    def test_std_stays_accurate_at_a_large_offset(self):
        """The variance still uses the per-window offset; only min/max moved
        to raw values. Shifting the data must not change the std."""
        base = np.random.default_rng(6).normal(size=300)
        plain = np.asarray(bs.Rolling(base, window=20).std())
        shifted = np.asarray(bs.Rolling(base + 1e8, window=20).std())
        np.testing.assert_allclose(shifted, plain, rtol=1e-6)


# ======================================================================
# Pearson / Spearman at extreme scales
# ======================================================================

class TestPearsonScale:
    @pytest.mark.parametrize("scale", [1e-100, 1e-60, 1e60, 1e100])
    def test_pearson_is_scale_invariant(self, scale):
        """(sxx * syy).sqrt() under/overflowed: r was inf at 1e-100 and 0 at
        1e100, with p-values 0 and 1."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=40)
        y = 0.5 * x + rng.normal(size=40)
        ref = bs.pearson_corr_test(x, y)
        out = bs.pearson_corr_test(scale * x, scale * y)
        assert close(out["correlation"], ref["correlation"], 1e-12)
        assert close(out["pvalue"], ref["pvalue"], 1e-9)
        assert abs(out["correlation"]) <= 1.0

    def test_pearson_matches_scipy(self, scipy_stats):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=50), rng.normal(size=50)
        ref = scipy_stats.pearsonr(x, y)
        out = bs.pearson_corr_test(x, y)
        assert close(out["correlation"], ref[0], 1e-12)
        assert close(out["pvalue"], ref[1], 1e-10)

    def test_spearman_matches_scipy(self, scipy_stats):
        rng = np.random.default_rng(9)
        x = rng.integers(0, 6, size=80).astype(float)  # ties
        y = x + rng.integers(0, 3, size=80)
        ref = scipy_stats.spearmanr(x, y)
        out = bs.spearman_corr_test(x, y)
        assert close(out["correlation"], ref[0], 1e-12)
        assert close(out["pvalue"], ref[1], 1e-9)


# ======================================================================
# Inputs that used to abort the interpreter
# ======================================================================

ABORT_CASES = [
    ("rolling over a frame with no columns",
     "bs.Rolling(np.zeros((5, 0)), window=3, axis=0).mean()", "no columns"),
    ("rolling over an empty frame",
     "bs.Rolling(np.zeros((0, 0)), window=3, axis=0).mean()", ""),
    ("f_test_var with a variance that overflows",
     "bs.f_test_var(np.array([1e200, -1e200, 1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0]))",
     "overflowed"),
    ("mean_diff_ci with alpha out of range",
     "bs.mean_diff_ci(np.arange(1.0, 6.0), None, 3.0)", "alpha must be in"),
    ("mean_diff_ci with alpha negative",
     "bs.mean_diff_ci(np.arange(1.0, 6.0), None, -1.0)", "alpha must be in"),
    ("mean_diff_ci with alpha NaN",
     "bs.mean_diff_ci(np.arange(1.0, 6.0), None, float('nan'))", "alpha must be in"),
    ("odds_ratio with confidence NaN",
     "bs.odds_ratio(np.array([[10.0, 5.0], [3.0, 12.0]]), confidence=float('nan'))",
     "confidence must be in"),
]


class TestNoProcessAbort:
    @pytest.mark.parametrize("label, call, message", ABORT_CASES, ids=[c[0] for c in ABORT_CASES])
    def test_raises_a_python_exception(self, label, call, message):
        """Each of these killed the interpreter (exit code 0xC0000409 on
        Windows, SIGABRT elsewhere) with no Python exception. Run in a
        subprocess: a regression would otherwise take pytest down with it."""
        done = run_isolated(
            "try:\n"
            f"    {call}\n"
            "except ValueError as exc:\n"
            "    print('ValueError:', exc)\n"
            "else:\n"
            "    print('NO EXCEPTION')\n"
        )
        assert done.returncode == 0, f"process died: {done.stderr[-300:]}"
        assert done.stdout.startswith("ValueError:"), done.stdout
        assert message in done.stdout

    @pytest.mark.parametrize("confidence", [float("nan"), 0.0, 1.0, 2.0, -0.5])
    def test_confidence_must_be_a_proper_fraction(self, confidence):
        """`c <= 0 || c >= 1` is false for NaN, which then reached the quantile
        function: an abort in odds_ratio, a silent zero-width interval in
        var_ci."""
        table = np.array([[10.0, 5.0], [3.0, 12.0]])
        x = np.arange(1.0, 9.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            bs.var_ci(x, confidence=confidence)
        with pytest.raises(ValueError, match="confidence must be in"):
            bs.corr_ci(x, x[::-1].copy() + np.array([0, 1, 0, 2, 0, 1, 0, 3.0]), confidence=confidence)
        if not np.isnan(confidence):  # the NaN case is the subprocess test above
            with pytest.raises(ValueError, match="confidence must be in"):
                bs.odds_ratio(table, confidence=confidence)

    def test_valid_calls_still_work(self, scipy_stats):
        x = np.random.default_rng(0).normal(size=30)
        y = np.random.default_rng(1).normal(0.0, 2.0, size=30)
        lo, hi = bs.mean_diff_ci(x, y, 0.05, False)
        ref = scipy_stats.ttest_ind(x, y, equal_var=False).confidence_interval()
        assert close(lo, ref.low, 1e-10) and close(hi, ref.high, 1e-10)
        f = bs.f_test_var(x, y)
        ratio = np.var(y, ddof=1) / np.var(x, ddof=1)
        assert close(f["statistic"], max(ratio, 1 / ratio), 1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_p = 2 * min(scipy_stats.f.sf(f["statistic"], 29, 29), scipy_stats.f.cdf(f["statistic"], 29, 29))
        assert close(f["pvalue"], ref_p, 1e-9)
        assert 0.0 < bs.var_ci(x, confidence=0.9)["ci_lower"] < np.var(x, ddof=1)
