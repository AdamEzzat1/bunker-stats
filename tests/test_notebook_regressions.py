"""Regression tests for defects found by reviewing the notebook layer.

Each class pins one reproduced failure:

* ``TestQnScale``            - Qn allocated n(n-1)/2 floats and aborted the
                               process on large columns.
* ``TestTinyPValues``        - t-based p-values cancelled to exactly 0.0.
* ``TestDegenerateTStatistics`` - infinite / NaN t statistics.
* ``TestCenteredRolling``    - even centered windows covered window+1 rows.
* ``TestMultiIndexColumns``  - every report raised on MultiIndex columns.
* ``TestKurtosisConvention`` - two reports disagreed by 3 under one name.
* ``TestBootstrapEstimate``  - ``estimate`` was the bootstrap mean.
* ``TestRenderLimit``        - highlights silently vanished on big frames.

Conventions used here, each learned from a test that passed while the code
was wrong:

* Small quantities are compared with ``close(..., rel)``, which sets
  ``abs=0``. ``pytest.approx`` otherwise adds an absolute tolerance of 1e-12,
  under which ``0.0 == approx(1e-140)`` is true.
* References are independent implementations (scipy, pandas, statsmodels, or
  an all-pairs computation), not a second copy of the kernel's formula. Where
  a test only checks self-consistency its name says so.
* Optional references are imported inside the tests that need them, so a
  missing package skips those tests and not the whole file.
* Subprocess tests assert that the child imported the same package.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="notebook layer requires pandas")

import bunker_stats as bs
from bunker_stats import notebook as nb

PACKAGE_DIR = str(pathlib.Path(bs.__file__).resolve().parent)


def close(value: float, reference: float, rel: float) -> bool:
    """Relative comparison with no absolute tolerance."""
    return value == pytest.approx(reference, rel=rel, abs=0)


@pytest.fixture(scope="module")
def scipy_stats():
    return pytest.importorskip("scipy.stats")


def run_isolated(body: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter that must import *this* package.

    A crash that aborts the process then fails one test instead of killing
    pytest. The child resolves ``bunker_stats`` from its own environment, so
    it is pointed at this checkout explicitly and checks what it imported.
    """
    root = str(pathlib.Path(PACKAGE_DIR).parent)
    code = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, {root!r})\n"
        "import numpy as np, pandas as pd, bunker_stats as bs\n"
        "from bunker_stats import notebook as nb\n"
        f"assert str(pathlib.Path(bs.__file__).resolve().parent) == {PACKAGE_DIR!r}, bs.__file__\n"
        + body
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
    )


# ======================================================================
# Qn scale: O(n) memory
# ======================================================================

QN_CONSTANT = 2.219144465985076   # 1 / (sqrt(2) * Phi^-1(5/8))


def _all_pairs_quartile(x: np.ndarray) -> float:
    """Rousseeuw & Croux's Qn evaluated the slow way: the constant times the
    k-th smallest of all pairwise distances, k = C(floor(n/2) + 1, 2). This
    is written from the paper's definition, not from the kernel; the kernel
    is also checked against statsmodels below (``test_qn_matches_statsmodels``)."""
    i, j = np.triu_indices(len(x), k=1)
    with np.errstate(invalid="ignore"):
        diffs = np.sort(np.abs(x[i] - x[j]))
    h = len(x) // 2 + 1
    return float(diffs[h * (h - 1) // 2 - 1] * QN_CONSTANT)


class TestQnScale:
    @pytest.mark.parametrize("n", [3, 4, 5, 10, 51, 200])
    def test_fast_path_equals_all_pairs_definition(self, n):
        x = np.random.default_rng(n).normal(size=n)
        assert bs.qn_scale(x) == _all_pairs_quartile(x)

    def test_heavy_ties_and_constant(self):
        x = np.random.default_rng(0).integers(0, 4, size=300).astype(float)
        assert bs.qn_scale(x) == _all_pairs_quartile(x)
        assert bs.qn_scale(np.full(40, 2.5)) == 0.0

    def test_large_offset_small_spread(self):
        x = 1e9 + np.random.default_rng(1).normal(scale=1e-3, size=150)
        assert bs.qn_scale(x) == _all_pairs_quartile(x)

    def test_large_columns_do_not_abort_the_process(self):
        """200,000 rows used to request a 160 GB buffer; the failed allocation
        killed the interpreter with no Python exception."""
        done = run_isolated(
            "x = np.random.default_rng(0).normal(size=200_000)\n"
            "frame = pd.DataFrame({'x': x})\n"
            "for helper in (nb.describe_fast, nb.robust_summary):\n"
            "    q = float(helper(frame).loc['x', 'qn_scale'])\n"
            "    assert 0.99 < q < 1.01, (helper.__name__, q)\n"  # consistent for sigma = 1
            "print('OK')\n"
        )
        assert done.returncode == 0, done.stderr[-400:]
        assert done.stdout.strip() == "OK"

    def test_one_infinity_in_a_large_sample_stays_finite(self):
        """The first rewrite returned NaN for any Inf once n passed 4096. A
        50%-breakdown scale must shrug off one infinite value."""
        x = np.random.default_rng(0).standard_normal(4097)
        x[0] = np.inf
        value = bs.qn_scale(x)
        assert np.isfinite(value)
        assert value == _all_pairs_quartile(x)

    @pytest.mark.parametrize(
        "n_pos, n_neg", [(1, 0), (0, 1), (2, 0), (1, 1), (3, 2), (40, 0), (25, 25)]
    )
    def test_infinities_follow_the_all_pairs_definition(self, n_pos, n_neg):
        """Differences sort as finite < +Inf < NaN (Inf - Inf); the result is
        whichever run the quartile rank lands in."""
        x = np.random.default_rng(n_pos * 10 + n_neg).normal(size=60)
        x[:n_pos] = np.inf
        x[n_pos:n_pos + n_neg] = -np.inf
        ours, ref = bs.qn_scale(x), _all_pairs_quartile(x)
        assert (np.isnan(ours) and np.isnan(ref)) or ours == ref

    def test_constant_is_the_normal_consistency_factor(self, scipy_stats):
        assert close(QN_CONSTANT, 1.0 / (np.sqrt(2.0) * scipy_stats.norm.ppf(0.625)), 1e-15)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 51, 100, 501, 2000])
    def test_qn_matches_statsmodels(self, n):
        """statsmodels implements Rousseeuw & Croux's definition with their
        own O(n log n) algorithm. The kernel used the first-quartile rank
        floor(C(n,2)/4) and a rounded constant, which agreed only for odd
        n <= 9 (to 0.1%); for normal samples it was typically 0.68x at n = 6
        and 0.82x at n = 10, and always 0.37x at n = 2."""
        robust_scale = pytest.importorskip("statsmodels.robust.scale")
        x = np.random.default_rng(n).normal(size=n)
        assert close(bs.qn_scale(x), float(robust_scale.qn_scale(x)), 1e-12)

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 250.0])
    def test_qn_estimates_sigma_for_normal_data(self, scale):
        x = np.random.default_rng(5).normal(0.0, scale, size=20000)
        assert close(bs.qn_scale(x), scale, 0.03)

    def test_qn_is_scale_and_shift_equivariant(self):
        x = np.random.default_rng(8).normal(size=40)
        base = bs.qn_scale(x)
        assert close(bs.qn_scale(3.5 * x), 3.5 * base, 1e-12)
        assert close(bs.qn_scale(-x), base, 1e-12)
        assert close(bs.qn_scale(x + 7.0), base, 1e-9)


# ======================================================================
# t-based p-values far in the tail
# ======================================================================

class TestTinyPValues:
    @pytest.fixture
    def strong(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=400)
        return a, a + 0.5 * rng.normal(size=400)

    def test_pearson_pvalue_matches_scipy(self, strong, scipy_stats):
        a, b = strong
        ref = scipy_stats.pearsonr(a, b)[1]
        assert 0.0 < ref < 1e-100  # the regime that used to give exactly 0.0
        assert close(bs.pearson_corr_test(a, b)["pvalue"], ref, 1e-9)

    def test_spearman_pvalue_matches_scipy(self, strong, scipy_stats):
        a, b = strong
        ref = scipy_stats.spearmanr(a, b)[1]
        assert 0.0 < ref < 1e-80
        assert close(bs.spearman_corr_test(a, b)["pvalue"], ref, 1e-9)

    def test_correlation_report_carries_the_kernel_pvalue(self, strong):
        a, b = strong
        table = nb.correlation_report(pd.DataFrame({"a": a, "b": b}), pvalues=True)
        assert table["pvalue"].iloc[0] == bs.pearson_corr_test(a, b)["pvalue"] > 0.0

    def test_two_sample_ttest_is_symmetric_in_the_tail(self, scipy_stats):
        """`1 - cdf(t)` underflowed only for t > 0, so swapping the samples
        changed a two-sided p-value from ~1e-60 to exactly 0."""
        rng = np.random.default_rng(3)
        x, y = rng.normal(3.0, 1.0, 200), rng.normal(0.0, 1.0, 200)
        p_xy = bs.t_test_2samp(x, y)["pvalue"]
        ref = scipy_stats.ttest_ind(x, y)[1]
        assert 0.0 < ref < 1e-50
        assert close(p_xy, ref, 1e-9)
        assert close(bs.t_test_2samp(y, x)["pvalue"], p_xy, 1e-12)

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_one_sided_tails(self, alternative, scipy_stats):
        rng = np.random.default_rng(4)
        shift = 3.0 if alternative == "greater" else -3.0
        x = rng.normal(shift, 1.0, 150)
        ref = scipy_stats.ttest_1samp(x, 0.0, alternative=alternative)[1]
        assert 0.0 < ref < 1e-40
        assert close(bs.t_test_1samp(x, 0.0, alternative)["pvalue"], ref, 1e-9)

    def test_paired_ttest_tail(self, scipy_stats):
        rng = np.random.default_rng(5)
        before = rng.normal(size=120)
        after = before + 2.0 + rng.normal(scale=0.3, size=120)
        ref = scipy_stats.ttest_rel(after, before)[1]
        assert 0.0 < ref < 1e-50
        assert close(bs.t_test_paired(after, before)["pvalue"], ref, 1e-9)

    def test_tiny_degrees_of_freedom(self, scipy_stats):
        """With df = 1 and t = 2e9 the incomplete beta inside statrs returns
        exactly 0 although the true p-value is 3e-10."""
        x = np.array([1.0, 1.000000001])
        ref = scipy_stats.ttest_1samp(x, 0.0).pvalue
        assert 1e-11 < ref < 1e-9
        assert close(bs.t_test_1samp(x, 0.0)["pvalue"], ref, 1e-9)

    def test_ordinary_pvalues_unchanged(self, scipy_stats):
        rng = np.random.default_rng(6)
        x, y = rng.normal(size=40), rng.normal(0.2, 1.0, size=45)
        assert close(bs.t_test_2samp(x, y)["pvalue"], scipy_stats.ttest_ind(x, y)[1], 1e-10)
        assert close(
            bs.pearson_corr_test(x, y[:40])["pvalue"],
            scipy_stats.pearsonr(x, y[:40])[1], 1e-10,
        )


class TestDegenerateTStatistics:
    """Zero standard error makes t infinite. An infinite statistic pointing
    *away* from a one-sided alternative is no evidence, so p = 1 -- these
    used to return 0."""

    @pytest.mark.parametrize("popmean", [0.0, 2.0])
    @pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
    def test_one_sample_constant_data(self, popmean, alternative, scipy_stats):
        data = np.array([1.0, 1.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = scipy_stats.ttest_1samp(data, popmean, alternative=alternative).pvalue
        assert ref in (0.0, 1.0)
        assert bs.t_test_1samp(data, popmean, alternative)["pvalue"] == ref

    @pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
    @pytest.mark.parametrize("swap", [False, True])
    def test_paired_constant_difference(self, alternative, swap, scipy_stats):
        a, b = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0])
        if swap:
            a, b = b, a
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = scipy_stats.ttest_rel(a, b, alternative=alternative).pvalue
        assert ref in (0.0, 1.0)
        assert bs.t_test_paired(a, b, alternative)["pvalue"] == ref

    def test_nan_statistic_gives_nan_pvalue(self):
        out = bs.t_test_1samp(np.array([1.0, 2.0, 3.0]), float("nan"))
        assert np.isnan(out["statistic"]) and np.isnan(out["pvalue"])


# ======================================================================
# Centered rolling windows
# ======================================================================

class TestCenteredRolling:
    @pytest.mark.parametrize("window", [1, 2, 3, 4, 5, 6, 7, 10])
    def test_matches_pandas_center_true(self, window):
        x = np.random.default_rng(window).normal(size=40)
        df = pd.DataFrame({"x": x})
        out = nb.rolling_report(
            df, "x", window, stats=("mean", "std", "min", "max"), alignment="centered"
        )
        ref = df["x"].rolling(window, center=True)
        np.testing.assert_allclose(out[f"x_roll{window}_mean"], ref.mean(), rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(out[f"x_roll{window}_min"], ref.min(), rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(out[f"x_roll{window}_max"], ref.max(), rtol=1e-12, equal_nan=True)
        if window > 1:
            np.testing.assert_allclose(
                out[f"x_roll{window}_std"], ref.std(), rtol=1e-9, equal_nan=True
            )
        # NaN in the same places, not merely "close where both are finite"
        assert (out[f"x_roll{window}_mean"].isna() == ref.mean().isna()).all()

    def test_even_window_uses_exactly_window_rows(self):
        df = pd.DataFrame({"x": np.arange(1.0, 11.0)})
        out = nb.rolling_report(df, "x", 4, stats=("mean", "count"), alignment="centered")
        full = out["x_roll4_mean"].notna()
        assert full.sum() == 7 and (out.loc[full, "x_roll4_count"] == 4).all()
        assert out["x_roll4_mean"].tolist()[2:9] == [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

    def test_edges_are_nan_unless_min_periods_is_lowered(self):
        df = pd.DataFrame({"x": np.arange(1.0, 11.0)})
        strict = nb.rolling_report(df, "x", 5, stats=("mean",), alignment="centered")
        assert strict["x_roll5_mean"].isna().tolist() == [True, True] + [False] * 6 + [True, True]
        loose = nb.rolling_report(
            df, "x", 5, stats=("mean",), alignment="centered", min_periods=3
        )
        ref = df["x"].rolling(5, center=True, min_periods=3).mean()
        assert not ref.isna().any()
        np.testing.assert_allclose(loose["x_roll5_mean"], ref, rtol=1e-12)

    def test_trailing_is_unchanged(self):
        x = np.random.default_rng(9).normal(size=30)
        df = pd.DataFrame({"x": x})
        out = nb.rolling_report(df, "x", 4, stats=("mean",))
        ref = df["x"].rolling(4).mean()
        np.testing.assert_allclose(out["x_roll4_mean"], ref, rtol=1e-12, equal_nan=True)
        assert out["x_roll4_mean"].isna().sum() == ref.isna().sum() == 3

    def test_unknown_alignment_is_a_clear_error(self):
        with pytest.raises(ValueError, match="alignment must be one of"):
            nb.rolling_report(pd.DataFrame({"x": np.arange(5.0)}), "x", 2, alignment="center")


# ======================================================================
# MultiIndex columns in reports
# ======================================================================

class TestMultiIndexColumns:
    @pytest.fixture
    def frame(self):
        columns = pd.MultiIndex.from_product([["g1", "g2"], ["a", "b"]], names=["grp", "var"])
        values = np.random.default_rng(0).normal(size=(30, 4))
        values[3, 1] = np.nan
        return pd.DataFrame(values, columns=columns)

    @pytest.fixture
    def flat(self, frame):
        out = frame.copy()
        out.columns = ["_".join(c) for c in frame.columns]
        return out

    @pytest.mark.parametrize(
        "name, kwargs",
        [
            ("robust_summary", {}),
            ("describe_fast", {}),
            ("outlier_report", {}),
            ("normality_report", {}),
            ("missingness_report", {}),
            ("bootstrap_ci_report", {"n_resamples": 200, "random_state": 0}),
        ],
    )
    def test_reports_keep_the_levels_and_the_numbers(self, frame, flat, name, kwargs):
        out = getattr(nb, name)(frame, **kwargs)
        ref = getattr(nb, name)(flat, **kwargs)
        assert isinstance(out.index, pd.MultiIndex)
        assert list(out.index) == list(frame.columns)
        assert list(out.index.names) == ["grp", "var"]
        # Row by label, and every value identical to the flat-column report.
        assert out.loc[("g1", "b")].equals(out.iloc[1])
        pd.testing.assert_frame_equal(
            out.reset_index(drop=True), ref.reset_index(drop=True)
        )

    def test_correlation_matrix_is_labelled_on_both_axes(self, frame):
        matrix = nb.correlation_report(frame)
        assert matrix.columns.equals(matrix.index)
        assert matrix.loc[("g1", "a"), ("g1", "a")] == 1.0
        ref = frame.corr()
        for row in frame.columns:
            for col in frame.columns:
                assert close(matrix.loc[row, col], ref.loc[row, col], 1e-10)

    def test_correlation_long_form(self, frame):
        long = nb.correlation_report(frame, pvalues=True)
        assert len(long) == 6
        assert (long["x"].iloc[0], long["y"].iloc[0]) == (("g1", "a"), ("g1", "b"))
        assert long["n"].iloc[0] == 29  # the NaN row is dropped pairwise

    @pytest.mark.parametrize(
        "name", ["robust_summary", "outlier_report", "correlation_report", "bootstrap_ci_report"]
    )
    def test_rich_reports(self, frame, name):
        report = getattr(nb, name)(frame, rich=True)
        assert len(report.to_frame()) == 4
        text = report.info()
        assert text.splitlines()[0] == report._title and "Rows:" in text

    def test_flat_columns_are_unaffected(self):
        frame = pd.DataFrame({"x": np.arange(5.0), "y": np.arange(5.0) ** 2})
        out = nb.robust_summary(frame)
        assert out.index.name == "column" and not isinstance(out.index, pd.MultiIndex)
        matrix = nb.correlation_report(frame)
        assert matrix.columns.name is None and list(matrix.columns) == ["x", "y"]


# ======================================================================
# One meaning for "kurtosis"
# ======================================================================

class TestKurtosisConvention:
    def test_reports_agree(self, scipy_stats):
        df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 50]})
        a = nb.normality_report(df).loc["x", "kurtosis"]
        b = nb.robust_summary(df).loc["x", "kurtosis"]
        # They used to be 5.974 and 2.974.
        assert close(a, b, 1e-12)
        assert close(a, scipy_stats.kurtosis(df["x"], fisher=True, bias=True), 1e-12)

    def test_is_excess_kurtosis(self):
        x = np.random.default_rng(0).normal(size=20_000)
        k = nb.normality_report(pd.DataFrame({"x": x})).loc["x", "kurtosis"]
        assert abs(k) < 0.15  # ~0 for normal data, not ~3

    def test_jarque_bera_is_untouched_by_the_conversion(self, scipy_stats):
        """A sample that is close to normal, so the p-value is moderate and
        the comparison is meaningful."""
        x = np.random.default_rng(1).normal(size=300)
        row = nb.normality_report(pd.DataFrame({"x": x})).loc["x"]
        ref = scipy_stats.jarque_bera(x)
        assert 0.01 < ref.pvalue < 0.99
        assert close(row["jb_statistic"], ref.statistic, 1e-10)
        assert close(row["jb_pvalue"], ref.pvalue, 1e-9)

    def test_jarque_bera_pvalue_in_the_far_tail(self, scipy_stats):
        """Was 0.0 while the tail was `1 - cdf` (scipy: 2.4e-40 here)."""
        x = np.random.default_rng(1).standard_t(df=5, size=300)
        row = nb.normality_report(pd.DataFrame({"x": x})).loc["x"]
        assert close(row["jb_pvalue"], scipy_stats.jarque_bera(x).pvalue, 1e-6)


# ======================================================================
# Bootstrap report: estimate and small-sample warning
# ======================================================================

class TestBootstrapEstimate:
    @pytest.mark.parametrize(
        "stat, ref",
        [("mean", np.mean), ("median", np.median), ("std", lambda v: np.std(v, ddof=0))],
    )
    def test_estimate_is_the_sample_statistic(self, stat, ref):
        x = np.random.default_rng(0).lognormal(size=80)
        out = nb.bootstrap_ci_report(
            pd.DataFrame({"x": x}), stat=stat, n_resamples=500, random_state=0
        )
        assert close(out.loc["x", "estimate"], ref(x), 1e-12)

    def test_std_estimate_uses_the_resampler_ddof(self):
        """The resampling kernel computes std with ddof=0: on [0, 1] no draw
        can exceed 0.5. Reporting the ddof=1 value (0.707) put the estimate
        outside its own interval."""
        x = np.array([0.0, 1.0])
        _, lo, hi = bs.BootstrapConfig(n_resamples=2000, stat="std", random_state=0).run(x)
        assert (lo, hi) == (0.0, 0.5)
        report = nb.bootstrap_ci_report(
            pd.DataFrame({"x": x}), stat="std", n_resamples=2000, random_state=0, rich=True
        )
        # ddof=0 value (0.5), not the ddof=1 value (0.7071). It is the ddof=1
        # kernel rescaled, so it agrees to rounding rather than bit for bit.
        assert close(report.data.loc["x", "estimate"], 0.5, 1e-12)
        assert report.meta["std_ddof"] == 0
        assert nb.bootstrap_ci_report(pd.DataFrame({"x": x}), rich=True).meta["std_ddof"] is None

    @pytest.mark.parametrize("stat", ["mean", "median", "std"])
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_estimate_lies_inside_its_interval_at_tiny_n(self, stat, n):
        """With mismatched estimators this failed for std in every trial at
        n = 3. Testing only large n, or only the mean, would not notice."""
        rng = np.random.default_rng(100 * n + len(stat))
        for seed in range(150):
            x = rng.lognormal(sigma=1.5, size=n)
            row = nb.bootstrap_ci_report(
                pd.DataFrame({"x": x}), stat=stat, n_resamples=200, random_state=seed
            ).iloc[0]
            # A resample that is a permutation of the sample reproduces the
            # estimate up to rounding, so allow a few ulps at the boundary.
            slack = 1e-12 * abs(row["estimate"])
            assert row["ci_lower"] - slack <= row["estimate"] <= row["ci_upper"] + slack, (seed, x)

    def test_interval_is_unchanged_by_the_estimate_fix(self):
        x = np.random.default_rng(1).normal(size=60)
        out = nb.bootstrap_ci_report(
            pd.DataFrame({"x": x}), stat="median", n_resamples=400, random_state=3
        )
        _, lo, hi = bs.BootstrapConfig(
            n_resamples=400, random_state=3, stat="median", nan_policy="omit"
        ).run(x)
        assert (out.loc["x", "ci_lower"], out.loc["x", "ci_upper"]) == (lo, hi)

    def test_small_sample_warning_boundary(self):
        rng = np.random.default_rng(2)

        def warnings_for(n):
            frame = pd.DataFrame({"x": rng.normal(size=n)})
            return nb.bootstrap_ci_report(
                frame, n_resamples=1000, random_state=0, rich=True
            ).warnings

        assert any("n < 30" in w and "too narrow" in w for w in warnings_for(29))
        assert warnings_for(30) == []


# ======================================================================
# pandas' render limit
# ======================================================================

class TestRenderLimit:
    @pytest.fixture(autouse=True)
    def _needs_jinja2(self):
        pytest.importorskip("jinja2")

    @pytest.fixture
    def frame(self):
        x = np.random.default_rng(0).normal(size=60)
        x[-1] = 40.0  # an outlier in the last row, beyond a small render limit
        return pd.DataFrame({"x": x, "y": np.arange(60.0)})

    @pytest.mark.parametrize(
        "name",
        ["outlier_style", "iqr_outlier_style", "zscore_style", "demean_style",
         "missingness_style", "scale_compare_style"],
    )
    def test_warning_fires_once_and_points_at_the_caller(self, frame, name):
        """`iqr_outlier_style` calls `outlier_style`; with a fixed stacklevel
        its warning was attributed to a line inside the library."""
        helper = getattr(nb, name)
        args = () if name in ("outlier_style", "missingness_style") else ("x",)
        with pd.option_context("styler.render.max_elements", 40):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                helper(frame, *args)
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        assert "styler.render.max_elements" in str(caught[0].message)
        assert caught[0].filename == __file__

    def test_corr_heatmap_warns_too(self):
        pytest.importorskip("matplotlib")
        frame = pd.DataFrame(np.random.default_rng(2).normal(size=(20, 3)), columns=list("abc"))
        with pd.option_context("styler.render.max_elements", 4):
            with pytest.warns(UserWarning, match="styler.render.max_elements") as caught:
                result = nb.corr_heatmap(frame, rich=True)
        assert len(caught) == 1 and caught[0].filename == __file__
        assert sum("truncated" in w for w in result.warnings) == 1

    def test_rich_result_reports_what_the_rendering_lost(self, frame):
        with pd.option_context("styler.render.max_elements", 40):
            with pytest.warns(UserWarning):
                result = nb.outlier_style(frame, ["x"], rich=True)
            html = result.styler.to_html()
        # The flagged cell exists in the metadata but not in the rendering.
        assert result.metadata["n_outliers_total"] == 1
        assert "#ff8a80" not in html
        notes = [w for w in result.warnings if "truncated" in w]
        assert len(notes) == 1 and "120 cells" in notes[0]

    def test_single_column_wrapper_does_not_duplicate_the_note(self, frame):
        with pd.option_context("styler.render.max_elements", 40):
            with pytest.warns(UserWarning):
                result = nb.iqr_outlier_style(frame, "x", rich=True)
        assert sum("truncated" in w for w in result.warnings) == 1

    def test_no_warning_at_or_below_the_limit(self, frame):
        with pd.option_context("styler.render.max_elements", frame.size):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = nb.outlier_style(frame, rich=True)
        assert [w for w in caught if "max_elements" in str(w.message)] == []
        assert not any("truncated" in w for w in result.warnings)

    def test_transforms_are_unaffected(self, frame):
        with pd.option_context("styler.render.max_elements", 40):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = nb.scale_columns(frame, rich=True)
        assert [w for w in caught if "max_elements" in str(w.message)] == []
        assert not any("truncated" in w for w in result.warnings)
