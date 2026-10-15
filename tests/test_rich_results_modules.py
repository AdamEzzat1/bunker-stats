"""Rich result objects for matrix, robust, rolling and resampling.

Companion to test_rich_results.py (which covers inference). Same contract:
default returns are unchanged; the rich object is opt-in and follows the shared
protocol. Matrix/outlier results are *array-like* (they behave like their
payload) rather than scalar-tuple-like, which these tests pin down.
"""
from __future__ import annotations

import numpy as np
import pytest

import bunker_stats as bs
from bunker_stats import matrix, robust
from bunker_stats.rolling import RollingResult


@pytest.fixture
def matrix_data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(90, 3))


@pytest.fixture
def sample():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 80)
    x[0] = 40.0   # a clear outlier
    x[1] = -35.0
    return x


# ======================================================================
# Matrix
# ======================================================================

class TestCorrelationMatrixResult:
    def test_default_is_ndarray(self, matrix_data):
        assert isinstance(bs.corr_matrix(matrix_data), np.ndarray)

    def test_rich_is_array_like(self, matrix_data):
        C = bs.corr_matrix(matrix_data, rich=True)
        assert isinstance(C, matrix.CorrelationMatrixResult)
        # np.asarray gives the raw matrix, equal to the default return
        np.testing.assert_allclose(np.asarray(C), bs.corr_matrix(matrix_data))
        assert C.shape == (3, 3)
        assert C[0, 0] == pytest.approx(1.0)

    def test_metadata(self, matrix_data):
        C = bs.corr_matrix(matrix_data, rich=True, columns=["a", "b", "c"])
        assert C.columns == ["a", "b", "c"]
        assert C.method == "pearson"
        assert C.n_obs == 90

    def test_columns_kwarg_not_forwarded_to_kernel(self, matrix_data):
        # `columns` is builder-only; passing it must not break the raw call.
        C = bs.corr_matrix(matrix_data, rich=True, columns=["a", "b", "c"])
        assert C.to_frame().columns.tolist() == ["a", "b", "c"]

    def test_columns_length_validated(self, matrix_data):
        C = bs.corr_matrix(matrix_data, rich=True, columns=["only", "two"])
        with pytest.raises(ValueError, match="length 2"):
            C.to_frame()

    def test_to_frame_is_labeled(self, matrix_data):
        pytest.importorskip("pandas")
        C = bs.corr_matrix(matrix_data, rich=True, columns=["x", "y", "z"])
        df = C.to_frame()
        assert df.index.tolist() == ["x", "y", "z"]
        assert df.columns.tolist() == ["x", "y", "z"]

    def test_style_heatmap(self, matrix_data):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("matplotlib")
        styler = bs.corr_matrix(matrix_data, rich=True).style_heatmap()
        assert isinstance(styler, pd.io.formats.style.Styler)
        assert "background-color" in styler.to_html()

    def test_to_dict(self, matrix_data):
        d = bs.corr_matrix(matrix_data, rich=True).to_dict()
        assert isinstance(d["matrix"], list)          # array -> list by default
        assert d["method"] == "pearson" and d["n_obs"] == 90
        d2 = bs.corr_matrix(matrix_data, rich=True).to_dict(array=True)
        assert isinstance(d2["matrix"], np.ndarray)

    def test_info(self, matrix_data):
        text = bs.corr_matrix(matrix_data, rich=True).info()
        assert "Correlation Matrix" in text and "Method:" in text
        text.encode("ascii")   # ASCII-safe


class TestCovarianceMatrixResult:
    def test_rich_and_ddof(self, matrix_data):
        V = bs.cov_matrix(matrix_data, rich=True)
        assert isinstance(V, matrix.CovarianceMatrixResult)
        assert V.ddof == 1
        np.testing.assert_allclose(np.asarray(V), bs.cov_matrix(matrix_data))

    def test_default_unchanged(self, matrix_data):
        assert isinstance(bs.cov_matrix(matrix_data), np.ndarray)


# ======================================================================
# Robust
# ======================================================================

class TestRobustFitResult:
    def test_default_is_tuple(self, sample):
        out = bs.robust_fit(sample)
        assert isinstance(out, tuple) and len(out) == 2

    def test_rich_unpacks_location_scale(self, sample):
        f = bs.robust_fit(sample, rich=True)
        assert isinstance(f, robust.RobustFitResult)
        location, scale = f
        assert location == pytest.approx(f.location)
        assert scale == pytest.approx(f.scale)
        assert f[0] == location

    def test_matches_raw(self, sample):
        raw = bs.robust_fit(sample)
        f = bs.robust_fit(sample, rich=True)
        assert (f.location, f.scale) == pytest.approx(raw)

    def test_methods_and_counts(self, sample):
        f = bs.robust_fit(sample, location="huber", scale="mad", rich=True)
        assert f.method_location == "huber" and f.method_scale == "mad"
        assert f.n == len(sample) and f.n_missing == 0

    def test_zscores_helper(self, sample):
        f = bs.robust_fit(sample, rich=True)
        z = f.zscores(sample)
        assert z.shape == sample.shape
        expected = (sample - f.location) / f.scale
        np.testing.assert_allclose(z, expected)

    def test_info_ascii(self, sample):
        bs.robust_fit(sample, rich=True).info().encode("ascii")


class TestOutlierResult:
    def test_default_is_bool_array(self, sample):
        m = bs.iqr_outliers(sample)
        assert isinstance(m, np.ndarray) and m.dtype == bool

    def test_rich_is_array_like(self, sample):
        o = bs.iqr_outliers(sample, rich=True)
        assert isinstance(o, robust.OutlierResult)
        np.testing.assert_array_equal(np.asarray(o), bs.iqr_outliers(sample))
        assert len(o) == len(sample)
        assert o[0] == True  # the planted outlier at index 0

    def test_counts_and_indices(self, sample):
        o = bs.iqr_outliers(sample, rich=True)
        assert o.n == len(sample)
        assert o.n_outliers == int(np.asarray(o).sum())
        assert o.proportion_outliers == pytest.approx(o.n_outliers / o.n)
        assert 0 in o.indices() and 1 in o.indices()

    def test_bounds_reported(self, sample):
        o = bs.iqr_outliers(sample, k=1.5, rich=True)
        assert o.method == "iqr" and o.threshold == 1.5
        assert o.lower_bound is not None and o.upper_bound is not None
        # Every flagged value lies outside the reported fence.
        flagged = sample[np.asarray(o)]
        assert np.all((flagged < o.lower_bound) | (flagged > o.upper_bound))

    def test_zscore_method(self, sample):
        o = bs.zscore_outliers(sample, threshold=3.0, rich=True)
        assert o.method == "zscore" and o.threshold == 3.0

    def test_default_zscore_unchanged(self, sample):
        assert isinstance(bs.zscore_outliers(sample), np.ndarray)

    def test_info_ascii(self, sample):
        bs.iqr_outliers(sample, rich=True).info().encode("ascii")


# ======================================================================
# Rolling
# ======================================================================

class TestRollingResult:
    @pytest.fixture
    def series(self):
        return np.arange(30.0)

    def test_aggregate_unchanged(self, series):
        d = bs.Rolling(series, window=5).aggregate("mean", "std")
        assert isinstance(d, dict) and set(d) == {"mean", "std"}

    def test_result_is_rich(self, series):
        r = bs.Rolling(series, window=5).result("mean", "std")
        assert isinstance(r, RollingResult)
        assert r.window == 5 and tuple(r.stats) == ("mean", "std")

    def test_dict_like_access(self, series):
        r = bs.Rolling(series, window=5).result("mean", "std")
        assert list(r) == ["mean", "std"]
        assert len(r) == 2
        np.testing.assert_allclose(
            r["mean"], bs.Rolling(series, window=5).aggregate("mean")["mean"]
        )

    def test_default_stats(self, series):
        r = bs.Rolling(series, window=5).result()
        assert tuple(r.stats) == ("mean", "std", "min", "max")

    def test_to_frame(self, series):
        pytest.importorskip("pandas")
        r = bs.Rolling(series, window=5).result("mean", "std")
        df = r.to_frame()
        assert df.columns.tolist() == ["roll5_mean", "roll5_std"]

    def test_to_dict(self, series):
        r = bs.Rolling(series, window=5).result("mean")
        d = r.to_dict()
        assert d["window"] == 5 and d["stats"] == ["mean"]
        assert isinstance(d["table"]["mean"], list)
        assert isinstance(r.to_dict(array=True)["table"]["mean"], np.ndarray)

    def test_info_ascii(self, series):
        bs.Rolling(series, window=5).result("mean").info().encode("ascii")

    def test_config_captured(self, series):
        r = bs.Rolling(series, window=7, min_periods=3, nan_policy="ignore").result("mean")
        assert r.window == 7 and r.min_periods == 3 and r.nan_policy == "ignore"


# ======================================================================
# Resampling
# ======================================================================

class TestResamplingRichResults:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(3)
        return rng.normal(5.0, 2.0, 300)

    def test_bootstrap_default_is_tuple(self, data):
        out = bs.bootstrap(data, random_state=1)
        assert isinstance(out, tuple) and len(out) == 3

    def test_bootstrap_rich(self, data):
        r = bs.bootstrap(data, random_state=1, rich=True)
        assert isinstance(r, bs.BootstrapResult)
        est, lo, hi = r
        assert (est, lo, hi) == pytest.approx(bs.bootstrap(data, random_state=1))
        assert r.n_resamples == 1000 and r.confidence_level == 0.95
        assert r.random_state == 1

    def test_bootstrap_rich_to_dict_and_info(self, data):
        r = bs.bootstrap(data, random_state=1, rich=True)
        d = r.to_dict()
        assert {"estimate", "ci_lower", "ci_upper"} <= set(d)
        r.info().encode("ascii")

    def test_permutation_default_is_tuple(self):
        rng = np.random.default_rng(4)
        out = bs.permutation_test(rng.normal(size=40), rng.normal(size=40), random_state=1)
        assert isinstance(out, tuple) and len(out) == 2

    def test_permutation_rich_with_conclusion(self):
        rng = np.random.default_rng(4)
        g1, g2 = rng.normal(0, 1, 60), rng.normal(1.0, 1, 60)
        r = bs.permutation_test(g1, g2, random_state=1, rich=True)
        assert isinstance(r, bs.PermutationTestResult)
        stat, pval = r
        assert stat == pytest.approx(r.statistic)
        assert "reject" in r.conclusion(0.05).lower()
        assert r.method == "mean_diff" and r.n_permutations == 1000

    def test_result_classes_exported(self):
        assert bs.BootstrapResult is not None
        assert bs.PermutationTestResult is not None


# ======================================================================
# Public surface across all new modules
# ======================================================================

class TestPublicSurface:
    NAMES = [
        "CorrelationMatrixResult", "CovarianceMatrixResult",
        "RobustFitResult", "OutlierResult", "RollingResult",
        "BootstrapResult", "PermutationTestResult",
    ]

    @pytest.mark.parametrize("name", NAMES)
    def test_exported_from_root(self, name):
        assert hasattr(bs, name)
        assert name in bs.__all__

    def test_submodule_exports(self):
        assert matrix.CorrelationMatrixResult is bs.CorrelationMatrixResult
        assert robust.OutlierResult is bs.OutlierResult

    def test_all_has_no_duplicates_for_new_names(self):
        for name in self.NAMES:
            assert bs.__all__.count(name) == 1

    def test_wrapped_funcs_advertise_rich(self):
        for fn in (bs.corr_matrix, bs.robust_fit, bs.iqr_outliers):
            assert "rich" in (fn.__doc__ or "").lower()
