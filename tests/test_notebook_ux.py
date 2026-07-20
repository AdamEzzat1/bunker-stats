"""Tests for the optional pandas/Jupyter notebook layer (`bunker_stats.notebook`).

Structure:

* ``TestLazyImport``   - the layer must not make pandas a runtime dependency.
* ``TestValidation``   - shared column/dtype/argument validation contract.
* one class per helper - happy path, NaN/inf behaviour, and edge cases.
* ``TestPublicSurface``- ``__all__`` alignment and backwards compatibility.

Styler assertions deliberately check *rendered CSS substrings and cell counts*
rather than exact HTML, so they survive pandas' markup churn.
"""
from __future__ import annotations

import builtins
import importlib
import re
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="notebook layer requires pandas")

import bunker_stats as bs
from bunker_stats import notebook as nb


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def clean_df():
    """50 rows, two well-behaved numeric columns and one string column."""
    rng = np.random.default_rng(20240719)
    return pd.DataFrame(
        {
            "x": rng.normal(loc=0.0, scale=1.0, size=50),
            "y": rng.normal(loc=5.0, scale=2.0, size=50),
            "label": ["a", "b"] * 25,
        }
    )


@pytest.fixture
def messy_df(clean_df):
    """`clean_df` plus a NaN, a +inf, a -inf and a large positive outlier."""
    df = clean_df.copy()
    df.loc[2, "x"] = np.nan
    df.loc[4, "x"] = 50.0          # unambiguous outlier
    df.loc[6, "y"] = np.inf
    df.loc[8, "y"] = -np.inf
    return df


@pytest.fixture
def empty_col_df():
    """A numeric column with no finite values at all."""
    return pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1.0, 2.0, 3.0]})


def render(styler) -> str:
    """Rendered HTML for a Styler, across pandas versions."""
    return styler.to_html()


def is_styler(obj) -> bool:
    return isinstance(obj, pd.io.formats.style.Styler)


_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}", re.S)
_CELL_SELECTOR = re.compile(r"#T_\w+_row\d+_col\d+")


def count_styled_cells(styler, declaration: str) -> int:
    """Number of *cells* whose CSS contains `declaration`.

    pandas coalesces identical rules into a single selector list --
    ``#T_x_row0_col0, #T_x_row1_col0 { background-color: red; }`` -- so
    ``html.count("red")`` counts rules, not cells. Parse the style block and
    count cell selectors instead. This stays correct regardless of how pandas
    groups rules, which is the brittleness we want to avoid.
    """
    html = styler.to_html()
    block = re.search(r"<style[^>]*>(.*?)</style>", html, re.S)
    if block is None:
        return 0
    return sum(
        len(_CELL_SELECTOR.findall(selectors))
        for selectors, body in _RULE.findall(block.group(1))
        if declaration in body
    )


# All report helpers that take (df, columns=None) and return a per-column frame.
PER_COLUMN_REPORTS = [
    nb.robust_summary,
    nb.describe_fast,
    nb.outlier_report,
    nb.normality_report,
    nb.bootstrap_ci_report,
]

# All helpers that validate a (df, columns=None) selection.
COLUMN_SELECTING = PER_COLUMN_REPORTS + [
    nb.scale_columns,
    nb.winsorize_columns,
    nb.outlier_style,
]


# ======================================================================
# Lazy import / optional dependency
# ======================================================================

class TestLazyImport:
    def test_core_package_does_not_import_pandas(self):
        """`import bunker_stats` must work with pandas absent.

        Rather than assert on an already-imported interpreter, re-import the
        package in a subprocess where `pandas` is blocked at the finder level.
        """
        import subprocess

        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name == 'pandas' or name.startswith('pandas.'):\n"
            "            raise ImportError('pandas blocked for test')\n"
            "        return None\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'pandas' or name.startswith('pandas.'):\n"
            "            raise ImportError('pandas blocked for test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "import numpy as np\n"
            "import bunker_stats\n"
            "import bunker_stats.notebook\n"          # module import must also work
            "assert 'pandas' not in sys.modules, 'pandas was imported eagerly'\n"
            "assert bunker_stats.mean(np.array([1.0, 2.0, 3.0])) == 2.0\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_call_without_pandas_raises_actionable_error(self, monkeypatch):
        """Calling a helper with pandas missing names the install command."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pandas" or name.startswith("pandas."):
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "pandas", raising=False)

        with pytest.raises(ImportError, match=r"bunker-stats-rs\[notebook\]"):
            nb._pd()

    def test_notebook_accessible_lazily_from_package(self):
        """`bs.notebook` resolves through the package __getattr__."""
        module = importlib.import_module("bunker_stats")
        assert module.notebook is nb
        assert "notebook" in dir(module)

    def test_unknown_package_attribute_still_raises(self):
        with pytest.raises(AttributeError, match="no attribute"):
            bs.definitely_not_a_real_attribute


# ======================================================================
# Shared validation contract
# ======================================================================

class TestValidation:
    @pytest.mark.parametrize("fn", COLUMN_SELECTING, ids=lambda f: f.__name__)
    def test_missing_column_raises_keyerror(self, fn, clean_df):
        with pytest.raises(KeyError, match="nope"):
            fn(clean_df, ["nope"])

    @pytest.mark.parametrize("fn", COLUMN_SELECTING, ids=lambda f: f.__name__)
    def test_non_numeric_column_raises_typeerror(self, fn, clean_df):
        with pytest.raises(TypeError, match="not numeric"):
            fn(clean_df, ["label"])

    @pytest.mark.parametrize("fn", COLUMN_SELECTING, ids=lambda f: f.__name__)
    def test_empty_column_list_raises_valueerror(self, fn, clean_df):
        with pytest.raises(ValueError, match="empty"):
            fn(clean_df, [])

    @pytest.mark.parametrize("fn", COLUMN_SELECTING, ids=lambda f: f.__name__)
    def test_non_dataframe_raises_typeerror(self, fn):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            fn({"x": [1, 2, 3]})

    @pytest.mark.parametrize("fn", COLUMN_SELECTING, ids=lambda f: f.__name__)
    def test_bare_string_columns_rejected(self, fn, clean_df):
        """`columns="x"` is a common slip; it must not silently iterate chars."""
        with pytest.raises(TypeError, match="sequence of column labels"):
            fn(clean_df, "x")

    @pytest.mark.parametrize("fn", COLUMN_SELECTING, ids=lambda f: f.__name__)
    def test_duplicate_columns_rejected(self, fn, clean_df):
        with pytest.raises(ValueError, match="duplicates"):
            fn(clean_df, ["x", "x"])

    def test_no_numeric_columns_raises(self):
        df = pd.DataFrame({"a": ["p", "q"], "b": ["r", "s"]})
        with pytest.raises(ValueError, match="no numeric columns"):
            nb.robust_summary(df)

    def test_columns_none_skips_non_numeric(self, clean_df):
        out = nb.robust_summary(clean_df)
        assert list(out.index) == ["x", "y"]
        assert "label" not in out.index


# ======================================================================
# robust_summary / describe_fast
# ======================================================================

class TestRobustSummary:
    def test_returns_dataframe_indexed_by_column(self, clean_df):
        out = nb.robust_summary(clean_df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.index) == ["x", "y"]
        assert out.index.name == "column"

    def test_has_documented_columns(self, clean_df):
        out = nb.robust_summary(clean_df)
        expected = {
            "n", "n_missing", "mean", "std", "min", "median", "max",
            "mad", "mad_std", "iqr", "qn_scale", "trimmed_mean",
            "skew", "kurtosis",
        }
        assert set(out.columns) == expected

    def test_matches_rust_kernels_directly(self, clean_df):
        """The report must be a pass-through, not a reimplementation."""
        out = nb.robust_summary(clean_df)
        x = clean_df["x"].to_numpy(dtype=float)
        assert out.loc["x", "mean"] == pytest.approx(bs.mean(x))
        assert out.loc["x", "std"] == pytest.approx(bs.std(x))
        assert out.loc["x", "median"] == pytest.approx(bs.median(x))
        assert out.loc["x", "mad"] == pytest.approx(bs.mad(x))
        assert out.loc["x", "iqr"] == pytest.approx(bs.iqr(x))
        assert out.loc["x", "qn_scale"] == pytest.approx(bs.qn_scale(x))
        assert out.loc["x", "skew"] == pytest.approx(bs.skew(x))
        assert out.loc["x", "kurtosis"] == pytest.approx(bs.kurtosis(x))

    def test_drops_non_finite_and_counts_them(self, messy_df):
        """NaN and +/-inf are excluded and reflected in n / n_missing."""
        out = nb.robust_summary(messy_df)
        assert out.loc["x", "n"] == 49 and out.loc["x", "n_missing"] == 1
        assert out.loc["y", "n"] == 48 and out.loc["y", "n_missing"] == 2
        # Without dropping, a strict kernel would return NaN here.
        assert np.isfinite(out.loc["x", "mean"])
        assert np.isfinite(out.loc["y", "mean"])

    def test_equals_manual_drop(self, messy_df):
        out = nb.robust_summary(messy_df)
        finite = messy_df["y"].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        assert out.loc["y", "mean"] == pytest.approx(bs.mean(finite))

    def test_all_nan_column_yields_nan_row_not_exception(self, empty_col_df):
        out = nb.robust_summary(empty_col_df)
        assert out.loc["x", "n"] == 0
        assert out.loc["x", "n_missing"] == 3
        assert np.isnan(out.loc["x", "mean"])
        # the healthy column is unaffected
        assert out.loc["y", "n"] == 3

    def test_short_column_gives_nan_not_error(self):
        """kurtosis needs n >= 4; a 2-row frame must still produce a row."""
        out = nb.robust_summary(pd.DataFrame({"x": [1.0, 2.0]}))
        assert out.loc["x", "n"] == 2
        assert np.isfinite(out.loc["x", "mean"])
        assert np.isnan(out.loc["x", "kurtosis"])

    def test_integer_and_nullable_dtypes(self):
        df = pd.DataFrame({"i": [1, 2, 3, 4], "n": pd.array([1, 2, None, 4], dtype="Int64")})
        out = nb.robust_summary(df)
        assert out.loc["i", "n"] == 4
        assert out.loc["n", "n"] == 3
        assert out.loc["n", "mean"] == pytest.approx(7 / 3)

    def test_does_not_mutate_input(self, messy_df):
        before = messy_df.copy()
        nb.robust_summary(messy_df)
        pd.testing.assert_frame_equal(messy_df, before)


class TestDescribeFast:
    def test_default_is_robust_and_superset_of_describe(self, clean_df):
        out = nb.describe_fast(clean_df)
        for name in ("n", "mean", "std", "min", "25%", "50%", "75%", "max"):
            assert name in out.columns
        for name in ("mad", "iqr", "qn_scale", "trimmed_mean", "skew", "kurtosis"):
            assert name in out.columns

    def test_robust_false_drops_robust_block(self, clean_df):
        out = nb.describe_fast(clean_df, robust=False)
        assert "mad" not in out.columns
        assert "qn_scale" not in out.columns
        assert "mean" in out.columns

    def test_quartiles_match_pandas(self, clean_df):
        out = nb.describe_fast(clean_df)
        ref = clean_df["x"].describe()
        assert out.loc["x", "25%"] == pytest.approx(ref["25%"])
        assert out.loc["x", "50%"] == pytest.approx(ref["50%"])
        assert out.loc["x", "75%"] == pytest.approx(ref["75%"])

    def test_all_nan_column(self, empty_col_df):
        out = nb.describe_fast(empty_col_df)
        assert out.loc["x", "n"] == 0
        assert np.isnan(out.loc["x", "50%"])


# ======================================================================
# outlier_report / outlier_style
# ======================================================================

class TestOutlierReport:
    @pytest.mark.parametrize("method", ["iqr", "zscore", "robust_zscore"])
    def test_all_methods_flag_a_planted_outlier(self, messy_df, method):
        out = nb.outlier_report(messy_df, ["x"], method=method)
        assert out.loc["x", "n_outliers"] >= 1
        assert out.loc["x", "method"] == method
        assert out.loc["x", "max"] == 50.0

    def test_iqr_bounds_match_manual_computation(self, clean_df):
        out = nb.outlier_report(clean_df, ["x"], method="iqr", k=1.5)
        x = clean_df["x"].to_numpy(dtype=float)
        q1, q3 = bs.percentile(x, 25.0), bs.percentile(x, 75.0)
        assert out.loc["x", "lower_bound"] == pytest.approx(q1 - 1.5 * (q3 - q1))
        assert out.loc["x", "upper_bound"] == pytest.approx(q3 + 1.5 * (q3 - q1))

    def test_robust_zscore_bounds_use_median_and_mad(self, clean_df):
        out = nb.outlier_report(clean_df, ["x"], method="robust_zscore", z_threshold=3.0)
        x = clean_df["x"].to_numpy(dtype=float)
        expected = bs.median(x) + 3.0 * bs.mad(x) * 1.4826
        assert out.loc["x", "upper_bound"] == pytest.approx(expected)

    def test_robust_method_less_swayed_by_contamination(self, clean_df):
        """The point of robust_zscore: one huge value must not inflate the fence."""
        contaminated = clean_df.copy()
        contaminated.loc[0, "x"] = 1e6
        classic = nb.outlier_report(contaminated, ["x"], method="zscore")
        robust = nb.outlier_report(contaminated, ["x"], method="robust_zscore")
        assert robust.loc["x", "upper_bound"] < classic.loc["x", "upper_bound"]
        assert robust.loc["x", "n_outliers"] >= classic.loc["x", "n_outliers"]

    def test_percentage_is_relative_to_finite_count(self, messy_df):
        out = nb.outlier_report(messy_df, ["x"])
        expected = 100.0 * out.loc["x", "n_outliers"] / out.loc["x", "n"]
        assert out.loc["x", "pct_outliers"] == pytest.approx(expected)

    def test_non_finite_never_counted_as_outlier(self, messy_df):
        """+/-inf must not be flagged; they are missing data, not outliers."""
        out = nb.outlier_report(messy_df, ["y"])
        assert out.loc["y", "n_missing"] == 2
        assert np.isfinite(out.loc["y", "min"])
        assert np.isfinite(out.loc["y", "max"])

    def test_all_nan_column(self, empty_col_df):
        out = nb.outlier_report(empty_col_df)
        assert out.loc["x", "n_outliers"] == 0
        assert np.isnan(out.loc["x", "pct_outliers"])

    def test_unknown_method_rejected(self, clean_df):
        with pytest.raises(ValueError, match="method must be one of"):
            nb.outlier_report(clean_df, method="magic")

    @pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": -1}, {"z_threshold": 0}])
    def test_non_positive_multipliers_rejected(self, clean_df, kwargs):
        with pytest.raises(ValueError, match="must be > 0"):
            nb.outlier_report(clean_df, **kwargs)


class TestOutlierStyle:
    def test_returns_styler(self, messy_df):
        assert is_styler(nb.outlier_style(messy_df, ["x"]))

    def test_html_contains_highlight_color(self, messy_df):
        html = render(nb.outlier_style(messy_df, ["x"], outlier_color="#ff8a80"))
        assert "background-color" in html
        assert "#ff8a80" in html

    def test_highlight_count_matches_report(self, messy_df):
        report = nb.outlier_report(messy_df, ["x"], method="iqr")
        styler = nb.outlier_style(messy_df, ["x"], method="iqr")
        assert count_styled_cells(styler, "#ff8a80") == int(report.loc["x", "n_outliers"])

    def test_styles_multiple_columns(self, messy_df):
        """The whole point of the new helper: not limited to one column."""
        single = count_styled_cells(nb.outlier_style(messy_df, ["x"]), "#ff8a80")
        both = count_styled_cells(nb.outlier_style(messy_df, ["x", "y"]), "#ff8a80")
        assert both >= single
        expected = nb.outlier_report(messy_df, ["x", "y"])["n_outliers"].sum()
        assert both == int(expected)

    def test_clean_column_produces_no_highlights(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert count_styled_cells(nb.outlier_style(df, ["x"]), "#ff8a80") == 0

    def test_all_nan_column_renders(self, empty_col_df):
        assert "<table" in render(nb.outlier_style(empty_col_df, ["x"]))

    def test_non_selected_columns_present_but_unstyled(self, messy_df):
        html = render(nb.outlier_style(messy_df, ["x"]))
        assert "label" in html  # full frame is displayed


# ======================================================================
# normality_report
# ======================================================================

class TestNormalityReport:
    def test_columns_and_kernel_agreement(self, clean_df):
        out = nb.normality_report(clean_df)
        expected = {
            "n", "skewness", "kurtosis", "jb_statistic", "jb_pvalue",
            "ad_statistic", "normal", "conclusion",
        }
        assert set(out.columns) == expected
        jb = bs.jarque_bera(clean_df["x"].to_numpy(dtype=float))
        assert out.loc["x", "jb_statistic"] == pytest.approx(jb["statistic"])
        assert out.loc["x", "jb_pvalue"] == pytest.approx(jb["pvalue"])

    def test_ad_statistic_reported(self, clean_df):
        """Anderson-Darling has no p-value in the kernel; A* must still appear."""
        out = nb.normality_report(clean_df)
        ad = bs.anderson_darling(clean_df["x"].to_numpy(dtype=float))
        assert out.loc["x", "ad_statistic"] == pytest.approx(ad["statistic"])
        assert "ad_pvalue" not in out.columns

    def test_normal_data_not_rejected(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"g": rng.normal(size=3000)})
        out = nb.normality_report(df)
        assert bool(out.loc["g", "normal"]) is True
        assert "cannot reject" in out.loc["g", "conclusion"]

    def test_skewed_data_rejected(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"e": rng.exponential(size=3000)})
        out = nb.normality_report(df)
        assert bool(out.loc["e", "normal"]) is False
        assert "reject normality" in out.loc["e", "conclusion"]

    def test_verdict_respects_alpha(self):
        rng = np.random.default_rng(11)
        df = pd.DataFrame({"g": rng.normal(size=200)})
        strict = nb.normality_report(df, alpha=0.999)
        assert bool(strict.loc["g", "normal"]) is False

    def test_nan_dropped(self, messy_df):
        out = nb.normality_report(messy_df, ["x"])
        assert out.loc["x", "n"] == 49
        assert np.isfinite(out.loc["x", "jb_statistic"])

    def test_all_nan_column_is_inconclusive(self, empty_col_df):
        out = nb.normality_report(empty_col_df, ["x"])
        assert out.loc["x", "n"] == 0
        assert out.loc["x", "normal"] is None
        assert "inconclusive" in out.loc["x", "conclusion"]

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_bad_alpha_rejected(self, clean_df, alpha):
        with pytest.raises(ValueError, match="alpha must be in"):
            nb.normality_report(clean_df, alpha=alpha)


# ======================================================================
# correlation_report / corr_heatmap
# ======================================================================

class TestCorrelationReport:
    def test_matrix_shape_and_diagonal(self, clean_df):
        out = nb.correlation_report(clean_df)
        assert out.shape == (2, 2)
        assert list(out.index) == ["x", "y"]
        np.testing.assert_allclose(np.diag(out.to_numpy()), 1.0)

    def test_matrix_symmetric(self, clean_df):
        out = nb.correlation_report(clean_df).to_numpy()
        np.testing.assert_allclose(out, out.T)

    def test_pearson_matches_pandas(self, clean_df):
        out = nb.correlation_report(clean_df, method="pearson")
        ref = clean_df[["x", "y"]].corr(method="pearson")
        np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), atol=1e-10)

    def test_spearman_matches_pandas(self, clean_df):
        out = nb.correlation_report(clean_df, method="spearman")
        ref = clean_df[["x", "y"]].corr(method="spearman")
        np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), atol=1e-8)

    def test_perfect_correlation_detected(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 4, 6, 8, 10]})
        assert nb.correlation_report(df).loc["a", "b"] == pytest.approx(1.0)

    def test_pvalues_long_form(self, clean_df):
        out = nb.correlation_report(clean_df, pvalues=True)
        assert list(out.columns) == ["x", "y", "n", "correlation", "statistic", "pvalue"]
        assert len(out) == 1  # one unique pair from two columns
        assert 0.0 <= out.loc[0, "pvalue"] <= 1.0

    def test_pvalue_rows_are_unique_pairs(self, clean_df):
        df = clean_df.assign(z=clean_df["x"] * 2)
        out = nb.correlation_report(df, pvalues=True)
        assert len(out) == 3  # C(3, 2)

    def test_pvalues_agree_with_matrix(self, clean_df):
        matrix = nb.correlation_report(clean_df)
        longform = nb.correlation_report(clean_df, pvalues=True)
        assert longform.loc[0, "correlation"] == pytest.approx(matrix.loc["x", "y"])

    def test_pairwise_complete_drops_only_affected_rows(self, messy_df):
        """x has 1 NaN and y has 2 infs, at different positions -> n = 47."""
        out = nb.correlation_report(messy_df, ["x", "y"], pvalues=True)
        assert out.loc[0, "n"] == 47
        assert np.isfinite(out.loc[0, "correlation"])

    def test_too_few_columns_rejected(self, clean_df):
        with pytest.raises(ValueError, match="at least 2 columns"):
            nb.correlation_report(clean_df, ["x"])

    def test_unknown_method_rejected(self, clean_df):
        with pytest.raises(ValueError, match="method must be one of"):
            nb.correlation_report(clean_df, method="kendall")

    def test_insufficient_overlap_gives_nan(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, np.nan], "b": [np.nan, np.nan, 3.0, 4.0]})
        out = nb.correlation_report(df, pvalues=True)
        assert out.loc[0, "n"] == 0
        assert np.isnan(out.loc[0, "correlation"])


class TestCorrHeatmap:
    def test_returns_styler_with_gradient(self, clean_df):
        pytest.importorskip("matplotlib", reason="background_gradient needs matplotlib")
        styler = nb.corr_heatmap(clean_df)
        assert is_styler(styler)
        html = render(styler)
        assert "background-color" in html
        assert "<table" in html

    def test_labels_present_in_html(self, clean_df):
        pytest.importorskip("matplotlib")
        html = render(nb.corr_heatmap(clean_df))
        assert "x" in html and "y" in html

    def test_spearman_supported(self, clean_df):
        pytest.importorskip("matplotlib")
        assert is_styler(nb.corr_heatmap(clean_df, method="spearman"))


# ======================================================================
# missingness_report
# ======================================================================

class TestMissingnessReport:
    def test_covers_every_column_including_non_numeric(self, messy_df):
        out = nb.missingness_report(messy_df)
        assert list(out.index) == ["x", "y", "label"]

    def test_counts_nan_and_infinities_separately(self, messy_df):
        out = nb.missingness_report(messy_df)
        # x: one NaN, no infs
        assert out.loc["x", "n_missing"] == 1
        assert out.loc["x", "n_infinite"] == 0
        assert out.loc["x", "n_finite"] == 49
        # y: no NaN, but +inf and -inf
        assert out.loc["y", "n_missing"] == 0
        assert out.loc["y", "n_infinite"] == 2
        assert out.loc["y", "n_finite"] == 48

    def test_non_numeric_finite_counts_are_nan(self, messy_df):
        out = nb.missingness_report(messy_df)
        assert np.isnan(out.loc["label", "n_finite"])
        assert np.isnan(out.loc["label", "n_infinite"])

    def test_percentage(self, messy_df):
        out = nb.missingness_report(messy_df)
        assert out.loc["x", "pct_missing"] == pytest.approx(2.0)

    def test_dtype_recorded(self, messy_df):
        assert "float" in out_dtype(nb.missingness_report(messy_df), "x")

    def test_clean_frame_reports_zero(self, clean_df):
        out = nb.missingness_report(clean_df)
        assert (out["n_missing"] == 0).all()

    def test_empty_frame(self):
        out = nb.missingness_report(pd.DataFrame({"x": pd.Series([], dtype=float)}))
        assert out.loc["x", "n_rows"] == 0
        assert np.isnan(out.loc["x", "pct_missing"])

    def test_non_dataframe_rejected(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            nb.missingness_report([1, 2, 3])


def out_dtype(report, column):
    return str(report.loc[column, "dtype"])


# ======================================================================
# rolling_report
# ======================================================================

class TestRollingReport:
    def test_shape_index_and_names(self, clean_df):
        out = nb.rolling_report(clean_df, "x", 5)
        assert len(out) == len(clean_df)
        pd.testing.assert_index_equal(out.index, clean_df.index)
        assert list(out.columns) == [
            "x_roll5_mean", "x_roll5_std", "x_roll5_min", "x_roll5_max",
        ]

    def test_trailing_windows_left_padded(self, clean_df):
        """First window-1 rows have no complete window -> NaN, index aligned."""
        out = nb.rolling_report(clean_df, "x", 5, stats=("mean",))
        assert out["x_roll5_mean"].iloc[:4].isna().all()
        assert np.isfinite(out["x_roll5_mean"].iloc[4])

    def test_values_match_pandas_rolling(self, clean_df):
        out = nb.rolling_report(clean_df, "x", 5, stats=("mean",))
        ref = clean_df["x"].rolling(5).mean()
        np.testing.assert_allclose(
            out["x_roll5_mean"].to_numpy(), ref.to_numpy(), atol=1e-10
        )

    def test_custom_stats_subset(self, clean_df):
        out = nb.rolling_report(clean_df, "x", 3, stats=("min", "max"))
        assert list(out.columns) == ["x_roll3_min", "x_roll3_max"]
        assert (out["x_roll3_min"].dropna() <= out["x_roll3_max"].dropna()).all()

    def test_centered_alignment_full_length(self, clean_df):
        out = nb.rolling_report(clean_df, "x", 5, stats=("mean",), alignment="centered")
        assert len(out) == len(clean_df)

    def test_nan_propagates_by_default(self, messy_df):
        """Documented behaviour: a NaN poisons every window containing it."""
        out = nb.rolling_report(messy_df, "x", 3, stats=("mean",))
        assert out["x_roll3_mean"].iloc[2:5].isna().all()

    def test_min_periods_enables_nan_skipping(self, messy_df):
        """nan_policy='ignore' only bites when min_periods < window."""
        out = nb.rolling_report(
            messy_df, "x", 3, stats=("mean",), min_periods=2, nan_policy="ignore"
        )
        assert out["x_roll3_mean"].iloc[2:5].notna().any()

    def test_preserves_custom_index(self, clean_df):
        df = clean_df.set_index(pd.date_range("2024-01-01", periods=len(clean_df)))
        out = nb.rolling_report(df, "x", 5, stats=("mean",))
        pd.testing.assert_index_equal(out.index, df.index)

    def test_missing_column_raises(self, clean_df):
        with pytest.raises(KeyError, match="nope"):
            nb.rolling_report(clean_df, "nope", 5)

    def test_non_numeric_column_raises(self, clean_df):
        with pytest.raises(TypeError, match="not numeric"):
            nb.rolling_report(clean_df, "label", 5)

    @pytest.mark.parametrize("window", [0, -3])
    def test_bad_window_rejected(self, clean_df, window):
        with pytest.raises(ValueError, match="window must be >= 1"):
            nb.rolling_report(clean_df, "x", window)

    def test_window_larger_than_frame_rejected(self, clean_df):
        with pytest.raises(ValueError, match="exceeds the number of rows"):
            nb.rolling_report(clean_df, "x", len(clean_df) + 1)

    def test_non_int_window_rejected(self, clean_df):
        with pytest.raises(TypeError, match="window must be an int"):
            nb.rolling_report(clean_df, "x", 5.0)

    def test_unknown_stat_rejected(self, clean_df):
        with pytest.raises(ValueError, match="Unsupported stat"):
            nb.rolling_report(clean_df, "x", 5, stats=("median",))

    def test_empty_stats_rejected(self, clean_df):
        with pytest.raises(ValueError, match="empty"):
            nb.rolling_report(clean_df, "x", 5, stats=())


# ======================================================================
# bootstrap_ci_report
# ======================================================================

class TestBootstrapCIReport:
    def test_shape_and_columns(self, clean_df):
        out = nb.bootstrap_ci_report(clean_df, n_resamples=200, random_state=0)
        assert list(out.index) == ["x", "y"]
        assert list(out.columns) == [
            "stat", "n", "n_missing", "estimate", "ci_lower", "ci_upper", "conf",
        ]

    def test_interval_brackets_estimate(self, clean_df):
        out = nb.bootstrap_ci_report(clean_df, n_resamples=500, random_state=1)
        for col in ("x", "y"):
            assert out.loc[col, "ci_lower"] <= out.loc[col, "estimate"]
            assert out.loc[col, "estimate"] <= out.loc[col, "ci_upper"]

    def test_interval_covers_true_mean(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"v": rng.normal(loc=10.0, scale=1.0, size=500)})
        out = nb.bootstrap_ci_report(df, n_resamples=1000, random_state=3)
        assert out.loc["v", "ci_lower"] < 10.0 < out.loc["v", "ci_upper"]

    def test_reproducible_with_random_state(self, clean_df):
        a = nb.bootstrap_ci_report(clean_df, n_resamples=200, random_state=42)
        b = nb.bootstrap_ci_report(clean_df, n_resamples=200, random_state=42)
        pd.testing.assert_frame_equal(a, b)

    @pytest.mark.parametrize("stat", ["mean", "median", "std"])
    def test_supported_stats(self, clean_df, stat):
        out = nb.bootstrap_ci_report(
            clean_df, ["x"], stat=stat, n_resamples=200, random_state=0
        )
        assert out.loc["x", "stat"] == stat
        assert np.isfinite(out.loc["x", "estimate"])

    def test_wider_conf_gives_wider_interval(self, clean_df):
        narrow = nb.bootstrap_ci_report(clean_df, ["x"], conf=0.80, n_resamples=1000, random_state=5)
        wide = nb.bootstrap_ci_report(clean_df, ["x"], conf=0.99, n_resamples=1000, random_state=5)
        narrow_width = narrow.loc["x", "ci_upper"] - narrow.loc["x", "ci_lower"]
        wide_width = wide.loc["x", "ci_upper"] - wide.loc["x", "ci_lower"]
        assert wide_width > narrow_width

    def test_nan_and_inf_dropped(self, messy_df):
        out = nb.bootstrap_ci_report(messy_df, n_resamples=200, random_state=0)
        assert out.loc["x", "n"] == 49 and out.loc["x", "n_missing"] == 1
        assert out.loc["y", "n"] == 48 and out.loc["y", "n_missing"] == 2
        assert np.isfinite(out.loc["y", "estimate"])

    def test_all_nan_column_yields_nan(self, empty_col_df):
        out = nb.bootstrap_ci_report(empty_col_df, ["x"], n_resamples=100, random_state=0)
        assert out.loc["x", "n"] == 0
        assert np.isnan(out.loc["x", "estimate"])

    def test_single_finite_value_yields_nan(self):
        df = pd.DataFrame({"x": [1.0, np.nan, np.nan]})
        out = nb.bootstrap_ci_report(df, n_resamples=100, random_state=0)
        assert out.loc["x", "n"] == 1
        assert np.isnan(out.loc["x", "ci_lower"])

    def test_bad_stat_rejected(self, clean_df):
        with pytest.raises((ValueError, KeyError)):
            nb.bootstrap_ci_report(clean_df, stat="mode", n_resamples=50)


# ======================================================================
# scale_columns / winsorize_columns
# ======================================================================

class TestScaleColumns:
    def test_adds_suffixed_columns_and_preserves_originals(self, clean_df):
        out = nb.scale_columns(clean_df, ["x"], method="robust")
        assert "x_robust" in out.columns
        assert "x" in out.columns
        pd.testing.assert_series_equal(out["x"], clean_df["x"])

    def test_does_not_mutate_input(self, clean_df):
        before = clean_df.copy()
        nb.scale_columns(clean_df)
        pd.testing.assert_frame_equal(clean_df, before)

    def test_all_numeric_columns_by_default(self, clean_df):
        out = nb.scale_columns(clean_df)
        assert "x_robust" in out.columns and "y_robust" in out.columns
        assert "label_robust" not in out.columns

    def test_zscore_is_standardised(self, clean_df):
        out = nb.scale_columns(clean_df, ["x"], method="zscore")
        assert out["x_zscore"].mean() == pytest.approx(0.0, abs=1e-10)
        assert out["x_zscore"].std(ddof=1) == pytest.approx(1.0, rel=1e-8)

    def test_zscore_matches_kernel(self, clean_df):
        out = nb.scale_columns(clean_df, ["x"], method="zscore")
        expected = np.asarray(bs.zscore(clean_df["x"].to_numpy(dtype=float)))
        np.testing.assert_allclose(out["x_zscore"].to_numpy(), expected)

    def test_minmax_spans_unit_interval(self, clean_df):
        out = nb.scale_columns(clean_df, ["x"], method="minmax")
        assert out["x_minmax"].min() == pytest.approx(0.0)
        assert out["x_minmax"].max() == pytest.approx(1.0)

    def test_robust_centres_median_at_zero(self, clean_df):
        out = nb.scale_columns(clean_df, ["x"], method="robust")
        assert out["x_robust"].median() == pytest.approx(0.0, abs=1e-10)

    def test_nan_positions_preserved_exactly(self, messy_df):
        """The scatter-back contract: no row shifting, NaN in == NaN out."""
        out = nb.scale_columns(messy_df, ["x"], method="zscore")
        assert len(out) == len(messy_df)
        source_bad = ~np.isfinite(messy_df["x"].to_numpy(dtype=float))
        result_bad = out["x_zscore"].isna().to_numpy()
        np.testing.assert_array_equal(source_bad, result_bad)

    def test_scaling_fit_on_finite_values_only(self, messy_df):
        """An inf in the column must not poison every scaled value."""
        out = nb.scale_columns(messy_df, ["y"], method="zscore")
        assert out["y_zscore"].notna().sum() == 48

    def test_custom_suffix(self, clean_df):
        out = nb.scale_columns(clean_df, ["x"], method="zscore", suffix="_std")
        assert "x_std" in out.columns
        assert "x_zscore" not in out.columns

    def test_replace_overwrites_in_place(self, clean_df):
        out = nb.scale_columns(clean_df, ["x"], method="zscore", replace=True)
        assert "x_zscore" not in out.columns
        assert out["x"].mean() == pytest.approx(0.0, abs=1e-10)
        assert list(out.columns) == list(clean_df.columns)

    def test_all_nan_column_stays_nan(self, empty_col_df):
        out = nb.scale_columns(empty_col_df, ["x"], method="zscore")
        assert out["x_zscore"].isna().all()

    def test_unknown_method_rejected(self, clean_df):
        with pytest.raises(ValueError, match="method must be one of"):
            nb.scale_columns(clean_df, method="quantile")

    def test_empty_suffix_rejected(self, clean_df):
        with pytest.raises(ValueError, match="suffix"):
            nb.scale_columns(clean_df, ["x"], suffix="")

    @pytest.mark.parametrize("factor", [0.0, -1.0, np.nan])
    def test_bad_scale_factor_rejected(self, clean_df, factor):
        with pytest.raises(ValueError, match="scale_factor"):
            nb.scale_columns(clean_df, ["x"], method="robust", scale_factor=factor)


class TestWinsorizeColumns:
    def test_clips_tails(self, clean_df):
        out = nb.winsorize_columns(clean_df, ["x"], lower_q=0.1, upper_q=0.9)
        assert out["x_winsor"].min() > clean_df["x"].min()
        assert out["x_winsor"].max() < clean_df["x"].max()

    def test_matches_kernel(self, clean_df):
        out = nb.winsorize_columns(clean_df, ["x"], lower_q=0.05, upper_q=0.95)
        expected = np.asarray(
            bs.winsorize(clean_df["x"].to_numpy(dtype=float), lower_q=0.05, upper_q=0.95)
        )
        np.testing.assert_allclose(out["x_winsor"].to_numpy(), expected)

    def test_extreme_outlier_pulled_in(self, messy_df):
        out = nb.winsorize_columns(messy_df, ["x"])
        assert out.loc[4, "x"] == 50.0            # original untouched
        assert out.loc[4, "x_winsor"] < 50.0      # winsorized value clipped

    def test_length_and_nan_positions_preserved(self, messy_df):
        out = nb.winsorize_columns(messy_df, ["x"])
        assert len(out) == len(messy_df)
        assert bool(np.isnan(out.loc[2, "x_winsor"]))

    def test_does_not_mutate_input(self, messy_df):
        before = messy_df.copy()
        nb.winsorize_columns(messy_df)
        pd.testing.assert_frame_equal(messy_df, before)

    def test_replace_mode(self, clean_df):
        out = nb.winsorize_columns(clean_df, ["x"], replace=True)
        assert "x_winsor" not in out.columns
        assert out["x"].max() < clean_df["x"].max()

    @pytest.mark.parametrize("bounds", [(-0.1, 0.9), (0.1, 1.5)])
    def test_out_of_range_quantiles_rejected(self, clean_df, bounds):
        lo, hi = bounds
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            nb.winsorize_columns(clean_df, lower_q=lo, upper_q=hi)

    def test_inverted_quantiles_rejected(self, clean_df):
        with pytest.raises(ValueError, match="must be <"):
            nb.winsorize_columns(clean_df, lower_q=0.9, upper_q=0.1)

    def test_all_nan_column(self, empty_col_df):
        out = nb.winsorize_columns(empty_col_df, ["x"])
        assert out["x_winsor"].isna().all()


class TestRobustScaleColumn:
    def test_backwards_compatible_behaviour(self, clean_df):
        out = nb.robust_scale_column(clean_df, "x")
        assert isinstance(out, pd.DataFrame)
        assert "x_robust" in out.columns
        assert out["x_robust"].median() == pytest.approx(0.0, abs=1e-10)

    def test_custom_suffix(self, clean_df):
        out = nb.robust_scale_column(clean_df, "x", add_suffix="_rs")
        assert "x_rs" in out.columns

    def test_missing_column_raises(self, clean_df):
        with pytest.raises(KeyError):
            nb.robust_scale_column(clean_df, "nope")


# ======================================================================
# style_significance / style_effect_size
# ======================================================================

@pytest.fixture
def results_df():
    return pd.DataFrame(
        {
            "test": ["a", "b", "c", "d", "e"],
            "pvalue": [0.0001, 0.006, 0.03, 0.40, np.nan],
            "effect": [1.2, -0.6, 0.3, 0.05, np.nan],
        }
    )


class TestStyleSignificance:
    def test_returns_styler(self, results_df):
        assert is_styler(nb.style_significance(results_df))

    def test_tiers_produce_distinct_colors(self, results_df):
        html = render(nb.style_significance(results_df))
        for color in ("#66bb6a", "#a5d6a7", "#e8f5e9", "#f5f5f5"):
            assert color in html

    def test_nan_pvalue_row_unstyled(self, results_df):
        """4 styled rows x 3 cells = 12 background-color declarations."""
        styler = nb.style_significance(results_df)
        assert count_styled_cells(styler, "background-color") == 12

    def test_cell_only_mode(self, results_df):
        styler = nb.style_significance(results_df, highlight_row=False)
        assert count_styled_cells(styler, "background-color") == 4

    def test_alpha_changes_classification(self, results_df):
        strict = render(nb.style_significance(results_df, alpha=0.001))
        assert "#f5f5f5" in strict  # 0.03 and 0.40 now non-significant

    def test_custom_pvalue_column(self, results_df):
        renamed = results_df.rename(columns={"pvalue": "p_adj"})
        assert is_styler(nb.style_significance(renamed, pvalue_column="p_adj"))

    def test_missing_pvalue_column_raises(self, results_df):
        with pytest.raises(KeyError, match="p-value column"):
            nb.style_significance(results_df, pvalue_column="nope")

    def test_non_numeric_pvalue_column_raises(self, results_df):
        with pytest.raises(TypeError, match="not numeric"):
            nb.style_significance(results_df, pvalue_column="test")

    def test_bad_alpha_raises(self, results_df):
        with pytest.raises(ValueError, match="alpha must be in"):
            nb.style_significance(results_df, alpha=0)

    def test_integrates_with_correlation_report(self, clean_df):
        report = nb.correlation_report(clean_df, pvalues=True)
        assert is_styler(nb.style_significance(report))


class TestStyleEffectSize:
    def test_returns_styler(self, results_df):
        assert is_styler(nb.style_effect_size(results_df, "effect"))

    def test_magnitude_buckets_are_symmetric(self):
        """-0.6 and +0.6 are the same magnitude -> same color."""
        df = pd.DataFrame({"d": [0.6, -0.6]})
        assert count_styled_cells(nb.style_effect_size(df, "d"), "#ffe082") == 2

    def test_all_four_buckets_reachable(self, results_df):
        html = render(nb.style_effect_size(results_df, "effect"))
        for color in ("#f5f5f5", "#fff9c4", "#ffe082", "#ffab91"):
            assert color in html

    def test_nan_effect_unstyled(self, results_df):
        styler = nb.style_effect_size(results_df, "effect")
        assert count_styled_cells(styler, "background-color") == 4  # 5 rows, 1 NaN

    def test_row_highlight_mode(self, results_df):
        styler = nb.style_effect_size(results_df, "effect", highlight_row=True)
        assert count_styled_cells(styler, "background-color") == 12

    def test_custom_thresholds(self, results_df):
        styler = nb.style_effect_size(results_df, "effect", thresholds=(0.1, 0.3, 0.5))
        assert is_styler(styler)

    def test_missing_column_raises(self, results_df):
        with pytest.raises(KeyError, match="Effect column"):
            nb.style_effect_size(results_df, "nope")

    def test_non_numeric_column_raises(self, results_df):
        with pytest.raises(TypeError, match="not numeric"):
            nb.style_effect_size(results_df, "test")

    def test_wrong_threshold_count_raises(self, results_df):
        with pytest.raises(ValueError, match="exactly 3 values"):
            nb.style_effect_size(results_df, "effect", thresholds=(0.2, 0.5))

    def test_unsorted_thresholds_raise(self, results_df):
        with pytest.raises(ValueError, match="strictly ascending"):
            nb.style_effect_size(results_df, "effect", thresholds=(0.8, 0.5, 0.2))


# ======================================================================
# Legacy single-column stylers
# ======================================================================

class TestLegacyStylers:
    def test_demean_style_adds_column_and_colors(self, clean_df):
        styler = nb.demean_style(clean_df, "x")
        assert is_styler(styler)
        html = render(styler)
        assert "x_demeaned" in html
        assert "#c8e6c9" in html  # above mean
        assert "#ffcdd2" in html  # below mean

    def test_demean_values_match_kernel(self, clean_df):
        expected = np.asarray(bs.demean_with_signs(clean_df["x"].to_numpy(dtype=float))[0])
        out = nb.demean_style(clean_df, "x").data
        np.testing.assert_allclose(out["x_demeaned"].to_numpy(), expected)

    def test_demean_style_handles_nan(self, messy_df):
        assert "<table" in render(nb.demean_style(messy_df, "x"))

    def test_zscore_style_highlights_extremes(self, messy_df):
        html = render(nb.zscore_style(messy_df, "x", threshold=2.0))
        assert "x_zscore" in html
        assert "#ffcc80" in html  # the planted +50 outlier

    def test_zscore_style_no_extremes_when_threshold_huge(self, clean_df):
        styler = nb.zscore_style(clean_df, "x", threshold=100.0)
        assert count_styled_cells(styler, "#ffcc80") == 0
        assert count_styled_cells(styler, "#bbdefb") == 0

    def test_zscore_style_bad_threshold(self, clean_df):
        with pytest.raises(ValueError, match="threshold must be > 0"):
            nb.zscore_style(clean_df, "x", threshold=0)

    def test_iqr_outlier_style_matches_report(self, messy_df):
        report = nb.outlier_report(messy_df, ["x"], method="iqr", k=1.5)
        styler = nb.iqr_outlier_style(messy_df, "x", k=1.5)
        assert count_styled_cells(styler, "#ff8a80") == int(report.loc["x", "n_outliers"])

    @pytest.mark.parametrize(
        "fn", [nb.demean_style, nb.zscore_style, nb.iqr_outlier_style],
        ids=lambda f: f.__name__,
    )
    def test_legacy_stylers_validate_column(self, fn, clean_df):
        with pytest.raises(KeyError):
            fn(clean_df, "nope")
        with pytest.raises(TypeError, match="not numeric"):
            fn(clean_df, "label")


# ======================================================================
# Degenerate / pathological data
# ======================================================================

class TestDegenerateData:
    """Zero-variance and single-row frames must degrade, never crash.

    Constant columns are extremely common in real data (a flag that never
    flips, a rate that never moved), and they make every scale estimate zero.
    These tests pin the resulting behaviour so a kernel change cannot silently
    start emitting outliers or exceptions here.
    """

    @pytest.fixture
    def constant_df(self):
        return pd.DataFrame({"k": [5.0] * 10})

    def test_all_scale_estimates_are_zero(self, constant_df):
        out = nb.robust_summary(constant_df)
        for stat in ("std", "mad", "iqr", "qn_scale"):
            assert out.loc["k", stat] == pytest.approx(0.0)
        assert out.loc["k", "median"] == pytest.approx(5.0)

    @pytest.mark.parametrize("method", ["iqr", "zscore", "robust_zscore"])
    def test_no_spurious_outliers(self, constant_df, method):
        """Zero spread must not make every point an outlier."""
        out = nb.outlier_report(constant_df, method=method)
        assert out.loc["k", "n_outliers"] == 0

    def test_outlier_style_renders(self, constant_df):
        assert count_styled_cells(nb.outlier_style(constant_df), "#ff8a80") == 0

    @pytest.mark.parametrize(
        "method,expected",
        [("minmax", 0.0), ("robust", 0.0), ("zscore", np.nan)],
    )
    def test_scaling_degrades_predictably(self, constant_df, method, expected):
        """Zero spread -> 0 for minmax/robust, NaN for zscore (0/0)."""
        col = nb.scale_columns(constant_df, method=method)[f"k_{method}"]
        if np.isnan(expected):
            assert col.isna().all()
        else:
            assert np.allclose(col.to_numpy(), expected)

    def test_winsorize_is_identity(self, constant_df):
        out = nb.winsorize_columns(constant_df)
        assert (out["k_winsor"] == 5.0).all()

    def test_normality_is_inconclusive_not_an_error(self, constant_df):
        out = nb.normality_report(constant_df)
        assert out.loc["k", "normal"] is None

    def test_bootstrap_collapses_to_the_constant(self, constant_df):
        out = nb.bootstrap_ci_report(constant_df, n_resamples=100, random_state=0)
        assert out.loc["k", "estimate"] == pytest.approx(5.0)
        assert out.loc["k", "ci_lower"] == pytest.approx(5.0)
        assert out.loc["k", "ci_upper"] == pytest.approx(5.0)

    def test_correlation_with_a_constant_column(self, constant_df):
        """Correlation with zero-variance data is undefined, not an exception."""
        df = constant_df.assign(v=np.arange(10.0))
        out = nb.correlation_report(df, pvalues=True)
        assert len(out) == 1
        assert np.isnan(out.loc[0, "correlation"]) or out.loc[0, "correlation"] == 0.0

    def test_single_row_frame(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        summary = nb.robust_summary(df)
        assert summary.loc["a", "n"] == 1
        assert summary.loc["a", "mean"] == pytest.approx(1.0)
        assert np.isnan(summary.loc["a", "skew"])
        assert nb.outlier_report(df).loc["a", "n_outliers"] == 0

    def test_single_row_rolling_window_of_one(self):
        df = pd.DataFrame({"a": [1.0]})
        out = nb.rolling_report(df, "a", 1, stats=("mean",))
        assert out["a_roll1_mean"].iloc[0] == pytest.approx(1.0)


# ======================================================================
# Public surface / import smoke tests
# ======================================================================

class TestPublicSurface:
    def test_every_all_name_is_importable_and_callable(self):
        for name in nb.__all__:
            attr = getattr(nb, name)
            assert callable(attr), f"{name} is not callable"

    def test_all_has_no_duplicates(self):
        assert len(nb.__all__) == len(set(nb.__all__))

    def test_no_private_names_exported(self):
        assert not [n for n in nb.__all__ if n.startswith("_")]

    def test_naming_convention_holds(self):
        """*_report -> DataFrame, *_style/style_* -> Styler, *_columns -> DataFrame."""
        reports = [n for n in nb.__all__ if n.endswith("_report")]
        stylers = [n for n in nb.__all__ if n.endswith("_style") or n.startswith("style_")]
        transforms = [n for n in nb.__all__ if n.endswith("_columns")]
        assert set(reports) == {
            "outlier_report", "normality_report", "correlation_report",
            "missingness_report", "rolling_report", "bootstrap_ci_report",
        }
        assert set(stylers) == {
            "outlier_style", "style_significance", "style_effect_size",
            "demean_style", "zscore_style", "iqr_outlier_style",
        }
        assert set(transforms) == {"scale_columns", "winsorize_columns"}

    def test_report_helpers_return_dataframes(self, clean_df):
        assert isinstance(nb.outlier_report(clean_df), pd.DataFrame)
        assert isinstance(nb.normality_report(clean_df), pd.DataFrame)
        assert isinstance(nb.correlation_report(clean_df), pd.DataFrame)
        assert isinstance(nb.missingness_report(clean_df), pd.DataFrame)
        assert isinstance(nb.rolling_report(clean_df, "x", 5), pd.DataFrame)
        assert isinstance(nb.robust_summary(clean_df), pd.DataFrame)
        assert isinstance(nb.describe_fast(clean_df), pd.DataFrame)

    def test_column_helpers_return_dataframes(self, clean_df):
        assert isinstance(nb.scale_columns(clean_df), pd.DataFrame)
        assert isinstance(nb.winsorize_columns(clean_df), pd.DataFrame)

    def test_style_helpers_return_stylers(self, clean_df, results_df):
        assert is_styler(nb.outlier_style(clean_df))
        assert is_styler(nb.demean_style(clean_df, "x"))
        assert is_styler(nb.zscore_style(clean_df, "x"))
        assert is_styler(nb.iqr_outlier_style(clean_df, "x"))
        assert is_styler(nb.style_significance(results_df))
        assert is_styler(nb.style_effect_size(results_df, "effect"))

    def test_notebook_discoverable_but_not_in_all(self):
        """`__all__` is a list of callables (see test_hardening_v030.py).

        The submodule must stay out of it -- and out of `import *` -- while
        remaining reachable by attribute and visible to tab-completion.
        """
        assert "notebook" not in bs.__all__
        assert all(callable(getattr(bs, n, None)) for n in bs.__all__)
        assert "notebook" in dir(bs)
        assert bs.notebook is nb

    def test_lazy_submodules_all_resolve(self):
        for name in ("notebook", "pandas", "pandas_helpers"):
            assert getattr(bs, name) is importlib.import_module(f"bunker_stats.{name}")

    def test_core_api_unaffected(self):
        """Adding the layer must not disturb the numpy-only core."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert bs.mean(x) == pytest.approx(2.5)
        assert bs.median(x) == pytest.approx(2.5)
        assert "mean" in bs.__all__ and "corr_matrix" in bs.__all__


class TestBackwardsCompatibility:
    def test_pandas_helpers_shim_reexports_originals(self):
        from bunker_stats import pandas_helpers as ph

        for name in ("demean_style", "zscore_style", "iqr_outlier_style",
                     "corr_heatmap", "robust_scale_column"):
            assert hasattr(ph, name)
            assert getattr(ph, name) is getattr(nb, name)

    def test_pandas_module_still_has_labeled_matrices(self, clean_df):
        import bunker_stats.pandas as bsp

        cov = bsp.cov_df(clean_df)
        corr = bsp.corr_df(clean_df)
        assert list(cov.index) == ["x", "y"]
        assert list(corr.columns) == ["x", "y"]
        np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)

    def test_corr_df_matches_pandas(self, clean_df):
        import bunker_stats.pandas as bsp

        ref = clean_df[["x", "y"]].corr()
        np.testing.assert_allclose(bsp.corr_df(clean_df).to_numpy(), ref.to_numpy(), atol=1e-10)

    def test_pandas_module_reexports_notebook_surface(self):
        import bunker_stats.pandas as bsp

        for name in nb.__all__:
            assert hasattr(bsp, name), f"bunker_stats.pandas is missing {name}"

    def test_old_positional_helper_calls_still_work(self, clean_df):
        """Original signatures: fn(df, column) and fn(df, columns)."""
        assert is_styler(nb.demean_style(clean_df, "x"))
        assert is_styler(nb.zscore_style(clean_df, "x"))
        assert is_styler(nb.iqr_outlier_style(clean_df, "x"))
        assert isinstance(nb.robust_scale_column(clean_df, "x"), pd.DataFrame)
        pytest.importorskip("matplotlib")
        assert is_styler(nb.corr_heatmap(clean_df, ["x", "y"]))
