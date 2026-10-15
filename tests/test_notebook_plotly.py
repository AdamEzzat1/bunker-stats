"""Tests for the Plotly visualization layer and rich notebook report objects.

Covers:
* ``*_report(..., rich=True)`` report objects: type, ``.to_frame``/``.to_dict``
  schema stability, ``.style``, ``.info``, Jupyter repr.
* Backward compatibility: default report returns stay plain DataFrames.
* Every ``.plot_*`` method returns a ``plotly.graph_objects.Figure`` that
  survives ``to_json()`` and never mutates the underlying report data.
* Optional-dependency behavior: a missing plotly produces the documented
  install hint, not an obscure ImportError.

pandas-dependent tests skip cleanly without pandas; plotly-dependent tests
skip cleanly without plotly.
"""
from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="notebook layer requires pandas")

import bunker_stats as bs
from bunker_stats import notebook as nb
from bunker_stats.resampling.config import BootstrapResult
from bunker_stats.tsa.types import ZivotAndrewsResult

go = pytest.importorskip(
    "plotly.graph_objects", reason="plotly layer requires plotly"
)


@pytest.fixture
def df():
    rng = np.random.default_rng(20260720)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=60),
            "b": rng.normal(2.0, 3.0, 60),
            "c": rng.normal(-1.0, 0.5, 60),
        }
    )
    frame.loc[3, "a"] = np.nan
    frame.loc[5, "a"] = 25.0  # planted outlier
    return frame


def assert_figure(fig):
    assert isinstance(fig, go.Figure)
    payload = fig.to_json()
    assert payload.startswith("{")
    return payload


# ======================================================================
# Report objects
# ======================================================================

class TestReportObjects:
    @pytest.mark.parametrize(
        "call,cls_name",
        [
            (lambda d: nb.robust_summary(d, rich=True), "SummaryReport"),
            (lambda d: nb.outlier_report(d, rich=True), "OutlierReport"),
            (lambda d: nb.correlation_report(d, rich=True), "CorrelationReport"),
            (
                lambda d: nb.bootstrap_ci_report(
                    d, n_resamples=200, random_state=0, rich=True
                ),
                "BootstrapCIReport",
            ),
        ],
        ids=["summary", "outlier", "correlation", "bootstrap"],
    )
    def test_rich_report_protocol(self, df, call, cls_name):
        report = call(df)
        assert type(report).__name__ == cls_name
        # to_frame returns a copy — mutating a numeric cell must not leak back
        frame = report.to_frame()
        assert isinstance(frame, pd.DataFrame)
        numeric_col = frame.select_dtypes(include="number").columns[0]
        frame.loc[frame.index[0], numeric_col] = -999
        assert not report.data.equals(frame)
        # stable to_dict schema (warnings always present, possibly empty)
        payload = report.to_dict()
        assert set(payload) == {"title", "data", "meta", "warnings"}
        assert isinstance(payload["meta"], dict)
        assert isinstance(payload["warnings"], list)
        # style + info + notebook repr
        assert report.style() is not None
        assert isinstance(report.info(), str)
        assert "Rows:" in report.info()
        assert "<table" in report._repr_html_()

    @pytest.mark.parametrize(
        "call",
        [
            lambda d: nb.robust_summary(d),
            lambda d: nb.outlier_report(d),
            lambda d: nb.correlation_report(d),
            lambda d: nb.bootstrap_ci_report(d, n_resamples=100, random_state=0),
        ],
        ids=["summary", "outlier", "correlation", "bootstrap"],
    )
    def test_default_returns_remain_dataframes(self, df, call):
        assert isinstance(call(df), pd.DataFrame)

    def test_rich_data_matches_default(self, df):
        default = nb.outlier_report(df)
        rich = nb.outlier_report(df, rich=True)
        pd.testing.assert_frame_equal(default, rich.data)

    def test_meta_records_method_context(self, df):
        rep = nb.bootstrap_ci_report(
            df, stat="median", n_resamples=300, conf=0.9, random_state=7, rich=True
        )
        assert rep.meta["stat"] == "median"
        assert rep.meta["n_resamples"] == 300
        assert rep.meta["conf"] == 0.9
        assert rep.meta["random_state"] == 7
        assert "nan_policy" in rep.meta

    def test_info_mentions_meta(self, df):
        rep = nb.outlier_report(df, method="robust_zscore", rich=True)
        assert "robust_zscore" in rep.info()


# ======================================================================
# Plotly figure methods
# ======================================================================

class TestPlotlyFigures:
    def test_outlier_plot_counts(self, df):
        rep = nb.outlier_report(df, rich=True)
        payload = assert_figure(rep.plot_counts())
        assert "n_outliers" in payload or "Outliers" in payload

    def test_outlier_hover_includes_method_and_pct(self, df):
        fig = nb.outlier_report(df, rich=True).plot_counts()
        hover = " ".join(fig.data[0].hovertext)
        assert "iqr" in hover and "%" in hover

    def test_corr_plot_heatmap(self, df):
        rep = nb.correlation_report(df, rich=True)
        assert_figure(rep.plot_heatmap())

    def test_corr_heatmap_hover_pairs(self, df):
        fig = nb.correlation_report(df, rich=True).plot_heatmap()
        hover = fig.data[0].hovertext
        assert "a vs b" in hover[0][1]

    def test_corr_heatmap_handles_nan(self):
        frame = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan, np.nan], "b": [np.nan, np.nan, 3.0, 4.0]}
        )
        rep = nb.correlation_report(frame, rich=True)
        fig = rep.plot_heatmap()
        hover = fig.data[0].hovertext
        assert "insufficient overlap" in hover[0][1]
        assert_figure(fig)

    def test_corr_heatmap_rejects_long_form(self, df):
        rep = nb.correlation_report(df, pvalues=True, rich=True)
        with pytest.raises(ValueError, match="matrix form"):
            rep.plot_heatmap()

    def test_bootstrap_plot_intervals(self, df):
        rep = nb.bootstrap_ci_report(df, n_resamples=200, random_state=0, rich=True)
        fig = rep.plot_intervals()
        assert_figure(fig)
        trace = fig.data[0]
        assert trace.error_y.array is not None

    def test_figures_do_not_mutate_reports(self, df):
        for rep, plot in [
            (nb.outlier_report(df, rich=True), "plot_counts"),
            (nb.correlation_report(df, rich=True), "plot_heatmap"),
            (
                nb.bootstrap_ci_report(df, n_resamples=100, random_state=0, rich=True),
                "plot_intervals",
            ),
        ]:
            before = rep.data.copy()
            getattr(rep, plot)()
            pd.testing.assert_frame_equal(rep.data, before)


# ======================================================================
# Result-object plot methods
# ======================================================================

class TestResultObjectPlots:
    def test_rolling_result_plot_traces(self):
        rng = np.random.default_rng(1)
        res = bs.Rolling(rng.normal(size=200), window=20).result("mean", "std", "min")
        fig = res.plot()
        assert_figure(fig)
        assert [t.name for t in fig.data] == ["mean", "std", "min"]

    def test_rolling_result_plot_rejects_2d(self):
        rng = np.random.default_rng(2)
        res = bs.Rolling(rng.normal(size=(100, 3)), window=10).result("mean")
        with pytest.raises(ValueError, match="1-D"):
            res.plot()

    def test_bootstrap_result_plot_distribution_with_draws(self):
        rng = np.random.default_rng(3)
        result = BootstrapResult(
            estimate=0.1,
            ci_lower=-0.2,
            ci_upper=0.4,
            draws=rng.normal(0.1, 0.15, 800),
            method="percentile",
            n_resamples=800,
            confidence_level=0.95,
        )
        assert_figure(result.plot_distribution())

    def test_bootstrap_result_plot_distribution_without_draws(self):
        result = BootstrapResult(estimate=0.1, ci_lower=-0.2, ci_upper=0.4)
        with pytest.raises(ValueError, match="draws were not retained"):
            result.plot_distribution()

    def test_zivot_andrews_plotly_scan(self):
        rng = np.random.default_rng(4)
        za = ZivotAndrewsResult(
            stat=-4.2,
            breakpoint=120,
            pval=0.01,
            stat_at_each_bp=rng.normal(-3, 0.5, 200),
            tested_breakpoints=np.arange(50, 250),
        )
        assert_figure(za.plot_breakpoint_scan_plotly())

    def test_zivot_andrews_plotly_needs_metadata(self):
        za = ZivotAndrewsResult(stat=-4.2, breakpoint=120, pval=0.01)
        with pytest.raises(ValueError, match="metadata=True"):
            za.plot_breakpoint_scan_plotly()


# ======================================================================
# Optional dependency behavior
# ======================================================================

class TestMissingPlotly:
    def test_helpful_error_when_plotly_missing(self, df, monkeypatch):
        """Simulate a missing plotly: the error must name the extra."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "plotly" or name.startswith("plotly."):
                raise ImportError("No module named 'plotly'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rep = nb.outlier_report(df, rich=True)
        with pytest.raises(ImportError, match=r"bunker-stats-rs\[notebook\]"):
            rep.plot_counts()

    def test_report_objects_usable_without_plotly(self, df, monkeypatch):
        """Everything except .plot_* works when plotly is absent."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "plotly" or name.startswith("plotly."):
                raise ImportError("No module named 'plotly'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rep = nb.correlation_report(df, rich=True)
        assert isinstance(rep.to_frame(), pd.DataFrame)
        assert isinstance(rep.to_dict(), dict)
        assert isinstance(rep.info(), str)


# ======================================================================
# Misuse-prevention warnings
# ======================================================================

class TestWarningsMetadata:
    def test_clean_data_has_no_warnings(self, df):
        rep = nb.robust_summary(df, rich=True)
        assert rep.warnings == []
        assert rep.to_dict()["warnings"] == []

    def test_all_nan_column_warned(self):
        frame = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1.0, 2.0, 3.0]})
        rep = nb.robust_summary(frame, rich=True)
        assert any("no finite values" in w for w in rep.warnings)
        assert "Warning:" in rep.info()

    def test_zero_variance_column_warned(self):
        frame = pd.DataFrame({"k": [5.0] * 20, "y": np.arange(20.0)})
        rep = nb.robust_summary(frame, rich=True)
        assert any("zero variance" in w for w in rep.warnings)

    def test_outlier_degenerate_fences_warned(self):
        frame = pd.DataFrame({"k": [5.0] * 20})
        rep = nb.outlier_report(frame, rich=True)
        assert any("zero spread" in w for w in rep.warnings)

    def test_bootstrap_low_resamples_warned(self, df):
        rep = nb.bootstrap_ci_report(df, n_resamples=200, random_state=0, rich=True)
        assert any("n_resamples=200 is low" in w for w in rep.warnings)

    def test_bootstrap_small_n_warned(self):
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        rep = nb.bootstrap_ci_report(frame, n_resamples=2000, random_state=0, rich=True)
        assert any("small sample" in w for w in rep.warnings)

    def test_inference_small_sample_warning(self):
        rng = np.random.default_rng(5)
        r = bs.t_test_2samp(rng.normal(size=5), rng.normal(size=6), rich=True)
        assert any("small sample" in w for w in r.warnings)
        assert "Warning:" in r.info()
        assert "warnings" in r.to_dict()

    def test_inference_large_sample_no_warning(self):
        rng = np.random.default_rng(6)
        r = bs.t_test_2samp(rng.normal(size=100), rng.normal(size=100), rich=True)
        assert r.warnings == []
        assert "warnings" not in r.to_dict()

    def test_anderson_darling_approximate_warning(self):
        rng = np.random.default_rng(7)
        r = bs.anderson_darling(rng.normal(size=100), rich=True)
        assert any("critical value" in w for w in r.warnings)

    def test_jarque_bera_tiny_n_warning(self):
        rng = np.random.default_rng(8)
        r = bs.jarque_bera(rng.normal(size=10), rich=True)
        assert any("n ~ 20" in w for w in r.warnings)

    def test_mann_whitney_small_n_exact_tables_warning(self):
        rng = np.random.default_rng(9)
        r = bs.mann_whitney_u(rng.normal(size=6), rng.normal(size=7), rich=True)
        assert any("exact tables" in w for w in r.warnings)

    def test_warnings_do_not_break_unpacking(self):
        rng = np.random.default_rng(10)
        r = bs.t_test_1samp(rng.normal(size=5), 0.0, rich=True)
        stat, pval = r  # warnings must not interfere with the tuple protocol
        assert isinstance(stat, float) and isinstance(pval, float)


# ======================================================================
# BootstrapConfig(return_draws=True) end-to-end
# ======================================================================

class TestBootstrapReturnDraws:
    def test_default_tuple_path_unchanged(self):
        from bunker_stats.resampling import BootstrapConfig

        rng = np.random.default_rng(11)
        x = rng.normal(size=300)
        out = BootstrapConfig(n_resamples=500, random_state=3).run(x)
        assert isinstance(out, tuple) and len(out) == 3

    def test_return_draws_yields_result_with_draws(self):
        from bunker_stats.resampling import BootstrapConfig

        rng = np.random.default_rng(11)
        x = rng.normal(size=300)
        res = BootstrapConfig(
            n_resamples=500, random_state=3, return_draws=True
        ).run(x)
        assert isinstance(res, BootstrapResult)
        assert res.draws is not None and res.draws.shape == (500,)
        assert res.method == "percentile"
        assert res.n_resamples == 500
        assert res.confidence_level == 0.95
        assert res.se is not None and res.se > 0

    def test_draws_ci_identical_to_tuple_ci(self):
        """Both paths share one kernel + RNG stream: results must be equal."""
        from bunker_stats.resampling import BootstrapConfig

        rng = np.random.default_rng(12)
        x = rng.normal(size=250)
        tuple_out = BootstrapConfig(n_resamples=800, random_state=7).run(x)
        rich_out = BootstrapConfig(
            n_resamples=800, random_state=7, return_draws=True
        ).run(x)
        assert tuple_out == (
            rich_out.estimate,
            rich_out.ci_lower,
            rich_out.ci_upper,
        )
        # draws reproduce the estimate
        assert abs(rich_out.draws.mean() - rich_out.estimate) < 1e-12

    def test_plot_distribution_works_end_to_end(self):
        from bunker_stats.resampling import BootstrapConfig

        rng = np.random.default_rng(13)
        x = rng.normal(size=200)
        res = BootstrapConfig(
            n_resamples=400, random_state=5, return_draws=True
        ).run(x)
        assert_figure(res.plot_distribution())

    def test_return_draws_respects_nan_policy_omit(self):
        from bunker_stats.resampling import BootstrapConfig

        rng = np.random.default_rng(14)
        x = rng.normal(size=100)
        x[::10] = np.nan
        res = BootstrapConfig(
            n_resamples=200, random_state=1, nan_policy="omit", return_draws=True
        ).run(x)
        assert np.all(np.isfinite(res.draws))
