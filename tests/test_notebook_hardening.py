"""Regression tests for the notebook-layer hardening pass.

* ``TestNormalityReport``   - the verdict came from Jarque-Bera alone, a
                              constant column read "sample too small", and
                              there was no ``n_missing`` and no ``rich``.
* ``TestStrictJson``        - ``to_dict()`` was documented as JSON-safe but
                              held NaN, numpy scalars, tuple keys and
                              Timestamps.
* ``TestMultiIndexNames``   - derived columns were named
                              ``"('g1', 'a')_zscore"``.
* ``TestPlainPathWarnings`` - conditions that change what the numbers mean
                              were visible only with ``rich=True``.
* ``TestRichWarnings``      - missing notes: unreachable z-threshold,
                              undefined correlations.
* ``TestMissingness``       - the report and the styler disagreed on object
                              and complex columns.
* ``TestValidation``        - bad arguments raised raw comparison errors or
                              were accepted.
* ``TestResultInterfaces``  - reports and style results had different part
                              names and four reports had no ``rich``.
* ``TestAndersonDarlingFarTail`` - every strong departure reported the same
                              floor p-value.
* ``TestOneNamePerOption``  - ``suffix`` / ``add_suffix`` and ``*_bound`` /
                              ``*_fence`` named the same things.
* ``TestLabeledMatrices``   - ``corr_df`` / ``cov_df`` returned all-NaN for a
                              frame with one NaN.

References are independent of this module (statsmodels, numpy, the flat-frame
result); comparisons of small quantities use ``close`` (no absolute
tolerance).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("jinja2")

from bunker_stats import notebook as nb  # noqa: E402


def close(value: float, reference: float, rel: float) -> bool:
    """Relative comparison with no absolute tolerance."""
    return value == pytest.approx(reference, rel=rel, abs=0)


def strict(result, **kwargs) -> dict:
    """Round-trip through strict JSON; raises if anything is not JSON."""
    text = json.dumps(result.to_dict(**kwargs), allow_nan=False)
    assert text == result.to_json(**kwargs)
    return json.loads(text)


def background_columns(styler) -> set:
    """Positions of the columns that carry a background color."""
    styler._compute()
    return {
        col for (_, col), css in styler.ctx.items()
        if any(prop == "background-color" for prop, _ in css)
    }


@pytest.fixture
def grouped():
    """A frame with MultiIndex columns and its flat twin."""
    values = np.random.default_rng(2).normal(size=(30, 3))
    values[4, 0] = 9.0
    flat = pd.DataFrame(values, columns=["a", "b", "c"])
    multi = pd.DataFrame(
        values,
        columns=pd.MultiIndex.from_tuples(
            [("g1", "a"), ("g1", "b"), ("g2", "c")], names=["grp", "var"]
        ),
    )
    return multi, flat


# ======================================================================
# normality_report
# ======================================================================

class TestNormalityReport:
    @pytest.mark.parametrize(
        "draw",
        [
            lambda rng: rng.normal(size=50),
            lambda rng: rng.uniform(size=40),
            lambda rng: rng.exponential(size=30),
            lambda rng: rng.standard_t(3, size=200),
            lambda rng: rng.normal(size=8),
        ],
    )
    def test_ad_pvalue_matches_statsmodels(self, draw):
        normal_ad = pytest.importorskip("statsmodels.stats.diagnostic").normal_ad
        x = draw(np.random.default_rng(0))
        row = nb.normality_report(pd.DataFrame({"x": x})).loc["x"]
        _, ref = normal_ad(x)
        assert 0.0 < ref < 1.0
        assert close(row["ad_pvalue"], ref, 1e-6), (row["ad_pvalue"], ref)

    def test_verdict_follows_anderson_darling_not_jarque_bera(self):
        """30 exponential draws (a seed where the two tests disagree, which
        about 1 in 20 do): Jarque-Bera cannot reject -- its chi-square
        approximation is poor at this n -- and Anderson-Darling does."""
        x = np.random.default_rng(63).exponential(size=30)
        row = nb.normality_report(pd.DataFrame({"x": x})).loc["x"]
        assert row["jb_pvalue"] > 0.05 > row["ad_pvalue"]
        assert row["normal"] is False or row["normal"] == False  # noqa: E712
        assert row["conclusion"] == "reject normality (Anderson-Darling p < 0.05)"

    def test_alpha_moves_the_verdict(self):
        x = np.random.default_rng(63).exponential(size=30)  # ad_pvalue ~ 0.030
        frame = pd.DataFrame({"x": x})
        assert not nb.normality_report(frame, alpha=0.05).loc["x", "normal"]
        assert nb.normality_report(frame, alpha=0.01).loc["x", "normal"]

    @pytest.mark.parametrize(
        "values, conclusion",
        [
            ([1.0, 2.0, 3.0, 4.0, 50.0], "inconclusive (n < 8)"),
            ([5.0] * 50, "constant column (no variance)"),
            ([np.nan] * 5, "no finite values"),
        ],
    )
    def test_no_verdict_cases_say_why(self, values, conclusion):
        row = nb.normality_report(pd.DataFrame({"x": values})).loc["x"]
        assert row["normal"] is None
        assert row["conclusion"] == conclusion
        assert np.isnan(row["ad_pvalue"])

    def test_n_missing_column(self):
        x = np.r_[np.random.default_rng(1).normal(size=20), np.nan, np.inf, -np.inf]
        row = nb.normality_report(pd.DataFrame({"x": x})).loc["x"]
        assert (row["n"], row["n_missing"]) == (20, 3)

    def test_pvalue_for_a_huge_statistic_is_tiny_not_above_one(self):
        """The fitted formula turns upward for huge statistics (see
        TestAndersonDarlingFarTail); the result must stay a tiny p-value."""
        x = np.r_[np.zeros(500), np.full(500, 1e6), 3.0]
        row = nb.normality_report(pd.DataFrame({"x": x})).loc["x"]
        assert row["ad_statistic"] > 100
        assert 0.0 <= row["ad_pvalue"] < 1e-100

    def test_rich_result_and_its_warnings(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame({
            "big": rng.standard_t(30, size=20000),
            "ok": np.r_[rng.normal(size=100), [np.nan] * 19900],
            "tiny": np.r_[[1.0, 2.0, 4.0], [np.nan] * 19997],
            "flat": 1.0,
        })
        report = nb.normality_report(frame, rich=True)
        assert isinstance(report, nb.NormalityReport)
        pd.testing.assert_frame_equal(report.data, nb.normality_report(frame))
        text = " | ".join(report.warnings)
        assert "large sample (n > 5000) in column(s): big" in text
        assert "constant column(s): flat" in text
        assert "fewer than 8 finite values in column(s): tiny" in text
        assert "ok" not in text.replace("look", "")
        assert report.meta["decided_by"].startswith("Anderson-Darling")


# ======================================================================
# to_dict() / to_json() are strict JSON
# ======================================================================

class TestStrictJson:
    def test_nan_becomes_none_in_a_report(self):
        frame = pd.DataFrame({"e": [np.nan, np.nan], "v": [1.0, 2.0]})
        data = strict(nb.robust_summary(frame, rich=True))["data"]
        assert data["e"]["mean"] is None and data["e"]["n"] == 0
        assert data["v"]["mean"] == 1.5

    def test_infinite_fences_become_none(self):
        result = nb.outlier_style(pd.DataFrame({"e": [np.nan] * 3}), rich=True)
        assert np.isnan(result.metadata["per_column"]["e"]["lower_fence"])
        assert strict(result)["metadata"]["per_column"]["e"]["lower_fence"] is None

    def test_every_rich_result_on_multiindex_columns(self, grouped):
        multi, _ = grouped
        col = ("g1", "a")
        results = [
            nb.robust_summary(multi, rich=True),
            nb.describe_fast(multi, rich=True),
            nb.outlier_report(multi, rich=True),
            nb.normality_report(multi, rich=True),
            nb.correlation_report(multi, rich=True),
            nb.correlation_report(multi, pvalues=True, rich=True),
            nb.missingness_report(multi, rich=True),
            nb.rolling_report(multi, col, 5, rich=True),
            nb.bootstrap_ci_report(multi, n_resamples=200, rich=True),
            nb.scale_columns(multi, rich=True),
            nb.winsorize_columns(multi, rich=True),
            nb.robust_scale_column(multi, col, rich=True),
            nb.outlier_style(multi, rich=True),
            nb.iqr_outlier_style(multi, col, rich=True),
            nb.corr_heatmap(multi, rich=True),
            nb.missingness_style(multi, rich=True),
            nb.demean_style(multi, col, rich=True),
            nb.zscore_style(multi, col, rich=True),
            nb.scale_compare_style(multi, col, rich=True),
        ]
        for result in results:
            payload = strict(result)
            assert payload["title"] and isinstance(payload["warnings"], list)
        # a tuple label survives as a readable string key
        assert "('g1', 'a')" in strict(results[0])["data"]

    def test_numpy_scalar_arguments(self):
        frame = pd.DataFrame({"a": np.random.default_rng(0).normal(size=40)})
        report = nb.bootstrap_ci_report(
            frame, n_resamples=200, conf=np.float32(0.9),
            random_state=np.int64(3), rich=True,
        )
        meta = strict(report)["meta"]
        assert meta["random_state"] == 3 and meta["conf"] == pytest.approx(0.9)

    def test_dates_in_labels_and_in_the_frame(self):
        when = pd.date_range("2024-01-01", periods=4)
        frame = pd.DataFrame({"t": when, "x": [1.0, 2.0, np.nan, 40.0]})
        payload = strict(nb.zscore_style(frame, "x", rich=True), frame=True)
        assert payload["frame"]["t"][0].startswith("2024-01-01")
        assert payload["frame"]["x"][2] is None

        wide = pd.DataFrame(
            np.random.default_rng(0).normal(size=(12, 2)),
            columns=list(pd.date_range("2024-01-01", periods=2)),
        )
        data = strict(nb.robust_summary(wide, rich=True))["data"]
        assert all(key.startswith("2024-01-0") for key in data)

    def test_original_values_stay_available(self):
        report = nb.robust_summary(pd.DataFrame({"e": [np.nan, np.nan]}), rich=True)
        assert np.isnan(report.data.loc["e", "mean"])


# ======================================================================
# MultiIndex columns: derived names
# ======================================================================

class TestMultiIndexNames:
    def test_transforms(self, grouped):
        multi, flat = grouped
        for fn, suffix in (
            (nb.scale_columns, "_robust"),
            (nb.winsorize_columns, "_winsor"),
        ):
            out, ref = fn(multi), fn(flat)
            assert isinstance(out.columns, pd.MultiIndex) and out.columns.nlevels == 2
            assert ("g1", f"a{suffix}") in out.columns
            assert ("g2", f"c{suffix}") in out.columns
            np.testing.assert_array_equal(
                out[("g1", f"a{suffix}")].to_numpy(), ref[f"a{suffix}"].to_numpy()
            )
        out = nb.robust_scale_column(multi, ("g1", "b"))
        assert ("g1", "b_robust") in out.columns

    def test_stylers(self, grouped):
        multi, flat = grouped
        col = ("g1", "a")
        z = nb.zscore_style(multi, col, rich=True)
        assert z.metadata["derived_column"] == ("g1", "a_zscore")
        np.testing.assert_array_equal(
            z.frame[("g1", "a_zscore")].to_numpy(),
            nb.zscore_style(flat, "a", rich=True).frame["a_zscore"].to_numpy(),
        )
        assert nb.demean_style(multi, col, rich=True).metadata["derived_column"] == (
            "g1", "a_demeaned"
        )
        both = nb.scale_compare_style(multi, col, rich=True)
        assert both.metadata["derived_columns"] == {
            "zscore": ("g1", "a_zscore"), "robust_z": ("g1", "a_robust_z"),
        }
        for result in (z, both):
            assert result.frame.columns.nlevels == 2
            assert not any(isinstance(c, str) for c in result.frame.columns)
            result.styler.to_html()  # the painted column must be addressable

    def test_rolling_report(self, grouped):
        multi, flat = grouped
        out = nb.rolling_report(multi, ("g1", "a"), 5)
        ref = nb.rolling_report(flat, "a", 5)
        assert list(out.columns) == [("g1", c) for c in ref.columns]
        assert list(out.columns.names) == ["grp", "var"]
        np.testing.assert_array_equal(out.to_numpy(), ref.to_numpy())

    def test_flat_names_are_unchanged(self, grouped):
        _, flat = grouped
        assert "a_robust" in nb.scale_columns(flat).columns
        assert list(nb.rolling_report(flat, "a", 5, stats=["mean"]).columns) == ["a_roll5_mean"]
        assert "0_zscore" in nb.zscore_style(
            pd.DataFrame({0: [1.0, 2.0, 4.0]}), 0, rich=True
        ).frame.columns


# ======================================================================
# UserWarnings on the plain (non-rich) path
# ======================================================================

class TestPlainPathWarnings:
    def test_large_integers_lose_precision(self):
        """2**53 + [1, 3, 5, 2, 4] collapses to three distinct floats: the std
        comes out as 2.236 where the true value is 1.581."""
        frame = pd.DataFrame({"big": 2 ** 53 + np.array([1, 3, 5, 2, 4])})
        with pytest.warns(UserWarning, match=r"'big' holds integers beyond 2\*\*53") as rec:
            out = nb.robust_summary(frame)
        assert rec[0].filename == __file__
        assert not close(out.loc["big", "std"], np.std([1, 3, 5, 2, 4], ddof=1), 1e-6)

    @pytest.mark.parametrize("dtype", ["int64", "Int64", "int32", "float64"])
    def test_ordinary_numbers_do_not_warn(self, dtype):
        frame = pd.DataFrame({"x": pd.array([1, 5, 2 ** 31 - 1, 7, 9], dtype=dtype)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.robust_summary(frame)
            nb.outlier_report(frame)

    def test_exactly_two_to_the_53_is_fine(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.describe_fast(pd.DataFrame({"x": [2 ** 53, -(2 ** 53), 0]}))

    def test_overwriting_an_existing_column(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 9.0], "a_robust": 0.0,
                              "a_winsor": 0.0, "a_zscore": 0.0, "a_demeaned": 0.0})
        calls = [
            lambda: nb.scale_columns(frame, ["a"]),
            lambda: nb.robust_scale_column(frame, "a"),
            lambda: nb.winsorize_columns(frame, ["a"]),
            lambda: nb.zscore_style(frame, "a"),
            lambda: nb.demean_style(frame, "a"),
            lambda: nb.scale_compare_style(frame, "a"),
        ]
        for call in calls:
            with pytest.warns(UserWarning, match="already existed in the input") as rec:
                call()
            assert rec[0].filename == __file__
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.scale_columns(frame, ["a"], replace=True)       # asked for
            nb.scale_columns(frame, ["a"], suffix="_new")      # no collision

    def test_zero_mad_scaling(self):
        frame = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0, 2.0]})
        with pytest.warns(UserWarning, match=r"MAD is 0 in column\(s\): a"):
            out = nb.scale_columns(frame)
        assert abs(out["a_robust"].iloc[-1]) > 1e9   # what the warning is about
        with pytest.warns(UserWarning, match="MAD is 0"):
            nb.robust_scale_column(frame, "a")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.scale_columns(frame, method="zscore")
            nb.scale_columns(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}))

    def test_too_few_bootstrap_resamples(self):
        frame = pd.DataFrame({"a": np.random.default_rng(0).normal(size=40)})
        with pytest.warns(UserWarning, match="n_resamples=1 is too few"):
            out = nb.bootstrap_ci_report(frame, n_resamples=1)
        assert out.loc["a", "ci_lower"] == out.loc["a", "ci_upper"]  # zero width
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.bootstrap_ci_report(frame, n_resamples=100)

    def test_rich_results_carry_the_note_instead_of_warning(self):
        frame = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0, 2.0], "a_robust": 0.0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = nb.scale_columns(frame, ["a"], rich=True)
            boot = nb.bootstrap_ci_report(frame, ["a"], n_resamples=5, rich=True)
        text = " | ".join(result.warnings)
        assert "MAD is 0" in text and "already existed" in text
        assert any("too few" in w for w in boot.warnings)


# ======================================================================
# Notes on rich results
# ======================================================================

class TestRichWarnings:
    def test_unreachable_z_threshold(self):
        """With n = 5 the largest possible z-score is 4/sqrt(5) = 1.79."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
        z = (frame["x"] - frame["x"].mean()) / frame["x"].std(ddof=1)
        assert z.abs().max() < 1.79
        for result in (
            nb.outlier_report(frame, method="zscore", rich=True),
            nb.outlier_style(frame, method="zscore", rich=True),
        ):
            assert any(
                "z_threshold 3 cannot be reached in column(s): x" in w
                for w in result.warnings
            ), result.warnings

    def test_reachable_threshold_and_other_methods_are_quiet(self):
        small = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
        large = pd.DataFrame({"x": np.random.default_rng(0).normal(size=50)})
        quiet = [
            nb.outlier_report(large, method="zscore", rich=True),
            nb.outlier_report(small, method="robust_zscore", rich=True),
            nb.outlier_report(small, method="iqr", rich=True),
            nb.outlier_report(small, method="zscore", z_threshold=1.5, rich=True),
        ]
        for result in quiet:
            assert not any("cannot be reached" in w for w in result.warnings)
        # the robust rule does flag the point the classical one cannot
        assert quiet[1].data.loc["x", "n_outliers"] == 1

    def test_undefined_correlations(self):
        two = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0], "c": [0.5, 0.7]})
        for result in (
            nb.corr_heatmap(two, rich=True),
            nb.correlation_report(two, rich=True),
            nb.correlation_report(two, pvalues=True, rich=True),
        ):
            assert any(
                "3 pair(s) have fewer than 3 overlapping rows" in w
                for w in result.warnings
            ), result.warnings

    def test_constant_column_and_small_overlap(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.normal(size=6), "b": rng.normal(size=6), "k": 1.0})
        report = nb.correlation_report(frame, rich=True)
        text = " | ".join(report.warnings)
        assert "constant column(s): k" in text
        assert "small overlap (min pairwise n = 6)" in text
        assert report.meta["pairwise_n_min"] == 6
        assert report.warnings == nb.corr_heatmap(frame, rich=True).warnings

    def test_clean_frame_has_no_notes(self):
        frame = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=list("abc"))
        assert nb.correlation_report(frame, rich=True).warnings == []
        assert nb.corr_heatmap(frame, rich=True).warnings == []
        long = nb.correlation_report(frame, pvalues=True, rich=True)
        assert long.warnings == [
            "3 tests with no multiple-comparison correction — consider "
            "bunker_stats.p_adjust"
        ]

    def test_long_form_style_colors_only_the_correlation(self):
        frame = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=list("abc"))
        report = nb.correlation_report(frame, pvalues=True, rich=True)
        styled = report.style()
        assert background_columns(styled) == {list(report.data.columns).index("correlation")}
        matrix = nb.correlation_report(frame, rich=True).style()
        assert background_columns(matrix) == {0, 1, 2}

    def test_significance_declared_adjusted(self):
        table = pd.DataFrame({"pvalue": [0.001, 0.02, 0.2, 0.6]})
        plain = nb.style_significance(table, rich=True)
        assert len(plain.warnings) == 1 and "no multiple-comparison" in plain.warnings[0]
        assert plain.metadata["expected_false_positives_if_all_null"] == pytest.approx(0.2)

        declared = nb.style_significance(table, adjusted=True, rich=True)
        assert declared.warnings == []
        assert declared.metadata["expected_false_positives_if_all_null"] is None
        assert "adjusted" in declared.metadata["multiple_comparison_correction"]
        # a declaration only: the shading is identical
        assert declared.styler.to_html(uuid="t") == plain.styler.to_html(uuid="t")

    def test_heatmap_uses_both_policy_keys(self):
        frame = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 2)), columns=list("ab"))
        meta = nb.corr_heatmap(frame, rich=True).metadata
        assert meta["nan_policy"] == meta["missing_policy"]
        assert meta["nan_policy"] == nb.correlation_report(frame, rich=True).meta["nan_policy"]


# ======================================================================
# missingness_report / missingness_style agree
# ======================================================================

class TestMissingness:
    @pytest.fixture
    def frame(self):
        inf = float("inf")
        return pd.DataFrame({
            "z": [1 + 2j, complex(inf, 0.0), complex(0.0, inf), complex(np.nan, 0.0)],
            "o": pd.Series([1.0, inf, -inf, np.nan], dtype=object),
            "t": pd.Series(["a", "b", None, "d"], dtype=object),
            "m": pd.Series([1, "x", 2.5, None], dtype=object),
            "b": [True, False, True, True],
            "f": [1.0, inf, np.nan, 4.0],
        })

    def test_counts(self, frame):
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # no ComplexWarning from a float cast
            out = nb.missingness_report(frame)
        assert out.loc["z", ["n_missing", "n_infinite", "n_finite"]].tolist() == [1, 2, 1]
        assert out.loc["o", ["n_missing", "n_infinite", "n_finite"]].tolist() == [1, 2, 1]
        assert out.loc["f", ["n_missing", "n_infinite", "n_finite"]].tolist() == [1, 1, 2]
        for label in ("t", "m", "b"):       # text, mixed, boolean: not meaningful
            assert np.isnan(out.loc[label, "n_infinite"])
            assert np.isnan(out.loc[label, "n_finite"])

    def test_report_and_style_agree(self, frame):
        report = nb.missingness_report(frame)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            style = nb.missingness_style(frame, rich=True)
        per_column = style.metadata["per_column"]
        for label in frame.columns:
            assert per_column[label]["n_missing"] == report.loc[label, "n_missing"]
            expected = report.loc[label, "n_infinite"]
            assert per_column[label]["n_infinite"] == (0 if np.isnan(expected) else expected)

    def test_infinite_cells_in_an_object_column_are_painted(self, frame):
        style = nb.missingness_style(frame, ["o"], rich=True)
        infinite = style.metadata["theme"]["colors"]["infinite"]
        html = style.styler.to_html()
        style.styler._compute()
        painted = [
            css for (_, col), css in style.styler.ctx.items()
            if col == list(frame.columns).index("o")
        ]
        colors = sorted(v for css in painted for k, v in css if k == "background-color")
        assert colors.count(infinite) == 2 and len(colors) == 3 and html

    def test_rich_report(self, frame):
        report = nb.missingness_report(frame, rich=True)
        assert isinstance(report, nb.MissingnessReport)
        assert report.meta["n_complete_rows"] == 1
        hollow = nb.missingness_report(pd.DataFrame({"a": [np.nan, np.nan]}), rich=True)
        assert any("entirely missing column(s): a" in w for w in hollow.warnings)
        assert any("no row is complete" in w for w in hollow.warnings)


# ======================================================================
# Argument validation
# ======================================================================

class TestValidation:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame({"x": np.random.default_rng(0).normal(size=30)})

    def test_non_numbers_name_the_argument(self, frame):
        with pytest.raises(TypeError, match="alpha must be a number, got str"):
            nb.normality_report(frame, alpha="x")
        with pytest.raises(TypeError, match="lower_q must be a number, got str"):
            nb.winsorize_columns(frame, lower_q="x")
        with pytest.raises(TypeError, match="conf must be a number, got str"):
            nb.bootstrap_ci_report(frame, conf="x")
        with pytest.raises(TypeError, match="n_resamples must be an int"):
            nb.bootstrap_ci_report(frame, n_resamples=10.5)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan")])
    def test_alpha_range(self, frame, alpha):
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
            nb.normality_report(frame, alpha=alpha)

    @pytest.mark.parametrize("conf", [0.0, 1.0, 95, float("nan")])
    def test_conf_range(self, frame, conf):
        with pytest.raises(ValueError, match=r"conf must be in \(0, 1\)"):
            nb.bootstrap_ci_report(frame, conf=conf)

    @pytest.mark.parametrize("n", [0, -5])
    def test_n_resamples_range(self, frame, n):
        with pytest.raises(ValueError, match="n_resamples must be >= 1"):
            nb.bootstrap_ci_report(frame, n_resamples=n)

    @pytest.mark.parametrize(
        "thresholds",
        [(-3, -2, -1), (0.0, 0.5, 0.8), (0.1, 0.2, float("inf")), (0.1, float("nan"), 0.8)],
    )
    def test_effect_thresholds_must_be_positive_and_finite(self, thresholds):
        table = pd.DataFrame({"d": [0.1, 0.4, 0.9, 2.0]})
        with pytest.raises(ValueError, match="`thresholds` must be finite and > 0"):
            nb.style_effect_size(table, "d", thresholds=thresholds)

    def test_effect_thresholds_type(self):
        table = pd.DataFrame({"d": [0.1, 0.4]})
        with pytest.raises(TypeError, match="`thresholds` must be a sequence"):
            nb.style_effect_size(table, "d", thresholds="abc")
        with pytest.raises(TypeError, match="each threshold must be a number"):
            nb.style_effect_size(table, "d", thresholds=(0.1, "b", 0.8))
        ok = nb.style_effect_size(
            table, "d", thresholds=np.array([0.2, 0.5, 0.8], dtype=np.float32), rich=True
        )
        assert ok.metadata["n_per_bucket"] == {
            "negligible": 1, "small": 1, "medium": 0, "large": 0,
        }


# ======================================================================
# One interface for reports and style results
# ======================================================================

class TestResultInterfaces:
    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})

    def test_every_report_takes_rich(self, frame):
        made = {
            nb.DescribeReport: nb.describe_fast(frame, rich=True),
            nb.NormalityReport: nb.normality_report(frame, rich=True),
            nb.MissingnessReport: nb.missingness_report(frame, rich=True),
            nb.RollingReport: nb.rolling_report(frame, "a", 5, rich=True),
        }
        plain = [
            nb.describe_fast(frame), nb.normality_report(frame),
            nb.missingness_report(frame), nb.rolling_report(frame, "a", 5),
        ]
        for (cls, report), expected in zip(made.items(), plain):
            assert type(report) is cls and isinstance(report, nb.NotebookReport)
            pd.testing.assert_frame_equal(report.data, expected)
            assert report.info() and isinstance(report.warnings, list)

    def test_reports_unpack_and_answer_to_both_names(self, frame):
        report = nb.robust_summary(frame, rich=True)
        data, meta = report
        assert data is report.data is report.frame
        assert meta is report.meta is report.metadata
        assert len(report) == 2 and report[0] is data

    def test_style_results_answer_to_both_names(self, frame):
        result = nb.zscore_style(frame, "a", rich=True)
        assert result.data is result.frame and result.meta is result.metadata
        copy = result.to_frame()
        assert copy is not result.frame
        pd.testing.assert_frame_equal(copy, result.frame)

    def test_describe_fast_rich_meta(self, frame):
        assert nb.describe_fast(frame, rich=True).meta["robust"] is True
        lean = nb.describe_fast(frame, robust=False, rich=True)
        assert lean.meta["robust"] is False and "mad" not in lean.data.columns
        flat = nb.describe_fast(pd.DataFrame({"k": [2.0, 2.0, 2.0]}), rich=True)
        assert flat.warnings == ["zero variance in column(s): k"]

    def test_rolling_report_rich(self, frame):
        report = nb.rolling_report(frame, "a", 5, stats=["mean", "max"], rich=True)
        assert report.meta["derived_columns"] == ["a_roll5_mean", "a_roll5_max"]
        assert report.meta["n_rows_without_a_value"] == 4
        assert report.meta["min_periods"] == 5 and report.warnings == []

        gappy = frame.copy()
        gappy.loc[10, "a"] = np.nan
        noted = nb.rolling_report(gappy, "a", 5, nan_policy="ignore", rich=True)
        assert any("has no effect" in w for w in noted.warnings)
        fixed = nb.rolling_report(
            gappy, "a", 5, nan_policy="ignore", min_periods=3, rich=True
        )
        assert fixed.warnings == [] and fixed.meta["n_missing"] == 1


# ======================================================================
# Follow-ups: one name per option, pairwise corr_df, the A-D far tail
# ======================================================================

class TestAndersonDarlingFarTail:
    def test_pvalue_never_increases_with_the_statistic(self):
        """The published fit turns back upward past its range; the p-value
        must keep falling so that stronger departures rank as stronger."""
        grid = np.r_[np.linspace(0.05, 13.0, 400), np.linspace(13.0, 400.0, 800)]
        p = np.array([nb._ad_pvalue(a, 1000) for a in grid])
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert np.all(np.diff(p) <= 0.0)
        assert p[0] > 0.99 and p[-1] == 0.0          # underflow only at the far end

    def test_no_plateau_beyond_the_fitted_range(self):
        beyond = [nb._ad_pvalue(a, 1000) for a in (14.0, 20.0, 50.0, 100.0)]
        assert all(b > 0.0 for b in beyond)
        assert all(later < earlier * 1e-10 for earlier, later in zip(beyond, beyond[1:]))

    def test_continuous_where_the_extrapolation_starts(self):
        n = 1000
        scale = (1.0 + 0.75 / n + 2.25 / n**2) / (1.0 + 4.0 / n - 25.0 / n**2)
        edge = 13.0 / scale                           # a_star whose z is 13
        below, above = nb._ad_pvalue(edge * (1 - 1e-9), n), nb._ad_pvalue(edge * (1 + 1e-9), n)
        assert close(above, below, 1e-6)

    def test_stronger_bimodality_gets_a_smaller_pvalue(self):
        rng = np.random.default_rng(0)
        pvalues = []
        for gap in (4.0, 10.0, 50.0):
            x = np.r_[rng.normal(size=400), rng.normal(gap, 1.0, 400)]
            pvalues.append(nb.normality_report(pd.DataFrame({"x": x})).loc["x", "ad_pvalue"])
        assert pvalues[0] > pvalues[1] > pvalues[2] > 0.0


class TestOneNamePerOption:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 40.0]})

    def test_suffix_everywhere(self, frame):
        assert "a_s" in nb.scale_columns(frame, suffix="_s").columns
        assert "a_s" in nb.winsorize_columns(frame, suffix="_s").columns
        assert "a_s" in nb.robust_scale_column(frame, "a", suffix="_s").columns
        assert nb.zscore_style(frame, "a", suffix="_s", rich=True).metadata["derived_column"] == "a_s"
        assert nb.demean_style(frame, "a", suffix="_s", rich=True).metadata["derived_column"] == "a_s"

    def test_add_suffix_still_works_and_defaults_are_unchanged(self, frame):
        assert "a_old" in nb.robust_scale_column(frame, "a", add_suffix="_old").columns
        assert nb.zscore_style(frame, "a", add_suffix="_old", rich=True).metadata["derived_column"] == "a_old"
        assert nb.demean_style(frame, "a", add_suffix="_old", rich=True).metadata["derived_column"] == "a_old"
        assert "a_robust" in nb.robust_scale_column(frame, "a").columns
        assert nb.zscore_style(frame, "a", rich=True).metadata["derived_column"] == "a_zscore"
        assert nb.demean_style(frame, "a", rich=True).metadata["derived_column"] == "a_demeaned"

    def test_conflicting_spellings_are_rejected(self, frame):
        for call in (nb.zscore_style, nb.demean_style, nb.robust_scale_column):
            with pytest.raises(TypeError, match="pass only `suffix`"):
                call(frame, "a", suffix="_x", add_suffix="_y")
            call(frame, "a", suffix="_x", add_suffix="_x")      # same value: fine
        with pytest.raises(ValueError, match="`suffix` must be a non-empty string"):
            nb.zscore_style(frame, "a", suffix="")

    def test_bounds_and_fences_are_the_same_numbers(self, frame):
        report = nb.outlier_report(frame).loc["a"]
        flat = nb.iqr_outlier_style(frame, "a", rich=True).metadata
        nested = nb.outlier_style(frame, rich=True).metadata["per_column"]["a"]
        for meta in (flat, nested):
            assert meta["lower_bound"] == meta["lower_fence"] == report["lower_bound"]
            assert meta["upper_bound"] == meta["upper_fence"] == report["upper_bound"]


class TestLabeledMatrices:
    @pytest.fixture
    def frame(self):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(40, 4)), columns=list("abcd")
        )
        frame.iloc[3, 0] = np.nan
        frame.iloc[7, 1] = np.nan
        frame.iloc[9, 2] = np.inf
        frame["k"] = 1.0
        frame["e"] = np.nan
        return frame

    def test_corr_df_is_pairwise_complete_like_pandas(self, frame):
        import bunker_stats.pandas as bsp

        ref = frame.replace([np.inf, -np.inf], np.nan).corr()
        got = bsp.corr_df(frame)
        assert list(got.columns) == list(ref.columns) == list(got.index)
        np.testing.assert_array_equal(got.isna().to_numpy(), ref.isna().to_numpy())
        np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-12, atol=1e-15)
        # and it agrees with the notebook layer's own pairwise report
        report = nb.correlation_report(frame[list("abcd")])
        np.testing.assert_allclose(
            got.loc[list("abcd"), list("abcd")].to_numpy(), report.to_numpy(), rtol=1e-12
        )

    def test_cov_df_is_pairwise_complete_like_pandas(self, frame):
        import bunker_stats.pandas as bsp

        ref = frame.replace([np.inf, -np.inf], np.nan).cov()
        got = bsp.cov_df(frame)
        np.testing.assert_array_equal(got.isna().to_numpy(), ref.isna().to_numpy())
        np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-12, atol=1e-15)

    def test_strict_mode_is_still_available(self, frame):
        import bunker_stats.pandas as bsp

        assert bsp.corr_df(frame[["a", "b"]], skipna=False).isna().all().all()
        assert bsp.cov_df(frame[["a", "b"]], skipna=False).isna().all().all()

    def test_complete_data_is_the_same_either_way(self):
        import bunker_stats.pandas as bsp

        frame = pd.DataFrame(np.random.default_rng(1).normal(size=(30, 3)), columns=list("abc"))
        np.testing.assert_allclose(
            bsp.corr_df(frame).to_numpy(), bsp.corr_df(frame, skipna=False).to_numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(bsp.corr_df(frame).to_numpy(), frame.corr().to_numpy(), rtol=1e-12)
