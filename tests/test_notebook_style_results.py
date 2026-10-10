"""Rich style/transform results, themes and input hardening for the notebook layer.

Covers what ``rich=True`` adds to the stylers and column transforms in
:mod:`bunker_stats.notebook`:

* the result protocol (``styler, frame, metadata`` unpacking, ``to_dict``,
  ``info``, ``warnings``),
* metadata values checked against hand-computed numbers,
* metadata counts checked against the cells actually painted,
* copy semantics (no helper mutates its input),
* input validation that used to produce wrong numbers or late errors,
* themes and text contrast,
* property tests for the metadata invariants.

Nothing here needs a notebook frontend: Stylers are rendered with
``Styler.to_html()`` and inspected as text.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="notebook layer requires pandas")
pytest.importorskip("jinja2", reason="pandas Styler requires jinja2")

import bunker_stats as bs
from bunker_stats import notebook as nb


# ======================================================================
# Helpers
# ======================================================================

_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}", re.S)
_CELL = re.compile(r"#T_\w+_(row\d+_col\d+)")
_BACKGROUND = re.compile(r"background-color:\s*([^;]+);")


def cell_backgrounds(styler) -> dict:
    """``{"rowR_colC": color}`` using the *last* background each cell is given.

    pandas emits every declaration applied to a cell in order, and CSS resolves
    duplicates last-wins, so the final ``background-color`` is what a browser
    paints.
    """
    block = re.search(r"<style[^>]*>(.*?)</style>", styler.to_html(), re.S)
    out: dict = {}
    if block is None:
        return out
    for selectors, body in _RULE.findall(block.group(1)):
        colors = _BACKGROUND.findall(body)
        if colors:
            for cell in _CELL.findall(selectors):
                out[cell] = colors[-1].strip()
    return out


def count_painted(styler, color: str) -> int:
    return sum(1 for c in cell_backgrounds(styler).values() if c == color)


def is_styler(obj) -> bool:
    return isinstance(obj, pd.io.formats.style.Styler)


def relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(a: str, b: str) -> float:
    hi, lo = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def small():
    """Five rows with one gross outlier; every statistic is hand-checkable.

    x = [1, 2, 3, 4, 100]: mean 22, sample sd sqrt(1902.5), median 3, MAD 1,
    Q1 2, Q3 4. The outlier's z-score is only 1.788 (it inflates the sd that
    is supposed to catch it), while its robust score is 97 / 1.4826.
    """
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 100.0],
            "y": [5.0, 3.0, 4.0, 1.0, 2.0],
            "label": list("abcde"),
        }
    )


@pytest.fixture
def gappy():
    """Twelve rows with NaN, +/-inf, a constant and an all-missing column."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 60.0],
            "y": [2.0, np.inf, 1.0, 3.0, -np.inf, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0],
            "k": [3.0] * 12,
            "void": [np.nan] * 12,
            "label": [None, "a"] * 6,
        }
    )


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "test": ["a", "b", "c", "d", "e"],
            "pvalue": [0.0005, 0.004, 0.03, 0.4, np.nan],
            "effect": [1.1, -0.6, 0.3, 0.05, np.nan],
        }
    )


def rich_makers():
    """(id, callable(df_small, df_results) -> rich result) for every helper."""
    return [
        ("outlier_style", lambda s, r: nb.outlier_style(s, rich=True)),
        ("iqr_outlier_style", lambda s, r: nb.iqr_outlier_style(s, "x", rich=True)),
        ("demean_style", lambda s, r: nb.demean_style(s, "x", rich=True)),
        ("zscore_style", lambda s, r: nb.zscore_style(s, "x", rich=True)),
        ("scale_compare_style", lambda s, r: nb.scale_compare_style(s, "x", rich=True)),
        ("corr_heatmap", lambda s, r: nb.corr_heatmap(s, rich=True)),
        ("missingness_style", lambda s, r: nb.missingness_style(s, rich=True)),
        ("style_significance", lambda s, r: nb.style_significance(r, rich=True)),
        ("style_effect_size", lambda s, r: nb.style_effect_size(r, "effect", rich=True)),
        ("scale_columns", lambda s, r: nb.scale_columns(s, rich=True)),
        ("winsorize_columns", lambda s, r: nb.winsorize_columns(s, rich=True)),
        ("robust_scale_column", lambda s, r: nb.robust_scale_column(s, "x", rich=True)),
    ]


RICH = rich_makers()
RICH_IDS = [name for name, _ in RICH]
NEEDS_MATPLOTLIB = {"corr_heatmap"}


@pytest.fixture(params=RICH, ids=RICH_IDS)
def rich_result(request, small, results):
    name, make = request.param
    if name in NEEDS_MATPLOTLIB:
        pytest.importorskip("matplotlib")
    return name, make(small, results)


# ======================================================================
# Imports and optional dependencies
# ======================================================================

class TestImports:
    def test_legacy_module_imports_without_pandas(self):
        """The regression this layer was born from: `pandas_helpers` failed at
        import. It must import -- and must not drag pandas in -- in a clean
        interpreter where pandas is unavailable."""
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'pandas' or name.startswith('pandas.'):\n"
            "            raise ImportError('pandas blocked for test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "import bunker_stats.pandas_helpers as ph\n"
            "import bunker_stats.notebook as nb\n"
            "assert 'pandas' not in sys.modules\n"
            "assert ph.zscore_style is nb.zscore_style\n"
            "try:\n"
            "    nb.zscore_style(None, 'x')\n"
            "except ImportError as exc:\n"
            "    assert 'bunker-stats-rs[notebook]' in str(exc), exc\n"
            "else:\n"
            "    raise AssertionError('expected ImportError')\n"
            "print('OK')\n"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "OK" in done.stdout

    def test_missing_jinja2_names_the_extra(self, small, monkeypatch):
        """pandas ships Styler as an optional feature; without jinja2 `df.style`
        raises. The helper must turn that into the install command."""
        def no_style(self):
            raise ImportError("Missing optional dependency 'Jinja2'.")

        monkeypatch.setattr(pd.DataFrame, "style", property(no_style))
        with pytest.raises(ImportError, match=r"jinja2.*bunker-stats-rs\[notebook\]"):
            nb.zscore_style(small, "x")

    def test_transforms_do_not_need_jinja2(self, small, monkeypatch):
        """`pip install bunker-stats-rs[pandas]` (no jinja2) must still be able
        to run a transform with rich metadata."""
        def no_style(self):
            raise ImportError("Missing optional dependency 'Jinja2'.")

        monkeypatch.setattr(pd.DataFrame, "style", property(no_style))
        result = nb.scale_columns(small, rich=True)
        frame, metadata = result
        assert "x_robust" in frame.columns and metadata["method"] == "robust"
        assert result.info() and result.to_dict()
        with pytest.raises(ImportError, match="jinja2"):
            result.styler

    def test_result_classes_are_public(self):
        for name in (
            "StyleResult", "OutlierStyleResult", "IQROutlierStyleResult",
            "DemeanStyleResult", "ZScoreStyleResult", "ScaleCompareStyleResult",
            "CorrelationHeatmapResult", "MissingnessStyleResult",
            "SignificanceStyleResult", "EffectSizeStyleResult",
            "TransformResult", "RobustScaleResult",
        ):
            assert name in nb.__all__
            assert issubclass(getattr(nb, name), nb.StyleResult)


# ======================================================================
# The result protocol, for every helper
# ======================================================================

EXPECTED_TYPE = {
    "outlier_style": "OutlierStyleResult",
    "iqr_outlier_style": "IQROutlierStyleResult",
    "demean_style": "DemeanStyleResult",
    "zscore_style": "ZScoreStyleResult",
    "scale_compare_style": "ScaleCompareStyleResult",
    "corr_heatmap": "CorrelationHeatmapResult",
    "missingness_style": "MissingnessStyleResult",
    "style_significance": "SignificanceStyleResult",
    "style_effect_size": "EffectSizeStyleResult",
    "scale_columns": "TransformResult",
    "winsorize_columns": "TransformResult",
    "robust_scale_column": "RobustScaleResult",
}


class TestResultProtocol:
    def test_type(self, rich_result):
        name, result = rich_result
        assert type(result).__name__ == EXPECTED_TYPE[name]

    def test_attributes(self, rich_result):
        _, result = rich_result
        assert is_styler(result.styler)
        assert isinstance(result.frame, pd.DataFrame)
        assert isinstance(result.metadata, dict) and result.metadata
        assert result.meta is result.metadata
        assert isinstance(result.metadata["frame_is_copy"], bool)

    def test_tuple_unpacking(self, rich_result):
        _, result = rich_result
        parts = tuple(result)
        assert len(parts) == len(result)
        if isinstance(result, nb.TransformResult):
            frame, metadata = result
        else:
            styler, frame, metadata = result
            assert styler is result.styler and result[0] is result.styler
        assert frame is result.frame and metadata is result.metadata
        assert result[-1] is result.metadata

    def test_to_dict_is_stable_and_json_safe(self, rich_result):
        _, result = rich_result
        d = result.to_dict()
        assert list(d) == ["title", "metadata", "warnings"]
        assert d["warnings"] == result.warnings
        round_tripped = json.loads(json.dumps(d))
        assert set(round_tripped["metadata"]) == set(result.metadata)

    def test_to_dict_can_include_the_frame(self, rich_result):
        _, result = rich_result
        d = result.to_dict(frame=True)
        assert set(d["frame"]) == {c for c in result.frame.columns}
        assert all(len(v) == len(result.frame) for v in d["frame"].values())

    def test_info_names_every_metadata_key(self, rich_result):
        _, result = rich_result
        text = result.info()
        assert text.splitlines()[0] == result._title
        for key in result.metadata:
            assert f"{key}:" in text
        for warning in result.warnings:
            assert f"Warning: {warning}" in text

    def test_repr_is_one_short_line(self, rich_result):
        _, result = rich_result
        text = repr(result)
        assert text.startswith(type(result).__name__ + "(")
        assert "\n" not in text and len(text) < 300

    def test_displays_as_a_table_in_jupyter(self, rich_result):
        _, result = rich_result
        assert "<table" in result._repr_html_()

    def test_warnings_are_copied_and_appendable(self, rich_result):
        _, result = rich_result
        before = result.warnings
        result.warnings.append("should not stick")
        assert result.warnings == before
        result.add_warning("custom note")
        assert "custom note" in result.warnings
        assert "Warning: custom note" in result.info()
        assert "custom note" in result.to_dict()["warnings"]


class TestRichIsOptIn:
    """`rich=True` must add context, never change what is drawn or computed."""

    @pytest.mark.parametrize("name, make", RICH[:9], ids=RICH_IDS[:9])
    def test_rich_styler_matches_plain_styler(self, name, make, small, results):
        if name in NEEDS_MATPLOTLIB:
            pytest.importorskip("matplotlib")
        rich = make(small, results)
        plain = getattr(nb, name)
        if name == "outlier_style":
            styler = plain(small)
        elif name in ("corr_heatmap", "missingness_style"):
            styler = plain(small)
        elif name == "style_significance":
            styler = plain(results)
        elif name == "style_effect_size":
            styler = plain(results, "effect")
        else:
            styler = plain(small, "x")
        assert is_styler(styler)
        assert styler.set_uuid("t").to_html() == rich.styler.set_uuid("t").to_html()

    def test_rich_transform_matches_plain_frame(self, small):
        pd.testing.assert_frame_equal(
            nb.scale_columns(small, rich=True).frame, nb.scale_columns(small)
        )
        pd.testing.assert_frame_equal(
            nb.winsorize_columns(small, rich=True).frame, nb.winsorize_columns(small)
        )
        pd.testing.assert_frame_equal(
            nb.robust_scale_column(small, "x", rich=True).frame,
            nb.robust_scale_column(small, "x"),
        )


# ======================================================================
# Metadata values on a hand-checkable frame
# ======================================================================

class TestDemeanMetadata:
    def test_values(self, small):
        m = nb.demean_style(small, "x", rich=True).metadata
        assert m["source_column"] == "x" and m["derived_column"] == "x_demeaned"
        assert (m["n"], m["n_missing"]) == (5, 0)
        assert m["mean"] == pytest.approx(22.0)
        assert (m["n_above"], m["n_below"], m["n_at_mean"]) == (1, 4, 0)
        assert m["frame_is_copy"] is True and m["overwrote_existing_column"] is False

    def test_counts_exact_ties_with_the_mean(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        m = nb.demean_style(df, "x", rich=True).metadata
        assert (m["n_above"], m["n_below"], m["n_at_mean"]) == (1, 1, 1)

    def test_counts_match_painted_cells(self, gappy):
        result = nb.demean_style(gappy, "x", rich=True)
        colors = result.metadata["theme"]["colors"]
        assert count_painted(result.styler, colors["above"]) == result.metadata["n_above"]
        assert count_painted(result.styler, colors["below"]) == result.metadata["n_below"]
        assert result.metadata["n_missing"] == 1

    def test_all_missing_column(self, gappy):
        result = nb.demean_style(gappy, "void", rich=True)
        assert result.metadata["n"] == 0 and math.isnan(result.metadata["mean"])
        assert any("no finite values" in w for w in result.warnings)
        assert cell_backgrounds(result.styler) == {}


class TestZScoreMetadata:
    def test_values(self, small):
        m = nb.zscore_style(small, "x", threshold=1.5, rich=True).metadata
        assert m["mean"] == pytest.approx(22.0)
        assert m["std"] == pytest.approx(math.sqrt(1902.5))
        assert m["ddof"] == 1 and m["threshold"] == 1.5
        assert (m["n_high"], m["n_low"], m["n_flagged"]) == (1, 0, 1)
        assert m["pct_flagged"] == pytest.approx(20.0)
        assert m["std"] == pytest.approx(bs.std(small["x"].to_numpy()))

    def test_expected_share_under_normality(self, small):
        m = nb.zscore_style(small, "x", threshold=1.959964, rich=True).metadata
        assert m["expected_pct_if_normal"] == pytest.approx(5.0, abs=1e-4)

    def test_masking_ceiling_is_reported(self, small):
        """With n=5 no z-score can exceed 4/sqrt(5) = 1.789, so the gross
        outlier is not flagged at the default threshold of 2."""
        result = nb.zscore_style(small, "x", rich=True)
        assert result.metadata["n_flagged"] == 0
        assert result.frame["x_zscore"].abs().max() < 4 / math.sqrt(5) + 1e-12
        assert any("no z-score can reach 2" in w for w in result.warnings)

    def test_no_ceiling_warning_when_reachable(self, small):
        assert nb.zscore_style(small, "x", threshold=1.5, rich=True).warnings == []

    def test_constant_column(self, gappy):
        result = nb.zscore_style(gappy, "k", rich=True)
        assert result.metadata["std"] == 0.0 and result.metadata["n_flagged"] == 0
        assert any("zero variance" in w for w in result.warnings)
        assert result.frame["k_zscore"].isna().all()

    def test_counts_match_painted_cells(self, gappy):
        result = nb.zscore_style(gappy, "x", threshold=2.0, rich=True)
        colors = result.metadata["theme"]["colors"]
        assert count_painted(result.styler, colors["high"]) == result.metadata["n_high"] == 1
        assert count_painted(result.styler, colors["low"]) == result.metadata["n_low"]
        assert result.metadata["n"] + result.metadata["n_missing"] == len(gappy)


class TestIQRMetadata:
    def test_values(self, small):
        m = nb.iqr_outlier_style(small, "x", rich=True).metadata
        assert (m["q1"], m["q3"], m["iqr"], m["k"]) == (2.0, 4.0, 2.0, 1.5)
        assert (m["lower_fence"], m["upper_fence"]) == (-1.0, 7.0)
        assert (m["n_outliers"], m["n_low"], m["n_high"]) == (1, 0, 1)
        assert (m["n"], m["n_missing"]) == (5, 0)
        assert m["frame_is_copy"] is False

    def test_quartiles_match_numpy_linear(self, gappy):
        m = nb.iqr_outlier_style(gappy, "x", rich=True).metadata
        finite = gappy["x"].dropna().to_numpy()
        assert m["q1"] == pytest.approx(np.percentile(finite, 25))
        assert m["q3"] == pytest.approx(np.percentile(finite, 75))
        assert m["quantile_method"] == "linear interpolation"

    def test_expected_share_is_the_textbook_figure(self, small):
        """Tukey's 1.5 x IQR fences flag about 0.7% of a normal sample."""
        m = nb.iqr_outlier_style(small, "x", rich=True).metadata
        assert m["expected_pct_if_normal"] == pytest.approx(0.698, abs=2e-3)

    def test_agrees_with_outlier_report(self, gappy):
        report = nb.outlier_report(gappy, ["x"], method="iqr", k=1.5).loc["x"]
        result = nb.iqr_outlier_style(gappy, "x", rich=True)
        m = result.metadata
        assert m["n_outliers"] == report["n_outliers"]
        assert m["lower_fence"] == pytest.approx(report["lower_bound"])
        assert m["upper_fence"] == pytest.approx(report["upper_bound"])
        assert count_painted(result.styler, m["theme"]["colors"]["outlier"]) == m["n_outliers"]


class TestOutlierStyleMetadata:
    @pytest.mark.parametrize("method", ["iqr", "zscore", "robust_zscore"])
    def test_per_column_agrees_with_report(self, gappy, method):
        report = nb.outlier_report(gappy, ["x", "y", "k"], method=method)
        result = nb.outlier_style(gappy, ["x", "y", "k"], method=method, rich=True)
        for col, detail in result.metadata["per_column"].items():
            assert detail["n_outliers"] == report.loc[col, "n_outliers"]
            assert detail["n_low"] + detail["n_high"] == detail["n_outliers"]
            assert detail["n"] == report.loc[col, "n"]
        total = result.metadata["n_outliers_total"]
        assert total == report["n_outliers"].sum()
        assert count_painted(result.styler, "#ff8a80") == total

    def test_unused_parameter_is_none(self, small):
        assert nb.outlier_style(small, method="iqr", rich=True).metadata["z_threshold"] is None
        assert nb.outlier_style(small, method="zscore", rich=True).metadata["k"] is None

    def test_zscore_method_carries_the_masking_caveat(self, small):
        assert "masking" in nb.outlier_style(small, method="zscore", rich=True).metadata["caveat"]
        assert "masking" not in nb.outlier_style(small, method="iqr", rich=True).metadata["caveat"]

    def test_zero_spread_warning(self, gappy):
        result = nb.outlier_style(gappy, ["k"], rich=True)
        assert any("zero spread" in w for w in result.warnings)


class TestScaleCompare:
    def test_exposes_masking(self, small):
        result = nb.scale_compare_style(small, "x", rich=True)
        m = result.metadata
        assert (m["mean"], m["median"], m["mad"]) == (pytest.approx(22.0), 3.0, 1.0)
        assert m["robust_scale"] == pytest.approx(1.4826)
        assert (m["n_flagged_classical"], m["n_flagged_robust"]) == (0, 1)
        assert (m["n_flagged_both"], m["n_robust_only"], m["n_classical_only"]) == (0, 1, 0)
        assert any("masking" in w for w in result.warnings)
        assert result.frame["x_robust_z"].iloc[-1] == pytest.approx(97 / 1.4826)

    def test_columns_match_the_kernels(self, gappy):
        frame = nb.scale_compare_style(gappy, "x", rich=True).frame
        finite = gappy["x"].dropna().to_numpy()
        np.testing.assert_allclose(frame["x_zscore"].dropna(), bs.zscore(finite))
        np.testing.assert_allclose(
            frame["x_robust_z"].dropna(), bs.robust_scale(finite, 1.4826)[0]
        )
        assert frame["x_zscore"].isna().sum() == 1

    def test_zero_mad_is_nan_not_huge(self):
        """MAD = 0 here; the raw kernel would return ~4e12 for the last value."""
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 5.0]})
        assert abs(bs.robust_scale(df["x"].to_numpy(), 1.4826)[0][-1]) > 1e9
        result = nb.scale_compare_style(df, "x", rich=True)
        assert result.metadata["zero_mad"] is True
        assert result.frame["x_robust_z"].isna().all()
        assert result.metadata["n_flagged_robust"] == 0
        assert any("MAD is 0" in w for w in result.warnings)

    def test_custom_suffixes_and_validation(self, small):
        frame = nb.scale_compare_style(small, "x", suffixes=("_z", "_rz"), rich=True).frame
        assert {"x_z", "x_rz"} <= set(frame.columns)
        with pytest.raises(ValueError, match="two different strings"):
            nb.scale_compare_style(small, "x", suffixes=("_z", "_z"))
        with pytest.raises(ValueError, match="non-empty string"):
            nb.scale_compare_style(small, "x", suffixes=("", "_rz"))


class TestCorrelationHeatmap:
    @pytest.fixture(autouse=True)
    def _needs_matplotlib(self):
        pytest.importorskip("matplotlib")

    def test_values(self, small):
        result = nb.corr_heatmap(small, rich=True)
        m = result.metadata
        r = np.corrcoef(small["x"], small["y"])[0, 1]
        assert m["columns"] == ["x", "y"] and m["shape"] == (2, 2)
        assert m["method"] == "pearson" and m["n_rows"] == 5
        assert m["min_correlation"] == pytest.approx(r)
        assert m["max_offdiag_correlation"] == pytest.approx(r)
        assert m["max_abs_offdiag_correlation"] == pytest.approx(abs(r))
        assert m["strongest_pairs"] == [
            {"x": "x", "y": "y", "correlation": pytest.approx(r), "n": 5}
        ]
        assert (m["pairwise_n_min"], m["pairwise_n_max"]) == (5, 5)
        assert result.warnings == [
            "small overlap (min pairwise n = 5): correlations from few rows are noisy"
        ]
        pd.testing.assert_frame_equal(result.frame, nb.correlation_report(small))

    def test_no_warnings_on_a_healthy_frame(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
        assert nb.corr_heatmap(df, rich=True).warnings == []

    def test_strongest_pairs_are_sorted_by_magnitude(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=200)
        df = pd.DataFrame(
            {
                "a": base,
                "b": -base + rng.normal(scale=0.1, size=200),
                "c": base + rng.normal(scale=1.0, size=200),
                "d": rng.normal(size=200),
            }
        )
        pairs = nb.corr_heatmap(df, rich=True).metadata["strongest_pairs"]
        magnitudes = [abs(p["correlation"]) for p in pairs]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert (pairs[0]["x"], pairs[0]["y"]) == ("a", "b") and pairs[0]["correlation"] < -0.9
        assert len(pairs) == 5  # capped below the 6 available pairs

    def test_degenerate_columns_are_reported(self, gappy):
        result = nb.corr_heatmap(gappy, rich=True)
        m = result.metadata
        assert m["constant_columns"] == ["k"] and m["all_missing_columns"] == ["void"]
        assert m["n_undefined_pairs"] == 5 and m["n_pairs"] == 6
        text = " | ".join(result.warnings)
        assert "constant column(s): k" in text and "no finite values" in text
        assert "pairwise deletion" in text

    def test_undefined_cells_use_the_missing_color_not_black(self, gappy):
        """matplotlib maps NaN to a 'bad' color that pandas renders as solid
        black -- which reads as the strongest possible correlation."""
        result = nb.corr_heatmap(gappy, rich=True)
        painted = cell_backgrounds(result.styler)
        matrix = result.frame.to_numpy()
        missing = result.metadata["theme"]["colors"]["missing"]
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = painted[f"row{i}_col{j}"]
                if np.isnan(matrix[i, j]):
                    assert color == missing
                else:
                    assert color not in (missing, "#000000")

    def test_spearman_and_cmap_are_recorded(self, small):
        m = nb.corr_heatmap(small, method="spearman", cmap="PuOr", rich=True).metadata
        assert m["method"] == "spearman" and m["theme"]["cmap"] == "PuOr"
        assert (m["theme"]["vmin"], m["theme"]["vmax"]) == (-1.0, 1.0)


class TestMissingnessStyle:
    def test_values(self, gappy):
        result = nb.missingness_style(gappy, rich=True)
        m = result.metadata
        assert m["n_rows"] == 12 and m["n_cells"] == 60
        assert m["n_missing_cells"] == 1 + 12 + 6
        assert m["n_infinite_cells"] == 2
        assert m["n_complete_rows"] == 0 and m["n_incomplete_rows"] == 12
        assert m["per_column"]["y"] == {
            "dtype": "float64", "n_missing": 0, "n_infinite": 2, "pct_missing": 0.0,
        }
        assert m["per_column"]["label"]["n_missing"] == 6
        text = " | ".join(result.warnings)
        assert "entirely missing column(s): void" in text and "no row is complete" in text

    def test_counts_match_painted_cells_and_report(self, gappy):
        result = nb.missingness_style(gappy, rich=True)
        colors = result.metadata["theme"]["colors"]
        assert count_painted(result.styler, colors["missing"]) == result.metadata["n_missing_cells"]
        assert count_painted(result.styler, colors["infinite"]) == result.metadata["n_infinite_cells"]
        report = nb.missingness_report(gappy)
        for col, detail in result.metadata["per_column"].items():
            assert detail["n_missing"] == report.loc[col, "n_missing"]

    def test_column_subset_and_complete_rows(self, gappy):
        m = nb.missingness_style(gappy, ["x", "y"], rich=True).metadata
        assert m["columns"] == ["x", "y"] and m["n_cells"] == 24
        assert m["n_complete_rows"] == 9 and m["n_incomplete_rows"] == 3

    def test_clean_frame_paints_nothing(self, small):
        result = nb.missingness_style(small, rich=True)
        assert cell_backgrounds(result.styler) == {} and result.warnings == []
        assert result.frame is small


class TestSignificanceAndEffect:
    def test_significance_values(self, results):
        result = nb.style_significance(results, rich=True)
        m = result.metadata
        assert (m["n_tests"], m["n_missing"], m["n_significant"]) == (4, 1, 3)
        assert list(m["n_per_tier"].values()) == [1, 1, 1, 1]
        assert m["tier_cutoffs"] == pytest.approx([0.001, 0.01, 0.05])
        assert m["expected_false_positives_if_all_null"] == pytest.approx(0.2)
        assert "none" in m["multiple_comparison_correction"]
        assert any("no multiple-comparison correction" in w for w in result.warnings)

    def test_single_test_has_no_multiplicity_warning(self):
        result = nb.style_significance(pd.DataFrame({"pvalue": [0.01]}), rich=True)
        assert result.warnings == []

    def test_adjusted_pvalues_shade_fewer_rows(self):
        """The documented remedy: adjust first, then style the adjusted column."""
        rng = np.random.default_rng(11)
        table = pd.DataFrame({"pvalue": rng.uniform(size=200)})
        table["p_adj"] = np.asarray(bs.p_adjust(table["pvalue"].to_numpy(), "holm"))
        raw = nb.style_significance(table, rich=True).metadata["n_significant"]
        adjusted = nb.style_significance(
            table, pvalue_column="p_adj", rich=True
        ).metadata["n_significant"]
        assert raw > 0 and adjusted <= raw

    def test_documented_adjust_recipe_survives_nan_pvalues(self):
        """`correlation_report(pvalues=True)` emits NaN for undefined pairs and
        `p_adjust` rejects NaN, so the documented recipe adjusts the finite
        values only. This runs that recipe on the layer's own output."""
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {"a": rng.normal(size=20), "b": rng.normal(size=20), "k": 1.0}
        )
        table = nb.correlation_report(frame, pvalues=True)
        assert table["pvalue"].isna().sum() == 2
        with pytest.raises(ValueError, match="NaN"):
            bs.p_adjust(table["pvalue"].to_numpy(), "holm")

        p = table["pvalue"].to_numpy(dtype=float)
        ok = np.isfinite(p)
        table["p_adj"] = np.nan
        table.loc[ok, "p_adj"] = bs.p_adjust(p[ok], "holm")

        result = nb.style_significance(table, pvalue_column="p_adj", rich=True)
        assert (result.metadata["n_tests"], result.metadata["n_missing"]) == (1, 2)
        assert "<table" in result.styler.to_html()

    @pytest.mark.parametrize(
        "alpha", [0.05, np.float64(0.05), np.float32(0.05), np.float16(0.05)]
    )
    def test_alpha_accepts_numpy_scalars(self, results, alpha):
        result = nb.style_significance(results, alpha=alpha, rich=True)
        assert isinstance(result.metadata["alpha"], float)
        assert result.metadata["n_significant"] == 3

    @pytest.mark.parametrize("alpha", [0, 1, 1.5, -0.1, True, "0.05", None])
    def test_alpha_out_of_range_or_wrong_type(self, results, alpha):
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
            nb.style_significance(results, alpha=alpha)

    def test_rich_coverage_matches_the_docs(self):
        """The docs name the helpers without a `rich` option; keep them honest."""
        import inspect

        without = {
            name for name in nb.__all__
            if inspect.isfunction(getattr(nb, name))
            and "rich" not in inspect.signature(getattr(nb, name)).parameters
        }
        assert without == {
            "describe_fast", "normality_report", "missingness_report",
            "rolling_report", "available_themes",
        }

    @pytest.mark.parametrize("bad", [-0.01, 1.5])
    def test_out_of_range_pvalues_are_rejected(self, bad):
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            nb.style_significance(pd.DataFrame({"pvalue": [0.2, bad]}))

    def test_row_and_cell_highlighting_paint_the_same_tiers(self, results):
        rows = nb.style_significance(results, highlight_row=True)
        cells = nb.style_significance(results, highlight_row=False)
        assert len(cell_backgrounds(rows)) == 4 * results.shape[1]
        assert len(cell_backgrounds(cells)) == 4

    def test_effect_values(self, results):
        result = nb.style_effect_size(results, "effect", rich=True)
        m = result.metadata
        assert (m["n"], m["n_missing"]) == (4, 1)
        assert m["n_per_bucket"] == {"negligible": 1, "small": 1, "medium": 1, "large": 1}
        assert m["thresholds"] == [0.2, 0.5, 0.8]
        palette = m["theme"]["colors"]["effect"]
        assert [count_painted(result.styler, c) for c in palette] == [1, 1, 1, 1]


class TestTransformMetadata:
    def test_robust_scale_column(self, small):
        result = nb.robust_scale_column(small, "x", rich=True)
        frame, m = result
        scaled, median, mad = bs.robust_scale(small["x"].to_numpy(), 1.4826)
        assert (m["median"], m["mad"]) == (median, mad) == (3.0, 1.0)
        assert m["scale"] == pytest.approx(1.4826) and m["scale_factor"] == 1.4826
        assert m["source_column"] == "x" and m["derived_column"] == "x_robust"
        assert m["zero_mad"] is False and result.warnings == []
        np.testing.assert_allclose(frame["x_robust"], scaled)

    def test_zero_mad_is_flagged_honestly(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 5.0]})
        result = nb.robust_scale_column(df, "x", rich=True)
        assert result.metadata["zero_mad"] is True
        assert "not a usable robust score" in result.metadata["zero_mad_behavior"]
        assert any("MAD is 0" in w for w in result.warnings)

    @pytest.mark.parametrize(
        "method, keys",
        [
            ("robust", {"median", "mad", "scale", "zero_mad"}),
            ("zscore", {"mean", "std", "ddof"}),
            ("minmax", {"min", "max", "range"}),
        ],
    )
    def test_scale_columns_parameters_reproduce_the_output(self, small, method, keys):
        result = nb.scale_columns(small, ["x"], method=method, rich=True)
        detail = result.metadata["per_column"]["x"]
        assert keys <= set(detail)
        x = small["x"].to_numpy()
        if method == "robust":
            rebuilt = (x - detail["median"]) / detail["scale"]
        elif method == "zscore":
            rebuilt = (x - detail["mean"]) / detail["std"]
        else:
            rebuilt = (x - detail["min"]) / detail["range"]
        np.testing.assert_allclose(result.frame[detail["derived_column"]], rebuilt)

    def test_scale_columns_degenerate_warnings(self, gappy):
        z = nb.scale_columns(gappy, ["k", "void"], method="zscore", rich=True)
        assert any("zero variance in column(s): k" in w for w in z.warnings)
        assert any("no finite values in column(s): void" in w for w in z.warnings)
        mm = nb.scale_columns(gappy, ["k"], method="minmax", rich=True)
        assert any("zero range" in w for w in mm.warnings)

    def test_winsorize_counts_and_bounds(self):
        df = pd.DataFrame({"x": np.arange(1.0, 21.0)})
        result = nb.winsorize_columns(df, lower_q=0.1, upper_q=0.9, rich=True)
        detail = result.metadata["per_column"]["x"]
        clipped = result.frame["x_winsor"].to_numpy()
        assert detail["n_clipped_low"] == int((clipped > df["x"]).sum()) == 2
        assert detail["n_clipped_high"] == int((clipped < df["x"]).sum()) == 2
        assert detail["lower_bound"] == clipped.min()
        assert detail["upper_bound"] == clipped.max()
        assert detail["pct_clipped"] == pytest.approx(20.0)
        assert (result.metadata["lower_q"], result.metadata["upper_q"]) == (0.1, 0.9)


# ======================================================================
# Copy semantics
# ======================================================================

ADDS_COLUMNS = {
    "demean_style", "zscore_style", "scale_compare_style",
    "scale_columns", "winsorize_columns", "robust_scale_column",
}
BUILDS_NEW_TABLE = {"corr_heatmap"}


class TestCopySemantics:
    def test_input_is_never_mutated(self, rich_result, small, results):
        """The fixtures were handed to the helper; compare against fresh ones."""
        fresh_small = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, 100.0], "y": [5.0, 3.0, 4.0, 1.0, 2.0],
             "label": list("abcde")}
        )
        pd.testing.assert_frame_equal(small, fresh_small)
        assert list(results.columns) == ["test", "pvalue", "effect"]

    def test_frame_identity_matches_the_metadata(self, rich_result, small, results):
        name, result = rich_result
        is_input = result.frame is small or result.frame is results
        assert result.metadata["frame_is_copy"] is (not is_input)
        assert is_input is (name not in ADDS_COLUMNS | BUILDS_NEW_TABLE)

    def test_derived_frame_keeps_index_and_source_columns(self, gappy):
        df = gappy.set_index(pd.Index(list("abcdefghijkl"), name="row"))
        for result in (
            nb.zscore_style(df, "x", rich=True),
            nb.demean_style(df, "x", rich=True),
            nb.scale_compare_style(df, "x", rich=True),
            nb.scale_columns(df, ["x"], rich=True),
            nb.winsorize_columns(df, ["x"], rich=True),
        ):
            assert result.frame.index.equals(df.index)
            pd.testing.assert_frame_equal(result.frame[df.columns], df)

    def test_writing_to_the_result_does_not_reach_the_input(self, small):
        result = nb.zscore_style(small, "x", rich=True)
        result.frame.loc[0, "x"] = -999.0
        assert small.loc[0, "x"] == 1.0

    def test_rerunning_a_cell_overwrites_and_says_so(self, small):
        """Re-running `df = helper(df)` is normal in a notebook, so a name
        clash replaces the old derived column instead of raising -- but the
        rich result reports it."""
        once = nb.zscore_style(small, "x", rich=True)
        twice = nb.zscore_style(once.frame, "x", rich=True)
        assert once.metadata["overwrote_existing_column"] is False
        assert twice.metadata["overwrote_existing_column"] is True
        assert any("already existed" in w for w in twice.warnings)
        assert list(twice.frame.columns) == list(once.frame.columns)
        scaled = nb.scale_columns(nb.scale_columns(small, ["x"]), ["x"], rich=True)
        assert scaled.metadata["overwrote_existing_column"] is True

    def test_replace_overwrites_only_in_the_copy(self, small):
        result = nb.scale_columns(small, ["x"], method="zscore", replace=True, rich=True)
        assert list(result.frame.columns) == list(small.columns)
        assert result.metadata["overwrote_existing_column"] is False
        assert small["x"].tolist() == [1.0, 2.0, 3.0, 4.0, 100.0]

    @pytest.mark.parametrize("fn", [nb.demean_style, nb.zscore_style])
    def test_empty_suffix_cannot_clobber_the_source(self, fn, small):
        with pytest.raises(ValueError, match="non-empty string"):
            fn(small, "x", add_suffix="")


# ======================================================================
# Input hardening
# ======================================================================

class TestInputHardening:
    def test_duplicate_column_labels_are_rejected_not_pooled(self):
        """`df['a']` is 2-D here; this used to pool both columns into one
        sample and report n=6 for a 3-row frame."""
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]], columns=["a", "a"])
        for call in (
            lambda: nb.robust_summary(df),
            lambda: nb.robust_summary(df, ["a"]),
            lambda: nb.outlier_report(df, ["a"]),
            lambda: nb.scale_columns(df),
            lambda: nb.zscore_style(df, "a"),
        ):
            with pytest.raises(ValueError, match="more than one column"):
                call()

    def test_missingness_report_survives_duplicate_labels(self):
        df = pd.DataFrame([[1.0, np.nan], [3.0, 4.0]], columns=["a", "a"])
        assert nb.missingness_report(df)["n_missing"].tolist() == [0, 1]

    def test_duplicate_index_fails_at_the_call_not_at_render(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 40.0]}, index=[0, 0, 1, 2])
        for call in (
            lambda: nb.zscore_style(df, "x"),
            lambda: nb.demean_style(df, "x"),
            lambda: nb.outlier_style(df),
            lambda: nb.missingness_style(df),
            lambda: nb.scale_compare_style(df, "x"),
            lambda: nb.style_effect_size(df, "x"),
        ):
            with pytest.raises(ValueError, match="unique index"):
                call()
        # Reports and transforms do not render, so they stay usable.
        assert nb.robust_summary(df).loc["x", "n"] == 4
        assert len(nb.scale_columns(df)) == 4

    def test_complex_columns_are_rejected_not_truncated(self):
        df = pd.DataFrame({"c": np.array([1 + 2j, 2, 3, 4, 5]), "x": np.arange(5.0)})
        with pytest.raises(TypeError, match="not numeric"):
            nb.zscore_style(df, "c")
        assert list(nb.robust_summary(df).index) == ["x"]

    def test_auto_selection_skips_timedelta_and_bool(self):
        """`select_dtypes(np.number)` returns timedelta64 columns, which then
        fail the float cast; auto-selection must skip them."""
        df = pd.DataFrame(
            {
                "x": np.arange(5.0),
                "td": pd.to_timedelta(range(5), unit="s"),
                "flag": [True, False, True, True, False],
                "when": pd.date_range("2024-01-01", periods=5),
            }
        )
        assert list(nb.robust_summary(df).index) == ["x"]
        assert nb.outlier_style(df, rich=True).metadata["columns"] == ["x"]

    @pytest.mark.parametrize("dtype", ["Int64", "Float64", "Int8"])
    def test_nullable_dtypes_count_na_as_missing(self, dtype):
        df = pd.DataFrame({"x": pd.array([1, 2, None, 4, 50, 6, 7], dtype=dtype)})
        z = nb.zscore_style(df, "x", rich=True)
        assert (z.metadata["n"], z.metadata["n_missing"]) == (6, 1)
        assert np.isnan(z.frame["x_zscore"].iloc[2])
        assert df["x"].dtype.name == dtype
        iqr = nb.iqr_outlier_style(df, "x", rich=True)
        assert iqr.metadata["n_high"] == 1

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
    def test_thresholds_must_be_positive_and_finite(self, small, bad):
        for call in (
            lambda: nb.zscore_style(small, "x", threshold=bad),
            lambda: nb.outlier_style(small, k=bad),
            lambda: nb.outlier_style(small, method="zscore", z_threshold=bad),
            lambda: nb.outlier_report(small, k=bad),
            lambda: nb.iqr_outlier_style(small, "x", k=bad),
            lambda: nb.scale_compare_style(small, "x", threshold=bad),
        ):
            with pytest.raises(ValueError, match="must be > 0"):
                call()

    @pytest.mark.parametrize("bad", ["2", None, True])
    def test_threshold_type_errors_are_clear(self, small, bad):
        with pytest.raises(TypeError, match="threshold must be a number"):
            nb.zscore_style(small, "x", threshold=bad)

    def test_empty_frame_renders(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        for result in (
            nb.zscore_style(df, "x", rich=True),
            nb.demean_style(df, "x", rich=True),
            nb.iqr_outlier_style(df, "x", rich=True),
            nb.scale_compare_style(df, "x", rich=True),
            nb.missingness_style(df, rich=True),
        ):
            assert "<table" in result.styler.to_html()
            assert result.metadata.get("n", 0) == 0
            json.dumps(result.to_dict())

    def test_one_row_frame_renders(self):
        """pandas 3 infers its `str` dtype for an all-string CSS frame, and a
        length-1 `.loc` assignment into it raises. Every styler must still
        handle a single-row DataFrame."""
        df = pd.DataFrame({"x": [0.0], "pvalue": [0.01], "label": ["a"]})
        stylers = [
            nb.zscore_style(df, "x"),
            nb.demean_style(df, "x"),
            nb.iqr_outlier_style(df, "x"),
            nb.outlier_style(df, ["x"]),
            nb.scale_compare_style(df, "x"),
            nb.missingness_style(df),
            nb.style_significance(df, highlight_row=False),
            nb.style_significance(df, highlight_row=True),
            nb.style_effect_size(df, "x"),
        ]
        for styler in stylers:
            assert "<table" in styler.to_html()
        assert count_painted(nb.demean_style(df, "x"), "#e0e0e0") == 1

    def test_non_string_and_multiindex_labels(self):
        ints = pd.DataFrame({0: [1.0, 2.0, 3.0, 40.0]})
        assert nb.zscore_style(ints, 0, rich=True).metadata["derived_column"] == "0_zscore"
        multi = pd.DataFrame({("g", "a"): [1.0, 2.0, 3.0, 40.0], ("g", "b"): [1.0, 2.0, 3.0, 4.0]})
        assert nb.outlier_style(multi, rich=True).metadata["columns"] == [("g", "a"), ("g", "b")]
        with pytest.raises((KeyError, ValueError)):
            nb.outlier_style(multi, ["g"])


# ======================================================================
# Themes
# ======================================================================

class TestThemes:
    def test_available_themes_is_a_copy(self):
        themes = nb.available_themes()
        assert {"default", "colorblind"} <= set(themes)
        themes["default"]["outlier"] = "#000000"
        assert nb.available_themes()["default"]["outlier"] != "#000000"

    def test_themes_share_one_set_of_roles(self):
        themes = nb.available_themes()
        assert all(set(t) == set(themes["default"]) for t in themes.values())

    @pytest.mark.parametrize("theme", sorted(nb.available_themes()))
    def test_text_contrast_meets_wcag_aa(self, theme):
        """Every background must keep 4.5:1 against the theme's text color."""
        roles = nb.available_themes()[theme]
        text = roles["text"]
        for role, value in roles.items():
            if role == "text":
                continue
            for color in (value if isinstance(value, tuple) else (value,)):
                assert contrast_ratio(color, text) >= 4.5, (theme, role, color)

    def test_every_styled_cell_sets_a_text_color(self, gappy, results):
        """A background without a text color is unreadable in dark notebooks."""
        stylers = [
            nb.zscore_style(gappy, "x"), nb.demean_style(gappy, "x"),
            nb.outlier_style(gappy), nb.missingness_style(gappy),
            nb.scale_compare_style(gappy, "x"),
            nb.style_significance(results), nb.style_effect_size(results, "effect"),
        ]
        for styler in stylers:
            block = re.search(r"<style[^>]*>(.*?)</style>", styler.to_html(), re.S).group(1)
            bodies = [body for _, body in _RULE.findall(block) if "background-color" in body]
            assert bodies and all(re.search(r"(?<!-)color:\s*#1a1a1a", b) for b in bodies)

    def test_default_theme_keeps_the_original_colors(self, gappy):
        html = nb.zscore_style(gappy, "x").to_html()
        assert "#ffcc80" in html
        assert count_painted(nb.outlier_style(gappy, ["x"]), "#ff8a80") == 1

    def test_named_theme_changes_the_paint(self, gappy):
        result = nb.outlier_style(gappy, ["x"], theme="colorblind", rich=True)
        expected = nb.available_themes()["colorblind"]["outlier"]
        assert result.metadata["theme"] == {
            "name": "colorblind", "colors": {"text": "#1a1a1a", "outlier": expected},
        }
        assert count_painted(result.styler, expected) == 1
        assert count_painted(result.styler, "#ff8a80") == 0

    def test_colorblind_theme_avoids_red_green_for_above_below(self):
        roles = nb.available_themes()["colorblind"]
        def channels(c):
            return [int(c[i:i + 2], 16) for i in (1, 3, 5)]
        above, below = channels(roles["above"]), channels(roles["below"])
        # blue-dominant vs red/orange-dominant: separable without red-green vision
        assert above[2] > above[0] and below[0] > below[2]

    def test_explicit_color_beats_the_theme(self, gappy):
        result = nb.outlier_style(
            gappy, ["x"], theme="colorblind", outlier_color="#ffd54f", rich=True
        )
        assert result.metadata["theme"]["name"] == "custom"
        assert count_painted(result.styler, "#ffd54f") == 1

    def test_mapping_theme_overrides_single_roles(self, small):
        result = nb.demean_style(small, "x", theme={"above": "#123456"}, rich=True)
        colors = result.metadata["theme"]["colors"]
        assert colors["above"] == "#123456"
        assert colors["below"] == nb.available_themes()["default"]["below"]
        assert count_painted(result.styler, "#123456") == 1

    def test_legacy_color_keywords_still_work(self, small):
        styler = nb.zscore_style(
            small, "x", threshold=1.5, high_color="#aa0000", low_color="#0000aa",
            zero_color="#eeeeee",
        )
        assert count_painted(styler, "#aa0000") == 1
        assert count_painted(styler, "#eeeeee") == 4

    def test_bad_themes_are_rejected(self, small):
        with pytest.raises(ValueError, match="Unknown theme 'neon'"):
            nb.outlier_style(small, theme="neon")
        with pytest.raises(ValueError, match="Unknown theme role"):
            nb.outlier_style(small, theme={"outliers": "#fff"})
        with pytest.raises(TypeError, match="theme must be"):
            nb.outlier_style(small, theme=3)
        with pytest.raises(ValueError, match="needs 3 colors"):
            nb.style_significance(pd.DataFrame({"pvalue": [0.1]}), theme={"significance": ("#fff",)})

    @pytest.mark.parametrize("color", ["red; } body { display: none", "", "<script>", 7])
    def test_colors_cannot_break_out_of_the_declaration(self, small, color):
        with pytest.raises(ValueError, match="Invalid CSS color"):
            nb.outlier_style(small, outlier_color=color)


# ======================================================================
# Property tests: metadata invariants
# ======================================================================

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

_values = st.lists(
    st.one_of(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.sampled_from([np.nan, np.inf, -np.inf, 0.0, 1.0]),
    ),
    min_size=0,
    max_size=40,
)
_thresholds = st.floats(min_value=0.1, max_value=6.0, allow_nan=False)
PROPERTY = settings(max_examples=60, deadline=None)


def _frame(values) -> "pd.DataFrame":
    index = pd.Index([f"r{i}" for i in range(len(values))], name="row")
    return pd.DataFrame({"x": np.asarray(values, dtype=float)}, index=index)


def _same(a: "pd.DataFrame", b: "pd.DataFrame") -> bool:
    return a.index.equals(b.index) and np.array_equal(
        a["x"].to_numpy(), b["x"].to_numpy(), equal_nan=True
    )


class TestProperties:
    @PROPERTY
    @given(_values, _thresholds)
    def test_zscore_counts_are_consistent(self, values, threshold):
        df = _frame(values)
        before = df.copy(deep=True)
        result = nb.zscore_style(df, "x", threshold=threshold, rich=True)
        m, z = result.metadata, result.frame["x_zscore"].to_numpy()

        assert _same(df, before)
        assert len(result.frame) == len(df) and result.frame.index.equals(df.index)
        assert m["n"] + m["n_missing"] == len(df)
        assert m["n"] == int(np.isfinite(df["x"]).sum())
        with np.errstate(invalid="ignore"):
            assert m["n_high"] == int((z >= threshold).sum())
            assert m["n_low"] == int((z <= -threshold).sum())
        assert m["n_flagged"] == m["n_high"] + m["n_low"] <= m["n"]
        assert not np.isfinite(z[~np.isfinite(df["x"].to_numpy())]).any()

    @PROPERTY
    @given(_values, _thresholds, _thresholds)
    def test_zscore_flags_shrink_as_the_threshold_grows(self, values, a, b):
        lo, hi = sorted((a, b))
        df = _frame(values)
        loose = nb.zscore_style(df, "x", threshold=lo, rich=True).metadata
        strict = nb.zscore_style(df, "x", threshold=hi, rich=True).metadata
        assert strict["n_flagged"] <= loose["n_flagged"]
        assert strict["expected_pct_if_normal"] <= loose["expected_pct_if_normal"]

    @PROPERTY
    @given(_values, _thresholds, _thresholds)
    def test_iqr_flags_shrink_as_k_grows(self, values, a, b):
        lo, hi = sorted((a, b))
        df = _frame(values)
        loose = nb.iqr_outlier_style(df, "x", k=lo, rich=True)
        strict = nb.iqr_outlier_style(df, "x", k=hi, rich=True)
        assert strict.metadata["n_outliers"] <= loose.metadata["n_outliers"]
        assert loose.frame is df

    @PROPERTY
    @given(_values, st.sampled_from(["iqr", "zscore", "robust_zscore"]))
    def test_outlier_counts_match_the_fences(self, values, method):
        df = _frame(values)
        m = nb.outlier_style(df, method=method, rich=True).metadata["per_column"]["x"]
        x = df["x"].to_numpy()
        finite = np.isfinite(x)
        assert m["n"] + m["n_missing"] == len(df)
        assert m["n_low"] + m["n_high"] == m["n_outliers"] <= m["n"]
        if np.isfinite(m["lower_fence"]) and np.isfinite(m["upper_fence"]):
            assert m["lower_fence"] <= m["upper_fence"]
            assert m["n_low"] == int((finite & (x < m["lower_fence"])).sum())
            assert m["n_high"] == int((finite & (x > m["upper_fence"])).sum())
        else:
            assert m["n_outliers"] == 0

    @PROPERTY
    @given(_values)
    def test_demean_counts_partition_the_finite_values(self, values):
        df = _frame(values)
        before = df.copy(deep=True)
        result = nb.demean_style(df, "x", rich=True)
        m = result.metadata
        assert _same(df, before)
        assert m["n_above"] + m["n_below"] + m["n_at_mean"] == m["n"]
        assert m["n"] + m["n_missing"] == len(df)
        assert result.frame.index.equals(df.index)

    @PROPERTY
    @given(_values, _thresholds)
    def test_scale_compare_agreement_counts_add_up(self, values, threshold):
        df = _frame(values)
        m = nb.scale_compare_style(df, "x", threshold=threshold, rich=True).metadata
        assert m["n_flagged_both"] + m["n_robust_only"] == m["n_flagged_robust"]
        assert m["n_flagged_both"] + m["n_classical_only"] == m["n_flagged_classical"]
        assert max(m["n_flagged_robust"], m["n_flagged_classical"]) <= m["n"]
        if m["zero_mad"]:
            assert m["n_flagged_robust"] == 0

    @PROPERTY
    @given(_values)
    def test_missingness_counts_add_up(self, values):
        df = _frame(values)
        m = nb.missingness_style(df, rich=True).metadata
        x = df["x"].to_numpy()
        assert m["n_missing_cells"] == int(np.isnan(x).sum())
        assert m["n_infinite_cells"] == int(np.isinf(x).sum())
        assert m["n_complete_rows"] + m["n_incomplete_rows"] == len(df)
        assert m["n_complete_rows"] == int(np.isfinite(x).sum())

    @PROPERTY
    @given(
        _values,
        st.floats(min_value=0.0, max_value=0.45),
        st.floats(min_value=0.55, max_value=1.0),
    )
    def test_winsorize_metadata_matches_the_frame(self, values, lower_q, upper_q):
        df = _frame(values)
        before = df.copy(deep=True)
        result = nb.winsorize_columns(df, lower_q=lower_q, upper_q=upper_q, rich=True)
        m = result.metadata["per_column"]["x"]
        x, w = df["x"].to_numpy(), result.frame["x_winsor"].to_numpy()
        finite = np.isfinite(x)
        assert _same(df, before)
        assert m["n"] == int(finite.sum())
        assert m["n_clipped_low"] + m["n_clipped_high"] <= m["n"]
        assert m["n_clipped_low"] == int((w[finite] > x[finite]).sum())
        assert m["n_clipped_high"] == int((w[finite] < x[finite]).sum())
        if m["n"]:
            assert m["lower_bound"] <= m["upper_bound"]
            assert np.all((w[finite] >= m["lower_bound"]) & (w[finite] <= m["upper_bound"]))

    @PROPERTY
    @given(_values, st.sampled_from(["robust", "zscore", "minmax"]))
    def test_scale_columns_preserves_shape_and_missing_positions(self, values, method):
        df = _frame(values)
        result = nb.scale_columns(df, method=method, rich=True)
        detail = result.metadata["per_column"]["x"]
        scaled = result.frame[detail["derived_column"]].to_numpy()
        assert len(scaled) == len(df) and result.frame.index.equals(df.index)
        assert detail["n"] + detail["n_missing"] == len(df)
        assert np.isnan(scaled[~np.isfinite(df["x"].to_numpy())]).all()
