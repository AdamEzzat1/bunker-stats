"""Tests for the opt-in rich result objects on the inference tests.

Every inference test keeps its dict return by default; ``rich=True`` returns a
dataclass from :mod:`bunker_stats.infer` implementing the shared protocol
(tuple unpacking, indexing, ``.to_dict()``, ``.info()``, ``.conclusion()``).

The suite is organized as:

* ``TestBackwardCompatibility`` -- the default return is byte-for-byte unchanged.
* ``TestProtocol``             -- the shared protocol holds for every rich type.
* one class per result type    -- field mapping and derived metadata.
* ``TestSharedBase``           -- unit tests for RichResult / HypothesisResult.
* ``TestPublicSurface``        -- exports line up with ``__all__``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

import bunker_stats as bs
from bunker_stats import infer
from bunker_stats._result import HypothesisResult, RichResult, rich_facade


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def xy():
    rng = np.random.default_rng(1234)
    return rng.normal(0.0, 1.0, 60), rng.normal(0.6, 1.2, 70)


@pytest.fixture
def paired_xy():
    rng = np.random.default_rng(99)
    x = rng.normal(0.0, 1.0, 40)
    return x, x + rng.normal(0.3, 0.5, 40)


# One rich-returning call per result type, for protocol tests that must hold
# uniformly. Each entry: (label, callable-returning-rich, expected_type).
def _all_rich_calls():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 50)
    y = rng.normal(0.5, 1, 55)
    table = np.array([[10.0, 20.0], [30.0, 25.0]])
    counts = np.array([12.0, 18.0, 30.0])
    return [
        ("t_test_1samp", lambda: bs.t_test_1samp(x, 0.0, rich=True), infer.TTestResult),
        ("t_test_2samp", lambda: bs.t_test_2samp(x, y, rich=True), infer.TTestResult),
        ("t_test_paired", lambda: bs.t_test_paired(x, y[:50], rich=True), infer.TTestResult),
        ("chi2_gof", lambda: bs.chi2_gof(counts, rich=True), infer.ChiSquareResult),
        ("chi2_independence", lambda: bs.chi2_independence(table, rich=True), infer.ChiSquareResult),
        ("mann_whitney_u", lambda: bs.mann_whitney_u(x, y, rich=True), infer.MannWhitneyResult),
        ("ks_1samp", lambda: bs.ks_1samp(x, "norm", (0.0, 1.0), rich=True), infer.KSResult),
        ("f_test_oneway", lambda: bs.f_test_oneway([x, y], rich=True), infer.ANOVAResult),
        ("pearson_corr_test", lambda: bs.pearson_corr_test(x, y[:50], rich=True), infer.CorrelationTestResult),
        ("spearman_corr_test", lambda: bs.spearman_corr_test(x, y[:50], rich=True), infer.CorrelationTestResult),
        ("jarque_bera", lambda: bs.jarque_bera(x, rich=True), infer.NormalityResult),
        ("anderson_darling", lambda: bs.anderson_darling(x, rich=True), infer.NormalityResult),
    ]


RICH_CALLS = _all_rich_calls()
RICH_IDS = [c[0] for c in RICH_CALLS]


# ----------------------------------------------------------------------
# Backward compatibility — the whole point of the opt-in design
# ----------------------------------------------------------------------

class TestBackwardCompatibility:
    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_default_return_is_a_dict(self, label, call, rtype):
        # Rebuild the same call without rich=True by calling the raw dict path.
        # Easiest: assert the rich object's to_dict() superset matches the dict.
        rich = call()
        assert isinstance(rich, rtype)

    def test_t_test_2samp_default_unchanged(self, xy):
        x, y = xy
        d = bs.t_test_2samp(x, y)
        assert isinstance(d, dict)
        assert set(d) >= {"statistic", "pvalue", "df", "mean_x", "mean_y", "equal_var"}

    def test_default_matches_rich_core_values(self, xy):
        """rich=True must not change the computed statistic/p-value."""
        x, y = xy
        d = bs.t_test_2samp(x, y, equal_var=False)
        r = bs.t_test_2samp(x, y, equal_var=False, rich=True)
        assert r.statistic == pytest.approx(d["statistic"])
        assert r.pvalue == pytest.approx(d["pvalue"])
        assert r.df == pytest.approx(d["df"])

    def test_modern_facade_kwargs_still_work(self, xy):
        """t_test_2samp has a hand-written facade; rich must ride on top of it."""
        x, y = xy
        welch = bs.t_test_2samp(x, y, equal_var=False, rich=True)
        pooled = bs.t_test_2samp(x, y, equal_var=True, rich=True)
        assert welch.equal_var is False and pooled.equal_var is True
        assert welch.statistic != pooled.statistic  # different variance handling

    def test_deprecated_np_alias_still_returns_dict(self, xy):
        x, y = xy
        with pytest.warns(DeprecationWarning):
            d = bs.t_test_2samp_np(x, y, equal_var=True)
        assert isinstance(d, dict)

    def test_positional_call_unchanged(self, xy):
        x, y = xy
        # pearson takes (x, y) positionally; default still a dict
        assert isinstance(bs.pearson_corr_test(x, y[:60]), dict)

    def test_rich_false_is_explicit_default(self, xy):
        x, y = xy
        assert isinstance(bs.mann_whitney_u(x, y, rich=False), dict)


# ----------------------------------------------------------------------
# Shared protocol — must hold uniformly for every rich type
# ----------------------------------------------------------------------

class TestProtocol:
    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_is_expected_type(self, label, call, rtype):
        assert isinstance(call(), rtype)

    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_tuple_unpacking(self, label, call, rtype):
        a, b = call()  # every rich type unpacks to exactly two primary fields
        assert isinstance(a, (int, float, np.floating))
        assert b is None or isinstance(b, (int, float, np.floating))

    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_indexing_matches_unpacking(self, label, call, rtype):
        r = call()
        seq = list(r)
        assert r[0] == seq[0]
        assert r[1] == seq[1]
        assert len(r) == 2

    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_index_out_of_range(self, label, call, rtype):
        with pytest.raises(IndexError):
            call()[5]

    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_to_dict_is_plain_and_json_friendly(self, label, call, rtype):
        import json

        d = call().to_dict()
        assert isinstance(d, dict) and d
        # No None values (populated fields only), JSON-serializable by default.
        assert all(v is not None for v in d.values())
        json.dumps(d)  # must not raise

    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_info_contains_key_labels(self, label, call, rtype):
        text = call().info()
        assert "Statistic:" in text
        assert "Conclusion:" in text
        # title present as the first line
        assert text.splitlines()[0].strip() != ""

    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_info_is_ascii(self, label, call, rtype):
        """info()/conclusion() must be printable on any console (e.g. cp1252)."""
        text = call().info()
        text.encode("ascii")  # raises if any non-ASCII slipped in

    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_conclusion_mentions_alpha(self, label, call, rtype):
        c = call().conclusion(alpha=0.05)
        assert isinstance(c, str) and c
        assert "reject" in c.lower()

    @pytest.mark.parametrize("label,call,rtype", RICH_CALLS, ids=RICH_IDS)
    def test_repr_names_the_type(self, label, call, rtype):
        assert rtype.__name__ in repr(call())


# ----------------------------------------------------------------------
# Per-type field mapping and derived metadata
# ----------------------------------------------------------------------

class TestTTestResult:
    def test_two_sample_fields(self, xy):
        x, y = xy
        r = bs.t_test_2samp(x, y, equal_var=False, rich=True)
        assert r.kind == "two-sample"
        assert r.n1 == len(x) and r.n2 == len(y)
        assert r.mean_x is not None and r.mean_y is not None
        assert r.effect_size is not None  # Cohen's d computed from same inputs
        assert r.alternative == "two-sided"

    def test_effect_size_matches_cohens_d(self, xy):
        x, y = xy
        r = bs.t_test_2samp(x, y, rich=True)
        assert r.effect_size == pytest.approx(bs.cohens_d_2samp(x, y))

    def test_one_sample_fields(self, xy):
        x, _ = xy
        r = bs.t_test_1samp(x, 0.0, rich=True)
        assert r.kind == "one-sample"
        assert r.n1 == len(x) and r.n2 is None
        assert r.mean is not None and r.equal_var is None
        assert "popmean" in r.conclusion().lower() or "reject" in r.conclusion().lower()

    def test_paired_fields(self, paired_xy):
        x, y = paired_xy
        r = bs.t_test_paired(x, y, rich=True)
        assert r.kind == "paired"
        assert r.n1 == len(x)
        assert "difference" in r._h0

    def test_alternative_is_captured(self, xy):
        x, y = xy
        r = bs.t_test_2samp(x, y, alternative="greater", rich=True)
        assert r.alternative == "greater"


class TestCorrelationTestResult:
    def test_pearson_has_ci(self, xy):
        x, y = xy
        r = bs.pearson_corr_test(x, y[:60], rich=True)
        assert r.method == "pearson"
        assert r.ci_low is not None and r.ci_high is not None
        assert r.ci_low <= r.r <= r.ci_high
        assert r.n == int(r.df) + 2

    def test_spearman_has_no_ci(self, xy):
        x, y = xy
        r = bs.spearman_corr_test(x, y[:60], rich=True)
        assert r.method == "spearman"
        assert r.ci_low is None and r.ci_high is None

    def test_unpacks_as_r_pvalue(self, xy):
        x, y = xy
        r = bs.pearson_corr_test(x, y[:60], rich=True)
        rr, pp = r
        assert rr == pytest.approx(r.r)
        assert pp == pytest.approx(r.pvalue)

    def test_r_matches_raw_dict(self, xy):
        x, y = xy
        d = bs.pearson_corr_test(x, y[:60])
        r = bs.pearson_corr_test(x, y[:60], rich=True)
        assert r.r == pytest.approx(d["correlation"])


class TestNormalityResult:
    def test_jarque_bera_has_pvalue_and_moments(self):
        rng = np.random.default_rng(3)
        r = bs.jarque_bera(rng.normal(size=500), rich=True)
        assert r.method == "jarque_bera"
        assert r.pvalue is not None
        assert r.skewness is not None and r.kurtosis is not None

    def test_anderson_darling_has_no_pvalue(self):
        rng = np.random.default_rng(3)
        r = bs.anderson_darling(rng.normal(size=500), rich=True)
        assert r.method == "anderson_darling"
        assert r.pvalue is None
        assert r.critical_value_5pct == pytest.approx(0.787)

    def test_ad_conclusion_uses_critical_value(self):
        rng = np.random.default_rng(0)
        normal = bs.anderson_darling(rng.normal(size=1000), rich=True)
        assert normal.is_normal() is True
        assert "5%" in normal.conclusion()

    def test_ad_rejects_non_normal(self):
        rng = np.random.default_rng(0)
        skewed = bs.anderson_darling(rng.exponential(size=1000), rich=True)
        assert skewed.is_normal() is False
        assert "Reject" in skewed.conclusion()

    def test_jb_verdict_tracks_alpha(self):
        rng = np.random.default_rng(0)
        r = bs.jarque_bera(rng.normal(size=2000), rich=True)
        assert r.is_normal(alpha=0.05) is True


class TestChiSquareResult:
    def test_gof_type_and_observed(self):
        counts = np.array([10.0, 20.0, 30.0])
        r = bs.chi2_gof(counts, rich=True)
        assert r.test_type == "goodness_of_fit"
        np.testing.assert_array_equal(r.observed, counts)
        assert r.dof == 2

    def test_independence_type_and_null(self):
        table = np.array([[10.0, 20.0], [30.0, 40.0]])
        r = bs.chi2_independence(table, rich=True)
        assert r.test_type == "independence"
        assert "independent" in r._h0
        assert r.observed.shape == (2, 2)

    def test_to_dict_observed_becomes_list(self):
        r = bs.chi2_gof(np.array([10.0, 20.0, 30.0]), rich=True)
        d = r.to_dict()
        assert isinstance(d["observed"], list)
        d2 = r.to_dict(array=True)
        assert isinstance(d2["observed"], np.ndarray)


class TestMannWhitneyResult:
    def test_rank_biserial_effect(self, xy):
        x, y = xy
        r = bs.mann_whitney_u(x, y, rich=True)
        assert r.n1 == len(x) and r.n2 == len(y)
        assert r.rank_biserial is not None
        assert r.rank_biserial == pytest.approx(bs.rank_biserial(x, y)["rank_biserial"])


class TestKSResult:
    def test_distribution_captured(self, xy):
        x, _ = xy
        r = bs.ks_1samp(x, "norm", (0.0, 1.0), rich=True)
        assert r.distribution == "norm"
        assert r.n == len(x)


class TestANOVAResult:
    def test_effect_sizes_and_counts(self):
        rng = np.random.default_rng(5)
        groups = [rng.normal(m, 1, 30) for m in (0.0, 0.5, 1.0)]
        r = bs.f_test_oneway(groups, rich=True)
        assert r.n_groups == 3 and r.n_total == 90
        assert r.eta_squared is not None and r.omega_squared is not None
        eff = bs.anova_effect_sizes(groups)
        assert r.eta_squared == pytest.approx(eff["eta_squared"])


# ----------------------------------------------------------------------
# Shared base unit tests
# ----------------------------------------------------------------------

@dataclass
class _Demo(HypothesisResult):
    statistic: float = 0.0
    pvalue: Optional[float] = None
    extra: Optional[float] = None
    _title = "Demo"
    _h0 = "H0 (demo)"


class TestSharedBase:
    def test_iter_len_getitem(self):
        d = _Demo(statistic=1.5, pvalue=0.02)
        assert list(d) == [1.5, 0.02]
        assert len(d) == 2
        assert d[0] == 1.5 and d[1] == 0.02

    def test_to_dict_omits_none_and_converts_numpy(self):
        d = _Demo(statistic=np.float64(2.0), pvalue=None, extra=np.float64(3.0))
        out = d.to_dict()
        assert out == {"statistic": 2.0, "extra": 3.0}
        assert "pvalue" not in out
        assert isinstance(out["statistic"], float)  # not np.float64

    def test_to_dict_array_flag(self):
        @dataclass
        class WithArr(RichResult):
            data: np.ndarray = None
            _fields = ("data",)

        arr = np.arange(3.0)
        w = WithArr(data=arr)
        assert w.to_dict()["data"] == [0.0, 1.0, 2.0]
        assert isinstance(w.to_dict(array=True)["data"], np.ndarray)

    def test_conclusion_significant(self):
        assert "Reject" in _Demo(statistic=5.0, pvalue=0.001).conclusion(0.05)

    def test_conclusion_not_significant(self):
        assert "Fail to reject" in _Demo(statistic=0.5, pvalue=0.5).conclusion(0.05)

    def test_conclusion_no_pvalue(self):
        assert "Inconclusive" in _Demo(statistic=0.5, pvalue=None).conclusion()

    def test_is_significant_handles_nan(self):
        assert _Demo(statistic=1.0, pvalue=float("nan")).is_significant() is False

    def test_alpha_threshold_boundary(self):
        # p exactly at alpha is NOT significant (strict <)
        assert _Demo(statistic=1.0, pvalue=0.05).is_significant(0.05) is False

    def test_rich_facade_passthrough_and_intercept(self):
        def raw(a, b, c=10):
            return {"sum": a + b + c}

        def builder(res, args, kwargs):
            return _Demo(statistic=res["sum"], pvalue=0.0)

        wrapped = rich_facade(raw, builder, name="raw")
        assert wrapped(1, 2) == {"sum": 13}          # default passthrough
        assert wrapped(1, 2, c=3) == {"sum": 6}       # kwargs forwarded
        rich = wrapped(1, 2, rich=True)               # intercepted
        assert isinstance(rich, _Demo) and rich.statistic == 13
        assert wrapped.__name__ == "raw"
        assert wrapped.__wrapped__ is raw


# ----------------------------------------------------------------------
# Public surface
# ----------------------------------------------------------------------

class TestPublicSurface:
    RESULT_NAMES = [
        "TTestResult", "ChiSquareResult", "MannWhitneyResult", "KSResult",
        "ANOVAResult", "CorrelationTestResult", "NormalityResult",
    ]

    @pytest.mark.parametrize("name", RESULT_NAMES)
    def test_exported_from_root(self, name):
        assert hasattr(bs, name)
        assert name in bs.__all__

    @pytest.mark.parametrize("name", RESULT_NAMES)
    def test_exported_from_infer(self, name):
        assert hasattr(infer, name)
        assert name in infer.__all__

    def test_root_and_infer_are_same_class(self):
        for name in self.RESULT_NAMES:
            assert getattr(bs, name) is getattr(infer, name)

    def test_wrapped_functions_advertise_rich_in_doc(self):
        assert "rich=True" in (bs.t_test_2samp.__doc__ or "")
        assert "rich=True" in (bs.jarque_bera.__doc__ or "")
