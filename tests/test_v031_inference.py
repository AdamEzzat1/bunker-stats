"""Smoke + sanity tests for the v0.3.1 inference functions.

These were ported from the parallel feature branch onto the hardened base. Each
must be callable via the facade and return sensible values (scipy/statsmodels
references where a closed form exists).
"""
from __future__ import annotations

import numpy as np
import pytest

bs = pytest.importorskip("bunker_stats")
scipy_stats = pytest.importorskip("scipy.stats")

RNG = np.random.default_rng(2025)


def test_t_test_paired_matches_scipy():
    x = RNG.normal(0, 1, 40)
    y = x + RNG.normal(0.3, 1, 40)
    r = bs.t_test_paired(x, y)
    ref = scipy_stats.ttest_rel(x, y)
    assert np.isclose(r["statistic"], ref.statistic, rtol=1e-6)
    assert np.isclose(r["pvalue"], ref.pvalue, rtol=1e-4)


def test_p_adjust_methods():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(bs.p_adjust(p, "bonferroni"), np.minimum(p * len(p), 1.0))
    holm = np.asarray(bs.p_adjust(p, "holm"))
    assert np.all(holm >= p - 1e-12) and np.all(holm <= 1.0)


def test_proportion_ztests_run():
    z1 = bs.proportion_ztest(45, 100, value=0.5)
    z2 = bs.two_proportions_ztest(45, 100, 60, 100)
    for r in (z1, z2):
        assert np.isfinite(r["statistic"]) and 0.0 <= r["pvalue"] <= 1.0


def test_corr_ci_brackets_estimate():
    x = RNG.normal(0, 1, 80)
    y = 0.6 * x + RNG.normal(0, 1, 80)
    r = bs.corr_ci(x, y, method="pearson", confidence=0.95)
    lo, hi = r["ci_lower"], r["ci_upper"]
    assert -1.0 <= lo <= r["correlation"] <= hi <= 1.0


def test_var_ci_contains_sample_variance():
    x = RNG.normal(0, 2, 100)
    r = bs.var_ci(x, confidence=0.95)
    assert r["ci_lower"] <= np.var(x, ddof=1) <= r["ci_upper"]


def test_odds_ratio_2x2():
    table = np.array([[10.0, 20.0], [30.0, 40.0]])
    r = bs.odds_ratio(table)
    assert np.isclose(r["odds_ratio"], (10 * 40) / (20 * 30), rtol=1e-9)
    assert r["ci_lower"] <= r["odds_ratio"] <= r["ci_upper"]


def test_effect_sizes_run():
    x = RNG.normal(0, 1, 50)
    y = RNG.normal(1, 1, 60)
    cd = bs.cliffs_delta(x, y)
    assert -1.0 <= cd["delta"] <= 1.0
    assert cd["magnitude"] in {"negligible", "small", "medium", "large"}
    rb = bs.rank_biserial(x, y)
    assert -1.0 <= rb["rank_biserial"] <= 1.0
    assert rb["u_statistic"] >= 0.0


def test_anova_effect_and_normality_summary():
    groups = np.vstack([RNG.normal(0, 1, 30), RNG.normal(0.5, 1, 30), RNG.normal(1, 1, 30)])
    ae = bs.anova_effect_sizes(groups)
    assert isinstance(ae, dict) and ae
    ns = bs.normality_summary(RNG.normal(0, 1, 200))
    assert isinstance(ns, dict) and ns
