"""Time-series correctness regressions and filed-bug documentation.

Passing tests pin fixes made in this hardening pass. The `xfail` tests document
CONFIRMED, REPRODUCED bugs that were intentionally NOT fixed here because a
correct fix requires validation against statsmodels reference values. They are
xfail (not skip) so that if/when the underlying kernel is fixed, the test starts
XPASSing and flags that the documentation is stale.
"""
from __future__ import annotations

import numpy as np
import pytest

bs = pytest.importorskip(
    "bunker_stats_rs",
    reason="build the extension first: `python -m maturin develop --release`",
)


def _bg_reference(e: np.ndarray, p: int) -> float:
    """Breusch-Godfrey LM = n*R^2 over the full sample with zero-padded
    pre-sample lags (statsmodels' `acorr_breusch_godfrey` convention), for the
    residuals-only auxiliary regression the kernel fits."""
    e = np.asarray(e, float)
    n = len(e)
    cols = [np.ones(n)]
    for j in range(1, p + 1):
        lag = np.zeros(n)
        lag[j:] = e[: n - j]  # e[t-j], zero for t < j
        cols.append(lag)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    resid = e - X @ beta
    rss = float(np.sum(resid**2))
    tss = float(np.sum((e - e.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    return n * r2


def test_breusch_godfrey_is_n_times_R2():
    """Regression for the fixed BG statistic (was n^2/T * R^2, now n * R^2 over
    the zero-padded full sample, matching statsmodels)."""
    rng = np.random.default_rng(7)
    # residuals with mild serial correlation
    e = np.zeros(80)
    for t in range(1, 80):
        e[t] = 0.4 * e[t - 1] + rng.standard_normal()
    p = 4
    stat, pval = bs.bg_test(e, p)
    expected = _bg_reference(e, p)
    assert abs(stat - expected) < 1e-6, f"BG stat {stat} != n*R^2 {expected}"
    assert 0.0 <= pval <= 1.0


def test_adf_honors_regression_argument():
    """ADF now honors `regression`: 'c' and 'ct' fit different designs."""
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(120))
    stat_c, _ = bs.adf_test(x, "c")
    stat_ct, _ = bs.adf_test(x, "ct")
    assert abs(stat_c - stat_ct) > 1e-6, "regression arg had no effect"


def test_adf_honors_max_lag_argument():
    """ADF now honors `max_lag`: augmenting lags change the statistic."""
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(120))
    stat_0, _ = bs.adf_test(x, "c", 0)
    stat_5, _ = bs.adf_test(x, "c", 5)
    assert abs(stat_0 - stat_5) > 1e-6, "max_lag arg had no effect"


def test_pp_honors_regression_argument():
    """PP now honors `regression` and applies a HAC correction + DF p-values."""
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(120))
    stat_c, _ = bs.pp_test(x, "c")
    stat_ct, _ = bs.pp_test(x, "ct")
    assert abs(stat_c - stat_ct) > 1e-6, "regression arg had no effect"


def test_adf_statistic_matches_statsmodels():
    """The ADF test statistic must match statsmodels exactly across
    regression/lag specifications (the p-value uses a simpler critical-value
    table, so only the statistic is pinned)."""
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    rng = np.random.default_rng(0)
    series = {
        "random_walk": np.cumsum(rng.standard_normal(200)),
        "stationary": rng.standard_normal(200),
    }
    for x in series.values():
        for reg in ("c", "ct"):
            for lag in (0, 4):
                bs_stat, _ = bs.adf_test(x, reg, lag)
                sm_stat = sm.adfuller(x, maxlag=lag, regression=reg, autolag=None)[0]
                assert abs(bs_stat - sm_stat) < 1e-6, (
                    f"ADF stat mismatch reg={reg} lag={lag}: {bs_stat} vs {sm_stat}"
                )


def test_adf_pp_decisions_are_correct():
    """Sanity: a stationary series rejects the unit-root null; a random walk
    (constant spec) does not."""
    rng = np.random.default_rng(7)
    rw = np.cumsum(rng.standard_normal(200))
    noise = rng.standard_normal(200)
    assert bs.adf_test(noise, "c")[1] < 0.05
    assert bs.adf_test(rw, "c")[1] > 0.10
    assert bs.pp_test(noise, "c")[1] < 0.05
    assert bs.pp_test(rw, "c")[1] > 0.10
