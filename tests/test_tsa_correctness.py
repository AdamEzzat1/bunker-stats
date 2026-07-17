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
    """Breusch-Godfrey LM = T*R^2 on the reduced auxiliary regression that the
    Rust kernel actually fits (drops the first `p` rows, no zero-padding)."""
    e = np.asarray(e, float)
    n = len(e)
    t0 = p
    t_len = n - t0
    y = e[t0:n]
    cols = [np.ones(t_len)] + [e[t0 - j : n - j] for j in range(1, p + 1)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    return t_len * r2


def test_breusch_godfrey_is_T_times_R2():
    """Regression for the fixed BG statistic (was n^2/T * R^2, now T * R^2)."""
    rng = np.random.default_rng(7)
    # residuals with mild serial correlation
    e = np.zeros(80)
    for t in range(1, 80):
        e[t] = 0.4 * e[t - 1] + rng.standard_normal()
    p = 4
    stat, pval = bs.bg_test(e, p)
    expected = _bg_reference(e, p)
    assert abs(stat - expected) < 1e-6, f"BG stat {stat} != T*R^2 {expected}"
    assert 0.0 <= pval <= 1.0


@pytest.mark.xfail(
    reason="FILED BUG (Role C, HIGH): adf_test ignores the `regression` argument "
    "(param is `_regression`, unused); 'c'/'ct'/'nc' all return identical stats.",
    strict=True,
)
def test_adf_honors_regression_argument():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(120))
    stat_c, _ = bs.adf_test(x, "c")
    stat_ct, _ = bs.adf_test(x, "ct")
    assert abs(stat_c - stat_ct) > 1e-6, "regression arg had no effect"


@pytest.mark.xfail(
    reason="FILED BUG (Role C, HIGH): adf_test ignores `max_lag` (param is "
    "`_max_lag`, unused); no lag augmentation is ever added (it is a plain DF, "
    "not ADF).",
    strict=True,
)
def test_adf_honors_max_lag_argument():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(120))
    stat_0, _ = bs.adf_test(x, "c", 0)
    stat_5, _ = bs.adf_test(x, "c", 5)
    assert abs(stat_0 - stat_5) > 1e-6, "max_lag arg had no effect"


@pytest.mark.xfail(
    reason="FILED BUG (Role C, HIGH): pp_test ignores `regression` (param is "
    "`_regression`), applies no Newey-West HAC correction, and uses the Normal "
    "CDF for p-values instead of the Dickey-Fuller distribution.",
    strict=True,
)
def test_pp_honors_regression_argument():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(120))
    stat_c, _ = bs.pp_test(x, "c")
    stat_ct, _ = bs.pp_test(x, "ct")
    assert abs(stat_c - stat_ct) > 1e-6, "regression arg had no effect"
