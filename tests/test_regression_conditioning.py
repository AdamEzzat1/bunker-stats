"""
The regression-based tests must not depend on where the series sits.

ADF with a constant, Zivot-Andrews, and Breusch-Godfrey with an intercept are
all mathematically unchanged by adding a constant to the data (the intercept
absorbs it). Numerically that only holds if the least-squares solve is stable:
the lagged level next to a constant is a badly conditioned design once the
level is far from zero. Solving through X'X squared that conditioning and the
ADF statistic at a level of 1e8 was off by several units; these tests pin the
QR-based solve.
"""

import numpy as np
import pytest

try:
    from bunker_stats import _rs as bsr
except ImportError:  # pragma: no cover
    pytest.skip("bunker_stats extension not built", allow_module_level=True)


@pytest.fixture(scope="module")
def ar_series():
    rng = np.random.default_rng(3)
    x = np.zeros(5000)
    for t in range(1, len(x)):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    return x


@pytest.fixture(scope="module")
def random_walk():
    return np.cumsum(np.random.default_rng(4).normal(size=5000))


# Tolerance grows with the level because the input itself loses digits there:
# at 1e8 a float64 holds the data to ~1e-8, and the statistic amplifies that.
LEVELS = [(1e3, 1e-8), (1e6, 1e-5), (1e8, 1e-3)]


@pytest.mark.parametrize("level,tol", LEVELS)
@pytest.mark.parametrize("regression,lag", [("c", 0), ("c", 4), ("ct", 4)])
def test_adf_statistic_ignores_the_level(ar_series, level, tol, regression, lag):
    base, _ = bsr.adf_test(ar_series, regression, lag)
    shifted, _ = bsr.adf_test(ar_series + level, regression, lag)
    assert abs(shifted - base) < tol * abs(base)


@pytest.mark.parametrize("level,tol", LEVELS)
def test_adf_on_a_random_walk_ignores_the_level(random_walk, level, tol):
    base, base_p = bsr.adf_test(random_walk, "c", 4)
    shifted, shifted_p = bsr.adf_test(random_walk + level, "c", 4)
    assert abs(shifted - base) < tol * abs(base)
    assert abs(shifted_p - base_p) < max(tol, 1e-6)


@pytest.mark.parametrize("level,tol", LEVELS)
def test_zivot_andrews_ignores_the_level(ar_series, level, tol):
    x = ar_series[:600]
    stat, brk, _ = bsr.zivot_andrews_test(x, 2)
    stat_s, brk_s, _ = bsr.zivot_andrews_test(x + level, 2)
    assert brk_s == brk
    assert abs(stat_s - stat) < tol * abs(stat)


def test_adf_matches_a_reference_least_squares_fit(ar_series):
    """t-ratio on y_{t-1} from numpy's SVD-based lstsq, for the 'c' model."""
    x = ar_series + 1.0e6
    lag = 3
    dy = np.diff(x)
    rows = np.arange(lag, len(dy))
    design = np.column_stack(
        [np.ones(len(rows)), x[rows]] + [dy[rows - i] for i in range(1, lag + 1)]
    )
    y = dy[rows]
    # Center the level column for the reference so it is itself well conditioned.
    design_c = design.copy()
    design_c[:, 1] -= design_c[:, 1].mean()
    beta, *_ = np.linalg.lstsq(design_c, y, rcond=None)
    resid = y - design_c @ beta
    sigma2 = resid @ resid / (len(y) - design_c.shape[1])
    cov = sigma2 * np.linalg.inv(design_c.T @ design_c)
    want = beta[1] / np.sqrt(cov[1, 1])

    got, _ = bsr.adf_test(x, "c", lag)
    assert got == pytest.approx(want, rel=1e-6)


def test_bg_test_ignores_regressor_scale():
    rng = np.random.default_rng(5)
    n = 2000
    resid = rng.normal(size=n)
    for t in range(1, n):
        resid[t] += 0.3 * resid[t - 1]
    exog = np.column_stack([np.ones(n), np.arange(n, dtype=float), rng.normal(size=n)])

    base = bsr.bg_test(resid, 3, exog)
    # Same column space: shifted trend, rescaled noise column.
    awkward = exog.copy()
    awkward[:, 1] += 1.0e7
    awkward[:, 2] *= 1.0e-6
    got = bsr.bg_test(resid, 3, awkward)

    assert got[0] == pytest.approx(base[0], rel=1e-6)
    assert got[1] == pytest.approx(base[1], rel=1e-6)


def test_degenerate_designs_return_nan_instead_of_noise():
    # A constant series: the lagged level is a multiple of the intercept.
    stat, p = bsr.adf_test(np.full(200, 7.5), "c", 1)
    assert np.isnan(stat) and np.isnan(p)
