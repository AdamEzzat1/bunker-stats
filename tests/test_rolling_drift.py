"""
Rolling kernels must describe the window they are on, not its history.

The O(1)-per-step kernels update running sums as values enter and leave. Left
alone, those sums keep the rounding error of everything that ever passed
through them: after a burst of huge values every later window was garbage, and
after a single NaN every later window of the strict kernels was NaN. These
tests compare each kernel against a direct per-window computation far
downstream of such an event.
"""

import warnings

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view as windows

try:
    from bunker_stats import _rs as bsr
except ImportError:  # pragma: no cover
    pytest.skip("bunker_stats extension not built", allow_module_level=True)


N, W, TAIL = 60_000, 50, 5_000


@pytest.fixture(scope="module")
def burst():
    """Ordinary data with a burst of ~1e12 early on, in both series."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    y = 0.5 * x + rng.normal(size=N)
    x[1000:1100] = 1e12 * (1 + rng.normal(size=100))
    y[1000:1100] = -1e12
    return x, y


def _tail_windows(a):
    return windows(a[-(TAIL + W - 1):], W)


def _direct_cov_corr(x, y):
    a, b = _tail_windows(x), _tail_windows(y)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    sxy = (a * b).sum(axis=1)
    return sxy / (W - 1), sxy / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


SINGLE = [
    ("rolling_mean_np", lambda x: bsr.rolling_mean_np(x, W), lambda w: w.mean(axis=1), 1e-9),
    ("rolling_std_np", lambda x: bsr.rolling_std_np(x, W), lambda w: w.std(axis=1, ddof=1), 1e-9),
    ("rolling_var_np", lambda x: bsr.rolling_var_np(x, W), lambda w: w.var(axis=1, ddof=1), 1e-9),
    ("rolling_mean_nan_np", lambda x: bsr.rolling_mean_nan_np(x, W), lambda w: w.mean(axis=1), 1e-9),
    ("rolling_std_nan_np", lambda x: bsr.rolling_std_nan_np(x, W), lambda w: w.std(axis=1, ddof=1), 1e-9),
    ("rolling_mean_axis0_np", lambda x: bsr.rolling_mean_axis0_np(x.reshape(-1, 1), W)[:, 0], lambda w: w.mean(axis=1), 1e-9),
    ("rolling_std_axis0_np", lambda x: bsr.rolling_std_axis0_np(x.reshape(-1, 1), W)[:, 0], lambda w: w.std(axis=1, ddof=1), 1e-9),
    ("rolling_cv", lambda x: bsr.rolling_cv(x + 10.0, W),
     lambda w: (w + 10.0).std(axis=1, ddof=1) / np.abs((w + 10.0).mean(axis=1)), 1e-9),
]


@pytest.mark.parametrize("name,kernel,direct,tol", SINGLE, ids=[c[0] for c in SINGLE])
def test_a_burst_leaves_no_trace_in_later_windows(burst, name, kernel, direct, tol):
    x, _ = burst
    got = np.asarray(kernel(x))[-TAIL:]
    want = direct(_tail_windows(x))
    scale = _tail_windows(x).std(axis=1, ddof=1)
    # Error measured against the spread of each window (a mean near zero has
    # no meaningful relative error).
    assert np.max(np.abs(got - want) / np.maximum(scale, np.abs(want))) < tol


PAIRS = [
    ("rolling_cov_np", lambda x, y: bsr.rolling_cov_np(x, y, W), 0),
    ("rolling_cov_nan_np", lambda x, y: bsr.rolling_cov_nan_np(x, y, W), 0),
    ("rolling_cov_skipna", lambda x, y: bsr.rolling_cov_skipna(x, y, W), 0),
    ("rolling_corr_np", lambda x, y: bsr.rolling_corr_np(x, y, W), 1),
    ("rolling_corr_nan_np", lambda x, y: bsr.rolling_corr_nan_np(x, y, W), 1),
    ("rolling_corr_skipna", lambda x, y: bsr.rolling_corr_skipna(x, y, W), 1),
]


@pytest.mark.parametrize("name,kernel,which", PAIRS, ids=[c[0] for c in PAIRS])
def test_a_burst_leaves_no_trace_in_later_pair_windows(burst, name, kernel, which):
    x, y = burst
    got = np.asarray(kernel(x, y))[-TAIL:]
    want = _direct_cov_corr(x, y)[which]
    assert np.max(np.abs(got - want)) < 1e-9


def test_beta_and_linreg_recover_after_a_burst(burst):
    x, y = burst
    a, b = _tail_windows(x), _tail_windows(y)
    am, bm = a.mean(axis=1, keepdims=True), b.mean(axis=1, keepdims=True)
    slope = ((a - am) * (b - bm)).sum(axis=1) / ((a - am) ** 2).sum(axis=1)
    intercept = bm[:, 0] - slope * am[:, 0]

    np.testing.assert_allclose(np.asarray(bsr.rolling_beta_skipna(x, y, W))[-TAIL:], slope, rtol=0, atol=1e-9)
    got_slope, got_icpt = bsr.rolling_linreg_skipna(x, y, W)
    np.testing.assert_allclose(np.asarray(got_slope)[-TAIL:], slope, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(got_icpt)[-TAIL:], intercept, rtol=0, atol=1e-9)


STRICT = ["rolling_mean_np", "rolling_std_np", "rolling_var_np", "rolling_zscore_np"]


@pytest.mark.parametrize("name", STRICT)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_strict_kernels_are_nan_only_while_the_bad_value_is_in_the_window(name, bad):
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    x[100] = bad
    w = 10
    out = np.asarray(getattr(bsr, name)(x, w))
    for k in range(len(out)):
        clean = not (k <= 100 < k + w)
        if clean:
            assert np.isfinite(out[k]), f"window {k} is clean but gave {out[k]}"
        elif name != "rolling_mean_np" or np.isnan(bad):
            assert np.isnan(out[k]), f"window {k} holds {bad} but gave {out[k]}"
        else:
            assert out[k] == bad  # the mean of a window with one infinity


def test_strict_kernels_match_numpy_window_by_window_with_nans():
    rng = np.random.default_rng(2)
    x = rng.normal(size=2000)
    x[rng.random(2000) < 0.02] = np.nan
    w = 7
    win = windows(x, w)
    np.testing.assert_allclose(bsr.rolling_mean_np(x, w), win.mean(axis=1), rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(bsr.rolling_std_np(x, w), win.std(axis=1, ddof=1), rtol=1e-10, atol=1e-12, equal_nan=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        np.testing.assert_allclose(
            bsr.rolling_std_nan_np(x, w)[w - 1:], np.nanstd(win, axis=1, ddof=1), rtol=1e-10, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            bsr.rolling_mean_nan_np(x, w)[w - 1:], np.nanmean(win, axis=1), rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_pair_kernels_recover_after_a_nan():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=300), rng.normal(size=300)
    x[50] = np.nan
    w = 8
    for name in ("rolling_cov_np", "rolling_corr_np", "rolling_cov_nan_np", "rolling_corr_nan_np"):
        out = np.asarray(getattr(bsr, name)(x, y, w))
        holds_nan = np.array([k <= 50 < k + w for k in range(len(out))])
        assert np.isnan(out[holds_nan]).all(), name
        assert np.isfinite(out[~holds_nan]).all(), name


@pytest.mark.parametrize("level", [1e6, 1e9])
def test_levels_far_from_zero_and_trends(level):
    rng = np.random.default_rng(4)
    noise = rng.normal(size=20_000)
    for series in (noise + level, noise + 0.5 * np.arange(len(noise)), np.cumsum(noise) + level):
        win = windows(series, 30)
        want = win.std(axis=1, ddof=1)
        np.testing.assert_allclose(bsr.rolling_std_np(series, 30), want, rtol=1e-6)
        np.testing.assert_allclose(bsr.rolling_std_axis0_np(series.reshape(-1, 1), 30)[:, 0], want, rtol=1e-6)
        np.testing.assert_allclose(bsr.rolling_std_nan_np(series, 30)[29:], want, rtol=1e-6)
