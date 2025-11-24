"""
Correctness + lightweight performance checks for advanced bunker-stats ops.

Run from project root (with venv active + `maturin develop` done):

    python benchmarks/test_advanced_ops.py
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import pandas as pd

import bunker_stats as bs


def time_it(name: str, fn, *args, repeats: int = 3, warmup: int = 1) -> float:
    """Tiny timing helper: best-of-N."""
    for _ in range(warmup):
        fn(*args)

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        end = time.perf_counter()
        best = min(best, end - start)
    print(f"{name:35s}: {best*1000:8.2f} ms")
    return best


# ---------- helpers for Python “reference” implementations ----------


def py_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def py_robust_scale(
    x: np.ndarray,
    scale_factor: float = 1.4826,
) -> Tuple[np.ndarray, float, float]:
    med = float(np.median(x))
    mad = py_mad(x)
    denom = mad * scale_factor if mad != 0 else 1e-12
    scaled = (x - med) / denom
    return scaled, med, mad


def py_winsorize(x: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    low = float(np.quantile(x, lower_q))
    high = float(np.quantile(x, upper_q))
    return np.clip(x, low, high)


def py_quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    # Note: different strategy than bunker-stats could give slightly different bin edges,
    # so we only check rough structure, not exact bin labels.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, quantiles)
    # np.digitize returns bin indices 1..n_bins; we'll subtract 1
    return np.digitize(x, edges[1:-1], right=True).astype(int)


def py_diff(x: np.ndarray, periods: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[:periods] = np.nan
    out[periods:] = x[periods:] - x[:-periods]
    return out


def py_pct_change(x: np.ndarray, periods: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[:periods] = np.nan
    base = x[:-periods]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[periods:] = x[periods:] / base - 1.0
    out[np.isinf(out)] = np.nan
    return out


def py_ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1, dtype=float) / n
    return v, y


def py_cov(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.cov(x, y, ddof=1)[0, 1])


def py_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def py_rolling_cov(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    assert x.shape == y.shape
    n = x.shape[0]
    if window <= 0 or window > n:
        return np.array([], dtype=float)
    out = []
    for i in range(n - window + 1):
        xr = x[i : i + window]
        yr = y[i : i + window]
        out.append(py_cov(xr, yr))
    return np.array(out, dtype=float)


def py_rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    assert x.shape == y.shape
    n = x.shape[0]
    if window <= 0 or window > n:
        return np.array([], dtype=float)
    out = []
    for i in range(n - window + 1):
        xr = x[i : i + window]
        yr = y[i : i + window]
        out.append(py_corr(xr, yr))
    return np.array(out, dtype=float)


# ---------- tests ----------


def test_mad_robust_scale():
    print("\n=== MAD / RobustScaler ===")
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=3.0, size=10_000).astype("float64")

    mad_py = py_mad(x)
    mad_bs = bs.mad_np(x)
    print("MAD numpy-style vs bunker:", mad_py, mad_bs)
    assert math.isclose(mad_py, mad_bs, rel_tol=1e-6, abs_tol=1e-6)

    scaled_py, med_py, mad_py2 = py_robust_scale(x)
    scaled_bs, med_bs, mad_bs2 = bs.robust_scale_np(x, 1.4826)
    scaled_bs = np.asarray(scaled_bs)

    print("median py vs bunker:", med_py, med_bs)
    print("mad py vs bunker   :", mad_py2, mad_bs2)
    assert math.isclose(med_py, med_bs, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(mad_py2, mad_bs2, rel_tol=1e-6, abs_tol=1e-6)
    assert np.allclose(scaled_py, scaled_bs, rtol=1e-6, atol=1e-6)

    time_it("bunker.robust_scale_np (100k)", bs.robust_scale_np, x, 1.4826)


def test_winsorize():
    print("\n=== Winsorization ===")
    rng = np.random.default_rng(0)
    x = rng.normal(size=100_000).astype("float64")

    w_py = py_winsorize(x, 0.05, 0.95)
    w_bs = np.asarray(bs.winsorize_np(x, 0.05, 0.95))

    # Values must be within [min_py, max_py] and should be close elementwise
    assert np.all(w_bs >= w_py.min() - 1e-12)
    assert np.all(w_bs <= w_py.max() + 1e-12)
    # not necessarily identical at every element due to quantile differences, but should be close overall
    assert np.allclose(np.quantile(w_bs, [0.05, 0.95]), np.quantile(w_py, [0.05, 0.95]), atol=1e-3)

    time_it("bunker.winsorize_np (100k)", bs.winsorize_np, x, 0.05, 0.95)


def test_quantile_bins():
    print("\n=== Quantile binning ===")
    rng = np.random.default_rng(1)
    x = rng.normal(size=100_000).astype("float64")
    n_bins = 5

    bins_bs = np.asarray(bs.quantile_bins_np(x, n_bins))
    # sanity checks on bins:
    assert bins_bs.min() >= 0
    assert bins_bs.max() <= n_bins - 1

    # rough check: bin counts are not ridiculously imbalanced
    counts = np.bincount(bins_bs, minlength=n_bins)
    print("bunker bin counts:", counts)
    assert counts.min() > 0

    time_it("bunker.quantile_bins_np (100k, 5 bins)", bs.quantile_bins_np, x, n_bins)


def test_diff_pct_cum():
    print("\n=== diff / pct_change / cumsum / cummean ===")
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype="float64")

    d_py = py_diff(x, 1)
    d_bs = np.asarray(bs.diff_np(x, 1))
    print("diff py vs bunker:", d_py, d_bs)
    assert np.allclose(d_py, d_bs, equal_nan=True)

    pct_py = py_pct_change(x, 1)
    pct_bs = np.asarray(bs.pct_change_np(x, 1))
    print("pct_change py vs bunker:", pct_py, pct_bs)
    assert np.allclose(pct_py, pct_bs, equal_nan=True)

    cumsum_py = np.cumsum(x)
    cumsum_bs = np.asarray(bs.cumsum_np(x))
    print("cumsum py vs bunker:", cumsum_py, cumsum_bs)
    assert np.allclose(cumsum_py, cumsum_bs, rtol=1e-12, atol=1e-12)

    cummean_py = np.cumsum(x) / np.arange(1, len(x) + 1)
    cummean_bs = np.asarray(bs.cummean_np(x))
    print("cummean py vs bunker:", cummean_py, cummean_bs)
    assert np.allclose(cummean_py, cummean_bs, rtol=1e-12, atol=1e-12)

    # light perf
    rng = np.random.default_rng(2)
    big = rng.normal(size=1_000_000).astype("float64")
    time_it("bunker.diff_np (1M)", bs.diff_np, big, 1)
    time_it("bunker.pct_change_np (1M)", bs.pct_change_np, big, 1)
    time_it("bunker.cumsum_np (1M)", bs.cumsum_np, big)
    time_it("bunker.cummean_np (1M)", bs.cummean_np, big)


def test_ecdf():
    print("\n=== ECDF ===")
    rng = np.random.default_rng(3)
    x = rng.normal(size=10_000).astype("float64")

    v_py, cdf_py = py_ecdf(x)
    v_bs, cdf_bs = bs.ecdf_np(x)
    v_bs = np.asarray(v_bs)
    cdf_bs = np.asarray(cdf_bs)

    # values should match sorted x
    assert np.allclose(v_py, v_bs, rtol=1e-12, atol=1e-12)
    # CDFs should be extremely close
    assert np.allclose(cdf_py, cdf_bs, rtol=1e-12, atol=1e-12)
    # monotonic checks
    assert np.all(np.diff(cdf_bs) >= 0)
    assert math.isclose(cdf_bs[-1], 1.0, rel_tol=1e-12, abs_tol=1e-12)

    time_it("bunker.ecdf_np (10k)", bs.ecdf_np, x)


def test_cov_corr_rolling():
    print("\n=== cov / corr / rolling_cov / rolling_corr ===")
    rng = np.random.default_rng(4)
    x = rng.normal(size=1_000).astype("float64")
    y = 0.5 * x + rng.normal(scale=0.1, size=1_000).astype("float64")

    cov_py = py_cov(x, y)
    cov_bs = bs.cov_np(x, y)
    print("cov py vs bunker:", cov_py, cov_bs)
    assert math.isclose(cov_py, cov_bs, rel_tol=1e-6, abs_tol=1e-6)

    corr_py = py_corr(x, y)
    corr_bs = bs.corr_np(x, y)
    print("corr py vs bunker:", corr_py, corr_bs)
    assert math.isclose(corr_py, corr_bs, rel_tol=1e-6, abs_tol=1e-6)

    # rolling
    window = 50
    rcov_py = py_rolling_cov(x, y, window)
    rcov_bs = np.asarray(bs.rolling_cov_np(x, y, window))
    print("rolling cov shapes py vs bunker:", rcov_py.shape, rcov_bs.shape)
    assert rcov_py.shape == rcov_bs.shape
    assert np.allclose(rcov_py, rcov_bs, rtol=1e-5, atol=1e-5)

    rcorr_py = py_rolling_corr(x, y, window)
    rcorr_bs = np.asarray(bs.rolling_corr_np(x, y, window))
    print("rolling corr shapes py vs bunker:", rcorr_py.shape, rcorr_bs.shape)
    assert rcorr_py.shape == rcorr_bs.shape
    assert np.allclose(rcorr_py, rcorr_bs, rtol=1e-5, atol=1e-5)

    # light perf on 100k
    big_n = 100_000
    xb = rng.normal(size=big_n).astype("float64")
    yb = 0.5 * xb + rng.normal(scale=0.2, size=big_n).astype("float64")
    time_it("bunker.cov_np (100k)", bs.cov_np, xb, yb)
    time_it("bunker.corr_np (100k)", bs.corr_np, xb, yb)
    time_it("bunker.rolling_cov_np (100k, window=50)", bs.rolling_cov_np, xb, yb, 50)
    time_it("bunker.rolling_corr_np (100k, window=50)", bs.rolling_corr_np, xb, yb, 50)


def test_kde():
    print("\n=== KDE (Gaussian) ===")
    rng = np.random.default_rng(5)
    x = rng.normal(size=5_000).astype("float64")

    grid, dens = bs.kde_gaussian_np(x, n_points=512, bandwidth=None)
    grid = np.asarray(grid)
    dens = np.asarray(dens)

    print("kde grid shape:", grid.shape)
    print("kde dens shape:", dens.shape)
    assert grid.shape == dens.shape
    assert grid.shape[0] == 512

    # basic sanity: density non-negative and roughly integrates to ~1
    assert np.all(dens >= 0)
    approx_int = np.trapz(dens, x=grid)
    print("approx integral of KDE density:", approx_int)
    # allow some tolerance
    assert 0.8 < approx_int < 1.2

    time_it("bunker.kde_gaussian_np (5k → 512)", bs.kde_gaussian_np, x, 512, None)


def test_outliers_iqr_zscore():
    print("\n=== Outlier flags (IQR & z-score) ===")
    x = np.array(
        [10, 12, 11, 13, 9, 8, 50, -20, 10, 11],
        dtype="float64",
    )

    # IQR-based
    mask_bs_iqr = np.asarray(bs.iqr_outliers_np(x, 1.5))
    print("IQR outlier mask (bunker):", mask_bs_iqr)

    # naive Python IQR detection
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask_py_iqr = (x < low) | (x > high)
    print("IQR outlier mask (py)    :", mask_py_iqr)

    assert np.array_equal(mask_bs_iqr, mask_py_iqr)

    # z-score-based
    m = x.mean()
    s = x.std(ddof=1)
    zs = (x - m) / s
    thresh = 2.0
    mask_py_z = np.abs(zs) > thresh
    mask_bs_z = np.asarray(bs.zscore_outliers_np(x, thresh))
    print("z-score outlier mask py:", mask_py_z)
    print("z-score outlier mask bs:", mask_bs_z)

    assert np.array_equal(mask_bs_z, mask_py_z)


def test_ewma_welford():
    print("\n=== EWMA & Welford one-pass mean/var ===")
    rng = np.random.default_rng(6)
    x = rng.normal(size=10_000).astype("float64")

    # EWMA vs numpy-style reference
    alpha = 0.2

    def py_ewma(arr: np.ndarray, a: float) -> np.ndarray:
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = a * arr[i] + (1.0 - a) * out[i - 1]
        return out

    ew_py = py_ewma(x, alpha)
    ew_bs = np.asarray(bs.ewma_np(x, alpha))
    print("EWMA first 5 py vs bunker:\n", ew_py[:5], "\n", ew_bs[:5])
    assert np.allclose(ew_py, ew_bs, rtol=1e-12, atol=1e-12)

    # Welford vs numpy mean/var
    mean_w, var_w, n_w = bs.welford_np(x)
    mean_np = float(np.mean(x))
    var_np = float(np.var(x, ddof=1))
    print("Welford mean vs numpy:", mean_w, mean_np)
    print("Welford var  vs numpy:", var_w, var_np)
    print("Welford count        :", n_w)
    assert n_w == x.shape[0]
    assert math.isclose(mean_w, mean_np, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(var_w, var_np, rel_tol=1e-12, abs_tol=1e-12)


def main():
    test_mad_robust_scale()
    test_winsorize()
    test_quantile_bins()
    test_diff_pct_cum()
    test_ecdf()
    test_cov_corr_rolling()
    test_kde()
    test_outliers_iqr_zscore()
    test_ewma_welford()
    print("\nAll advanced tests finished.\n")


if __name__ == "__main__":
    main()
