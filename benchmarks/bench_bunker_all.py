import time
import math
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import bunker_stats_rs as bs

# Optional SciPy-based comparisons
try:
    import scipy.stats as stats

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ========== Simple timing helper ==========


def bench(fn, n_repeat=5):
    """
    Run fn() n_repeat times, return (best_time_ms, last_result).
    Using best-of-N to reduce noise from Python/OS jitter.
    """
    best = float("inf")
    last_res = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        res = fn()
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0
        if dt < best:
            best = dt
        last_res = res
    return best, last_res


def allclose(a, b, rtol=1e-7, atol=1e-7):
    """
    Compare two arrays or scalars with numpy's allclose semantics.
    """
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def max_abs_diff(a, b):
    """
    Max absolute difference between two arrays or scalars.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.nanmax(np.abs(a_arr - b_arr)))


# ========== Data generators ==========


def make_1d(n=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=n)


def make_2d(n_rows=200_000, n_cols=10, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, n_cols))


def make_pandas_series(x):
    return pd.Series(x)


def make_pandas_df(x2d):
    return pd.DataFrame(x2d)


# ========== CSV writer & summary helper ==========


def write_csv(rows, filename="bench_bunker_results.csv"):
    out_path = Path(filename)
    fieldnames = [
        "group",
        "op",
        "backend",
        "time_ms",
        "compare_to",
        "allclose",
        "max_abs_diff",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote results to {out_path.resolve()}")


def print_all_comparisons(path="bench_bunker_results.csv"):
    """
    Read the benchmark CSV and print *all* bunker vs reference comparisons,
    not just the top speedups.

    For each (group, op) where backend == "bunker" and compare_to is set,
    we compute the speedup or slowdown and print a line.
    """
    out_path = Path(path)
    if not out_path.exists():
        print(f"No CSV found at {out_path}, skipping comparison summary.")
        return

    with out_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Index rows by (group, op, backend)
    idx = {(r["group"], r["op"], r["backend"]): r for r in rows}

    comparisons = []

    for r in rows:
        if r.get("backend") != "bunker":
            continue
        ref_backend = r.get("compare_to") or ""
        if not ref_backend:
            continue

        ref_row = idx.get((r["group"], r["op"], ref_backend))
        if ref_row is None:
            continue

        try:
            t_bunker = float(r["time_ms"])
            t_ref = float(ref_row["time_ms"])
        except (TypeError, ValueError):
            continue

        if t_bunker <= 0.0 or t_ref <= 0.0:
            continue

        speed_ratio = t_ref / t_bunker
        if speed_ratio >= 1.0:
            direction = "faster than"
            factor = speed_ratio
        else:
            direction = "slower than"
            factor = 1.0 / speed_ratio

        comparisons.append(
            (
                r["group"],
                r["op"],
                ref_backend,
                t_ref,
                t_bunker,
                direction,
                factor,
                r.get("allclose", ""),
                r.get("max_abs_diff", ""),
            )
        )

    if not comparisons:
        print("No bunker comparisons found to summarize.")
        return

    # Sort by group then op for a nice stable listing
    comparisons.sort(key=lambda x: (x[0], x[1]))

    print("\nFull bunker_stats_rs vs reference comparisons:")
    print("------------------------------------------------")
    for (
        group,
        op,
        ref_backend,
        t_ref,
        t_bunker,
        direction,
        factor,
        allclose_flag,
        max_diff,
    ) in comparisons:
        print(
            f"[{group}] {op}: bunker is x{factor:.2f} {direction} {ref_backend} "
            f"(ref {t_ref:.3f} ms vs bunker {t_bunker:.3f} ms, "
            f"allclose={allclose_flag}, max_abs_diff={max_diff})"
        )


# ========== BASIC STATS ==========


def bench_basic_stats(rows):
    """
    mean / std / var / percentile / iqr / mad
    Compare NumPy vs bunker_stats_rs; IQR/MAD also have pure-Python refs.
    """
    group = "basic_stats"
    print(f"\n== {group} ==")

    x = make_1d()

    # mean
    print("  mean...")
    t_np, res_np = bench(lambda: np.mean(x))
    t_bs, res_bs = bench(lambda: bs.mean_np(x))
    rows.append(
        {
            "group": group,
            "op": "mean",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "mean",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(res_np, res_bs),
            "max_abs_diff": max_abs_diff(res_np, res_bs),
        }
    )

    # std
    print("  std...")
    t_np, res_np = bench(lambda: np.std(x, ddof=1))
    t_bs, res_bs = bench(lambda: bs.std_np(x))
    rows.append(
        {
            "group": group,
            "op": "std",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "std",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(res_np, res_bs),
            "max_abs_diff": max_abs_diff(res_np, res_bs),
        }
    )

    # var
    print("  var...")
    t_np, res_np = bench(lambda: np.var(x, ddof=1))
    t_bs, res_bs = bench(lambda: bs.var_np(x))
    rows.append(
        {
            "group": group,
            "op": "var",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "var",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(res_np, res_bs),
            "max_abs_diff": max_abs_diff(res_np, res_bs),
        }
    )

    # percentile
    print("  percentile (q=0.9)...")
    q = 0.9
    t_np, res_np = bench(lambda: float(np.percentile(x, q * 100)))
    t_bs, res_bs = bench(lambda: bs.percentile_np(x, q))
    rows.append(
        {
            "group": group,
            "op": "percentile_0.9",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "percentile_0.9",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(res_np, res_bs),
            "max_abs_diff": max_abs_diff(res_np, res_bs),
        }
    )

    # IQR (Python ref via NumPy percentiles)
    print("  iqr (python_ref)...")
    t_py, res_py = bench(lambda: iqr_py(x))
    t_bs, res_bs = bench(lambda: bs.iqr_np(x))
    rows.append(
        {
            "group": group,
            "op": "iqr_pyref",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "iqr_pyref",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(res_py, res_bs),
            "max_abs_diff": max_abs_diff(res_py, res_bs),
        }
    )

    # MAD (Python ref via NumPy median)
    print("  mad (python_ref)...")
    t_py, res_py = bench(lambda: mad_py(x))
    t_bs, res_bs = bench(lambda: bs.mad_np(x))
    rows.append(
        {
            "group": group,
            "op": "mad_pyref",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "mad_pyref",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(res_py, res_bs),
            "max_abs_diff": max_abs_diff(res_py, res_bs),
        }
    )


def iqr_py(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    return (float(q1), float(q3), float(q3 - q1))


def mad_py(x):
    med = np.median(x)
    devs = np.abs(x - med)
    return float(np.median(devs))


# ========== ROLLING STATS ==========


def bench_rolling(rows):
    """
    Rolling mean / std / zscore / ewma.
    Compare vs pandas.Series.rolling().
    """
    group = "rolling"
    print(f"\n== {group} ==")

    x = make_1d()
    s = make_pandas_series(x)
    window = 50

    # rolling mean
    print("  rolling mean...")
    t_pd, res_pd_full = bench(lambda: s.rolling(window).mean().to_numpy())
    res_pd = res_pd_full[window - 1 :]  # align shapes with bunker (n - w + 1)
    t_bs, res_bs = bench(lambda: bs.rolling_mean_np(x, window))
    rows.append(
        {
            "group": group,
            "op": "rolling_mean",
            "backend": "pandas",
            "time_ms": t_pd,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "rolling_mean",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "pandas",
            "allclose": allclose(res_pd, res_bs),
            "max_abs_diff": max_abs_diff(res_pd, res_bs),
        }
    )

    # rolling std
    print("  rolling std...")
    t_pd, res_pd_full = bench(lambda: s.rolling(window).std(ddof=1).to_numpy())
    res_pd = res_pd_full[window - 1 :]
    t_bs, res_bs = bench(lambda: bs.rolling_std_np(x, window))
    rows.append(
        {
            "group": group,
            "op": "rolling_std",
            "backend": "pandas",
            "time_ms": t_pd,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "rolling_std",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "pandas",
            "allclose": allclose(res_pd, res_bs),
            "max_abs_diff": max_abs_diff(res_pd, res_bs),
        }
    )

    # rolling zscore (z of last element)
    print("  rolling zscore (z of last element in window)...")
    t_py, res_py = bench(lambda: rolling_zscore_py(x, window))
    t_bs, res_bs = bench(lambda: bs.rolling_zscore_np(x, window))
    rows.append(
        {
            "group": group,
            "op": "rolling_zscore",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "rolling_zscore",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(res_py, res_bs),
            "max_abs_diff": max_abs_diff(res_py, res_bs),
        }
    )

    # ewma
    print("  ewma (alpha=0.1)...")
    alpha = 0.1
    t_np, res_np = bench(lambda: ewma_np_ref(x, alpha))
    t_bs, res_bs = bench(lambda: bs.ewma_np(x, alpha))
    rows.append(
        {
            "group": group,
            "op": "ewma",
            "backend": "numpy_ref",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "ewma",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy_ref",
            "allclose": allclose(res_np, res_bs),
            "max_abs_diff": max_abs_diff(res_np, res_bs),
        }
    )


def rolling_zscore_py(x, window):
    n = len(x)
    if window <= 0 or window > n:
        return np.array([], dtype=float)

    out = np.empty(n - window + 1, dtype=float)
    for i in range(window - 1, n):
        w_slice = x[i - window + 1 : i + 1]
        m = w_slice.mean()
        s = w_slice.std(ddof=1)
        out[i - window + 1] = (x[i] - m) / s
    return out


def ewma_np_ref(x, alpha):
    out = np.empty_like(x, dtype=float)
    prev = x[0]
    out[0] = prev
    for i in range(1, len(x)):
        val = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = val
        prev = val
    return out


# ========== OUTLIERS & SCALING ==========


def bench_outliers_scaling(rows):
    group = "outliers_scaling"
    print(f"\n== {group} ==")

    x = make_1d()

    # IQR outliers
    print("  iqr_outliers...")
    t_py, res_py = bench(lambda: iqr_outliers_py(x, 1.5))
    t_bs, res_bs = bench(lambda: bs.iqr_outliers_np(x, 1.5))
    rows.append(
        {
            "group": group,
            "op": "iqr_outliers",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "iqr_outliers",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": np.array_equal(res_py, res_bs),
            "max_abs_diff": float(np.sum(res_py != res_bs)),
        }
    )

    # zscore_outliers
    print("  zscore_outliers...")
    t_py, res_py = bench(lambda: zscore_outliers_py(x, 3.0))
    t_bs, res_bs = bench(lambda: bs.zscore_outliers_np(x, 3.0))
    rows.append(
        {
            "group": group,
            "op": "zscore_outliers",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "zscore_outliers",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": np.array_equal(res_py, res_bs),
            "max_abs_diff": float(np.sum(res_py != res_bs)),
        }
    )

    # minmax_scale
    print("  minmax_scale...")
    t_py, (scaled_py, min_py, max_py) = bench(lambda: minmax_scale_py(x))
    t_bs, (scaled_bs, min_bs, max_bs) = bench(lambda: bs.minmax_scale_np(x))
    rows.append(
        {
            "group": group,
            "op": "minmax_scale",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "minmax_scale",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(scaled_py, scaled_bs)
            and math.isclose(min_py, min_bs)
            and math.isclose(max_py, max_bs),
            "max_abs_diff": max_abs_diff(scaled_py, scaled_bs),
        }
    )

    # robust_scale
    print("  robust_scale...")
    scale_factor = 1.4826
    t_py, (scaled_py, med_py, mad_py_val) = bench(
        lambda: robust_scale_py(x, scale_factor)
    )
    t_bs, (scaled_bs, med_bs, mad_bs) = bench(
        lambda: bs.robust_scale_np(x, scale_factor)
    )
    rows.append(
        {
            "group": group,
            "op": "robust_scale",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "robust_scale",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(scaled_py, scaled_bs)
            and math.isclose(med_py, med_bs)
            and math.isclose(mad_py_val, mad_bs),
            "max_abs_diff": max_abs_diff(scaled_py, scaled_bs),
        }
    )

    # winsorize
    print("  winsorize...")
    t_py, res_py = bench(lambda: winsorize_py(x, 0.05, 0.95))
    t_bs, res_bs = bench(lambda: bs.winsorize_np(x, 0.05, 0.95))
    rows.append(
        {
            "group": group,
            "op": "winsorize",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "winsorize",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(res_py, res_bs),
            "max_abs_diff": max_abs_diff(res_py, res_bs),
        }
    )


def iqr_outliers_py(x, k):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return (x < low) | (x > high)


def zscore_outliers_py(x, threshold):
    m = x.mean()
    s = x.std(ddof=1)
    z = (x - m) / s
    return np.abs(z) > threshold


def minmax_scale_py(x):
    if x.size == 0:
        return (x.copy(), float("nan"), float("nan"))
    mn = float(x.min())
    mx = float(x.max())
    if mx == mn:
        return (np.zeros_like(x, dtype=float), mn, mx)
    scaled = (x - mn) / (mx - mn)
    return (scaled, mn, mx)


def robust_scale_py(x, scale_factor):
    if x.size == 0:
        return (x.copy(), float("nan"), float("nan"))
    med = float(np.median(x))
    devs = np.abs(x - med)
    mad_val = float(np.median(devs))
    denom = mad_val * scale_factor if mad_val != 0.0 else 1e-12
    scaled = (x - med) / denom
    return (scaled, med, mad_val)


def winsorize_py(x, lower_q, upper_q):
    if x.size == 0:
        return x.copy()
    low = float(np.percentile(x, lower_q * 100))
    high = float(np.percentile(x, upper_q * 100))
    return np.clip(x, low, high)


# ========== DIFF / CUM / ECDF / BINS / SIGNS ==========


def bench_diff_cum_etc(rows):
    group = "diff_cum_etc"
    print(f"\n== {group} ==")

    x = make_1d()
    s = make_pandas_series(x)

    # diff
    print("  diff(periods=1)...")
    periods = 1
    t_pd, res_pd = bench(lambda: s.diff(periods).to_numpy())
    t_bs, res_bs = bench(lambda: bs.diff_np(x, periods))
    rows.append(
        {
            "group": group,
            "op": "diff_1",
            "backend": "pandas",
            "time_ms": t_pd,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "diff_1",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "pandas",
            "allclose": allclose(res_pd, res_bs),
            "max_abs_diff": max_abs_diff(res_pd, res_bs),
        }
    )

    # pct_change
    print("  pct_change(periods=1)...")
    t_pd, res_pd = bench(lambda: s.pct_change(periods).to_numpy())
    t_bs, res_bs = bench(lambda: bs.pct_change_np(x, periods))
    rows.append(
        {
            "group": group,
            "op": "pct_change_1",
            "backend": "pandas",
            "time_ms": t_pd,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "pct_change_1",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "pandas",
            "allclose": allclose(res_pd, res_bs),
            "max_abs_diff": max_abs_diff(res_pd, res_bs),
        }
    )

    # cumsum
    print("  cumsum...")
    t_np, res_np = bench(lambda: np.cumsum(x))
    t_bs, res_bs = bench(lambda: bs.cumsum_np(x))
    rows.append(
        {
            "group": group,
            "op": "cumsum",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "cumsum",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(res_np, res_bs),
            "max_abs_diff": max_abs_diff(res_np, res_bs),
        }
    )

    # cummean
    print("  cummean...")
    t_py, res_py = bench(lambda: cummean_py(x))
    t_bs, res_bs = bench(lambda: bs.cummean_np(x))
    rows.append(
        {
            "group": group,
            "op": "cummean",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "cummean",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(res_py, res_bs),
            "max_abs_diff": max_abs_diff(res_py, res_bs),
        }
    )

    # ecdf
    print("  ecdf...")
    t_py, (vals_py, cdf_py) = bench(lambda: ecdf_py(x))
    t_bs, (vals_bs, cdf_bs) = bench(lambda: bs.ecdf_np(x))
    rows.append(
        {
            "group": group,
            "op": "ecdf",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "ecdf",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(vals_py, vals_bs)
            and allclose(cdf_py, cdf_bs),
            "max_abs_diff": max(
                max_abs_diff(vals_py, vals_bs),
                max_abs_diff(cdf_py, cdf_bs),
            ),
        }
    )

    # quantile_bins (vs qcut)
    print("  quantile_bins (10 bins vs pd.qcut)...")
    n_bins = 10
    t_pd, bins_pd = bench(
        lambda: pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
    )
    bins_pd = bins_pd.to_numpy()
    t_bs, bins_bs = bench(lambda: bs.quantile_bins_np(x, n_bins))
    rows.append(
        {
            "group": group,
            "op": "quantile_bins_10",
            "backend": "pandas",
            "time_ms": t_pd,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "quantile_bins_10",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "pandas",
            "allclose": np.array_equal(bins_pd, bins_bs),
            "max_abs_diff": float(np.sum(bins_pd != bins_bs)),
        }
    )

    # sign_mask
    print("  sign_mask...")
    t_py, res_py = bench(lambda: sign_mask_py(x))
    t_bs, res_bs = bench(lambda: bs.sign_mask_np(x))
    rows.append(
        {
            "group": group,
            "op": "sign_mask",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "sign_mask",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": np.array_equal(res_py, res_bs),
            "max_abs_diff": float(np.sum(res_py != res_bs)),
        }
    )

    # demean_with_signs
    print("  demean_with_signs...")
    t_py, (demeaned_py, signs_py) = bench(lambda: demean_with_signs_py(x))
    t_bs, (demeaned_bs, signs_bs) = bench(lambda: bs.demean_with_signs_np(x))
    rows.append(
        {
            "group": group,
            "op": "demean_with_signs",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "demean_with_signs",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(demeaned_py, demeaned_bs)
            and np.array_equal(signs_py, signs_bs),
            "max_abs_diff": max(
                max_abs_diff(demeaned_py, demeaned_bs),
                float(np.sum(signs_py != signs_bs)),
            ),
        }
    )


def cummean_py(x):
    out = np.empty_like(x, dtype=float)
    s = 0.0
    for i in range(len(x)):
        s += x[i]
        out[i] = s / (i + 1)
    return out


def ecdf_py(x):
    if x.size == 0:
        return (x.copy(), x.copy())
    vals = np.sort(x)
    n = len(vals)
    cdf = np.arange(1, n + 1, dtype=float) / n
    return (vals, cdf)


def sign_mask_py(x):
    out = np.empty_like(x, dtype=np.int8)
    out[x > 0.0] = 1
    out[x < 0.0] = -1
    out[x == 0.0] = 0
    return out


def demean_with_signs_py(x):
    m = x.mean()
    demeaned = x - m
    signs = sign_mask_py(demeaned)
    return (demeaned, signs)


# ========== COVARIANCE / CORRELATION ==========


def bench_cov_corr(rows):
    """
    cov / corr for vectors and matrices, rolling cov/corr.
    Compare vs NumPy and pandas.
    """
    group = "cov_corr"
    print(f"\n== {group} ==")

    # vector cov/corr
    print("  cov_np / corr_np (vector)...")
    x = make_1d()
    y = make_1d(seed=123)

    # cov
    t_np, res_np = bench(lambda: float(np.cov(x, y, ddof=1)[0, 1]))
    t_bs, res_bs = bench(lambda: bs.cov_np(x, y))
    rows.append(
        {
            "group": group,
            "op": "cov_pair",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "cov_pair",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(res_np, res_bs),
            "max_abs_diff": max_abs_diff(res_np, res_bs),
        }
    )

    # corr
    t_np, res_np = bench(lambda: float(np.corrcoef(x, y)[0, 1]))
    t_bs, res_bs = bench(lambda: bs.corr_np(x, y))
    rows.append(
        {
            "group": group,
            "op": "corr_pair",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "corr_pair",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(res_np, res_bs),
            "max_abs_diff": max_abs_diff(res_np, res_bs),
        }
    )

    # matrix cov/corr
    print("  cov_matrix_np / corr_matrix_np (matrix)...")
    x2d = make_2d()
    t_np, cov_np = bench(lambda: np.cov(x2d, rowvar=False, ddof=1))
    t_bs, cov_bs = bench(lambda: bs.cov_matrix_np(x2d))
    rows.append(
        {
            "group": group,
            "op": "cov_matrix",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "cov_matrix",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(cov_np, cov_bs),
            "max_abs_diff": max_abs_diff(cov_np, cov_bs),
        }
    )

    t_np, corr_np = bench(lambda: np.corrcoef(x2d, rowvar=False))
    t_bs, corr_bs = bench(lambda: bs.corr_matrix_np(x2d))
    rows.append(
        {
            "group": group,
            "op": "corr_matrix",
            "backend": "numpy",
            "time_ms": t_np,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "corr_matrix",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "numpy",
            "allclose": allclose(corr_np, corr_bs),
            "max_abs_diff": max_abs_diff(corr_np, corr_bs),
        }
    )

    # rolling covariance / correlation (vector)
    print("  rolling_cov / rolling_corr (vector)...")
    window = 50
    s_x = make_pandas_series(x)
    s_y = make_pandas_series(y)

    def rolling_cov_pd():
        rc = s_x.rolling(window).cov(s_y)
        return rc.to_numpy()[window - 1 :]

    def rolling_corr_pd():
        rc = s_x.rolling(window).corr(s_y)
        return rc.to_numpy()[window - 1 :]

    t_pd, cov_pd = bench(rolling_cov_pd)
    t_bs, cov_bs = bench(lambda: bs.rolling_cov_np(x, y, window))
    rows.append(
        {
            "group": group,
            "op": "rolling_cov",
            "backend": "pandas",
            "time_ms": t_pd,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "rolling_cov",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "pandas",
            "allclose": allclose(cov_pd, cov_bs),
            "max_abs_diff": max_abs_diff(cov_pd, cov_bs),
        }
    )

    t_pd, corr_pd = bench(rolling_corr_pd)
    t_bs, corr_bs = bench(lambda: bs.rolling_corr_np(x, y, window))
    rows.append(
        {
            "group": group,
            "op": "rolling_corr",
            "backend": "pandas",
            "time_ms": t_pd,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "rolling_corr",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "pandas",
            "allclose": allclose(corr_pd, corr_bs),
            "max_abs_diff": max_abs_diff(corr_pd, corr_bs),
        }
    )


# ========== KDE & SciPy-only comparisons ==========


def bench_kde(rows):
    """
    KDE smoothing vs pure-NumPy reference.
    """
    group = "kde"
    print(f"\n== {group} ==")

    x = make_1d()
    n_points = 512

    print("  kde_gaussian (python_ref)...")
    t_py, (grid_py, dens_py) = bench(
        lambda: kde_gaussian_py(x, n_points, None)
    )
    t_bs, (grid_bs, dens_bs) = bench(
        lambda: bs.kde_gaussian_np(x, n_points, None)
    )

    rows.append(
        {
            "group": group,
            "op": "kde_gaussian_pyref",
            "backend": "python_ref",
            "time_ms": t_py,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "kde_gaussian_pyref",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "python_ref",
            "allclose": allclose(grid_py, grid_bs)
            and allclose(dens_py, dens_bs, rtol=1e-5, atol=1e-7),
            "max_abs_diff": max(
                max_abs_diff(grid_py, grid_bs),
                max_abs_diff(dens_py, dens_bs),
            ),
        }
    )


def kde_gaussian_py(xs, n_points, bandwidth):
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    if n == 0 or n_points == 0:
        return (np.array([], dtype=float), np.array([], dtype=float))

    s = xs.std(ddof=1)
    if bandwidth is None or bandwidth <= 0.0:
        bw = 1.06 * s * n ** (-1.0 / 5.0)
    else:
        bw = bandwidth

    mn = xs.min()
    mx = xs.max()
    if mx == mn:
        grid = np.full(n_points, mn, dtype=float)
        dens = np.zeros_like(grid)
        return (grid, dens)

    grid = np.linspace(mn, mx, n_points)
    norm_factor = 1.0 / (bw * math.sqrt(2.0 * math.pi))

    dens = np.empty(n_points, dtype=float)
    for i in range(n_points):
        x0 = grid[i]
        z = (x0 - xs) / bw
        dens[i] = norm_factor * np.exp(-0.5 * z * z).mean()

    return (grid, dens)


def bench_scipy_vs_bunker(rows):
    """
    Compare bunker_stats against SciPy for operations that NumPy doesn't provide
    directly: IQR, MAD, KDE (gaussian_kde).
    """
    if not HAVE_SCIPY:
        print("\n== scipy_compare ==\nSciPy not installed; skipping SciPy-based benchmarks.")
        return

    group = "scipy_compare"
    print(f"\n== {group} ==")

    x = make_1d()

    # SciPy IQR
    print("  scipy.stats.iqr...")
    t_sp, iqr_sp = bench(lambda: iqr_scipy(x))
    t_bs, iqr_bs = bench(lambda: bs.iqr_np(x))
    rows.append(
        {
            "group": group,
            "op": "iqr_scipy",
            "backend": "scipy",
            "time_ms": t_sp,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "iqr_scipy",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "scipy",
            "allclose": allclose(iqr_sp, iqr_bs),
            "max_abs_diff": max_abs_diff(iqr_sp, iqr_bs),
        }
    )

    # SciPy MAD
    print("  scipy.stats.median_abs_deviation...")
    t_sp, mad_sp = bench(lambda: mad_scipy(x))
    t_bs, mad_bs = bench(lambda: bs.mad_np(x))
    rows.append(
        {
            "group": group,
            "op": "mad_scipy",
            "backend": "scipy",
            "time_ms": t_sp,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "mad_scipy",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "scipy",
            "allclose": allclose(mad_sp, mad_bs),
            "max_abs_diff": max_abs_diff(mad_sp, mad_bs),
        }
    )

    # SciPy KDE (gaussian_kde)
    print("  scipy.stats.gaussian_kde...")
    n_points = 512
    t_sp, (grid_sp, dens_sp) = bench(lambda: kde_scipy(x, n_points))
    t_bs, (grid_bs, dens_bs) = bench(lambda: bs.kde_gaussian_np(x, n_points, None))
    rows.append(
        {
            "group": group,
            "op": "kde_gaussian_scipy",
            "backend": "scipy",
            "time_ms": t_sp,
            "compare_to": "",
            "allclose": True,
            "max_abs_diff": 0.0,
        }
    )
    rows.append(
        {
            "group": group,
            "op": "kde_gaussian_scipy",
            "backend": "bunker",
            "time_ms": t_bs,
            "compare_to": "scipy",
            "allclose": allclose(grid_sp, grid_bs)
            and allclose(dens_sp, dens_bs, rtol=1e-5, atol=1e-7),
            "max_abs_diff": max(
                max_abs_diff(grid_sp, grid_bs),
                max_abs_diff(dens_sp, dens_bs),
            ),
        }
    )


def iqr_scipy(x):
    q1_q3 = stats.iqr(x, rng=(25, 75), scale=1.0, nan_policy="omit", keepdims=False)
    # scipy.stats.iqr returns Q3-Q1; we reconstruct Q1, Q3 via percentiles for consistency
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    return (float(q1), float(q3), float(q1_q3))


def mad_scipy(x):
    # SciPy returns MAD; we match bunker-stats behavior (no scaling by default)
    return float(stats.median_abs_deviation(x, scale=1.0, nan_policy="omit"))


def kde_scipy(x, n_points):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or n_points == 0:
        return (np.array([], dtype=float), np.array([], dtype=float))
    kde = stats.gaussian_kde(x)
    mn = x.min()
    mx = x.max()
    if mx == mn:
        grid = np.full(n_points, mn, dtype=float)
        dens = np.zeros_like(grid)
        return (grid, dens)
    grid = np.linspace(mn, mx, n_points)
    dens = kde(grid)
    return (grid, dens)


# ========== MAIN ==========


if __name__ == "__main__":
    rows = []
    bench_basic_stats(rows)
    bench_rolling(rows)
    bench_outliers_scaling(rows)
    bench_diff_cum_etc(rows)
    bench_cov_corr(rows)
    bench_kde(rows)
    bench_scipy_vs_bunker(rows)

    write_csv(rows)
    print_all_comparisons()
