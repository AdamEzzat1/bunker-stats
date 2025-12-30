
"""
bench_suite_all.py

Benchmarks bunker-stats functions against:
- pure Python references (where reasonable)
- NumPy
- pandas
- SciPy (if installed)

Measures:
- Speed: per-run wall time (ms) across many repeats
- Stability: stddev / mean of timing (coefficient of variation, CV)
- Accuracy: allclose checks + max_abs_diff (when a meaningful reference exists)

Usage:
    python bench_suite_all.py --n 1000000 --window 50 --repeats 25 --warmup 3 --seed 0

Notes:
- Rolling functions in bunker-stats (non-NaN-aware) return length (n - window + 1).
  pandas rolling references are sliced to match this truncated shape.
- ddof semantics: bunker-stats uses sample statistics (ddof=1).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional deps
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import scipy
    import scipy.stats as st
except Exception:  # pragma: no cover
    scipy = None
    st = None

import bunker_stats as bs


@dataclass
class BenchResult:
    func: str
    impl: str
    n: int
    window: int
    repeats: int
    warmup: int
    median_ms: float
    mean_ms: float
    std_ms: float
    cv: float
    ok: Optional[bool]
    max_abs_diff: Optional[float]
    note: str = ""


def _now_ns() -> int:
    return time.perf_counter_ns()


def timeit(fn: Callable[[], Any], warmup: int, repeats: int) -> Tuple[np.ndarray, Any]:
    # warmup
    out = None
    for _ in range(warmup):
        out = fn()

    times = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = _now_ns()
        out = fn()
        t1 = _now_ns()
        times[i] = (t1 - t0) / 1e6  # ms
    return times, out


def max_abs_diff(a: Any, b: Any) -> float:
    # Handle tuple outputs recursively (e.g., (arr, scalar), (arr1, arr2))
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return float("inf")
        diffs = [max_abs_diff(ai, bi) for ai, bi in zip(a, b)]
        if any(isinstance(d, float) and np.isnan(d) for d in diffs):
            return float("nan")
        return float(max(diffs)) if diffs else 0.0

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        return float("inf")

    if a.size == 0 and b.size == 0:
        return 0.0

    # Boolean outputs: use XOR mismatch rate
    if a.dtype == np.bool_ or b.dtype == np.bool_:
        mism = np.logical_xor(a, b)
        return float(np.mean(mism))

    # Numeric diff path
    try:
        d = np.abs(a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False))
        if np.isnan(d).all():
            return float("nan")
        return float(np.nanmax(d))
    except Exception:
        mism = (a != b)
        return float(np.mean(mism))


def allclose(a: Any, b: Any, *, rtol=1e-7, atol=1e-9) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(allclose(ai, bi, rtol=rtol, atol=atol) for ai, bi in zip(a, b))

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype == np.bool_ or b.dtype == np.bool_:
        return bool(np.array_equal(a, b))
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


# -------------------------
# Reference implementations
# -------------------------

def py_mean(xs: np.ndarray) -> float:
    s = 0.0
    for x in xs:
        s += float(x)
    return s / float(len(xs)) if len(xs) else float("nan")


def py_var_sample(xs: np.ndarray) -> float:
    n = len(xs)
    if n <= 1:
        return float("nan")
    m = py_mean(xs)
    acc = 0.0
    for x in xs:
        d = float(x) - m
        acc += d * d
    return acc / float(n - 1)


def py_std_sample(xs: np.ndarray) -> float:
    v = py_var_sample(xs)
    return math.sqrt(v) if not math.isnan(v) else float("nan")


def py_percentile(xs: np.ndarray, q: float) -> float:
    if len(xs) == 0:
        return float("nan")
    v = sorted(float(x) for x in xs)
    n = len(v)
    if n == 1:
        return v[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return v[lo]
    w = pos - lo
    return (1.0 - w) * v[lo] + w * v[hi]


def py_iqr(xs: np.ndarray) -> Tuple[float, float, float]:
    if len(xs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    v = np.sort(xs.astype(np.float64, copy=True))
    q1 = py_percentile(v, 0.25)
    q3 = py_percentile(v, 0.75)
    return (q1, q3, q3 - q1)


def py_mad(xs: np.ndarray) -> float:
    if len(xs) == 0:
        return float("nan")
    v = np.sort(xs.astype(np.float64, copy=True))
    n = len(v)
    med = v[n // 2] if (n % 2 == 1) else 0.5 * (v[n // 2 - 1] + v[n // 2])
    dev = np.abs(xs - med)
    dev = np.sort(dev)
    m = len(dev)
    return float(dev[m // 2] if (m % 2 == 1) else 0.5 * (dev[m // 2 - 1] + dev[m // 2]))


def ref_rolling_slice(arr_full: np.ndarray, window: int) -> np.ndarray:
    # slice pandas rolling outputs to match bunker-stats truncated outputs
    return arr_full[window - 1 :]


def ref_rolling_mean_pandas(x: np.ndarray, window: int) -> np.ndarray:
    if pd is None:
        raise RuntimeError("pandas not installed")
    s = pd.Series(x)
    out = s.rolling(window, min_periods=window).mean().to_numpy()
    return ref_rolling_slice(out, window)


def ref_rolling_std_pandas(x: np.ndarray, window: int) -> np.ndarray:
    if pd is None:
        raise RuntimeError("pandas not installed")
    s = pd.Series(x)
    out = s.rolling(window, min_periods=window).std(ddof=1).to_numpy()
    return ref_rolling_slice(out, window)


def ref_rolling_zscore_pandas(x: np.ndarray, window: int) -> np.ndarray:
    if pd is None:
        raise RuntimeError("pandas not installed")
    s = pd.Series(x)
    roll = s.rolling(window, min_periods=window)
    m = roll.mean()
    sd = roll.std(ddof=1)
    out = ((s - m) / sd).to_numpy()
    return ref_rolling_slice(out, window)


def ref_ewma_pandas(x: np.ndarray, alpha: float) -> np.ndarray:
    if pd is None:
        raise RuntimeError("pandas not installed")
    s = pd.Series(x)
    # adjust=False matches recursive definition used in bs.ewma_np
    return s.ewm(alpha=alpha, adjust=False).mean().to_numpy()


def ref_cov_matrix_numpy(X: np.ndarray) -> np.ndarray:
    # ddof=1 matches sample covariance. rowvar=False expects columns = features
    return np.cov(X, rowvar=False, ddof=1)


def ref_corr_matrix_numpy(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


def ref_cov_numpy(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.cov(x, y, ddof=1)[0, 1])


def ref_corr_numpy(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def ref_kde_scipy(x: np.ndarray, n_points: int, bandwidth: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if st is None:
        raise RuntimeError("scipy not installed")
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or n_points == 0:
        return (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    mn, mx = float(np.min(x)), float(np.max(x))
    grid = np.linspace(mn, mx, n_points, dtype=np.float64)
    kde = st.gaussian_kde(x)  # note: different default bandwidth than bs
    dens = kde(grid)
    return grid, dens


def ref_kde_python_same_rule(x: np.ndarray, n_points: int, bandwidth: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    # matches bs.kde_gaussian_np bandwidth fallback: 1.06*std*n^(-1/5)
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or n_points == 0:
        return (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    mn, mx = float(np.min(x)), float(np.max(x))
    grid = np.linspace(mn, mx, n_points, dtype=np.float64)

    if n <= 1:
        bw = 1e-6
    else:
        std = float(np.std(x, ddof=1))
        bw = bandwidth if (bandwidth is not None and bandwidth > 0.0) else (1e-6 if std == 0.0 else 1.06 * std * (n ** (-1.0 / 5.0)))

    if mx == mn:
        return np.full(n_points, mn, dtype=np.float64), np.zeros(n_points, dtype=np.float64)

    inv = 1.0 / (bw * math.sqrt(2.0 * math.pi))
    dens = np.empty(n_points, dtype=np.float64)
    for i, x0 in enumerate(grid):
        z = (x0 - x) / bw
        dens[i] = inv * float(np.mean(np.exp(-0.5 * z * z)))
    return grid, dens


# -------------------------
# Suite definition
# -------------------------

@dataclass
class Case:
    name: str
    make_args: Callable[[], Tuple]
    impls: List[Tuple[str, Callable[..., Any]]]
    ref: Optional[Callable[..., Any]] = None
    ref_note: str = ""
    rtol: float = 1e-7
    atol: float = 1e-9


def run_case(case: Case, n: int, window: int, warmup: int, repeats: int) -> List[BenchResult]:
    args = case.make_args()
    ref_out = None
    if case.ref is not None:
        ref_out = case.ref(*args)

    results: List[BenchResult] = []
    for impl_name, impl_fn in case.impls:
        def call():
            return impl_fn(*args)

        times, out = timeit(call, warmup=warmup, repeats=repeats)

        ok = None
        mad = None
        note = case.ref_note
        if ref_out is not None:
            ok = allclose(out, ref_out, rtol=case.rtol, atol=case.atol)
            mad = max_abs_diff(out, ref_out)

        results.append(
            BenchResult(
                func=case.name,
                impl=impl_name,
                n=n,
                window=window,
                repeats=repeats,
                warmup=warmup,
                median_ms=float(np.median(times)),
                mean_ms=float(np.mean(times)),
                std_ms=float(np.std(times, ddof=1)) if repeats > 1 else 0.0,
                cv=float(np.std(times, ddof=1) / np.mean(times)) if repeats > 1 and np.mean(times) > 0 else 0.0,
                ok=ok,
                max_abs_diff=mad,
                note=note,
            )
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--p", type=int, default=32, help="feature count for matrix benchmarks")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench_suite_results.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = int(args.n)
    p = int(args.p)
    window = int(args.window)

    x = rng.normal(size=n).astype(np.float64)
    y = rng.normal(size=n).astype(np.float64)
    X = rng.normal(size=(n, p)).astype(np.float64)

    alpha = 0.1
    q = 0.95
    lower_q, upper_q = 0.05, 0.95
    n_bins = 10
    kde_points = 256

    # Build cases
    cases: List[Case] = []

    # Scalars
    cases += [
        Case(
            "mean",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.mean_np),
                ("numpy", np.mean),
                ("python", py_mean),
            ],
            ref=np.mean,
        ),
        Case(
            "var_sample(ddof=1)",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.var_np),
                ("numpy", lambda v: np.var(v, ddof=1)),
                ("python", py_var_sample),
            ],
            ref=lambda v: np.var(v, ddof=1),
        ),
        Case(
            "std_sample(ddof=1)",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.std_np),
                ("numpy", lambda v: np.std(v, ddof=1)),
                ("python", py_std_sample),
            ],
            ref=lambda v: np.std(v, ddof=1),
        ),
        Case(
            "percentile(q=0.95)",
            make_args=lambda: (x, q),
            impls=[
                ("bunker_stats", bs.percentile_np),
                ("numpy", lambda v, qq: float(np.quantile(v, qq))),
                ("python", py_percentile),
            ],
            ref=lambda v, qq: float(np.quantile(v, qq)),
        ),
        Case(
            "iqr",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.iqr_np),
                ("python", py_iqr),
                ("scipy", (lambda v: (float(st.scoreatpercentile(v, 25)), float(st.scoreatpercentile(v, 75)), float(st.iqr(v, rng=(25, 75)))))) if st is not None else ("scipy_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("scipy not installed"))),
            ],
            ref=py_iqr,
            ref_note="SciPy IQR uses its own implementation; python ref matches bs interpolation style.",
        ),
        Case(
            "mad",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.mad_np),
                ("python", py_mad),
                ("scipy", (lambda v: float(st.median_abs_deviation(v, scale=1.0))) if st is not None else ("scipy_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("scipy not installed"))),
            ],
            ref=py_mad,
            ref_note="SciPy MAD may differ slightly by definition/scale options; python ref matches bs median/MAD.",
        ),
    ]

    # Multi-D
    cases += [
        Case(
            "mean_axis(axis=0)",
            make_args=lambda: (X, 0, False),
            impls=[
                ("bunker_stats", bs.mean_axis_np),
                ("numpy", lambda A, axis, skipna: np.mean(A, axis=axis)),
                ("pandas", (lambda A, axis, skipna: pd.DataFrame(A).mean(axis=axis, skipna=skipna).to_numpy()) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=lambda A, axis, skipna: np.mean(A, axis=axis),
        ),
        Case(
            "mean_over_last_axis_dyn",
            make_args=lambda: (X,),
            impls=[
                ("bunker_stats", bs.mean_over_last_axis_dyn_np),
                ("numpy", lambda A: np.mean(A, axis=-1).reshape(-1)),
            ],
            ref=lambda A: np.mean(A, axis=-1).reshape(-1),
        ),
    ]

    # Rolling
    cases += [
        Case(
            "rolling_mean(window)",
            make_args=lambda: (x, window),
            impls=[
                ("bunker_stats", bs.rolling_mean_np),
                ("pandas", (lambda v, w: ref_rolling_mean_pandas(v, w)) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=(lambda v, w: ref_rolling_mean_pandas(v, w)) if pd is not None else None,
        ),
        Case(
            "rolling_std(window)",
            make_args=lambda: (x, window),
            impls=[
                ("bunker_stats", bs.rolling_std_np),
                ("pandas", (lambda v, w: ref_rolling_std_pandas(v, w)) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=(lambda v, w: ref_rolling_std_pandas(v, w)) if pd is not None else None,
        ),
        Case(
            "rolling_mean_std(window)",
            make_args=lambda: (x, window),
            impls=[
                ("bunker_stats", bs.rolling_mean_std_np),
            ],
            ref=(lambda v, w: (ref_rolling_mean_pandas(v, w), ref_rolling_std_pandas(v, w))) if pd is not None else None,
            ref_note="Reference is (pandas rolling mean, pandas rolling std) sliced to match truncated output.",
        ),
        Case(
            "rolling_zscore(window)",
            make_args=lambda: (x, window),
            impls=[
                ("bunker_stats", bs.rolling_zscore_np),
                ("pandas", (lambda v, w: ref_rolling_zscore_pandas(v, w)) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=(lambda v, w: ref_rolling_zscore_pandas(v, w)) if pd is not None else None,
        ),
        Case(
            "ewma(alpha=0.1)",
            make_args=lambda: (x, alpha),
            impls=[
                ("bunker_stats", bs.ewma_np),
                ("pandas", (lambda v, a: ref_ewma_pandas(v, a)) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=(lambda v, a: ref_ewma_pandas(v, a)) if pd is not None else None,
            ref_note="pandas reference uses adjust=False to match recursive EWMA.",
        ),
    ]

    # Outliers / scaling (refs are simple numpy)
    cases += [
        Case(
            "iqr_outliers(k=1.5)",
            make_args=lambda: (x, 1.5),
            impls=[
                ("bunker_stats", bs.iqr_outliers_np),
                ("numpy", lambda v, k: ((v < (np.quantile(v, 0.25) - k*(np.quantile(v, 0.75)-np.quantile(v, 0.25)))) | (v > (np.quantile(v, 0.75) + k*(np.quantile(v, 0.75)-np.quantile(v, 0.25)))))),
            ],
            ref=lambda v, k: ((v < (np.quantile(v, 0.25) - k*(np.quantile(v, 0.75)-np.quantile(v, 0.25)))) | (v > (np.quantile(v, 0.75) + k*(np.quantile(v, 0.75)-np.quantile(v, 0.25))))),
            atol=0.0,
            rtol=0.0,
        ),
        Case(
            "zscore_outliers(threshold=3)",
            make_args=lambda: (x, 3.0),
            impls=[
                ("bunker_stats", bs.zscore_outliers_np),
                ("numpy", lambda v, t: (np.abs((v - np.mean(v)) / np.std(v, ddof=1)) > t)),
            ],
            ref=lambda v, t: (np.abs((v - np.mean(v)) / np.std(v, ddof=1)) > t),
            atol=0.0,
            rtol=0.0,
        ),
        Case(
            "minmax_scale",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.minmax_scale_np),
                ("numpy", lambda v: ((v - np.min(v)) / (np.max(v) - np.min(v)), float(np.min(v)), float(np.max(v))) if np.max(v) != np.min(v) else (np.zeros_like(v), float(np.min(v)), float(np.max(v)))),
            ],
            ref=lambda v: ((v - np.min(v)) / (np.max(v) - np.min(v)), float(np.min(v)), float(np.max(v))) if np.max(v) != np.min(v) else (np.zeros_like(v), float(np.min(v)), float(np.max(v))),
        ),
        Case(
            "robust_scale(scale_factor=1.4826)",
            make_args=lambda: (x, 1.4826),
            impls=[
                ("bunker_stats", bs.robust_scale_np),
                ("numpy", lambda v, sf: ( (v - np.median(v)) / ( (py_mad(v)*sf) if py_mad(v) != 0.0 else 1e-12 ), float(np.median(v)), float(py_mad(v)) )),
            ],
            ref=lambda v, sf: ( (v - np.median(v)) / ( (py_mad(v)*sf) if py_mad(v) != 0.0 else 1e-12 ), float(np.median(v)), float(py_mad(v)) ),
        ),
        Case(
            "winsorize(0.05,0.95)",
            make_args=lambda: (x, lower_q, upper_q),
            impls=[
                ("bunker_stats", bs.winsorize_np),
                ("numpy", lambda v, lq, uq: np.clip(v, np.quantile(v, lq), np.quantile(v, uq))),
                ("scipy", (lambda v, lq, uq: st.mstats.winsorize(v, limits=(lq, 1-uq)).astype(np.float64)) if st is not None else ("scipy_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("scipy not installed"))),
            ],
            ref=lambda v, lq, uq: np.clip(v, np.quantile(v, lq), np.quantile(v, uq)),
        ),
    ]

    # diff / pct_change / cumsum / cummean / ecdf / bins / signs
    def ref_diff_full(v: np.ndarray, periods: int) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        n = v.size
        if n == 0 or periods == 0:
            return np.zeros(n, dtype=np.float64)
        p = abs(int(periods))
        if p >= n:
            return np.full(n, np.nan, dtype=np.float64)
        out = np.full(n, np.nan, dtype=np.float64)
        if periods > 0:
            out[p:] = v[p:] - v[:-p]
        else:
            out[: n - p] = v[: n - p] - v[p:]
        return out

    def ref_pct_change_full(v: np.ndarray, periods: int) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        n = v.size
        if n == 0 or periods == 0:
            return np.full(n, np.nan, dtype=np.float64)
        p = abs(int(periods))
        if p >= n:
            return np.full(n, np.nan, dtype=np.float64)
        out = np.full(n, np.nan, dtype=np.float64)
        if periods > 0:
            base = v[:-p]
            num = v[p:] - base
            out[p:] = np.where(base == 0.0, np.nan, num / base)
        else:
            base = v[p:]
            num = v[:-p] - base
            out[: n - p] = np.where(base == 0.0, np.nan, num / base)
        return out

    cases += [
        Case(
            "diff(periods=1)",
            make_args=lambda: (x, 1),
            impls=[
                ("bunker_stats", bs.diff_np),
                ("numpy", ref_diff_full),
                ("pandas", (lambda v, p: pd.Series(v).diff(p).to_numpy()) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=ref_diff_full,
        ),
        Case(
            "pct_change(periods=1)",
            make_args=lambda: (x, 1),
            impls=[
                ("bunker_stats", bs.pct_change_np),
                ("numpy", ref_pct_change_full),
                ("pandas", (lambda v, p: pd.Series(v).pct_change(p).to_numpy()) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=ref_pct_change_full,
        ),
        Case(
            "cumsum",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.cumsum_np),
                ("numpy", np.cumsum),
            ],
            ref=np.cumsum,
        ),
        Case(
            "cummean",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.cummean_np),
                ("numpy", lambda v: np.cumsum(v) / np.arange(1, v.size + 1, dtype=np.float64)),
            ],
            ref=lambda v: np.cumsum(v) / np.arange(1, v.size + 1, dtype=np.float64),
        ),
        Case(
            "ecdf",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.ecdf_np),
                ("python", lambda v: (np.sort(v), (np.arange(1, v.size + 1, dtype=np.float64) / v.size))),
            ],
            ref=lambda v: (np.sort(v), (np.arange(1, v.size + 1, dtype=np.float64) / v.size)),
        ),
        Case(
            "quantile_bins(n_bins=10)",
            make_args=lambda: (x, n_bins),
            impls=[
                ("bunker_stats", bs.quantile_bins_np),
                ("pandas", (lambda v, nb: pd.qcut(v, q=nb, labels=False, duplicates="drop").to_numpy(dtype=np.int64)) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=None,
            ref_note="Quantile bin edges may differ from pandas qcut when duplicates occur; speed-only comparison.",
        ),
        Case(
            "sign_mask",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.sign_mask_np),
                ("numpy", lambda v: np.sign(v).astype(np.int8)),
            ],
            ref=lambda v: np.sign(v).astype(np.int8),
            atol=0.0,
            rtol=0.0,
        ),
        Case(
            "demean_with_signs",
            make_args=lambda: (x,),
            impls=[
                ("bunker_stats", bs.demean_with_signs_np),
                ("numpy", lambda v: (v - np.mean(v), np.sign(v - np.mean(v)).astype(np.int8))),
            ],
            ref=lambda v: (v - np.mean(v), np.sign(v - np.mean(v)).astype(np.int8)),
        ),
    ]

    # Cov/corr
    cases += [
        Case(
            "cov(x,y)",
            make_args=lambda: (x, y),
            impls=[
                ("bunker_stats", bs.cov_np),
                ("numpy", ref_cov_numpy),
                ("pandas", (lambda a, b: float(pd.Series(a).cov(pd.Series(b)))) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=ref_cov_numpy,
        ),
        Case(
            "corr(x,y)",
            make_args=lambda: (x, y),
            impls=[
                ("bunker_stats", bs.corr_np),
                ("numpy", ref_corr_numpy),
                ("pandas", (lambda a, b: float(pd.Series(a).corr(pd.Series(b)))) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=ref_corr_numpy,
        ),
        Case(
            "cov_matrix",
            make_args=lambda: (X,),
            impls=[
                ("bunker_stats", bs.cov_matrix_np),
                ("numpy", ref_cov_matrix_numpy),
                ("pandas", (lambda A: pd.DataFrame(A).cov().to_numpy()) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=ref_cov_matrix_numpy,
        ),
        Case(
            "corr_matrix",
            make_args=lambda: (X,),
            impls=[
                ("bunker_stats", bs.corr_matrix_np),
                ("numpy", ref_corr_matrix_numpy),
                ("pandas", (lambda A: pd.DataFrame(A).corr().to_numpy()) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=ref_corr_matrix_numpy,
        ),
        Case(
            "rolling_cov(window)",
            make_args=lambda: (x, y, window),
            impls=[
                ("bunker_stats", bs.rolling_cov_np),
                ("pandas", (lambda a, b, w: ref_rolling_slice(pd.Series(a).rolling(w, min_periods=w).cov(pd.Series(b)).to_numpy(), w)) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=(lambda a, b, w: ref_rolling_slice(pd.Series(a).rolling(w, min_periods=w).cov(pd.Series(b)).to_numpy(), w)) if pd is not None else None,
        ),
        Case(
            "rolling_corr(window)",
            make_args=lambda: (x, y, window),
            impls=[
                ("bunker_stats", bs.rolling_corr_np),
                ("pandas", (lambda a, b, w: ref_rolling_slice(pd.Series(a).rolling(w, min_periods=w).corr(pd.Series(b)).to_numpy(), w)) if pd is not None else ("pandas_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("pandas not installed"))),
            ],
            ref=(lambda a, b, w: ref_rolling_slice(pd.Series(a).rolling(w, min_periods=w).corr(pd.Series(b)).to_numpy(), w)) if pd is not None else None,
        ),
    ]

    # KDE: speed + two refs (scipy default bandwidth differs)
    cases += [
        Case(
            "kde_gaussian(n_points=256)",
            make_args=lambda: (x, kde_points, None),
            impls=[
                ("bunker_stats", bs.kde_gaussian_np),
                ("python_same_rule", ref_kde_python_same_rule),
                ("scipy_default", ref_kde_scipy) if st is not None else ("scipy_missing", lambda *_: (_ for _ in ()).throw(RuntimeError("scipy not installed"))),
            ],
            ref=ref_kde_python_same_rule,
            ref_note="SciPy gaussian_kde uses a different default bandwidth than bunker-stats; accuracy is checked against python_same_rule.",
            rtol=1e-6,
            atol=1e-8,
        ),
    ]

    # Run suite
    all_results: List[BenchResult] = []
    for case in cases:
        # Skip placeholder impl tuples produced above
        impls_clean: List[Tuple[str, Callable[..., Any]]] = []
        for item in case.impls:
            if isinstance(item[0], str) and callable(item[1]):
                impls_clean.append(item)
            else:
                # item may be ("scipy_missing", fn)
                impls_clean.append(item)  # type: ignore
        case.impls = impls_clean

        try:
            all_results.extend(run_case(case, n=n, window=window, warmup=args.warmup, repeats=args.repeats))
            print(f"[ok] {case.name}")
        except Exception as e:
            print(f"[skip] {case.name}: {e}")

    # Write CSV
    import csv
    out_path = args.out
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "func","impl","n","window","repeats","warmup",
            "median_ms","mean_ms","std_ms","cv",
            "ok","max_abs_diff","note"
        ])
        for r in all_results:
            w.writerow([
                r.func, r.impl, r.n, r.window, r.repeats, r.warmup,
                f"{r.median_ms:.6f}", f"{r.mean_ms:.6f}", f"{r.std_ms:.6f}", f"{r.cv:.6f}",
                "" if r.ok is None else str(bool(r.ok)),
                "" if r.max_abs_diff is None else f"{r.max_abs_diff:.6e}",
                r.note,
            ])

    print(f"\nWrote results to: {out_path}")
    print("Tip: sort by func then median_ms to see fastest implementations per operation.")


if __name__ == "__main__":
    main()
