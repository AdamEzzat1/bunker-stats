#!/usr/bin/env python
"""
bench_full_suite_all_refs_subproc.py

Safer benchmark runner for bunker-stats that:

- Iterates over EVERYTHING exported by `bunker_stats.__all__`
- Compares each op against up to FOUR reference backends:
    python / numpy / pandas / scipy
- Runs *each case in its own subprocess* so a Rust panic (or segfault)
  does not kill the whole run.
- Measures:
    speed (median over repeats),
    accuracy (max abs diff + allclose),
    stability (timing coefficient of variation)

Key design choice:
- You cannot toggle Rust `--features parallel` at runtime.
  So this script benchmarks whichever wheel is installed in the
  Python environment you run it from.

Typical usage (serial build installed):
  python bench_full_suite_all_refs_subproc.py --out bench_serial.csv

Typical usage (parallel build installed in a different venv):
  python bench_full_suite_all_refs_subproc.py --out bench_parallel.csv

Then compare:
  python bench_full_suite_parallel_compare_subproc.py --serial bench_serial.csv --parallel bench_parallel.csv --out bench_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# pandas / scipy are optional
try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

try:
    from scipy import stats as sp_stats
except Exception:
    sp_stats = None  # type: ignore


# ------------------------------
# helpers (in-parent process)
# ------------------------------

def _cov(times: List[float]) -> float:
    if len(times) < 2:
        return float("nan")
    m = statistics.mean(times)
    if m == 0:
        return float("nan")
    return statistics.pstdev(times) / m


def _as_np(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    if pd is not None and isinstance(a, (pd.Series, pd.DataFrame)):
        return a.to_numpy()
    return np.asarray(a)


def _max_abs_diff(a: Any, b: Any) -> float:
    A = _as_np(a)
    B = _as_np(b)
    # scalars
    if A.shape == () and B.shape == ():
        av = float(A)
        bv = float(B)
        if (math.isnan(av) and math.isnan(bv)) or av == bv:
            return 0.0
        return abs(av - bv)
    # bools
    if A.dtype == bool or B.dtype == bool:
        try:
            return float(np.count_nonzero(np.logical_xor(A, B)))
        except Exception:
            return float("nan")
    try:
        d = np.abs(A - B)
        if d.size == 0:
            return 0.0
        return float(np.nanmax(d))
    except Exception:
        return float("nan")


def _allclose(a: Any, b: Any, rtol: float = 1e-10, atol: float = 1e-12) -> bool:
    A = _as_np(a)
    B = _as_np(b)
    try:
        return bool(np.allclose(A, B, rtol=rtol, atol=atol, equal_nan=True))
    except Exception:
        return False


# ------------------------------
# python reference helpers
# ------------------------------

def py_mean(xs: List[float]) -> float:
    s = 0.0
    n = 0
    for v in xs:
        if math.isnan(v):
            continue
        s += v
        n += 1
    return s / n if n else float("nan")


def py_var(xs: List[float], ddof: int = 1) -> float:
    mu = py_mean(xs)
    if math.isnan(mu):
        return float("nan")
    s2 = 0.0
    n = 0
    for v in xs:
        if math.isnan(v):
            continue
        d = v - mu
        s2 += d * d
        n += 1
    if n - ddof <= 0:
        return float("nan")
    return s2 / (n - ddof)


def py_std(xs: List[float], ddof: int = 1) -> float:
    v = py_var(xs, ddof=ddof)
    return math.sqrt(v) if not math.isnan(v) else float("nan")


def py_zscore(xs: List[float]) -> List[float]:
    mu = py_mean(xs)
    sd = py_std(xs, ddof=1)
    out = []
    for v in xs:
        if math.isnan(v) or math.isnan(mu) or math.isnan(sd) or sd == 0:
            out.append(float("nan"))
        else:
            out.append((v - mu) / sd)
    return out


def py_diff(xs: List[float]) -> List[float]:
    return [xs[i] - xs[i - 1] for i in range(1, len(xs))]


def py_cumsum(xs: List[float]) -> List[float]:
    out = []
    s = 0.0
    for v in xs:
        s += v
        out.append(s)
    return out


def py_cummean(xs: List[float]) -> List[float]:
    out = []
    s = 0.0
    for i, v in enumerate(xs, start=1):
        s += v
        out.append(s / i)
    return out


def py_ewma(xs: List[float], alpha: float) -> List[float]:
    out: List[float] = []
    if not xs:
        return out
    y = xs[0]
    out.append(y)
    for v in xs[1:]:
        y = alpha * v + (1.0 - alpha) * y
        out.append(y)
    return out


# ------------------------------
# Case definition
# ------------------------------

@dataclass
class Case:
    name: str
    call_expr: str
    # backend -> python expression returning the reference value
    refs_expr: Dict[str, str]
    shape: str
    notes: str = ""


def _make_inputs(n: int, p: int, window: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).astype(np.float64)
    y = rng.normal(loc=0.2, size=n).astype(np.float64)
    x_nan = x.copy()
    if n >= 10:
        idx = rng.choice(n, size=max(1, n // 50), replace=False)
        x_nan[idx] = np.nan
    y_nan = y.copy()
    if n >= 10:
        idx = rng.choice(n, size=max(1, n // 60), replace=False)
        y_nan[idx] = np.nan

    X = rng.normal(size=(n, p)).astype(np.float64)
    if n >= 10:
        ridx = rng.choice(n, size=max(1, n // 80), replace=False)
        cidx = rng.integers(0, p, size=ridx.size)
        X[ridx, cidx] = np.nan

    obs = rng.integers(0, 30, size=10).astype(np.float64)
    exp = np.full_like(obs, obs.mean() if obs.size else 1.0)
    tab = rng.integers(1, 30, size=(3, 4)).astype(np.float64)

    return dict(
        x=x, y=y, x_nan=x_nan, y_nan=y_nan, X=X, obs=obs, exp=exp, tab=tab,
        window=window, alpha=0.1
    )


def _build_cases(module_name: str, n: int, p: int, window: int, seed: int, n_py: int) -> List[Case]:
    # We build expressions that run inside the subprocess where `bs` is imported.
    # That subprocess will recreate identical inputs using the same RNG seed.
    # (So serial vs parallel runs are comparable.)
    alpha = 0.1
    q = 95.0
    lo, hi = 0.05, 0.95
    k = 1.5
    z = 3.0
    qb = 10

    # Note: we will filter cases inside the subprocess by checking hasattr(bs, func_name).
    cases: List[Case] = []

    def add(name: str, call_expr: str, refs_expr: Dict[str, str], shape: str, notes: str = ""):
        cases.append(Case(name=name, call_expr=call_expr, refs_expr=refs_expr, shape=shape, notes=notes))

    # ---- scalar stats ----
    add("mean_np", "bs.mean_np(x)", {
        "python": "py_mean(x_py)",
        "numpy": "float(np.mean(x))",
    }, "(n,)")

    add("var_np", "bs.var_np(x)", {
        "python": "py_var(x_py, ddof=1)",
        "numpy": "float(np.var(x, ddof=1))",
    }, "(n,)")

    add("std_np", "bs.std_np(x)", {
        "python": "py_std(x_py, ddof=1)",
        "numpy": "float(np.std(x, ddof=1))",
    }, "(n,)")

    add("zscore_np", "bs.zscore_np(x)", {
        "python": "np.array(py_zscore(x_py), dtype=np.float64)",
        "numpy": "(x - np.mean(x)) / np.std(x, ddof=1)",
    }, "(n,)")

    add("percentile_np", f"bs.percentile_np(x, {q})", {
        "numpy": f"float(np.percentile(x, {q}))",
    }, "(n,)", notes=f"q={q}")

    add("iqr_np", "bs.iqr_np(x)", {
        "numpy": "float(np.subtract(*np.percentile(x, [75, 25])))",
    }, "(n,)")

    add("mad_np", "bs.mad_np(x)", {
        "numpy": "float(np.median(np.abs(x - np.median(x))))",
    }, "(n,)")

    # ---- NaN-aware scalar stats ----
    add("mean_nan_np", "bs.mean_nan_np(x_nan)", {
        "python": "py_mean(x_py)",
        "numpy": "float(np.nanmean(x_nan)) if x_nan.size else float('nan')",
    }, "(n,)")

    add("var_nan_np", "bs.var_nan_np(x_nan)", {
        "python": "py_var(x_py, ddof=1)",
        "numpy": "float(np.nanvar(x_nan, ddof=1)) if x_nan.size else float('nan')",
    }, "(n,)")

    add("std_nan_np", "bs.std_nan_np(x_nan)", {
        "python": "py_std(x_py, ddof=1)",
        "numpy": "float(np.nanstd(x_nan, ddof=1)) if x_nan.size else float('nan')",
    }, "(n,)")

    # ---- rolling (1D) ----
    add("rolling_mean_np", "bs.rolling_mean_np(x, window)", {
        "pandas": "pd.Series(x).rolling(window).mean().to_numpy()[window-1:]",
        "numpy": "np.convolve(x, np.ones(window), 'valid') / window",
    }, "(n-w+1,)", notes=f"window={window}")

    add("rolling_std_np", "bs.rolling_std_np(x, window)", {
        "pandas": "pd.Series(x).rolling(window).std(ddof=1).to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    add("rolling_mean_std_np", "bs.rolling_mean_std_np(x, window)", {
        "pandas": "(pd.Series(x).rolling(window).mean().to_numpy()[window-1:], pd.Series(x).rolling(window).std(ddof=1).to_numpy()[window-1:])",
    }, "(2, n-w+1)", notes=f"window={window}")

    add("rolling_zscore_np", "bs.rolling_zscore_np(x, window)", {
        "pandas": "((pd.Series(x) - pd.Series(x).rolling(window).mean()) / pd.Series(x).rolling(window).std(ddof=1)).to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    add("ewma_np", f"bs.ewma_np(x, {alpha})", {
        "python": f"np.array(py_ewma(x_py, {alpha}), dtype=np.float64)",
        "pandas": f"pd.Series(x).ewm(alpha={alpha}, adjust=False).mean().to_numpy()",
    }, "(n,)", notes=f"alpha={alpha}")

    # ---- rolling NaN-aware (1D) ----
    add("rolling_mean_nan_np", "bs.rolling_mean_nan_np(x_nan, window)", {
        "pandas": "pd.Series(x_nan).rolling(window).mean().to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    add("rolling_std_nan_np", "bs.rolling_std_nan_np(x_nan, window)", {
        "pandas": "pd.Series(x_nan).rolling(window).std(ddof=1).to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    add("rolling_zscore_nan_np", "bs.rolling_zscore_nan_np(x_nan, window)", {
        "pandas": "((pd.Series(x_nan) - pd.Series(x_nan).rolling(window).mean()) / pd.Series(x_nan).rolling(window).std(ddof=1)).to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    # ---- diffs / cumulatives / ECDF ----
    add("diff_np", "bs.diff_np(x)", {
        "python": "np.array(py_diff(x_py), dtype=np.float64)",
        "numpy": "np.diff(x)",
    }, "(n-1,)")

    add("pct_change_np", "bs.pct_change_np(x)", {
        "pandas": "pd.Series(x).pct_change().to_numpy()[1:]",
    }, "(n-1,)")

    add("cumsum_np", "bs.cumsum_np(x)", {
        "python": "np.array(py_cumsum(x_py), dtype=np.float64)",
        "numpy": "np.cumsum(x)",
    }, "(n,)")

    add("cummean_np", "bs.cummean_np(x)", {
        "python": "np.array(py_cummean(x_py), dtype=np.float64)",
        "numpy": "np.cumsum(x) / np.arange(1, x.size + 1)",
    }, "(n,)")

    add("ecdf_np", "bs.ecdf_np(x)", {
        "numpy": "(np.sort(x), (np.arange(1, x.size + 1) / x.size))",
    }, "(2, n)")

    # ---- scaling / winsorize / bins / masks ----
    add("minmax_scale_np", "bs.minmax_scale_np(x)", {
        "numpy": "(x - x.min()) / (x.max() - x.min()) if x.size else x",
    }, "(n,)")

    add("robust_scale_np", "bs.robust_scale_np(x)", {
        "numpy": "(x - np.median(x)) / (np.subtract(*np.percentile(x, [75, 25])))",
    }, "(n,)")

    add("winsorize_np", f"bs.winsorize_np(x, {lo}, {hi})", {
        "numpy": f"np.clip(x, np.percentile(x, {lo}), np.percentile(x, {hi}))",
    }, "(n,)", notes=f"lo={lo},hi={hi}")

    add("quantile_bins_np", f"bs.quantile_bins_np(x, {qb})", {
        "pandas": f"pd.qcut(x, {qb}, labels=False, duplicates='drop').to_numpy()",
    }, "(n,)", notes=f"q={qb}")

    add("sign_mask_np", "bs.sign_mask_np(x)", {
        "numpy": "np.signbit(x).astype(np.bool_)",
    }, "(n,)")

    add("demean_with_signs_np", "bs.demean_with_signs_np(x)", {
        "numpy": "x - x.mean()",
    }, "(n,)")

    add("welford_np", "bs.welford_np(x_nan)", {
        "numpy": "(float(np.nanmean(x_nan)), float(np.nanvar(x_nan, ddof=1)), int(np.sum(~np.isnan(x_nan))))",
    }, "(3,)")

    add("iqr_outliers_np", f"bs.iqr_outliers_np(x, {k})", {
        "numpy": f"((x < (np.percentile(x, 0.25) - {k}*(np.percentile(x, 0.75)-np.percentile(x,0.25)))) | (x > (np.percentile(x, 0.75) + {k}*(np.percentile(x,0.75)-np.percentile(x,0.25)))))",
    }, "(n,)", notes=f"k={k}")

    add("zscore_outliers_np", f"bs.zscore_outliers_np(x, {z})", {
        "numpy": f"(np.abs((x - x.mean())/np.std(x, ddof=1)) > {z})",
    }, "(n,)", notes=f"z={z}")

    # ---- cov/corr ----
    add("cov_np", "bs.cov_np(x, y)", {
        "numpy": "float(np.cov(x, y, ddof=1)[0, 1])",
    }, "scalar")

    add("corr_np", "bs.corr_np(x, y)", {
        "numpy": "float(np.corrcoef(x, y)[0, 1])",
    }, "scalar")

    add("cov_nan_np", "bs.cov_nan_np(x_nan, y_nan)", {
        "numpy": "float(np.cov(x_nan[~(np.isnan(x_nan)|np.isnan(y_nan))], y_nan[~(np.isnan(x_nan)|np.isnan(y_nan))], ddof=1)[0,1]) if np.sum(~(np.isnan(x_nan)|np.isnan(y_nan)))>=2 else float('nan')",
    }, "scalar")

    add("corr_nan_np", "bs.corr_nan_np(x_nan, y_nan)", {
        "numpy": "float(np.corrcoef(x_nan[~(np.isnan(x_nan)|np.isnan(y_nan))], y_nan[~(np.isnan(x_nan)|np.isnan(y_nan))])[0,1]) if np.sum(~(np.isnan(x_nan)|np.isnan(y_nan)))>=2 else float('nan')",
    }, "scalar")

    add("rolling_cov_np", "bs.rolling_cov_np(x, y, window)", {
        "pandas": "pd.Series(x).rolling(window).cov(pd.Series(y)).to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    add("rolling_corr_np", "bs.rolling_corr_np(x, y, window)", {
        "pandas": "pd.Series(x).rolling(window).corr(pd.Series(y)).to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    add("rolling_cov_nan_np", "bs.rolling_cov_nan_np(x_nan, y_nan, window)", {
        "pandas": "pd.Series(x_nan).rolling(window).cov(pd.Series(y_nan)).to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    add("rolling_corr_nan_np", "bs.rolling_corr_nan_np(x_nan, y_nan, window)", {
        "pandas": "pd.Series(x_nan).rolling(window).corr(pd.Series(y_nan)).to_numpy()[window-1:]",
    }, "(n-w+1,)", notes=f"window={window}")

    add("cov_matrix_np", "bs.cov_matrix_np(X)", {
        "numpy": "np.cov(X, rowvar=False, ddof=1)",
    }, "(p,p)")

    add("corr_matrix_np", "bs.corr_matrix_np(X)", {
        "numpy": "np.corrcoef(X, rowvar=False)",
    }, "(p,p)")

    # ---- axis0 rolling (2D) ----
    add("rolling_mean_axis0_np", "bs.rolling_mean_axis0_np(X, window)", {
        "pandas": "pd.DataFrame(X).rolling(window).mean().to_numpy()[window-1:]",
    }, "(n-w+1,p)", notes=f"window={window}")

    add("rolling_std_axis0_np", "bs.rolling_std_axis0_np(X, window)", {
        "pandas": "pd.DataFrame(X).rolling(window).std(ddof=1).to_numpy()[window-1:]",
    }, "(n-w+1,p)", notes=f"window={window}")

    add("rolling_mean_std_axis0_np", "bs.rolling_mean_std_axis0_np(X, window)", {
        "pandas": "(pd.DataFrame(X).rolling(window).mean().to_numpy()[window-1:], pd.DataFrame(X).rolling(window).std(ddof=1).to_numpy()[window-1:])",
    }, "(2,n-w+1,p)", notes=f"window={window}")

    # ---- inference ----
    add("t_test_1samp_np", "bs.t_test_1samp_np(x, 0.0, 'two-sided')", {
        "scipy": "sp_stats.ttest_1samp(x, 0.0, alternative='two-sided')",
    }, "dict")

    add("t_test_2samp_np_pooled", "bs.t_test_2samp_np(x, y, True, 'two-sided')", {
        "scipy": "sp_stats.ttest_ind(x, y, equal_var=True, alternative='two-sided')",
    }, "dict", notes="equal_var=True")

    add("t_test_2samp_np_welch", "bs.t_test_2samp_np(x, y, False, 'two-sided')", {
        "scipy": "sp_stats.ttest_ind(x, y, equal_var=False, alternative='two-sided')",
    }, "dict", notes="equal_var=False")

    add("chi2_gof_np", "bs.chi2_gof_np(obs, exp)", {
        "scipy": "sp_stats.chisquare(obs, exp)",
    }, "dict")

    add("chi2_independence_np", "bs.chi2_independence_np(tab)", {
        "scipy": "sp_stats.chi2_contingency(tab, correction=False)",
    }, "dict")

    add("mean_diff_ci_np_1samp", "bs.mean_diff_ci_np(x, None, 0.05, True)", {
        "scipy": "sp_stats.t.interval(0.95, df=x.size-1, loc=float(np.mean(x)), scale=float(np.std(x, ddof=1))/math.sqrt(x.size))",
    }, "(2,)")

    # MWU + KS (small sample)
    add("mann_whitney_u_np", "bs.mann_whitney_u_np(x[:200], y[:180], 'two-sided')", {
        "scipy": "sp_stats.mannwhitneyu(x[:200], y[:180], alternative='two-sided')",
    }, "dict", notes="(small sample)")

    add("ks_1samp_np", "bs.ks_1samp_np(x[:200], 'norm', 'two-sided')", {
        "scipy": "sp_stats.kstest(x[:200], 'norm', alternative='two-sided')",
    }, "dict", notes="(small sample)")

    return cases


# ------------------------------
# subprocess runner
# ------------------------------

_SUBPROC_TEMPLATE = r"""
import json, math, os, statistics, time
import numpy as np

MAX_JSON_ELEMS = int(__MAX_JSON_ELEMS__)

def _maybe_value(obj):
    try:
        a = np.asarray(obj)
        if isinstance(obj, np.ndarray) and a.size > MAX_JSON_ELEMS:
            return None
    except Exception:
        pass
    return _jsonify(obj)

def _mad(a, b):
    try:
        A = np.asarray(a)
        B = np.asarray(b)
        if A.shape != B.shape:
            return float("nan")
        d = np.abs(A - B)
        if d.size == 0:
            return 0.0
        return float(np.nanmax(d))
    except Exception:
        return float("nan")

def _allclose(a, b, rtol=1e-7, atol=1e-9):
    try:
        A = np.asarray(a)
        B = np.asarray(b)
        if A.shape != B.shape:
            return False
        return bool(np.allclose(A, B, rtol=rtol, atol=atol, equal_nan=True))
    except Exception:
        return False

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy import stats as sp_stats
except Exception:
    sp_stats = None

import {module_name} as bs

# python refs
def py_mean(xs):
    s=0.0; n=0
    for v in xs:
        if math.isnan(v): continue
        s += v; n += 1
    return s/n if n else float('nan')

def py_var(xs, ddof=1):
    mu = py_mean(xs)
    if math.isnan(mu): return float('nan')
    s2=0.0; n=0
    for v in xs:
        if math.isnan(v): continue
        d=v-mu; s2 += d*d; n += 1
    if n-ddof<=0: return float('nan')
    return s2/(n-ddof)

def py_std(xs, ddof=1):
    v = py_var(xs, ddof=ddof)
    return math.sqrt(v) if not math.isnan(v) else float('nan')

def py_zscore(xs):
    mu = py_mean(xs)
    sd = py_std(xs, ddof=1)
    out=[]
    for v in xs:
        if math.isnan(v) or math.isnan(mu) or math.isnan(sd) or sd==0:
            out.append(float('nan'))
        else:
            out.append((v-mu)/sd)
    return out

def py_diff(xs): return [xs[i]-xs[i-1] for i in range(1,len(xs))]
def py_cumsum(xs):
    out=[]; s=0.0
    for v in xs:
        s += v; out.append(s)
    return out
def py_cummean(xs):
    out=[]; s=0.0
    for i,v in enumerate(xs, start=1):
        s += v; out.append(s/i)
    return out
def py_ewma(xs, alpha):
    out=[]
    if not xs: return out
    y = xs[0]; out.append(y)
    for v in xs[1:]:
        y = alpha*v + (1.0-alpha)*y
        out.append(y)
    return out

def _time_call(fn, warmup, repeats):
    for _ in range(max(0,warmup)):
        fn()
    times=[]
    for _ in range(max(1,repeats)):
        t0=time.perf_counter()
        fn()
        t1=time.perf_counter()
        times.append(t1-t0)
    return float(statistics.median(times)), float(statistics.mean(times)), times

def _cov(times):
    if len(times)<2: return float('nan')
    m = statistics.mean(times)
    if m==0: return float('nan')
    return float(statistics.pstdev(times)/m)

def main():
    payload = json.loads({payload_json})
    name = payload["name"]
    call_expr = payload["call_expr"]
    refs_expr = payload["refs_expr"]
    n = int(payload["n"]); p = int(payload["p"]); window = int(payload["window"]); seed = int(payload["seed"]); n_py = int(payload["n_py"])
    warmup = int(payload["warmup"]); repeats = int(payload["repeats"])

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).astype(np.float64)
    y = rng.normal(loc=0.2, size=n).astype(np.float64)
    x_nan = x.copy()
    if n>=10:
        idx = rng.choice(n, size=max(1,n//50), replace=False)
        x_nan[idx]=np.nan
    y_nan = y.copy()
    if n>=10:
        idx = rng.choice(n, size=max(1,n//60), replace=False)
        y_nan[idx]=np.nan

    X = rng.normal(size=(n,p)).astype(np.float64)
    if n>=10:
        ridx = rng.choice(n, size=max(1,n//80), replace=False)
        cidx = rng.integers(0,p,size=ridx.size)
        X[ridx,cidx]=np.nan

    obs = rng.integers(0,30,size=10).astype(np.float64)
    exp = np.full_like(obs, obs.mean() if obs.size else 1.0)
    tab = rng.integers(1,30,size=(3,4)).astype(np.float64)
    alpha = 0.1

    x_py = x_nan[:n_py].tolist()
    y_py = y_nan[:n_py].tolist()

    # availability checks
    fn_name = name.split("_np")[0] + "_np" if name.endswith("_np") else name
    base = name
    # In case names like t_test_2samp_np_pooled:
    if base.startswith("t_test_2samp_np"):
        avail = hasattr(bs, "t_test_2samp_np")
    elif base in ("t_test_1samp_np","chi2_gof_np","chi2_independence_np","mean_diff_ci_np_1samp","mean_diff_ci_np"):
        avail = hasattr(bs, "t_test_1samp_np") if base=="t_test_1samp_np" else hasattr(bs, base.replace("_1samp",""))
        if base=="mean_diff_ci_np_1samp":
            avail = hasattr(bs, "mean_diff_ci_np")
    else:
        # strip suffixes we introduced
        canonical = base.replace("_pooled","").replace("_welch","")
        avail = hasattr(bs, canonical)

    if not avail:
        print(json.dumps({{"status":"skip_missing"}}))
        return

    # backend availability
    if any(k=="pandas" for k in refs_expr.keys()) and pd is None:
        print(json.dumps({{"status":"skip_no_pandas"}})); return
    if any(k=="scipy" for k in refs_expr.keys()) and sp_stats is None:
        print(json.dumps({{"status":"skip_no_scipy"}})); return

    # Evaluate bs output once (for accuracy) + time it
    def bs_call():
        return eval(call_expr, globals(), locals())

    try:
        out_bs = bs_call()
    except Exception as e:
        print(json.dumps({{"status":"bs_error","error":repr(e)}}))
        return

    bs_med, bs_mean, bs_times = _time_call(bs_call, warmup, repeats)

    out = {{
        "status":"ok",
        "bs_median_s": bs_med,
        "bs_mean_s": bs_mean,
        "bs_cov": _cov(bs_times),
        "refs": {{}},
    }}

    # Refs
    for be, expr in refs_expr.items():
        def ref_call(expr=expr):
            return eval(expr, globals(), locals())
        try:
            out_ref = ref_call()
        except Exception as e:
            out["refs"][be] = {{"status":"ref_error","error":repr(e)}}
            continue
        ref_med, ref_mean, ref_times = _time_call(ref_call, warmup, repeats)
        out["refs"][be] = {{"status":"ok","median_s":ref_med,"mean_s":ref_mean,"cov":_cov(ref_times),"value":out_ref}}

    # Return both values for parent-side accuracy calc
    out["bs_value"] = out_bs
    print(json.dumps(out, default=str))

if __name__ == "__main__":
    main()
"""


def _run_case_in_subproc(
    module_name: str,
    case: Case,
    n: int,
    p: int,
    window: int,
    seed: int,
    n_py: int,
    warmup: int,
    repeats: int,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {
        "name": case.name,
        "call_expr": case.call_expr,
        "refs_expr": case.refs_expr,
        "n": n,
        "p": p,
        "window": window,
        "seed": seed,
        "n_py": n_py,
        "warmup": warmup,
        "repeats": repeats,
    }
    code = _SUBPROC_TEMPLATE.format(
        module_name=module_name,
        payload_json=json.dumps(json.dumps(payload)),
    )

    env = os.environ.copy()
    # prevent python from writing .pyc in repo
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}

    # Rust panic -> nonzero returncode, stderr has panic text
    if proc.returncode != 0:
        return {
            "status": "crash",
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1000:],
            "stdout_tail": proc.stdout[-1000:],
        }

    out = proc.stdout.strip()
    if not out:
        return {"status": "empty_output", "stderr_tail": proc.stderr[-1000:]}
    try:
        return json.loads(out)
    except Exception:
        return {"status": "bad_json", "stdout_tail": out[-1000:], "stderr_tail": proc.stderr[-1000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", type=str, default="bunker_stats", help="Module to benchmark (default: bunker_stats)")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench_full_suite_all_refs_subproc.csv")
    ap.add_argument("--n-py", type=int, default=5_000, dest="n_py")
    ap.add_argument("--timeout-s", type=int, default=120)
    args = ap.parse_args()

    module_name = args.module

    cases = _build_cases(module_name=module_name, n=args.n, p=args.p, window=args.window, seed=args.seed, n_py=args.n_py)

    rows: List[Dict[str, Any]] = []
    backends = ["python", "numpy", "pandas", "scipy"]

    for case in cases:
        result = _run_case_in_subproc(
            module_name=module_name,
            case=case,
            n=args.n,
            p=args.p,
            window=args.window,
            seed=args.seed,
            n_py=args.n_py,
            warmup=args.warmup,
            repeats=args.repeats,
            timeout_s=args.timeout_s,
            max_json_elems=args.max_json_elems,
        )

        row: Dict[str, Any] = {
            "name": case.name,
            "shape": case.shape,
            "notes": case.notes,
            "status": result.get("status", "unknown"),
        }

        if row["status"] != "ok":
            row["error"] = result.get("error") or result.get("stderr_tail") or result.get("stdout_tail") or ""
            rows.append(row)
            print(f"[skip] {case.name} ({row['status']})")
            continue

        # times
        row["bs_median_s"] = result["bs_median_s"]
        row["bs_mean_s"] = result["bs_mean_s"]
        row["bs_cov"] = result["bs_cov"]

        bs_val = result.get("bs_value")
        refs = result.get("refs", {})

        for be in backends:
            ref = refs.get(be)
            if not ref:
                row[f"{be}_status"] = "NA"
                continue
            row[f"{be}_status"] = ref.get("status", "unknown")
            row[f"{be}_median_s"] = ref.get("median_s", float("nan"))
            row[f"{be}_mean_s"] = ref.get("mean_s", float("nan"))
            row[f"{be}_cov"] = ref.get("cov", float("nan"))

            if ref.get("status") != "ok":
                row[f"{be}_error"] = ref.get("error", "")
                continue

            # Prefer accuracy computed inside subprocess (avoids serializing huge arrays)
            if "mad" in ref and "allclose" in ref:
                row[f"{be}_mad"] = ref.get("mad")
                row[f"{be}_allclose"] = bool(ref.get("allclose"))
                continue

            ref_val = ref.get("value")

            # accuracy:
            if isinstance(bs_val, dict):
                # normalize scipy-like objects
                out_ref_norm: Dict[str, float] = {}
                if hasattr(ref_val, "statistic") and hasattr(ref_val, "pvalue"):
                    out_ref_norm = {"statistic": float(ref_val.statistic), "pvalue": float(ref_val.pvalue)}
                elif isinstance(ref_val, (list, tuple)) and len(ref_val) >= 3 and be == "scipy":
                    out_ref_norm = {"statistic": float(ref_val[0]), "pvalue": float(ref_val[1]), "df": float(ref_val[2])}
                elif isinstance(ref_val, dict):
                    out_ref_norm = {k: float(v) for k, v in ref_val.items() if k in ("statistic","pvalue","df")}
                mad = float("nan")
                close = True
                for k in ("statistic","pvalue","df"):
                    if k in bs_val and k in out_ref_norm:
                        dk = abs(float(bs_val[k]) - float(out_ref_norm[k]))
                        mad = dk if math.isnan(mad) else max(mad, dk)
                        close = close and (dk <= 1e-9 or math.isclose(float(bs_val[k]), float(out_ref_norm[k]), rel_tol=1e-9, abs_tol=1e-12))
                row[f"{be}_mad"] = mad
                row[f"{be}_allclose"] = close
            else:
                row[f"{be}_mad"] = _max_abs_diff(bs_val, ref_val)
                row[f"{be}_allclose"] = _allclose(bs_val, ref_val)

        rows.append(row)
        print(f"[done] {case.name}")

    # write CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("bs_median_s"), (float, int))]
    ok_rows.sort(key=lambda r: r["bs_median_s"])
    print("\nTop 10 fastest (bunker-stats median):")
    for r in ok_rows[:10]:
        print(f"  {r['name']:<28} {r['bs_median_s']:.6f}s")

    print("\nTop 10 slowest (bunker-stats median):")
    for r in ok_rows[-10:]:
        print(f"  {r['name']:<28} {r['bs_median_s']:.6f}s")

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
