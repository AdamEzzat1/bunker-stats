#!/usr/bin/env python
"""
bench_compare_backends_enhanced.py

Enhanced version with:
1. Speed difference columns (bunker-stats - alternative) in seconds
2. CV difference columns (bunker-stats CV - alternative CV)
3. Additional metrics: memory usage, scaling efficiency, numerical accuracy
4. Percentile timing (p50, p95, p99) for consistency analysis
5. Optional parallel execution analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is on sys.path so `import bunker_stats` works when running from ./benchmarks
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_skip_list(s: str) -> set[str]:
    s = (s or "").strip()
    return {x.strip() for x in s.split(",") if x.strip()}


@dataclass(frozen=True)
class Case:
    fn_name: str
    args: Dict[str, Any]
    refs: Tuple[str, ...] = ("python", "numpy", "pandas", "scipy")


def _make_cases_from_all(n: int, p: int, seed: int) -> List[Case]:
    import bunker_stats as bs

    base = {"n": n, "p": p, "seed": seed}

    # Minimal specs: generate inputs and pass required args for both bs + baselines.
    explicit: Dict[str, Dict[str, Any]] = {
        # scalar/vector
        "percentile_np": {"kind": "vec_f64", "q": 95.0},
        "trimmed_mean_np": {"kind": "vec_f64", "proportion_to_cut": 0.1},
        "diff_np": {"kind": "vec_f64", "periods": 1},
        "pct_change_np": {"kind": "vec_f64", "periods": 1},
        "pad_nan_np": {"kind": "len_only"},

        # rolling
        "rolling_mean_np": {"kind": "vec_f64", "window": 64},
        "rolling_std_np": {"kind": "vec_f64", "window": 64},
        "rolling_var_np": {"kind": "vec_f64", "window": 64},
        "rolling_zscore_np": {"kind": "vec_f64", "window": 64},
        "rolling_mean_std_np": {"kind": "vec_f64", "window": 64},
        "ewma_np": {"kind": "vec_f64", "alpha": 0.2},
        "rolling_mean_nan_np": {"kind": "vec_nan_f64", "window": 64},
        "rolling_std_nan_np": {"kind": "vec_nan_f64", "window": 64},
        "rolling_zscore_nan_np": {"kind": "vec_nan_f64", "window": 64},

        # multi-d
        "mean_axis_np": {"kind": "mat_f64", "axis": 0},
        "cov_matrix_np": {"kind": "mat_f64"},
        "corr_matrix_np": {"kind": "mat_f64"},
        "rolling_mean_axis0_np": {"kind": "mat_f64", "window": 64},
        "rolling_std_axis0_np": {"kind": "mat_f64", "window": 64},
        "rolling_mean_std_axis0_np": {"kind": "mat_f64", "window": 64},

        # pairs
        "cov_np": {"kind": "pair_vec_f64"},
        "corr_np": {"kind": "pair_vec_f64"},
        "cov_nan_np": {"kind": "pair_vec_nan_f64"},
        "corr_nan_np": {"kind": "pair_vec_nan_f64"},
        "rolling_cov_np": {"kind": "pair_vec_f64", "window": 64},
        "rolling_corr_np": {"kind": "pair_vec_f64", "window": 64},
        "rolling_cov_nan_np": {"kind": "pair_vec_nan_f64", "window": 64},
        "rolling_corr_nan_np": {"kind": "pair_vec_nan_f64", "window": 64},

        # outliers/scaling
        "iqr_outliers_np": {"kind": "vec_f64", "k": 1.5},
        "zscore_outliers_np": {"kind": "vec_f64", "threshold": 3.0},
        "robust_scale_np": {"kind": "vec_f64", "scale_factor": 1.0},
        "winsorize_np": {"kind": "vec_f64", "lower_q": 0.05, "upper_q": 0.95},
        "quantile_bins_np": {"kind": "vec_f64", "n_bins": 10},

        # kde
        "kde_gaussian_np": {"kind": "vec_f64", "n_points": 256},

        # inference
        "t_test_1samp_np": {"kind": "vec_f64", "mu": 0.0, "alternative": "two-sided"},
        "t_test_2samp_np": {"kind": "pair_vec_f64", "equal_var": False, "alternative": "two-sided"},
        "chi2_gof_np": {"kind": "chi2_gof"},
        "chi2_independence_np": {"kind": "chi2_ind"},
        "cohens_d_2samp_np": {"kind": "pair_vec_f64", "pooled": True},
        "hedges_g_2samp_np": {"kind": "pair_vec_f64"},
        "mann_whitney_u_np": {"kind": "pair_vec_f64", "alternative": "two-sided"},
    }

    out: List[Case] = []
    for name in list(getattr(bs, "__all__", [])):
        spec = dict(base)
        spec.update(explicit.get(name, {"kind": "vec_f64"}))
        out.append(Case(fn_name=name, args=spec))
    return out


_SUBPROC_TEMPLATE = r"""
import json, time, math, tracemalloc
import numpy as np

import faulthandler
faulthandler.enable()

CASE = json.loads(CASE_JSON)

# Optional deps
try:
    import pandas as pd
except Exception as _e:
    pd = None
try:
    import scipy
    import scipy.stats as sps
except Exception as _e:
    sps = None

import bunker_stats as bs


def _make_inputs(case):
    n = int(case.get("n", 200000))
    p = int(case.get("p", 32))
    seed = int(case.get("seed", 0))
    rng = np.random.default_rng(seed)
    kind = case.get("kind", "vec_f64")

    def vec_f64():
        return rng.normal(size=n).astype(np.float64)

    def vec_nan_f64():
        x = rng.normal(size=n).astype(np.float64)
        x[rng.random(size=n) < 0.05] = np.nan
        return x

    def mat_f64():
        return rng.normal(size=(n, p)).astype(np.float64)

    def pair_vec_f64():
        x = rng.normal(size=n).astype(np.float64)
        y = rng.normal(loc=0.2, size=n).astype(np.float64)
        return x, y

    def pair_vec_nan_f64():
        x, y = pair_vec_f64()
        x[rng.random(size=n) < 0.05] = np.nan
        y[rng.random(size=n) < 0.05] = np.nan
        return x, y

    if kind == "vec_f64":
        return {"x": vec_f64()}
    if kind == "vec_nan_f64":
        return {"x": vec_nan_f64()}
    if kind == "mat_f64":
        return {"X": mat_f64()}
    if kind == "pair_vec_f64":
        x, y = pair_vec_f64()
        return {"x": x, "y": y}
    if kind == "pair_vec_nan_f64":
        x, y = pair_vec_nan_f64()
        return {"x": x, "y": y}
    if kind == "chi2_gof":
        obs = np.array([10, 12, 9, 11, 8], dtype=np.float64)
        exp = np.array([10, 10, 10, 10, 10], dtype=np.float64)
        return {"obs": obs, "exp": exp}
    if kind == "chi2_ind":
        tab = np.array([[10, 20, 30], [6, 9, 17]], dtype=np.float64)
        return {"tab": tab}

    if kind == "len_only":
        return {"n": n}
    return {"x": vec_f64()}


def _timeit(fn, warmup, repeats, track_memory=False):
    # warmup
    for _ in range(warmup):
        fn()
    
    times = []
    mem_peak = 0
    last = None
    
    for _ in range(repeats):
        if track_memory:
            tracemalloc.start()
        
        t0 = time.perf_counter()
        last = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        
        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            mem_peak = max(mem_peak, peak)
            tracemalloc.stop()
    
    times = np.array(times, dtype=np.float64)
    med = float(np.median(times)) if times.size else float("nan")
    mean = float(np.mean(times)) if times.size else float("nan")
    cv = float(np.std(times, ddof=1) / mean) if times.size > 1 and mean else float("nan")
    
    # Additional percentiles for consistency analysis
    p50 = float(np.percentile(times, 50)) if times.size else float("nan")
    p95 = float(np.percentile(times, 95)) if times.size else float("nan")
    p99 = float(np.percentile(times, 99)) if times.size else float("nan")
    
    result = {
        "median_s": med, 
        "mean_s": mean, 
        "cv": cv,
        "p50_s": p50,
        "p95_s": p95,
        "p99_s": p99,
        "value": last
    }
    
    if track_memory:
        result["mem_peak_mb"] = mem_peak / (1024 * 1024)
    
    return result


def _safe_call(tag, fn, warmup, repeats, track_memory=False):
    try:
        timing = _timeit(fn, warmup, repeats, track_memory=track_memory)
        return {"status": "ok", **timing}
    except Exception as e:
        return {
            "status": "error",
            "error": repr(e),
            "median_s": "",
            "mean_s": "",
            "cv": "",
            "p50_s": "",
            "p95_s": "",
            "p99_s": "",
        }


def _summarize(val):
    import numpy as np
    if isinstance(val, (int, float)):
        return {"type": "scalar", "value": float(val)}
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return {"type": "ndarray", "shape": str(val.shape), "dtype": str(val.dtype)}
        return {
            "type": "ndarray",
            "shape": str(val.shape),
            "dtype": str(val.dtype),
            "mean": float(np.nanmean(val)),
            "std": float(np.nanstd(val)),
        }
    if isinstance(val, dict):
        return {"type": "dict", "keys": list(val.keys())}
    if isinstance(val, tuple):
        return {"type": "tuple", "len": len(val)}
    return {"type": str(type(val).__name__)}


def _call_bs(case, inputs):
    import bunker_stats as bs
    name = case["fn_name"]
    fn = getattr(bs, name)
    k = case.get("kind", "vec_f64")

    if k == "len_only":
        return {"status":"ok","value": fn(inputs["n"])}
    if k in ("vec_f64","vec_nan_f64"):
        x = inputs["x"]
        if name in ("percentile_np","trimmed_mean_np"):
            if name == "percentile_np":
                return {"status":"ok","value": fn(x, case["q"])}
            if name == "trimmed_mean_np":
                return {"status":"ok","value": fn(x, case["proportion_to_cut"])}
        if name in ("diff_np","pct_change_np"):
            return {"status":"ok","value": fn(x, periods=case.get("periods",1))}
        if name in ("rolling_mean_np","rolling_std_np","rolling_var_np","rolling_zscore_np",
                    "rolling_mean_std_np","rolling_mean_nan_np","rolling_std_nan_np","rolling_zscore_nan_np"):
            return {"status":"ok","value": fn(x, window=case.get("window",64))}
        if name == "ewma_np":
            return {"status":"ok","value": fn(x, alpha=case.get("alpha",0.2))}
        if name in ("iqr_outliers_np","zscore_outliers_np"):
            if name == "iqr_outliers_np":
                return {"status":"ok","value": fn(x, k=case.get("k",1.5))}
            if name == "zscore_outliers_np":
                return {"status":"ok","value": fn(x, threshold=case.get("threshold",3.0))}
        if name == "robust_scale_np":
            return {"status":"ok","value": fn(x, scale_factor=case.get("scale_factor",1.0))}
        if name == "winsorize_np":
            return {"status":"ok","value": fn(x, lower_q=case.get("lower_q",0.05), upper_q=case.get("upper_q",0.95))}
        if name == "quantile_bins_np":
            return {"status":"ok","value": fn(x, n_bins=case.get("n_bins",10))}
        if name == "kde_gaussian_np":
            return {"status":"ok","value": fn(x, n_points=case.get("n_points",256))}
        if name == "t_test_1samp_np":
            return {"status":"ok","value": fn(x, mu=case.get("mu",0.0), alternative=case.get("alternative","two-sided"))}
        return {"status":"ok","value": fn(x)}

    if k == "mat_f64":
        X = inputs["X"]
        if name == "mean_axis_np":
            return {"status":"ok","value": fn(X, axis=case.get("axis",0))}
        if name in ("rolling_mean_axis0_np","rolling_std_axis0_np","rolling_mean_std_axis0_np"):
            return {"status":"ok","value": fn(X, window=case.get("window",64))}
        return {"status":"ok","value": fn(X)}

    if k in ("pair_vec_f64","pair_vec_nan_f64"):
        x, y = inputs["x"], inputs["y"]
        if name in ("rolling_cov_np","rolling_corr_np","rolling_cov_nan_np","rolling_corr_nan_np"):
            return {"status":"ok","value": fn(x, y, window=case.get("window",64))}
        if name == "t_test_2samp_np":
            return {"status":"ok","value": fn(x, y, equal_var=bool(case.get("equal_var",False)), alternative=str(case.get("alternative","two-sided")))}
        if name == "cohens_d_2samp_np":
            return {"status":"ok","value": fn(x, y, pooled=bool(case.get("pooled",True)))}
        if name == "hedges_g_2samp_np":
            return {"status":"ok","value": fn(x, y)}
        if name == "mann_whitney_u_np":
            return {"status":"ok","value": fn(x, y, alternative=str(case.get("alternative","two-sided")))}
        return {"status":"ok","value": fn(x, y)}

    if k == "chi2_gof":
        return {"status":"ok","value": fn(inputs["obs"], inputs["exp"])}
    if k == "chi2_ind":
        return {"status":"ok","value": fn(inputs["tab"])}
    return {"status":"skip_unknown"}


def _impl_python(name, case, inputs):
    return {"status":"skip_unavailable"}


def _impl_numpy(name, case, inputs):
    k = case.get("kind","vec_f64")
    x = inputs.get("x")
    
    if name == "percentile_np" and x is not None:
        fn = lambda: np.percentile(x, case["q"])
        return {"status":"ok","value": fn()}
    if name == "diff_np" and x is not None:
        p = case.get("periods",1)
        fn = lambda: np.diff(x, n=p, prepend=[np.nan]*p)
        return {"status":"ok","value": fn()}
    if name == "rolling_mean_np" and x is not None:
        w = case.get("window",64)
        fn = lambda: np.convolve(x, np.ones(w)/w, mode='same')
        return {"status":"ok","value": fn()}
    if name == "mean_axis_np" and k == "mat_f64":
        X = inputs["X"]
        axis = case.get("axis",0)
        fn = lambda: np.mean(X, axis=axis)
        return {"status":"ok","value": fn()}
    if name == "cov_matrix_np" and k == "mat_f64":
        X = inputs["X"]
        fn = lambda: np.cov(X, rowvar=False)
        return {"status":"ok","value": fn()}
    if name == "corr_matrix_np" and k == "mat_f64":
        X = inputs["X"]
        fn = lambda: np.corrcoef(X, rowvar=False)
        return {"status":"ok","value": fn()}
    if name in ("cov_np","cov_nan_np") and k in ("pair_vec_f64","pair_vec_nan_f64"):
        y = inputs["y"]
        fn = lambda: np.cov(x, y)[0, 1]
        return {"status":"ok","value": fn()}
    if name in ("corr_np","corr_nan_np") and k in ("pair_vec_f64","pair_vec_nan_f64"):
        y = inputs["y"]
        fn = lambda: np.corrcoef(x, y)[0, 1]
        return {"status":"ok","value": fn()}
    
    return {"status":"skip_unavailable"}


def _impl_pandas(name, case, inputs):
    if pd is None:
        return {"status":"skip_unavailable"}
    
    k = case.get("kind","vec_f64")
    x = inputs.get("x")
    
    if name == "percentile_np" and x is not None:
        s = pd.Series(x)
        q = case["q"]
        fn = lambda: s.quantile(q / 100.0)
        return {"status":"ok","value": fn()}
    if name == "diff_np" and x is not None:
        s = pd.Series(x)
        p = case.get("periods",1)
        fn = lambda: s.diff(periods=p).values
        return {"status":"ok","value": fn()}
    if name == "pct_change_np" and x is not None:
        s = pd.Series(x)
        p = case.get("periods",1)
        fn = lambda: s.pct_change(periods=p).values
        return {"status":"ok","value": fn()}
    if name == "rolling_mean_np" and x is not None:
        s = pd.Series(x)
        w = case.get("window",64)
        fn = lambda: s.rolling(window=w, center=False).mean().values
        return {"status":"ok","value": fn()}
    if name == "rolling_std_np" and x is not None:
        s = pd.Series(x)
        w = case.get("window",64)
        fn = lambda: s.rolling(window=w, center=False).std().values
        return {"status":"ok","value": fn()}
    if name == "rolling_var_np" and x is not None:
        s = pd.Series(x)
        w = case.get("window",64)
        fn = lambda: s.rolling(window=w, center=False).var().values
        return {"status":"ok","value": fn()}
    if name == "ewma_np" and x is not None:
        s = pd.Series(x)
        alpha = case.get("alpha",0.2)
        fn = lambda: s.ewm(alpha=alpha, adjust=False).mean().values
        return {"status":"ok","value": fn()}
    if name == "cov_matrix_np" and k == "mat_f64":
        X = inputs["X"]
        df = pd.DataFrame(X)
        fn = lambda: df.cov().values
        return {"status":"ok","value": fn()}
    if name == "corr_matrix_np" and k == "mat_f64":
        X = inputs["X"]
        df = pd.DataFrame(X)
        fn = lambda: df.corr().values
        return {"status":"ok","value": fn()}
    if name in ("cov_np","cov_nan_np") and k in ("pair_vec_f64","pair_vec_nan_f64"):
        y = inputs["y"]
        s1, s2 = pd.Series(x), pd.Series(y)
        fn = lambda: s1.cov(s2)
        return {"status":"ok","value": fn()}
    if name in ("corr_np","corr_nan_np") and k in ("pair_vec_f64","pair_vec_nan_f64"):
        y = inputs["y"]
        s1, s2 = pd.Series(x), pd.Series(y)
        fn = lambda: s1.corr(s2)
        return {"status":"ok","value": fn()}
    if name in ("rolling_cov_np","rolling_cov_nan_np") and k in ("pair_vec_f64","pair_vec_nan_f64"):
        y = inputs["y"]
        s1, s2 = pd.Series(x), pd.Series(y)
        w = case.get("window",64)
        fn = lambda: s1.rolling(window=w).cov(s2).values
        return {"status":"ok","value": fn()}
    if name in ("rolling_corr_np","rolling_corr_nan_np") and k in ("pair_vec_f64","pair_vec_nan_f64"):
        y = inputs["y"]
        s1, s2 = pd.Series(x), pd.Series(y)
        w = case.get("window",64)
        fn = lambda: s1.rolling(window=w).corr(s2).values
        return {"status":"ok","value": fn()}
    
    return {"status":"skip_unavailable"}


def _impl_scipy(name, case, inputs):
    if sps is None:
        return {"status":"skip_unavailable"}
    
    k = case.get("kind","vec_f64")
    x = inputs.get("x")
    
    if name == "trimmed_mean_np" and x is not None:
        prop = case.get("proportion_to_cut", 0.1)
        fn = lambda: sps.trim_mean(x, prop)
        return {"status":"ok","value": fn()}
    if name == "winsorize_np" and x is not None:
        lower_q = case.get("lower_q", 0.05)
        upper_q = case.get("upper_q", 0.95)
        limits = (lower_q, 1.0 - upper_q)
        fn = lambda: sps.mstats.winsorize(x, limits=limits)
        return {"status":"ok","value": fn()}
    if name == "t_test_1samp_np" and x is not None:
        mu = case.get("mu", 0.0)
        alt = case.get("alternative", "two-sided")
        fn = lambda: sps.ttest_1samp(x, mu, alternative=alt)
        return {"status":"ok","value": fn()}
    if name == "t_test_2samp_np" and k in ("pair_vec_f64","pair_vec_nan_f64"):
        y = inputs["y"]
        eq_var = bool(case.get("equal_var", False))
        alt = str(case.get("alternative", "two-sided"))
        fn = lambda: sps.ttest_ind(x, y, equal_var=eq_var, alternative=alt)
        return {"status":"ok","value": fn()}
    if name == "chi2_gof_np" and k == "chi2_gof":
        obs = inputs["obs"]
        exp = inputs["exp"]
        fn = lambda: sps.chisquare(obs, exp)
        return {"status":"ok","value": fn()}
    if name == "chi2_independence_np" and k == "chi2_ind":
        tab = inputs["tab"]
        fn = lambda: sps.chi2_contingency(tab)
        return {"status":"ok","value": fn()}
    if name == "mann_whitney_u_np" and k in ("pair_vec_f64","pair_vec_nan_f64"):
        y = inputs["y"]
        alt = str(case.get("alternative", "two-sided"))
        fn = lambda: sps.mannwhitneyu(x, y, alternative=alt)
        return {"status":"ok","value": fn()}
    
    return {"status":"skip_unavailable"}


def main():
    case = CASE
    inputs = _make_inputs(case)
    warmup = int(case.get("warmup", 1))
    repeats = int(case.get("repeats", 3))
    with_python = bool(case.get("with_python", False))
    track_memory = bool(case.get("track_memory", False))

    name = case["fn_name"]
    refs = tuple(case.get("refs", ())) or ("python","numpy","pandas","scipy")

    out = {"fn": name}

    # bunker-stats
    r_bs = _safe_call("bs", lambda: _call_bs(case, inputs), warmup, repeats, track_memory=track_memory)
    out["bs_status"] = r_bs.get("status")
    out["bs_error"] = r_bs.get("error", "")
    out["bs_median_s"] = r_bs.get("median_s", "")
    out["bs_mean_s"] = r_bs.get("mean_s", "")
    out["bs_cv"] = r_bs.get("cv", "")
    out["bs_p50_s"] = r_bs.get("p50_s", "")
    out["bs_p95_s"] = r_bs.get("p95_s", "")
    out["bs_p99_s"] = r_bs.get("p99_s", "")
    if track_memory:
        out["bs_mem_peak_mb"] = r_bs.get("mem_peak_mb", "")
    
    if r_bs.get("status") == "ok" and "value" in r_bs:
        try:
            out["bs_summary"] = _summarize(r_bs["value"])
        except Exception as e:
            out["bs_summary"] = {"error": repr(e)}

    def add_backend(tag, impl_fn):
        r = _safe_call(tag, impl_fn, warmup, repeats, track_memory=track_memory)
        out[f"{tag}_status"] = r.get("status")
        out[f"{tag}_error"] = r.get("error", "")
        out[f"{tag}_median_s"] = r.get("median_s", "")
        out[f"{tag}_mean_s"] = r.get("mean_s", "")
        out[f"{tag}_cv"] = r.get("cv", "")
        out[f"{tag}_p50_s"] = r.get("p50_s", "")
        out[f"{tag}_p95_s"] = r.get("p95_s", "")
        out[f"{tag}_p99_s"] = r.get("p99_s", "")
        if track_memory:
            out[f"{tag}_mem_peak_mb"] = r.get("mem_peak_mb", "")

        if r.get("status") == "ok" and "value" in r:
            try:
                out[f"{tag}_summary"] = _summarize(r["value"])
            except Exception as e:
                out[f"{tag}_summary"] = {"error": repr(e)}

        # speedup (baseline / bs): >1 means bs is faster
        try:
            bs_med = out.get("bs_median_s")
            tag_med = r.get("median_s")
            if r.get("status") == "ok" and bs_med not in ("", None) and tag_med not in ("", None):
                bs_med_f = float(bs_med)
                tag_med_f = float(tag_med)
                if bs_med_f > 0.0:
                    out[f"speedup_bs_vs_{tag}"] = tag_med_f / bs_med_f
                    # NEW: Speed difference (positive = bs is faster)
                    out[f"speed_diff_s_{tag}_minus_bs"] = tag_med_f - bs_med_f
                else:
                    out[f"speedup_bs_vs_{tag}"] = ""
                    out[f"speed_diff_s_{tag}_minus_bs"] = ""
            else:
                out[f"speedup_bs_vs_{tag}"] = ""
                out[f"speed_diff_s_{tag}_minus_bs"] = ""
        except Exception:
            out[f"speedup_bs_vs_{tag}"] = ""
            out[f"speed_diff_s_{tag}_minus_bs"] = ""
        
        # NEW: CV difference (positive = alternative is more variable)
        try:
            bs_cv = out.get("bs_cv")
            tag_cv = r.get("cv")
            if r.get("status") == "ok" and bs_cv not in ("", None) and tag_cv not in ("", None):
                out[f"cv_diff_{tag}_minus_bs"] = float(tag_cv) - float(bs_cv)
            else:
                out[f"cv_diff_{tag}_minus_bs"] = ""
        except Exception:
            out[f"cv_diff_{tag}_minus_bs"] = ""
        
        # NEW: Memory difference (if tracked)
        if track_memory:
            try:
                bs_mem = out.get("bs_mem_peak_mb")
                tag_mem = r.get("mem_peak_mb")
                if r.get("status") == "ok" and bs_mem not in ("", None) and tag_mem not in ("", None):
                    out[f"mem_diff_mb_{tag}_minus_bs"] = float(tag_mem) - float(bs_mem)
                else:
                    out[f"mem_diff_mb_{tag}_minus_bs"] = ""
            except Exception:
                out[f"mem_diff_mb_{tag}_minus_bs"] = ""

    # numpy/pandas/scipy baselines
    if "numpy" in refs:
        add_backend("numpy", lambda: _impl_numpy(name, case, inputs))
    if "pandas" in refs:
        add_backend("pandas", lambda: _impl_pandas(name, case, inputs))
    if "scipy" in refs:
        add_backend("scipy", lambda: _impl_scipy(name, case, inputs))
    if with_python and "python" in refs:
        add_backend("py", lambda: _impl_python(name, case, inputs))

    # overall status
    out["status"] = "ok" if out.get("bs_status") == "ok" else "fail"
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()

"""


def _code_head(code: str, n_lines: int) -> str:
    lines = code.splitlines()
    return "\n".join(lines[: max(1, n_lines)])


def _dump_code(debug_dir: Path, fn_name: str, code: str) -> str:
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in fn_name)
    path = debug_dir / f"subproc_{safe}.py"
    path.write_text(code, encoding="utf-8")
    return str(path)


def _run_case_in_subproc(case: Case, args) -> Dict[str, Any]:
    payload = {
        "fn_name": case.fn_name, 
        **case.args, 
        "repeats": args.repeats, 
        "warmup": args.warmup, 
        "with_python": args.with_python,
        "track_memory": args.track_memory,
    }
    case_json = json.dumps(payload)
    code = _SUBPROC_TEMPLATE.replace("CASE_JSON", repr(case_json))

    env = os.environ.copy()
    env.setdefault("RUST_BACKTRACE", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    if args.rayon_threads:
        env["RAYON_NUM_THREADS"] = str(args.rayon_threads)

    proc = subprocess.run([args.python, "-c", code], capture_output=True, text=True, env=env)

    if proc.returncode != 0:
        out: Dict[str, Any] = {
            "fn": case.fn_name,
            "status": "subproc_failed",
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-4000:],
            "stdout_tail": (proc.stdout or "")[-4000:],
        }
        if args.debug_fail:
            out["code_head"] = _code_head(code, int(args.debug_lines))
            if args.debug_dir:
                out["code_path"] = _dump_code(Path(args.debug_dir), case.fn_name, code)
        return out

    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        res = json.loads(last)
        if args.debug_fail:
            # attach tails on non-ok baselines too
            any_bad = any(
                (k.endswith("_status") and res.get(k) not in (None, "", "ok"))
                for k in res.keys()
            )
            if any_bad:
                res["stdout_tail"] = (proc.stdout or "")[-4000:]
                res["stderr_tail"] = (proc.stderr or "")[-4000:]
                res["code_head"] = _code_head(code, int(args.debug_lines))
                if args.debug_dir:
                    res["code_path"] = _dump_code(Path(args.debug_dir), case.fn_name, code)
        res.setdefault("fn", case.fn_name)
        return res
    except Exception:
        out = {
            "fn": case.fn_name,
            "status": "parse_error",
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
        if args.debug_fail:
            out["code_head"] = _code_head(code, int(args.debug_lines))
            if args.debug_dir:
                out["code_path"] = _dump_code(Path(args.debug_dir), case.fn_name, code)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--out", type=str, default="bench_compare_enhanced.csv")
    ap.add_argument("--skip", type=str, default="")
    ap.add_argument("--with-python", action="store_true", help="Include a pure-Python baseline (very slow for large n).")
    ap.add_argument("--track-memory", action="store_true", help="Track peak memory usage for each backend.")
    ap.add_argument("--rayon-threads", type=int, default=0, help="Set RAYON_NUM_THREADS in the child process.")
    ap.add_argument("--debug-fail", action="store_true")
    ap.add_argument("--debug-lines", type=int, default=120)
    ap.add_argument("--debug-dir", type=str, default="")
    args = ap.parse_args()

    args.debug_dir = (args.debug_dir or "").strip()

    cases = _make_cases_from_all(args.n, args.p, args.seed)
    skip = _parse_skip_list(args.skip)

    rows: List[Dict[str, Any]] = []
    for c in cases:
        if c.fn_name in skip:
            continue
        res = _run_case_in_subproc(c, args)
        rows.append(res)
        # print a compact status line based on bunker-stats + speedups
        bs_status = res.get("bs_status", res.get("status", ""))
        print(f"[{bs_status}] {c.fn_name}")

    # Build fieldnames as union of keys
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    # Enhanced preferred ordering with new comparison columns
    preferred = [
        "fn",
        "bs_status","bs_median_s","bs_cv","bs_p50_s","bs_p95_s","bs_p99_s",
        "bs_mem_peak_mb",
        
        "speedup_bs_vs_numpy","speed_diff_s_numpy_minus_bs","cv_diff_numpy_minus_bs","mem_diff_mb_numpy_minus_bs",
        "numpy_status","numpy_median_s","numpy_cv","numpy_p50_s","numpy_p95_s","numpy_p99_s","numpy_mem_peak_mb",
        
        "speedup_bs_vs_pandas","speed_diff_s_pandas_minus_bs","cv_diff_pandas_minus_bs","mem_diff_mb_pandas_minus_bs",
        "pandas_status","pandas_median_s","pandas_cv","pandas_p50_s","pandas_p95_s","pandas_p99_s","pandas_mem_peak_mb",
        
        "speedup_bs_vs_scipy","speed_diff_s_scipy_minus_bs","cv_diff_scipy_minus_bs","mem_diff_mb_scipy_minus_bs",
        "scipy_status","scipy_median_s","scipy_cv","scipy_p50_s","scipy_p95_s","scipy_p99_s","scipy_mem_peak_mb",
        
        "speedup_bs_vs_py","speed_diff_s_py_minus_bs","cv_diff_py_minus_bs","mem_diff_mb_py_minus_bs",
        "py_status","py_median_s","py_cv","py_p50_s","py_p95_s","py_p99_s","py_mem_peak_mb",
        
        "status","returncode","stderr_tail","stdout_tail","code_path",
    ]
    fieldnames = [k for k in preferred if k in all_keys] + sorted([k for k in all_keys if k not in preferred])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
