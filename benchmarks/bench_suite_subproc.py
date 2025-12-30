# benchmarks/bench_init_all_refs_subproc_fixed2.py
"""
Subprocess benchmark runner for bunker-stats.

Goals:
- Run each case in a subprocess (survive Rust panics / segfaults)
- Compare bunker_stats vs reference backends (numpy / pandas / scipy / pure python when appropriate)
- Measure speed (median/mean), accuracy (allclose + max abs diff), and stability (cv of timings)
- Avoid ALL .format() brace explosions by using sentinel replacement (__CASE_JSON__)

Usage examples:
  python benchmarks\\bench_init_all_refs_subproc_fixed2.py --out bench.csv
  python benchmarks\\bench_init_all_refs_subproc_fixed2.py --n 200000 --p 32 --repeats 5 --warmup 2 --out bench_full.csv --skip diff_np,pct_change_np
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics as stats
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ----------------------------
# Helpers
# ----------------------------

def _cov(xs: List[float]) -> float:
    """Coefficient of variation for timings (std/mean)."""
    if not xs:
        return float("nan")
    m = sum(xs) / len(xs)
    if m == 0:
        return float("inf")
    if len(xs) < 2:
        return 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v) / m

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))

def _max_abs_diff(a: Any, b: Any) -> float:
    """
    Robust max absolute diff between:
      - scalars
      - list/tuple
      - numpy-like string repr fallback (we avoid crashing the runner)
    """
    try:
        # scalar
        if _is_number(a) and _is_number(b):
            return abs(float(a) - float(b))

        # list/tuple
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return float("inf")
            ds = []
            for ai, bi in zip(a, b):
                ds.append(_max_abs_diff(ai, bi))
            return float(max(ds)) if ds else 0.0

        # dicts (compare common keys)
        if isinstance(a, dict) and isinstance(b, dict):
            keys = set(a.keys()) & set(b.keys())
            if not keys:
                return float("nan")
            return float(max(_max_abs_diff(a[k], b[k]) for k in keys))

        # numpy arrays might come through as repr strings from subprocess
        # (should not happen if JSON encoding is correct, but be defensive)
        if isinstance(a, str) or isinstance(b, str):
            return float("nan")

        # last resort
        return float("nan")
    except Exception:
        return float("nan")

def _allclose(a: Any, b: Any, rtol: float = 1e-8, atol: float = 1e-8) -> bool:
    """
    Loose allclose that works for scalars, tuples/lists, dicts (common keys).
    This runs in the parent process, comparing JSON-decoded results.
    """
    try:
        if _is_number(a) and _is_number(b):
            return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)

        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(_allclose(ai, bi, rtol=rtol, atol=atol) for ai, bi in zip(a, b))

        if isinstance(a, dict) and isinstance(b, dict):
            keys = set(a.keys()) & set(b.keys())
            if not keys:
                return False
            return all(_allclose(a[k], b[k], rtol=rtol, atol=atol) for k in keys)

        return False
    except Exception:
        return False

def _shorten(s: str, n: int = 1200) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n...[truncated]..."

def _first_n_lines(s: str, n: int = 60) -> str:
    lines = (s or "").splitlines()
    return "\n".join(lines[:n])

# ----------------------------
# Case specification
# ----------------------------

@dataclass
class Case:
    name: str
    fn_name: str
    args: Dict[str, Any]
    refs: Dict[str, Dict[str, Any]]  # backend -> config (like {"enabled": True})

# ----------------------------
# Subprocess template
# NOTE: NO .format() usage. We inject JSON via sentinel replacement only.
# ----------------------------

_SUBPROC_TEMPLATE = r"""
import json, math, os, sys, time, traceback

CASE = json.loads("__CASE_JSON__")

def _nan_to_none(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    return x

def _json_safe(obj):
    # Convert numpy scalars, arrays, etc. to JSON-safe structures
    try:
        import numpy as np
        if isinstance(obj, (np.generic,)):
            return _nan_to_none(obj.item())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return _nan_to_none(obj)

def _time_call(fn, repeats, warmup):
    # warmup
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return ts, out

def main():
    out = {
        "status": "ok",
        "name": CASE.get("name"),
        "fn_name": CASE.get("fn_name"),
        "args": CASE.get("args"),
        "bs": {},
        "refs": {}
    }

    # Import bunker-stats
    try:
        import bunker_stats as bs
    except Exception as e:
        out["status"] = "import_error"
        out["error"] = repr(e)
        print(json.dumps(out))
        return

    fn_name = CASE["fn_name"]
    if not hasattr(bs, fn_name):
        out["status"] = "skip_missing"
        print(json.dumps(out))
        return

    # Build the bunker-stats callable
    bs_fn = getattr(bs, fn_name)
    args = CASE.get("args", {})

    # Numpy and friends are optional, refs will self-skip if missing
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        from scipy import stats as sp_stats
    except Exception:
        sp_stats = None

    # Re-hydrate arrays from args (they arrive as plain lists)
    def _rehydrate(v):
        if isinstance(v, dict) and v.get("__kind__") == "ndarray":
            if np is None:
                raise RuntimeError("numpy missing")
            arr = np.array(v["data"], dtype=v.get("dtype", "float64"))
            if "shape" in v:
                arr = arr.reshape(tuple(v["shape"]))
            return arr
        return v

    real_args = {k: _rehydrate(v) for k, v in args.items()}

    # Run bunker-stats
    try:
        def _call_bs():
            return bs_fn(**real_args)
        bs_times, bs_val = _time_call(_call_bs, CASE["repeats"], CASE["warmup"])
        out["bs"] = {
            "status": "ok",
            "median_s": float(sorted(bs_times)[len(bs_times)//2]),
            "mean_s": float(sum(bs_times)/len(bs_times)),
            "cov": float((_nan_to_none(__import__("math").sqrt(sum((t - (sum(bs_times)/len(bs_times)))**2 for t in bs_times)/(len(bs_times)-1))/ (sum(bs_times)/len(bs_times))) if len(bs_times)>1 and (sum(bs_times)/len(bs_times))!=0 else 0.0)),
            "times_s": [float(t) for t in bs_times],
            "value": _json_safe(bs_val),
        }
    except Exception as e:
        out["bs"] = {"status": "bs_error", "error": repr(e), "traceback": traceback.format_exc()}
        out["status"] = "bs_error"
        print(json.dumps(out))
        return

    # Reference backends per case
    # Each backend function is looked up by name in a small registry below.
    backends = CASE.get("refs", {})

    def ref_numpy():
        if np is None:
            raise RuntimeError("numpy missing")
        # dispatch per fn_name
        if fn_name == "mean_np":
            return float(np.mean(real_args["a"]))
        if fn_name == "var_np":
            return float(np.var(real_args["a"], ddof=1))
        if fn_name == "std_np":
            return float(np.std(real_args["a"], ddof=1))
        if fn_name == "zscore_np":
            x = real_args["a"]
            m = np.mean(x)
            s = np.std(x, ddof=1)
            return ((x - m) / s).tolist()
        if fn_name == "mean_nan_np":
            return float(np.nanmean(real_args["a"])) if real_args["a"].size else float("nan")
        if fn_name == "var_nan_np":
            return float(np.nanvar(real_args["a"], ddof=1)) if real_args["a"].size else float("nan")
        if fn_name == "std_nan_np":
            return float(np.nanstd(real_args["a"], ddof=1)) if real_args["a"].size else float("nan")
        if fn_name == "percentile_np":
            return float(np.percentile(real_args["a"], real_args["q"]))
        if fn_name == "iqr_np":
            x = real_args["a"]
            q1 = np.percentile(x, 25)
            q3 = np.percentile(x, 75)
            return (float(q1), float(q3), float(q3 - q1))
        if fn_name == "mad_np":
            x = real_args["a"]
            med = np.median(x)
            return float(np.median(np.abs(x - med)))
        if fn_name == "cumsum_np":
            return np.cumsum(real_args["a"]).tolist()
        if fn_name == "cummean_np":
            x = real_args["a"]
            return (np.cumsum(x) / (np.arange(x.size) + 1)).tolist()
        if fn_name == "cov_matrix_np":
            x2 = real_args["x"]
            return np.cov(x2, rowvar=False, ddof=1).tolist()
        if fn_name == "corr_matrix_np":
            x2 = real_args["x"]
            return np.corrcoef(x2, rowvar=False).tolist()
        if fn_name == "chi2_gof_np":
            # SciPy usually; but fallback
            obs = real_args["observed"]
            exp = real_args["expected"]
            stat = float(((obs - exp) ** 2 / exp).sum())
            # pvalue needs chi2 sf; if scipy missing, return NaN
            return {"statistic": stat, "pvalue": float("nan"), "df": float(obs.size - 1)}
        raise RuntimeError("no numpy ref for this fn")

    def ref_pandas():
        if pd is None:
            raise RuntimeError("pandas missing")
        import numpy as np
        if fn_name == "rolling_mean_np":
            s = pd.Series(real_args["a"])
            return s.rolling(real_args["window"]).mean().to_numpy().tolist()
        if fn_name == "rolling_std_np":
            s = pd.Series(real_args["a"])
            return s.rolling(real_args["window"]).std(ddof=1).to_numpy().tolist()
        if fn_name == "rolling_mean_std_np":
            s = pd.Series(real_args["a"])
            m = s.rolling(real_args["window"]).mean().to_numpy()
            sd = s.rolling(real_args["window"]).std(ddof=1).to_numpy()
            return (m.tolist(), sd.tolist())
        if fn_name == "rolling_zscore_np":
            s = pd.Series(real_args["a"])
            m = s.rolling(real_args["window"]).mean()
            sd = s.rolling(real_args["window"]).std(ddof=1)
            return ((s - m) / sd).to_numpy().tolist()
        if fn_name == "ewma_np":
            s = pd.Series(real_args["a"])
            return s.ewm(alpha=real_args["alpha"], adjust=real_args.get("adjust", True)).mean().to_numpy().tolist()

        # axis0 rolling (DataFrame)
        if fn_name == "rolling_mean_axis0_np":
            x = real_args["x"]
            df = pd.DataFrame(x)
            return df.rolling(real_args["window"]).mean().to_numpy().tolist()
        if fn_name == "rolling_std_axis0_np":
            x = real_args["x"]
            df = pd.DataFrame(x)
            return df.rolling(real_args["window"]).std(ddof=1).to_numpy().tolist()
        if fn_name == "rolling_mean_std_axis0_np":
            x = real_args["x"]
            df = pd.DataFrame(x)
            m = df.rolling(real_args["window"]).mean().to_numpy()
            sd = df.rolling(real_args["window"]).std(ddof=1).to_numpy()
            return (m.tolist(), sd.tolist())

        raise RuntimeError("no pandas ref for this fn")

    def ref_scipy():
        if sp_stats is None:
            raise RuntimeError("scipy missing")
        import numpy as np
        if fn_name == "chi2_gof_np":
            obs = real_args["observed"]
            exp = real_args["expected"]
            r = sp_stats.chisquare(obs, exp)
            return {"statistic": float(r.statistic), "pvalue": float(r.pvalue), "df": float(obs.size - 1)}
        raise RuntimeError("no scipy ref for this fn")

    REF_REG = {
        "numpy": ref_numpy,
        "pandas": ref_pandas,
        "scipy": ref_scipy,
    }

    for be, cfg in backends.items():
        if not cfg.get("enabled", True):
            continue
        if be not in REF_REG:
            out["refs"][be] = {"status": "skip_unknown_backend"}
            continue
        try:
            def _call_ref():
                return REF_REG[be]()
            ref_times, ref_val = _time_call(_call_ref, CASE["repeats"], CASE["warmup"])
            out["refs"][be] = {
                "status": "ok",
                "median_s": float(sorted(ref_times)[len(ref_times)//2]),
                "mean_s": float(sum(ref_times)/len(ref_times)),
                "cov": float((_nan_to_none(__import__("math").sqrt(sum((t - (sum(ref_times)/len(ref_times)))**2 for t in ref_times)/(len(ref_times)-1))/ (sum(ref_times)/len(ref_times))) if len(ref_times)>1 and (sum(ref_times)/len(ref_times))!=0 else 0.0)),
                "times_s": [float(t) for t in ref_times],
                "value": _json_safe(ref_val),
            }
        except Exception as e:
            out["refs"][be] = {"status": "ref_error", "error": repr(e), "traceback": traceback.format_exc()}

    print(json.dumps(out))

if __name__ == "__main__":
    main()
""".strip()


def _run_case_in_subproc(
    case: Case,
    repeats: int,
    warmup: int,
    python_exe: str,
    show_code_on_fail: bool = True,
) -> Dict[str, Any]:
    payload = {
        "name": case.name,
        "fn_name": case.fn_name,
        "args": case.args,
        "refs": case.refs,
        "repeats": repeats,
        "warmup": warmup,
    }
    code = _SUBPROC_TEMPLATE.replace("__CASE_JSON__", json.dumps(payload))

    proc = subprocess.run(
        [python_exe, "-c", code],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        # Subprocess failed (syntax error, import error, panic causing abort, etc.)
        err = _shorten(proc.stderr, 4000)
        out = {
            "status": "subproc_failed",
            "name": case.name,
            "fn_name": case.fn_name,
            "returncode": proc.returncode,
            "stderr": err,
            "stdout": _shorten(proc.stdout, 2000),
        }
        if show_code_on_fail:
            out["code_head"] = _first_n_lines(code, 60)
        return out

    # Parse JSON result from stdout
    txt = (proc.stdout or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        out = {
            "status": "parse_error",
            "name": case.name,
            "fn_name": case.fn_name,
            "stdout": _shorten(txt, 4000),
            "stderr": _shorten(proc.stderr, 2000),
        }
        if show_code_on_fail:
            out["code_head"] = _first_n_lines(code, 60)
        return out

# ----------------------------
# Building cases from __init__.__all__
# ----------------------------

def _ndarray_payload(x, dtype="float64", shape=None) -> Dict[str, Any]:
    d = {"__kind__": "ndarray", "dtype": dtype, "data": x}
    if shape is not None:
        d["shape"] = list(shape)
    return d

def _make_default_inputs(n: int, p: int, seed: int = 0) -> Dict[str, Any]:
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n).astype(np.float64)
    x2 = rng.normal(size=n * p).astype(np.float64).reshape(n, p)
    return {"a": a, "x2": x2}

def _cases_from_init_all(n: int, p: int) -> List[Case]:
    import numpy as np
    import bunker_stats as bs

    inputs = _make_default_inputs(n=n, p=p)
    a = inputs["a"]
    x2 = inputs["x2"]

    # Discover exports
    exported = list(getattr(bs, "__all__", []))

    # We create best-effort cases only for functions we know how to reference.
    # Anything else is skipped (no crash).
    cases: List[Case] = []

    def add_case(name: str, fn: str, args: Dict[str, Any], refs: Dict[str, Dict[str, Any]]):
        cases.append(Case(name=name, fn_name=fn, args=args, refs=refs))

    # scalar stats
    if "mean_np" in exported:
        add_case("mean_np", "mean_np", {"a": _ndarray_payload(a.tolist())}, {"numpy": {"enabled": True}})
    if "var_np" in exported:
        add_case("var_np", "var_np", {"a": _ndarray_payload(a.tolist())}, {"numpy": {"enabled": True}})
    if "std_np" in exported:
        add_case("std_np", "std_np", {"a": _ndarray_payload(a.tolist())}, {"numpy": {"enabled": True}})
    if "zscore_np" in exported:
        add_case("zscore_np", "zscore_np", {"a": _ndarray_payload(a.tolist())}, {"numpy": {"enabled": True}})

    # percentile / iqr / mad
    if "percentile_np" in exported:
        add_case("percentile_np", "percentile_np", {"a": _ndarray_payload(a.tolist()), "q": 95.0}, {"numpy": {"enabled": True}})
    if "iqr_np" in exported:
        add_case("iqr_np", "iqr_np", {"a": _ndarray_payload(a.tolist())}, {"numpy": {"enabled": True}})
    if "mad_np" in exported:
        add_case("mad_np", "mad_np", {"a": _ndarray_payload(a.tolist())}, {"numpy": {"enabled": True}})

    # NaN aware scalar
    a_nan = a.copy()
    a_nan[::53] = np.nan
    if "mean_nan_np" in exported:
        add_case("mean_nan_np", "mean_nan_np", {"a": _ndarray_payload(a_nan.tolist())}, {"numpy": {"enabled": True}})
    if "var_nan_np" in exported:
        add_case("var_nan_np", "var_nan_np", {"a": _ndarray_payload(a_nan.tolist())}, {"numpy": {"enabled": True}})
    if "std_nan_np" in exported:
        add_case("std_nan_np", "std_nan_np", {"a": _ndarray_payload(a_nan.tolist())}, {"numpy": {"enabled": True}})

    # rolling 1d (pandas reference)
    window = 250
    if "rolling_mean_np" in exported:
        add_case("rolling_mean_np", "rolling_mean_np", {"a": _ndarray_payload(a.tolist()), "window": window}, {"pandas": {"enabled": True}})
    if "rolling_std_np" in exported:
        add_case("rolling_std_np", "rolling_std_np", {"a": _ndarray_payload(a.tolist()), "window": window}, {"pandas": {"enabled": True}})
    if "rolling_mean_std_np" in exported:
        add_case("rolling_mean_std_np", "rolling_mean_std_np", {"a": _ndarray_payload(a.tolist()), "window": window}, {"pandas": {"enabled": True}})
    if "rolling_zscore_np" in exported:
        add_case("rolling_zscore_np", "rolling_zscore_np", {"a": _ndarray_payload(a.tolist()), "window": window}, {"pandas": {"enabled": True}})
    if "ewma_np" in exported:
        add_case("ewma_np", "ewma_np", {"a": _ndarray_payload(a.tolist()), "alpha": 0.05, "adjust": True}, {"pandas": {"enabled": True}})

    # rolling nan (pandas reference)
    if "rolling_mean_nan_np" in exported:
        add_case("rolling_mean_nan_np", "rolling_mean_nan_np", {"a": _ndarray_payload(a_nan.tolist()), "window": window}, {"pandas": {"enabled": True}})
    if "rolling_std_nan_np" in exported:
        add_case("rolling_std_nan_np", "rolling_std_nan_np", {"a": _ndarray_payload(a_nan.tolist()), "window": window}, {"pandas": {"enabled": True}})
    if "rolling_zscore_nan_np" in exported:
        add_case("rolling_zscore_nan_np", "rolling_zscore_nan_np", {"a": _ndarray_payload(a_nan.tolist()), "window": window}, {"pandas": {"enabled": True}})

    # cumulatives
    if "cumsum_np" in exported:
        add_case("cumsum_np", "cumsum_np", {"a": _ndarray_payload(a.tolist())}, {"numpy": {"enabled": True}})
    if "cummean_np" in exported:
        add_case("cummean_np", "cummean_np", {"a": _ndarray_payload(a.tolist())}, {"numpy": {"enabled": True}})

    # matrix ops
    if "cov_matrix_np" in exported:
        add_case("cov_matrix_np", "cov_matrix_np", {"x": _ndarray_payload(x2.reshape(-1).tolist(), shape=x2.shape)}, {"numpy": {"enabled": True}})
    if "corr_matrix_np" in exported:
        add_case("corr_matrix_np", "corr_matrix_np", {"x": _ndarray_payload(x2.reshape(-1).tolist(), shape=x2.shape)}, {"numpy": {"enabled": True}})

    # axis0 rolling (pandas)
    if "rolling_mean_axis0_np" in exported:
        add_case("rolling_mean_axis0_np", "rolling_mean_axis0_np", {"x": _ndarray_payload(x2.reshape(-1).tolist(), shape=x2.shape), "window": 250}, {"pandas": {"enabled": True}})
    if "rolling_std_axis0_np" in exported:
        add_case("rolling_std_axis0_np", "rolling_std_axis0_np", {"x": _ndarray_payload(x2.reshape(-1).tolist(), shape=x2.shape), "window": 250}, {"pandas": {"enabled": True}})
    if "rolling_mean_std_axis0_np" in exported:
        add_case("rolling_mean_std_axis0_np", "rolling_mean_std_axis0_np", {"x": _ndarray_payload(x2.reshape(-1).tolist(), shape=x2.shape), "window": 250}, {"pandas": {"enabled": True}})

    # inference (scipy)
    obs = np.array([10, 12, 9, 11, 8], dtype=np.float64)
    exp = np.array([10, 10, 10, 10, 10], dtype=np.float64)
    if "chi2_gof_np" in exported:
        add_case("chi2_gof_np", "chi2_gof_np", {"observed": _ndarray_payload(obs.tolist()), "expected": _ndarray_payload(exp.tolist())}, {"scipy": {"enabled": True}})

    return cases

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--out", type=str, default="bench_init.csv")
    ap.add_argument("--skip", type=str, default="", help="comma-separated function names to skip")
    ap.add_argument("--debug-fail", action="store_true", help="print subprocess code head + stderr on failure")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    # Build cases
    cases = _cases_from_init_all(args.n, args.p)
    if skip:
        cases = [c for c in cases if c.fn_name not in skip]

    rows: List[Dict[str, Any]] = []

    for c in cases:
        res = _run_case_in_subproc(c, args.repeats, args.warmup, args.python, show_code_on_fail=args.debug_fail)

        status = res.get("status", "unknown")
        if status == "ok":
            print(f"[done] {c.name}")
        else:
            reason = status
            # If bs stage has status field
            bs = res.get("bs", {})
            if isinstance(bs, dict) and bs.get("status") == "bs_error":
                reason = "bs_error"
            print(f"[skip] {c.name} ({reason})")

        row: Dict[str, Any] = {
            "name": c.name,
            "fn_name": c.fn_name,
            "status": status,
        }

        # bunker-stats timing/value
        bs = res.get("bs", {})
        if isinstance(bs, dict) and bs.get("status") == "ok":
            row.update({
                "bs_median_s": bs.get("median_s"),
                "bs_mean_s": bs.get("mean_s"),
                "bs_cov": bs.get("cov"),
            })
            bs_val = bs.get("value")
        else:
            bs_val = None

        # refs
        refs = res.get("refs", {}) if isinstance(res.get("refs", {}), dict) else {}
        for be, r in refs.items():
            if not isinstance(r, dict):
                continue
            row[f"{be}_status"] = r.get("status")
            if r.get("status") == "ok":
                row[f"{be}_median_s"] = r.get("median_s")
                row[f"{be}_mean_s"] = r.get("mean_s")
                row[f"{be}_cov"] = r.get("cov")

                ref_val = r.get("value")
                row[f"{be}_allclose"] = _allclose(bs_val, ref_val)
                row[f"{be}_mad"] = _max_abs_diff(bs_val, ref_val)

        # debugging extras
        if status in ("subproc_failed", "parse_error", "bs_error", "import_error"):
            row["stderr"] = res.get("stderr") or res.get("error") or ""
            if args.debug_fail:
                row["code_head"] = res.get("code_head", "")

        rows.append(row)

    # Rank fastest/slowest by bunker-stats median where present
    ok_rows = [r for r in rows if isinstance(r.get("bs_median_s"), (int, float))]
    ok_rows_sorted = sorted(ok_rows, key=lambda r: r["bs_median_s"])
    print("\nTop 10 fastest (bunker-stats median):")
    for r in ok_rows_sorted[:10]:
        print(f"  {r['fn_name']:<28} {r['bs_median_s']:.6f}s")

    print("\nTop 10 slowest (bunker-stats median):")
    for r in ok_rows_sorted[-10:]:
        print(f"  {r['fn_name']:<28} {r['bs_median_s']:.6f}s")

    # Write CSV
    # Collect all columns
    cols = sorted({k for r in rows for k in r.keys()})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote: {args.out}")

if __name__ == "__main__":
    main()
