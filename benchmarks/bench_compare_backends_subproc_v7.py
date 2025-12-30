#!/usr/bin/env python
"""
bench_compare_backends_subproc_v7.py

Based on your debug runner v5 (subprocess-per-case) but extended to benchmark
bunker_stats vs numpy vs pandas vs scipy (and optional pure-Python baseline).

Key design points:
- Each CASE runs in a fresh subprocess (survive Rust panics, isolate imports).
- Child creates inputs ONCE, then times each backend with warmup/repeats.
- For each backend we report median/mean/cv and speedup vs bunker_stats.
- Backends that are not applicable or missing deps are marked as skip_*.

Notes:
- Pure Python baseline is extremely slow for large n. Use --with-python only
  when you really want it, or reduce --n.
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
        "pad_nan_np": {"kind": "vec_nan_f64", "n_pad": 8},

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
        # ks_1samp_np needs callable; we’ll skip it.
        # mann_whitney_u_np may be intentionally missing; we'll skip if not exported.
    }

    out: List[Case] = []
    for name in list(getattr(bs, "__all__", [])):
        spec = dict(base)
        spec.update(explicit.get(name, {"kind": "vec_f64"}))
        out.append(Case(fn_name=name, args=spec))
    return out


_SUBPROC_TEMPLATE = r"""
import json, time, math
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
    return {"x": vec_f64()}


def _timeit(fn, warmup, repeats):
    # warmup
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times, dtype=np.float64)
    med = float(np.median(times)) if times.size else float("nan")
    mean = float(np.mean(times)) if times.size else float("nan")
    cv = float(np.std(times, ddof=1) / mean) if times.size > 1 and mean else float("nan")
    return {"median_s": med, "mean_s": mean, "cv": cv}


def _safe_call(label, thunk, warmup, repeats):
    try:
        t = _timeit(thunk, warmup, repeats)
        return {"status": "ok", **t}
    except AttributeError as e:
        # common when __init__.py exports a placeholder that raises AttributeError
        return {"status": "skip_missing_export", "error": repr(e)}
    except Exception as e:
        return {"status": "error", "error": repr(e)}


# ---- Baseline implementations (best-effort, function-by-function) ----

def _impl_python(name, case, inputs):
    # Optional and intentionally limited (very slow on large n).
    import statistics as st
    k = case.get("kind", "vec_f64")
    if k not in ("vec_f64", "vec_nan_f64", "pair_vec_f64", "pair_vec_nan_f64"):
        raise NotImplementedError("python baseline only for vectors/pairs")

    if k in ("vec_f64", "vec_nan_f64"):
        x = inputs["x"]
        xs = [float(v) for v in x.tolist() if not (math.isnan(v) if isinstance(v, float) else False)]
        if name == "mean_np":
            return st.fmean(xs)
        if name == "std_np":
            # sample stdev
            return st.stdev(xs)
        if name == "var_np":
            return st.variance(xs)
        raise NotImplementedError("python baseline not defined for this fn")

    if k in ("pair_vec_f64", "pair_vec_nan_f64"):
        x = inputs["x"]; y = inputs["y"]
        xs = [float(v) for v in x.tolist()]
        ys = [float(v) for v in y.tolist()]
        if name == "cov_np":
            mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
            return sum((a-mx)*(b-my) for a,b in zip(xs,ys)) / (len(xs)-1)
        if name == "corr_np":
            mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
            num = sum((a-mx)*(b-my) for a,b in zip(xs,ys))
            denx = math.sqrt(sum((a-mx)**2 for a in xs))
            deny = math.sqrt(sum((b-my)**2 for b in ys))
            return num/(denx*deny)
        raise NotImplementedError("python baseline not defined for this fn")
    raise NotImplementedError("python baseline not defined")


def _impl_numpy(name, case, inputs):
    k = case.get("kind", "vec_f64")
    if k in ("vec_f64", "vec_nan_f64"):
        x = inputs["x"]
        if name == "mean_np": return float(np.nanmean(x) if k=="vec_nan_f64" else np.mean(x))
        if name == "std_np": return float(np.nanstd(x, ddof=1) if k=="vec_nan_f64" else np.std(x, ddof=1))
        if name == "var_np": return float(np.nanvar(x, ddof=1) if k=="vec_nan_f64" else np.var(x, ddof=1))
        if name == "zscore_np":
            m = np.mean(x); s = np.std(x, ddof=1)
            return (x - m) / (s if s else 1.0)
        if name == "percentile_np": return float(np.percentile(x, float(case.get("q",95.0))))
        if name == "diff_np": return np.diff(x, n=int(case.get("periods",1)))
        if name == "cumsum_np": return np.cumsum(x)
        if name == "cummean_np": return np.cumsum(x) / np.arange(1, x.size+1)
        # rolling: numpy doesn't have a standard rolling; skip
        raise NotImplementedError("numpy baseline not defined for this fn")
    if k in ("pair_vec_f64", "pair_vec_nan_f64"):
        x = inputs["x"]; y = inputs["y"]
        if name == "cov_np":
            return float(np.cov(x, y, ddof=1)[0,1])
        if name == "corr_np":
            return float(np.corrcoef(x, y)[0,1])
        raise NotImplementedError("numpy baseline not defined for this fn")
    if k == "mat_f64":
        X = inputs["X"]
        if name == "cov_matrix_np":
            return np.cov(X, rowvar=False, ddof=1)
        if name == "corr_matrix_np":
            return np.corrcoef(X, rowvar=False)
        raise NotImplementedError("numpy baseline not defined for this fn")
    raise NotImplementedError("numpy baseline not defined")


def _impl_pandas(name, case, inputs):
    if pd is None:
        raise ImportError("pandas not installed in this env")
    k = case.get("kind", "vec_f64")
    if k in ("vec_f64", "vec_nan_f64"):
        x = inputs["x"]
        s = pd.Series(x)
        w = int(case.get("window",64))
        if name == "rolling_mean_np": return s.rolling(w).mean().to_numpy()
        if name == "rolling_std_np": return s.rolling(w).std(ddof=1).to_numpy()
        if name == "rolling_var_np": return s.rolling(w).var(ddof=1).to_numpy()
        if name == "rolling_zscore_np":
            m = s.rolling(w).mean()
            sd = s.rolling(w).std(ddof=1)
            return ((s - m) / sd).to_numpy()
        if name == "rolling_mean_std_np":
            m = s.rolling(w).mean().to_numpy()
            sd = s.rolling(w).std(ddof=1).to_numpy()
            return m, sd
        if name == "ewma_np":
            a = float(case.get("alpha",0.2))
            # pandas uses span/alpha; match alpha directly
            return s.ewm(alpha=a, adjust=False).mean().to_numpy()
        if name == "diff_np":
            p = int(case.get("periods",1))
            return s.diff(p).to_numpy()
        if name == "pct_change_np":
            p = int(case.get("periods",1))
            return s.pct_change(p).to_numpy()
        if name == "cumsum_np": return s.cumsum().to_numpy()
        if name == "cummean_np": return s.expanding().mean().to_numpy()
        raise NotImplementedError("pandas baseline not defined for this fn")

    if k == "mat_f64":
        X = inputs["X"]
        df = pd.DataFrame(X)
        w = int(case.get("window",64))
        if name == "mean_axis_np":
            axis = int(case.get("axis",0))
            return df.mean(axis=axis).to_numpy()
        if name == "cov_matrix_np": return df.cov(ddof=1).to_numpy()
        if name == "corr_matrix_np": return df.corr().to_numpy()
        if name == "rolling_mean_axis0_np": return df.rolling(w).mean().to_numpy()
        if name == "rolling_std_axis0_np": return df.rolling(w).std(ddof=1).to_numpy()
        if name == "rolling_mean_std_axis0_np":
            m = df.rolling(w).mean().to_numpy()
            sd = df.rolling(w).std(ddof=1).to_numpy()
            return m, sd
        raise NotImplementedError("pandas baseline not defined for this fn")

    if k in ("pair_vec_f64","pair_vec_nan_f64"):
        x = inputs["x"]; y = inputs["y"]
        sx = pd.Series(x); sy = pd.Series(y)
        w = int(case.get("window",64))
        if name == "cov_np": return float(sx.cov(sy))
        if name == "corr_np": return float(sx.corr(sy))
        if name == "rolling_cov_np": return sx.rolling(w).cov(sy).to_numpy()
        if name == "rolling_corr_np": return sx.rolling(w).corr(sy).to_numpy()
        raise NotImplementedError("pandas baseline not defined for this fn")

    raise NotImplementedError("pandas baseline not defined")


def _impl_scipy(name, case, inputs):
    if sps is None:
        raise ImportError("scipy not installed in this env")
    k = case.get("kind", "vec_f64")
    if k in ("vec_f64","vec_nan_f64"):
        x = inputs["x"]
        if name == "skew_np": return float(sps.skew(x, nan_policy="omit" if k=="vec_nan_f64" else "propagate"))
        if name == "kurtosis_np": return float(sps.kurtosis(x, nan_policy="omit" if k=="vec_nan_f64" else "propagate"))
        if name == "trimmed_mean_np":
            p = float(case.get("proportion_to_cut", 0.1))
            return float(sps.trim_mean(x, proportiontocut=p))
        if name == "t_test_1samp_np":
            mu = float(case.get("mu",0.0))
            alt = str(case.get("alternative","two-sided"))
            return sps.ttest_1samp(x, popmean=mu, alternative=alt)
        raise NotImplementedError("scipy baseline not defined for this fn")
    if k in ("pair_vec_f64","pair_vec_nan_f64"):
        x = inputs["x"]; y = inputs["y"]
        if name == "t_test_2samp_np":
            alt = str(case.get("alternative","two-sided"))
            ev = bool(case.get("equal_var", False))
            return sps.ttest_ind(x, y, equal_var=ev, alternative=alt)
        if name == "mann_whitney_u_np":
            alt = str(case.get("alternative","two-sided"))
            return sps.mannwhitneyu(x, y, alternative=alt)
        raise NotImplementedError("scipy baseline not defined for this fn")
    if k == "chi2_gof":
        obs = inputs["obs"]; exp = inputs["exp"]
        return sps.chisquare(f_obs=obs, f_exp=exp)
    if k == "chi2_ind":
        tab = inputs["tab"]
        return sps.chi2_contingency(tab)
    raise NotImplementedError("scipy baseline not defined")


def _call_bs(case, inputs):
    fn = getattr(bs, case["fn_name"], None)
    if fn is None:
        return {"status":"skip_missing_export"}

    name = case["fn_name"]
    k = case.get("kind", "vec_f64")

    if name == "ks_1samp_np":
        return {"status":"skip_needs_callable"}

    if k in ("vec_f64","vec_nan_f64"):
        x = inputs["x"]
        if name == "percentile_np":
            return {"status":"ok","value": fn(x, float(case.get("q",95.0)))}
        if name == "trimmed_mean_np":
            return {"status":"ok","value": fn(x, float(case.get("proportion_to_cut", 0.1)))}
        if name in ("diff_np","pct_change_np"):
            return {"status":"ok","value": fn(x, int(case.get("periods", 1)))}
        if name in ("rolling_mean_np","rolling_std_np","rolling_var_np","rolling_mean_std_np","rolling_zscore_np",
                    "rolling_mean_nan_np","rolling_std_nan_np","rolling_zscore_nan_np"):
            return {"status":"ok","value": fn(x, int(case.get("window", 64)))}
        if name == "ewma_np":
            return {"status":"ok","value": fn(x, float(case.get("alpha", 0.2)))}
        if name == "iqr_outliers_np":
            return {"status":"ok","value": fn(x, float(case.get("k", 1.5)))}
        if name == "zscore_outliers_np":
            return {"status":"ok","value": fn(x, float(case.get("threshold", 3.0)))}
        if name == "robust_scale_np":
            return {"status":"ok","value": fn(x, float(case.get("scale_factor", 1.0)))}
        if name == "winsorize_np":
            return {"status":"ok","value": fn(x, float(case.get("lower_q", 0.05)), float(case.get("upper_q", 0.95)))}
        if name == "quantile_bins_np":
            return {"status":"ok","value": fn(x, int(case.get("n_bins", 10)))}
        if name == "kde_gaussian_np":
            return {"status":"ok","value": fn(x, int(case.get("n_points", 256)))}
        if name == "pad_nan_np":
            return {"status":"ok","value": fn(x, int(case.get("n_pad", 8)))}
        if name == "t_test_1samp_np":
            return {"status":"ok","value": fn(x, float(case.get("mu",0.0)), str(case.get("alternative","two-sided")))}
        return {"status":"ok","value": fn(x)}

    if k == "mat_f64":
        X = inputs["X"]
        if name == "mean_axis_np":
            return {"status":"ok","value": fn(X, int(case.get("axis", 0)))}
        if name in ("rolling_mean_axis0_np","rolling_std_axis0_np","rolling_mean_std_axis0_np"):
            return {"status":"ok","value": fn(X, int(case.get("window", 64)))}
        return {"status":"ok","value": fn(X)}

    if k in ("pair_vec_f64","pair_vec_nan_f64"):
        x = inputs["x"]; y = inputs["y"]
        if name in ("rolling_cov_np","rolling_corr_np","rolling_cov_nan_np","rolling_corr_nan_np"):
            return {"status":"ok","value": fn(x, y, int(case.get("window", 64)))}
        if name == "t_test_2samp_np":
            return {"status":"ok","value": fn(x, y, bool(case.get("equal_var", False)), str(case.get("alternative","two-sided")))}
        if name == "cohens_d_2samp_np":
            return {"status":"ok","value": fn(x, y, bool(case.get("pooled", True)))}
        if name == "hedges_g_2samp_np":
            return {"status":"ok","value": fn(x, y)}
        if name == "mann_whitney_u_np":
            return {"status":"ok","value": fn(x, y, str(case.get("alternative","two-sided")))}
        return {"status":"ok","value": fn(x, y)}

    if k == "chi2_gof":
        return {"status":"ok","value": fn(inputs["obs"], inputs["exp"])}
    if k == "chi2_ind":
        return {"status":"ok","value": fn(inputs["tab"])}

    return {"status":"skip_unknown"}


def main():
    case = dict(CASE)
    inputs = _make_inputs(case)
    warmup = int(case.get("warmup", 1))
    repeats = int(case.get("repeats", 3))
    with_python = bool(case.get("with_python", False))

    name = case["fn_name"]

    out = {"fn": name}

    # bunker-stats
    r_bs = _safe_call("bs", lambda: _call_bs(case, inputs), warmup, repeats)
    out["bs_status"] = r_bs.get("status")
    out["bs_error"] = r_bs.get("error", "")
    out["bs_median_s"] = r_bs.get("median_s", "")
    out["bs_mean_s"] = r_bs.get("mean_s", "")
    out["bs_cv"] = r_bs.get("cv", "")

    # If bs isn't ok, don't compare.
    if out["bs_status"] != "ok":
        print(json.dumps(out))
        return

    # baselines
    def add_backend(tag, impl_fn):
        r = _safe_call(tag, impl_fn, warmup, repeats)
        out[f"{tag}_status"] = r.get("status")
        out[f"{tag}_error"] = r.get("error", "")
        out[f"{tag}_median_s"] = r.get("median_s", "")
        out[f"{tag}_mean_s"] = r.get("mean_s", "")
        out[f"{tag}_cv"] = r.get("cv", "")
        # speedup (baseline / bs) i.e., >1 means bs is faster
        try:
            if r.get("status") == "ok":
                out[f"speedup_bs_vs_{tag}"] = float(r["median_s"]) / float(out["bs_median_s"])
            else:
                out[f"speedup_bs_vs_{tag}"] = ""
        except Exception:
            out[f"speedup_bs_vs_{tag}"] = ""

    if with_python:
        add_backend("py", lambda: _impl_python(name, case, inputs))

    add_backend("numpy", lambda: _impl_numpy(name, case, inputs))
    add_backend("pandas", lambda: _impl_pandas(name, case, inputs))
    add_backend("scipy", lambda: _impl_scipy(name, case, inputs))

    print(json.dumps(out))


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
    payload = {"fn_name": case.fn_name, **case.args, "repeats": args.repeats, "warmup": args.warmup, "with_python": args.with_python}
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
    ap.add_argument("--out", type=str, default="bench_compare.csv")
    ap.add_argument("--skip", type=str, default="")
    ap.add_argument("--with-python", action="store_true", help="Include a pure-Python baseline (very slow for large n).")
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

    preferred = [
        "fn",
        "bs_status","bs_median_s","bs_cv",
        "speedup_bs_vs_numpy","numpy_status","numpy_median_s","numpy_cv",
        "speedup_bs_vs_pandas","pandas_status","pandas_median_s","pandas_cv",
        "speedup_bs_vs_scipy","scipy_status","scipy_median_s","scipy_cv",
        "speedup_bs_vs_py","py_status","py_median_s","py_cv",
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
