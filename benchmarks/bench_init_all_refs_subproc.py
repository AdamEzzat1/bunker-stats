#!/usr/bin/env python
"""
bench_init_all_refs_subproc.py

Benchmarks *everything in bunker_stats.__all__* against reference backends:
- python (pure python fallback, when implemented)
- numpy
- pandas
- scipy

Autosafe:
- Each case runs in a subprocess so Rust panics don't kill the whole run.
- Accuracy metrics are computed inside the subprocess (no huge array JSON).
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
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def _cov(xs: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = statistics.mean(xs)
    if m == 0.0:
        return float("nan")
    s2 = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(s2) / m


def _parse_skip_list(s: str) -> set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


@dataclass(frozen=True)
class Case:
    name: str
    fn_name: str
    args: Dict[str, Any]
    refs: Tuple[str, ...] = ("python", "numpy", "pandas", "scipy")


def _make_cases_from_all(n: int, p: int, seed: int) -> List[Case]:
    import bunker_stats as bs

    names = list(getattr(bs, "__all__", []))

    base = {"n": n, "p": p, "seed": seed}

    # Explicit argument specs (fallback to vec_f64 for unknown)
    explicit: Dict[str, Dict[str, Any]] = {
        "mean_np": {"kind": "vec_f64"},
        "std_np": {"kind": "vec_f64", "ddof": 1},
        "var_np": {"kind": "vec_f64", "ddof": 1},
        "zscore_np": {"kind": "vec_f64"},
        "percentile_np": {"kind": "vec_f64", "q": 95.0},  # NumPy q in [0,100]
        "iqr_np": {"kind": "vec_f64"},
        "mad_np": {"kind": "vec_f64"},
        "mean_nan_np": {"kind": "vec_nan_f64"},
        "std_nan_np": {"kind": "vec_nan_f64", "ddof": 1},
        "var_nan_np": {"kind": "vec_nan_f64", "ddof": 1},
        "rolling_mean_np": {"kind": "vec_f64", "window": 64},
        "rolling_std_np": {"kind": "vec_f64", "window": 64, "ddof": 1},
        "rolling_mean_std_np": {"kind": "vec_f64", "window": 64, "ddof": 1},
        "rolling_zscore_np": {"kind": "vec_f64", "window": 64},
        "ewma_np": {"kind": "vec_f64", "alpha": 0.2},
        "rolling_mean_nan_np": {"kind": "vec_nan_f64", "window": 64},
        "rolling_std_nan_np": {"kind": "vec_nan_f64", "window": 64, "ddof": 1},
        "rolling_zscore_nan_np": {"kind": "vec_nan_f64", "window": 64},
        "diff_np": {"kind": "vec_f64", "periods": 1},
        "pct_change_np": {"kind": "vec_f64", "periods": 1},
        "cumsum_np": {"kind": "vec_f64"},
        "cummean_np": {"kind": "vec_f64"},
        "cov_np": {"kind": "pair_vec_f64"},
        "corr_np": {"kind": "pair_vec_f64"},
        "cov_nan_np": {"kind": "pair_vec_nan_f64"},
        "corr_nan_np": {"kind": "pair_vec_nan_f64"},
        "rolling_cov_np": {"kind": "pair_vec_f64", "window": 64},
        "rolling_corr_np": {"kind": "pair_vec_f64", "window": 64},
        "rolling_cov_nan_np": {"kind": "pair_vec_nan_f64", "window": 64},
        "rolling_corr_nan_np": {"kind": "pair_vec_nan_f64", "window": 64},
        "cov_matrix_np": {"kind": "mat_f64"},
        "corr_matrix_np": {"kind": "mat_f64"},
        "rolling_mean_axis0_np": {"kind": "mat_f64", "window": 64},
        "rolling_std_axis0_np": {"kind": "mat_f64", "window": 64, "ddof": 1},
        "rolling_mean_std_axis0_np": {"kind": "mat_f64", "window": 64, "ddof": 1},
        "chi2_gof_np": {"kind": "chi2_gof"},
        "chi2_independence_np": {"kind": "chi2_ind"},
        "t_test_1samp_np": {"kind": "vec_f64", "mu": 0.0, "alternative": "two-sided"},
        "t_test_2samp_np": {"kind": "pair_vec_f64", "equal_var": False, "alternative": "two-sided"},
    }

    cases: List[Case] = []
    for nm in names:
        spec = dict(base)
        spec.update(explicit.get(nm, {"kind": "vec_f64"}))
        cases.append(Case(name=nm, fn_name=nm, args=spec))
    return cases


_SUBPROC_TEMPLATE = r"""
import json, math, os, time
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None
try:
    from scipy import stats
except Exception:
    stats = None

import bunker_stats as bs

CASE = __CASE_JSON__

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
        m = rng.random(size=n) < 0.05
        x[m] = np.nan
        return x

    def mat_f64():
        return rng.normal(size=(n, p)).astype(np.float64)

    def pair_vec_f64():
        x = rng.normal(size=n).astype(np.float64)
        y = rng.normal(loc=0.2, size=n).astype(np.float64)
        return x, y

    def pair_vec_nan_f64():
        x, y = pair_vec_f64()
        mx = rng.random(size=n) < 0.05
        my = rng.random(size=n) < 0.05
        x[mx] = np.nan
        y[my] = np.nan
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

def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None

def _max_abs_diff(a, b):
    A = _as_numpy(a)
    B = _as_numpy(b)
    if A is None or B is None:
        try:
            return float(abs(float(a) - float(b)))
        except Exception:
            return float("nan")
    if A.shape != B.shape:
        return float("inf")
    if A.size == 0:
        return 0.0
    d = np.abs(A - B)
    return float(np.nanmax(d))

def _allclose(a, b, rtol=1e-7, atol=1e-9):
    A = _as_numpy(a)
    B = _as_numpy(b)
    if A is None or B is None:
        try:
            return bool(abs(float(a) - float(b)) <= (atol + rtol * abs(float(b))))
        except Exception:
            return False
    return bool(np.allclose(A, B, rtol=rtol, atol=atol, equal_nan=True))

def _run_bs(case, inputs):
    fn = getattr(bs, case["fn_name"], None)
    if fn is None:
        return {"status": "skip_missing"}

    name = case["fn_name"]
    k = case.get("kind", "vec_f64")

    if k in ("vec_f64","vec_nan_f64"):
        x = inputs["x"]
        if name == "percentile_np":
            return {"status":"ok","value": fn(x, float(case.get("q", 95.0)))}
        if name in ("diff_np","pct_change_np"):
            return {"status":"ok","value": fn(x, int(case.get("periods", 1)))}
        if name in ("rolling_mean_np","rolling_std_np","rolling_mean_std_np","rolling_zscore_np",
                    "rolling_mean_nan_np","rolling_std_nan_np","rolling_zscore_nan_np"):
            return {"status":"ok","value": fn(x, int(case.get("window", 64)))}
        if name == "ewma_np":
            return {"status":"ok","value": fn(x, float(case.get("alpha", 0.2)))}
        if name == "t_test_1samp_np":
            return {"status":"ok","value": fn(x, float(case.get("mu",0.0)), str(case.get("alternative","two-sided")))}
        return {"status":"ok","value": fn(x)}

    if k == "mat_f64":
        X = inputs["X"]
        if name in ("rolling_mean_axis0_np","rolling_std_axis0_np","rolling_mean_std_axis0_np"):
            return {"status":"ok","value": fn(X, int(case.get("window", 64)))}
        return {"status":"ok","value": fn(X)}

    if k in ("pair_vec_f64","pair_vec_nan_f64"):
        x = inputs["x"]; y = inputs["y"]
        if name in ("rolling_cov_np","rolling_corr_np","rolling_cov_nan_np","rolling_corr_nan_np"):
            return {"status":"ok","value": fn(x, y, int(case.get("window", 64)))}
        if name == "t_test_2samp_np":
            return {"status":"ok","value": fn(x, y, bool(case.get("equal_var", False)), str(case.get("alternative","two-sided")))}
        return {"status":"ok","value": fn(x, y)}

    if k == "chi2_gof":
        return {"status":"ok","value": fn(inputs["obs"], inputs["exp"])}
    if k == "chi2_ind":
        return {"status":"ok","value": fn(inputs["tab"])}

    return {"status":"skip_unknown"}

def _run_refs(case, inputs, bs_val):
    out = {}
    name = case["fn_name"]

    def ref_numpy():
        import numpy as np
        if name == "mean_np": return float(np.mean(inputs["x"]))
        if name == "std_np": return float(np.std(inputs["x"], ddof=1))
        if name == "var_np": return float(np.var(inputs["x"], ddof=1))
        if name == "zscore_np":
            x = inputs["x"]; m = np.mean(x); s = np.std(x, ddof=1)
            return (x - m) / s if s != 0 else np.full_like(x, np.nan, dtype=np.float64)
        if name == "percentile_np": return float(np.percentile(inputs["x"], float(case.get("q",95.0))))
        if name == "iqr_np":
            q1 = np.percentile(inputs["x"], 25.0); q3 = np.percentile(inputs["x"], 75.0)
            return (float(q1), float(q3), float(q3-q1))
        if name == "mad_np":
            x = inputs["x"]; med = np.median(x); return float(np.median(np.abs(x - med)))
        if name == "mean_nan_np": return float(np.nanmean(inputs["x"]))
        if name == "std_nan_np": return float(np.nanstd(inputs["x"], ddof=1))
        if name == "var_nan_np": return float(np.nanvar(inputs["x"], ddof=1))
        if name == "cumsum_np": return np.cumsum(inputs["x"])
        if name == "cummean_np": return np.cumsum(inputs["x"]) / (np.arange(inputs["x"].size, dtype=np.float64) + 1.0)
        if name == "cov_matrix_np": return np.cov(inputs["X"], rowvar=False, ddof=1)
        if name == "corr_matrix_np": return np.corrcoef(inputs["X"], rowvar=False)
        if name == "cov_np":
            x=inputs["x"]; y=inputs["y"]; return float(np.cov(x, y, ddof=1)[0,1])
        if name == "corr_np":
            x=inputs["x"]; y=inputs["y"]; return float(np.corrcoef(x, y)[0,1])
        raise KeyError("no numpy ref")

    def ref_pandas():
        if pd is None: raise RuntimeError("no pandas")
        if name == "rolling_mean_np":
            return pd.Series(inputs["x"]).rolling(int(case.get("window",64))).mean().to_numpy()
        if name == "rolling_std_np":
            return pd.Series(inputs["x"]).rolling(int(case.get("window",64))).std(ddof=1).to_numpy()
        if name == "rolling_mean_std_np":
            s = pd.Series(inputs["x"]).rolling(int(case.get("window",64)))
            return (s.mean().to_numpy(), s.std(ddof=1).to_numpy())
        if name == "rolling_zscore_np":
            w = int(case.get("window",64))
            s = pd.Series(inputs["x"]).rolling(w)
            m = s.mean(); sd = s.std(ddof=1)
            return ((pd.Series(inputs["x"]) - m) / sd).to_numpy()
        if name == "ewma_np":
            return pd.Series(inputs["x"]).ewm(alpha=float(case.get("alpha",0.2)), adjust=False).mean().to_numpy()
        if name in ("rolling_mean_axis0_np","rolling_std_axis0_np","rolling_mean_std_axis0_np"):
            w = int(case.get("window",64))
            df = pd.DataFrame(inputs["X"])
            r = df.rolling(w)
            if name == "rolling_mean_axis0_np": return r.mean().to_numpy()
            if name == "rolling_std_axis0_np": return r.std(ddof=1).to_numpy()
            if name == "rolling_mean_std_axis0_np": return (r.mean().to_numpy(), r.std(ddof=1).to_numpy())
        raise KeyError("no pandas ref")

    def ref_scipy():
        if stats is None: raise RuntimeError("no scipy")
        if name == "t_test_1samp_np":
            r = stats.ttest_1samp(inputs["x"], float(case.get("mu",0.0)), alternative=str(case.get("alternative","two-sided")))
            return {"statistic": float(r.statistic), "pvalue": float(r.pvalue)}
        if name == "t_test_2samp_np":
            r = stats.ttest_ind(inputs["x"], inputs["y"], equal_var=bool(case.get("equal_var", False)), alternative=str(case.get("alternative","two-sided")))
            return {"statistic": float(r.statistic), "pvalue": float(r.pvalue)}
        if name == "chi2_gof_np":
            r = stats.chisquare(inputs["obs"], inputs["exp"])
            return {"statistic": float(r.statistic), "pvalue": float(r.pvalue)}
        if name == "chi2_independence_np":
            r = stats.chi2_contingency(inputs["tab"], correction=False)
            return {"statistic": float(r[0]), "pvalue": float(r[1]), "df": float(r[2])}
        raise KeyError("no scipy ref")

    def ref_python():
        if name == "mean_np":
            xs = [float(v) for v in inputs["x"]]
            return sum(xs)/len(xs) if xs else float("nan")
        raise KeyError("no python ref")

    backends = {"python": ref_python, "numpy": ref_numpy, "pandas": ref_pandas, "scipy": ref_scipy}

    for be, f in backends.items():
        try:
            val = f()
            out[be] = {"status":"ok", "max_abs_diff": _max_abs_diff(bs_val, val), "allclose": _allclose(bs_val, val)}
        except KeyError:
            out[be] = {"status":"skip_no_ref"}
        except Exception as e:
            out[be] = {"status":"ref_error","error":repr(e)}
    return out

def main():
    case = dict(CASE)
    inputs = _make_inputs(case)

    warmup = int(case.get("warmup", 2))
    repeats = int(case.get("repeats", 5))

    bs_times = []
    last = None

    try:
        for _ in range(warmup):
            last = _run_bs(case, inputs)
            if last.get("status") != "ok":
                print(json.dumps({{"status": last.get("status","bs_error")}}))
                return

        for _ in range(repeats):
            t0 = time.perf_counter()
            last = _run_bs(case, inputs)
            t1 = time.perf_counter()
            if last.get("status") != "ok":
                print(json.dumps({{"status": last.get("status","bs_error")}}))
                return
            bs_times.append(t1 - t0)

    except Exception as e:
        print(json.dumps({{"status":"bs_error","error":repr(e)}}))
        return

    refs = _run_refs(case, inputs, last.get("value", None))

    mean_s = float(np.mean(bs_times)) if bs_times else float("nan")
    cv = float(np.std(bs_times, ddof=1) / mean_s) if len(bs_times) > 1 and mean_s != 0 else float("nan")

    out = {{
        "status": "ok",
        "fn": case["fn_name"],
        "median_s": float(np.median(bs_times)) if bs_times else float("nan"),
        "mean_s": mean_s,
        "cv": cv,
        "refs": refs,
    }}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
"""

def _run_case_in_subproc(case, repeats, warmup, python_exe):
    payload = {"fn_name": case.fn_name, **case.args, "repeats": repeats, "warmup": warmup}
    code = _SUBPROC_TEMPLATE.replace("__CASE_JSON__", json.dumps(payload))
    env = os.environ.copy()
    env.setdefault("RUST_BACKTRACE", "1")
    proc = subprocess.run([python_exe, "-c", code], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return {"status":"subproc_failed","returncode":proc.returncode,"stderr":proc.stderr[-4000:],"stdout":proc.stdout[-4000:]}
    out = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(out)
    except Exception:
        return {"status":"parse_error","stdout":proc.stdout[-4000:],"stderr":proc.stderr[-4000:]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--out", type=str, default="bench_init_all.csv")
    ap.add_argument("--skip", type=str, default="")
    args = ap.parse_args()

    skip = _parse_skip_list(args.skip)
    cases = _make_cases_from_all(args.n, args.p, args.seed)

    rows: List[Dict[str, Any]] = []
    for c in cases:
        if c.fn_name in skip:
            print(f"[skip] {c.fn_name} (user_skip)")
            continue
        res = _run_case_in_subproc(c, args.repeats, args.warmup, args.python)
        if res.get("status") != "ok":
            print(f"[skip] {c.fn_name} ({res.get('status')})")
            rows.append({"fn": c.fn_name, "status": res.get("status",""), "median_s": "", "mean_s": "", "cv": ""})
            continue
        refs = res.get("refs", {})
        row = {
            "fn": c.fn_name,
            "status": "ok",
            "median_s": res.get("median_s",""),
            "mean_s": res.get("mean_s",""),
            "cv": res.get("cv",""),
        }
        for be in ("python","numpy","pandas","scipy"):
            r = refs.get(be, {})
            row[f"{be}_status"] = r.get("status","")
            row[f"{be}_max_abs_diff"] = r.get("max_abs_diff","")
            row[f"{be}_allclose"] = r.get("allclose","")
        rows.append(row)
        print(f"[done] {c.fn_name}")

    if not rows:
        print("No rows produced.")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "ok" and r.get("median_s") not in ("", None)]
    ok_sorted = sorted(ok, key=lambda r: float(r["median_s"]))
    print("\nTop 10 fastest (bunker-stats median):")
    for r in ok_sorted[:10]:
        print(f"  {r['fn']:<28} {float(r['median_s']):.6f}s")
    print("\nTop 10 slowest (bunker-stats median):")
    for r in ok_sorted[-10:]:
        print(f"  {r['fn']:<28} {float(r['median_s']):.6f}s")
    print(f"\nWrote: {args.out}")

if __name__ == "__main__":
    main()
