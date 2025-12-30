#!/usr/bin/env python
"""
bench_init_all_refs_subproc_debug_v5.py

Fixes v4 issue:
- v4 embedded CASE as a raw JSON literal in Python source: CASE = __CASE_JSON__
  That breaks when JSON contains `true/false/null` (not valid Python).
  v5 embeds it as a JSON *string* and parses via json.loads.

Also improves harness coverage:
- Adds default arguments for several functions that require extra params
  (trimmed_mean_np, mean_axis_np, outliers/scaling/winsorize/bins/kde, pad_nan_np, cohens_d_2samp_np, hedges_g_2samp_np, mann_whitney_u_np).
- Leaves ks_1samp_np as bs_error/skip unless you want to model a callable CDF contract.
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

    # Minimal specs: generate inputs and pass required args for the benchmark harness.
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
        "mann_whitney_u_np": {"kind": "pair_vec_f64", "alternative": "two-sided"},
        # ks_1samp_np needs a callable cdf; you can skip it or define a contract later.
    }

    out: List[Case] = []
    for name in list(getattr(bs, "__all__", [])):
        spec = dict(base)
        spec.update(explicit.get(name, {"kind": "vec_f64"}))
        out.append(Case(fn_name=name, args=spec))
    return out


# Child process template.
# IMPORTANT: CASE_JSON is embedded as a JSON string literal and parsed with json.loads.
_SUBPROC_TEMPLATE = r"""
import json, time
import numpy as np

import faulthandler
faulthandler.enable()

import bunker_stats as bs

CASE = json.loads(CASE_JSON)

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

def _call_bs(case, inputs):
    fn = getattr(bs, case["fn_name"], None)
    if fn is None:
        return {"status":"skip_missing"}

    name = case["fn_name"]
    k = case.get("kind", "vec_f64")

    if name == "ks_1samp_np":
        # Requires a callable CDF contract; skip for now.
        return {"status":"skip_needs_callable"}

    # Positional-arg calling to match your PyO3 signatures.
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

    try:
        for _ in range(warmup):
            r = _call_bs(case, inputs)
            if r.get("status") != "ok":
                print(json.dumps(r))
                return

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            r = _call_bs(case, inputs)
            t1 = time.perf_counter()
            if r.get("status") != "ok":
                print(json.dumps(r))
                return
            times.append(t1 - t0)

        out = {
            "status": "ok",
            "fn": case["fn_name"],
            "median_s": float(np.median(times)) if times else float("nan"),
            "mean_s": float(np.mean(times)) if times else float("nan"),
            "cv": float(np.std(times, ddof=1) / np.mean(times)) if len(times) > 1 and np.mean(times) else float("nan"),
        }
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"status":"bs_error","error":repr(e)}))

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
    payload = {"fn_name": case.fn_name, **case.args, "repeats": args.repeats, "warmup": args.warmup}
    case_json = json.dumps(payload)  # JSON text
    code = _SUBPROC_TEMPLATE.replace("CASE_JSON", repr(case_json))  # embed as a Python string literal

    env = os.environ.copy()
    env.setdefault("RUST_BACKTRACE", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")

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
        if args.debug_fail and res.get("status") != "ok":
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
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--out", type=str, default="bench_debug.csv")
    ap.add_argument("--skip", type=str, default="")
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
        print(f"[{res.get('status','')}] {c.fn_name}")

    if not rows:
        print("No rows produced")
        return

    preferred = [
        "fn","status","median_s","mean_s","cv",
        "returncode","error",
        "stderr_tail","stdout_tail","code_head","code_path",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = [k for k in preferred if k in all_keys] + sorted([k for k in all_keys if k not in preferred])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
