from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import scipy.stats as st  # type: ignore
except Exception:
    st = None

import bunker_stats as bs

from bs_debug import rich_enable, debug_array, assert_close, max_abs_diff, timeit, summarize_times


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
    mad: Optional[float]
    note: str = ""


@dataclass
class Case:
    name: str
    make_args: Callable[[], Tuple]
    impls: List[Tuple[str, Callable[..., Any]]]
    ref: Optional[Callable[..., Any]] = None
    rtol: float = 1e-7
    atol: float = 1e-9
    note: str = ""


def ref_rolling_slice(arr_full: np.ndarray, window: int) -> np.ndarray:
    return arr_full[window - 1:]


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


def run_case(case: Case, *, n: int, window: int, warmup: int, repeats: int, debug: bool) -> List[BenchResult]:
    args = case.make_args()

    ref_out = None
    if case.ref is not None:
        ref_out = case.ref(*args)

    results: List[BenchResult] = []
    for impl_name, impl_fn in case.impls:
        def call():
            return impl_fn(*args)

        try:
            times, out = timeit(call, warmup=warmup, repeats=repeats)
        except Exception as e:
            results.append(BenchResult(case.name, impl_name, n, window, repeats, warmup,
                                       float("nan"), float("nan"), float("nan"), float("nan"),
                                       None, None, note=f"SKIP: {e}"))
            continue

        ok = None
        mad = None
        note = case.note

        if ref_out is not None:
            try:
                assert_close(out, ref_out, rtol=case.rtol, atol=case.atol)
                ok = True
            except AssertionError as e:
                ok = False
                note = f"{note} | FAIL: {e}".strip(" |")
                if debug:
                    debug_array(out, f"{case.name}:{impl_name}:out")
                    debug_array(ref_out, f"{case.name}:{impl_name}:ref")

            mad = max_abs_diff(out, ref_out)

        t = summarize_times(times)
        results.append(BenchResult(case.name, impl_name, n, window, repeats, warmup,
                                   t.median_ms, t.mean_ms, t.std_ms, t.cv, ok, mad, note))
    return results


def main() -> None:
    rich_enable()

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench_suite_results.csv")
    ap.add_argument("--debug", action="store_true", help="Print rich debug info on failures")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = int(args.n)
    p = int(args.p)
    w = int(args.window)

    x = rng.normal(size=n).astype(np.float64)
    X = rng.normal(size=(n, p)).astype(np.float64)

    cases: List[Case] = [
        Case("mean", lambda: (x,), [("bunker_stats", bs.mean_np), ("numpy", np.mean)], ref=np.mean),
        Case("std_sample(ddof=1)", lambda: (x,), [("bunker_stats", bs.std_np), ("numpy", lambda v: np.std(v, ddof=1))],
             ref=lambda v: np.std(v, ddof=1)),
        Case("rolling_mean(window)", lambda: (x, w), [("bunker_stats", bs.rolling_mean_np),
                                                      ("pandas", ref_rolling_mean_pandas)] if pd is not None else
             [("bunker_stats", bs.rolling_mean_np)], ref=ref_rolling_mean_pandas if pd is not None else None),
        Case("rolling_std(window)", lambda: (x, w), [("bunker_stats", bs.rolling_std_np),
                                                     ("pandas", ref_rolling_std_pandas)] if pd is not None else
             [("bunker_stats", bs.rolling_std_np)], ref=ref_rolling_std_pandas if pd is not None else None),
    ]

    all_results: List[BenchResult] = []
    for c in cases:
        all_results.extend(run_case(c, n=n, window=w, warmup=args.warmup, repeats=args.repeats, debug=args.debug))
        print(f"[done] {c.name}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["func","impl","n","window","repeats","warmup","median_ms","mean_ms","std_ms","cv","ok","max_abs_diff","note"])
        for r in all_results:
            wr.writerow([
                r.func, r.impl, r.n, r.window, r.repeats, r.warmup,
                f"{r.median_ms:.6f}", f"{r.mean_ms:.6f}", f"{r.std_ms:.6f}", f"{r.cv:.6f}",
                "" if r.ok is None else str(bool(r.ok)),
                "" if r.mad is None else f"{r.mad:.6e}",
                r.note
            ])

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
