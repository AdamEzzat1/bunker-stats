#!/usr/bin/env python
"""
Benchmark: multi-column rolling mean/std (axis=0) vs pandas.

Compares:
- bunker_stats.rolling_mean_axis0_np / rolling_std_axis0_np / rolling_mean_std_axis0_np
- pandas.DataFrame.rolling(window).mean() / std(ddof=1)

Outputs a CSV with median, mean, std, and coefficient of variation (cv) for timings.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import bunker_stats as bs


def _time_one(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def summarize(times_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(times_ms, dtype=float)
    mean = float(arr.mean()) if arr.size else float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    med = float(np.median(arr)) if arr.size else float("nan")
    cv = float(std / mean) if mean != 0.0 else float("inf")
    return {"median_ms": med, "mean_ms": mean, "std_ms": std, "cv": cv}


@dataclass
class Case:
    name: str
    impl: str
    fn: Callable[[], object]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="rows")
    ap.add_argument("--p", type=int, default=32, help="columns")
    ap.add_argument("--window", type=int, default=50, help="rolling window")
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench_rolling_axis0_results.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, args.p)).astype(np.float64, copy=False)

    # pandas wants 2D columns; create DataFrame view once
    df = pd.DataFrame(X, copy=False)

    w = args.window

    def bs_mean():
        return bs.rolling_mean_axis0_np(X, w)

    def bs_std():
        return bs.rolling_std_axis0_np(X, w)

    def bs_mean_std():
        return bs.rolling_mean_std_axis0_np(X, w)

    def pd_mean():
        # pandas returns full-length with NaNs; to be comparable, we can slice off the first w-1 rows
        return df.rolling(window=w).mean().iloc[w-1:].to_numpy()

    def pd_std():
        return df.rolling(window=w).std(ddof=1).iloc[w-1:].to_numpy()

    cases: List[Case] = [
        Case("rolling_mean_axis0(window)", "bunker_stats", bs_mean),
        Case("rolling_std_axis0(window)", "bunker_stats", bs_std),
        Case("rolling_mean_std_axis0(window)", "bunker_stats", bs_mean_std),
        Case("rolling_mean_axis0(window)", "pandas", pd_mean),
        Case("rolling_std_axis0(window)", "pandas", pd_std),
    ]

    rows: List[Dict[str, object]] = []

    for case in cases:
        # warmup
        for _ in range(args.warmup):
            case.fn()

        times: List[float] = []
        for _ in range(args.repeats):
            times.append(_time_one(case.fn))

        stats = summarize(times)
        rows.append({
            "func": case.name,
            "impl": case.impl,
            "n": args.n,
            "p": args.p,
            "window": w,
            **stats
        })
        print(f"[done] {case.impl} {case.name} median={stats['median_ms']:.3f}ms")

    out_df = pd.DataFrame(rows).sort_values(["func", "impl"])
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
