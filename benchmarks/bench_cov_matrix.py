#!/usr/bin/env python
"""
Benchmark: covariance matrix kernel (cov_matrix_np) end to end.

Every case runs in its own subprocess so that peak memory is attributable to
one call and so the Rayon pool size can be pinned per run. Timings INCLUDE the
NumPy -> Rust conversion (that is what a user pays), and np.cov is reported
only as context.

Per case this records:
- wall time: best and median of --repeats calls, bunker-stats and np.cov
- extra peak memory of one bunker-stats call (process high-water mark delta)
- numerical error against a math.fsum reference on sampled (i, j) pairs,
  scaled by sqrt(var_i * var_j) so near-zero covariances do not dominate
- determinism: sha256 of the output bytes across repeated runs, and across
  RAYON_NUM_THREADS=1 vs the default pool

Build a portable (wheel-like) extension first; .cargo/config.toml sets
target-cpu=native, which is not what users install:

    RUSTFLAGS="" maturin develop --release
    python benchmarks/bench_cov_matrix.py --label after --out benchmarks/bench_cov_matrix_after.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List

# (name, n, p, layout, data)
#   layout: c | f | strided | f32        data: normal | offset
CASES = [
    ("small",        200,       5,    "c",       "normal"),
    ("medium",       10_000,    20,   "c",       "normal"),
    ("tall",         1_000_000, 8,    "c",       "normal"),
    ("large",        100_000,   100,  "c",       "normal"),
    ("wide",         5_000,     500,  "c",       "normal"),
    ("square",       500,       500,  "c",       "normal"),
    ("wide_short",   200,       1000, "c",       "normal"),
    ("fortran",      100_000,   20,   "f",       "normal"),
    ("strided",      100_000,   20,   "strided", "normal"),
    ("float32",      100_000,   20,   "f32",     "normal"),
    ("large_offset", 100_000,   20,   "c",       "offset"),
]
QUICK = {"small", "medium", "square", "fortran", "large_offset"}

REF_PAIRS = 12


def _make(n: int, p: int, layout: str, data: str):
    """Build the input directly in its target layout (no large transients)."""
    import numpy as np

    rng = np.random.default_rng(12345)
    loc = 1.0e8 if data == "offset" else 0.0

    if layout == "f32":
        return rng.standard_normal((n, p), dtype=np.float32)
    if layout == "strided":
        base = np.empty((n, 2 * p))
        view = base[:, ::2]
    elif layout == "f":
        base = np.empty((n, p), order="F")
        view = base
    else:
        base = np.empty((n, p))
        view = base
    base[...] = 0.0
    step = max(1, 2_000_000 // max(p, 1))
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        view[r0:r1, :] = rng.standard_normal((r1 - r0, p)) + loc
    return view


def _peak_bytes() -> int:
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows high-water mark
    if peak is not None:
        return int(peak)
    import resource  # Linux: KiB, macOS: bytes

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru if sys.platform == "darwin" else ru * 1024)


def _rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _reference_error(x, cov) -> float:
    """Max |cov - ref| / sqrt(var_i var_j) over sampled pairs (fsum reference)."""
    import numpy as np

    n, p = x.shape
    rng = np.random.default_rng(7)
    pairs = {(0, 0), (0, p - 1), (p - 1, p - 1)}
    while len(pairs) < min(REF_PAIRS, p * (p + 1) // 2):
        i, j = sorted(int(v) for v in rng.integers(0, p, 2))
        pairs.add((i, j))

    cols: Dict[int, object] = {}

    def centered(c: int):
        if c not in cols:
            col = np.ascontiguousarray(x[:, c], dtype=np.float64)
            cols[c] = col - math.fsum(col) / n
        return cols[c]

    def ref(i: int, j: int) -> float:
        return math.fsum(centered(i) * centered(j)) / (n - 1)

    worst = 0.0
    for i, j in sorted(pairs):
        scale = math.sqrt(ref(i, i) * ref(j, j))
        if scale > 0.0:
            worst = max(worst, abs(float(cov[i, j]) - ref(i, j)) / scale)
    return worst


def child(spec: dict) -> None:
    import numpy as np
    from bunker_stats import _rs as bsr

    n, p = spec["n"], spec["p"]
    x = _make(n, p, spec["layout"], spec["data"])

    # Memory first, while the process high-water mark is still the input.
    rss_before = _rss_bytes()
    out = bsr.cov_matrix_np(x)
    extra_peak = max(0, _peak_bytes() - rss_before)

    digests = {hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()}
    if spec["hash_only"]:
        print(json.dumps({"digest": sorted(digests)[0]}))
        return

    def timeit(fn) -> List[float]:
        ts = []
        for _ in range(spec["repeats"]):
            t0 = time.perf_counter()
            res = fn()
            ts.append((time.perf_counter() - t0) * 1e3)
            if fn is bs_call:
                digests.add(hashlib.sha256(np.ascontiguousarray(res).tobytes()).hexdigest())
        return ts

    def bs_call():
        return bsr.cov_matrix_np(x)

    def np_call():
        return np.cov(x, rowvar=False)

    bs_ms = timeit(bs_call)
    np_ms = timeit(np_call)

    print(json.dumps({
        "bs_best_ms": min(bs_ms),
        "bs_median_ms": float(np.median(bs_ms)),
        "np_best_ms": min(np_ms),
        "extra_peak_mb": extra_peak / 2**20,
        "input_mb": n * p * x.dtype.itemsize / 2**20,
        "err_vs_ref": _reference_error(x, out),
        "digest": sorted(digests)[0],
        "run_to_run_identical": len(digests) == 1,
    }))


def _run_child(spec: dict, threads: str | None) -> dict:
    env = dict(os.environ)
    env.pop("RAYON_NUM_THREADS", None)
    if threads is not None:
        env["RAYON_NUM_THREADS"] = threads
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec)],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"child failed for {spec}:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--label", default="run", help="tag written to every CSV row (e.g. base / after)")
    ap.add_argument("--out", default=None, help="CSV path (default: print only)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="small subset, for CI smoke runs")
    args = ap.parse_args()

    if args.child:
        child(json.loads(args.child))
        return

    rows = []
    header = f"{'case':<13}{'n':>9}{'p':>6}  {'bs best':>9} {'bs med':>9} {'np.cov':>9} {'peak+MB':>8} {'err':>8}  {'rerun':>5} {'1thr':>5}"
    print(header)
    print("-" * len(header))
    for name, n, p, layout, data in CASES:
        if args.quick and name not in QUICK:
            continue
        spec = {"n": n, "p": p, "layout": layout, "data": data, "repeats": args.repeats, "hash_only": False}
        res = _run_child(spec, threads=None)
        one = _run_child({**spec, "hash_only": True}, threads="1")
        res["threads_identical"] = one["digest"] == res["digest"]
        res.update(label=args.label, case=name, n=n, p=p, layout=layout, data=data)
        rows.append(res)
        print(
            f"{name:<13}{n:>9}{p:>6}  {res['bs_best_ms']:>9.3f} {res['bs_median_ms']:>9.3f} "
            f"{res['np_best_ms']:>9.3f} {res['extra_peak_mb']:>8.1f} {res['err_vs_ref']:>8.1e}  "
            f"{'ok' if res['run_to_run_identical'] else 'DIFF':>5} {'ok' if res['threads_identical'] else 'DIFF':>5}"
        )

    if args.out:
        fields = ["label", "case", "n", "p", "layout", "data", "bs_best_ms", "bs_median_ms", "np_best_ms",
                  "extra_peak_mb", "input_mb", "err_vs_ref", "run_to_run_identical", "threads_identical", "digest"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
