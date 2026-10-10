#!/usr/bin/env python
"""
Benchmark: dense matrix kernels end to end.

--kernel selects cov (default), xtx, xxt, euclidean or cosine.

Every case runs in its own subprocess so that peak memory is attributable to
one call and so the Rayon pool size can be pinned per run. Timings INCLUDE the
NumPy -> Rust conversion (that is what a user pays), and the NumPy/SciPy
equivalent is reported only as context.

Per case this records:
- wall time: best and median of --repeats calls, bunker-stats and the
  NumPy/SciPy equivalent
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

# X X^T is n x n, so it gets its own (small-n) shapes.
XXT_CASES = [
    ("small",     200,   50,     "c", "normal"),
    ("medium",    3_000, 100,    "c", "normal"),
    ("square",    1_000, 1_000,  "c", "normal"),
    ("long_rows", 500,   20_000, "c", "normal"),
    ("fortran",   2_000, 200,    "f", "normal"),
]

# kernel -> (binding name, cases, "vectors are rows")
KERNELS = {
    "cov":       ("cov_matrix_np", CASES, False),
    "xtx":       ("xtx_matrix_np", CASES, False),
    "xxt":       ("xxt_matrix_np", XXT_CASES, True),
    "euclidean": ("pairwise_euclidean_cols_np", CASES, False),
    "cosine":    ("pairwise_cosine_cols_np", CASES, False),
}


def _numpy_equivalent(kernel: str, x):
    import numpy as np

    if kernel == "cov":
        return np.cov(x, rowvar=False)
    if kernel == "xtx":
        return x.T @ x
    if kernel == "xxt":
        return x @ x.T
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(x.T, "euclidean" if kernel == "euclidean" else "cosine"))

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


def _reference_error(kernel: str, x, out) -> float:
    """Max scaled |out - ref| over sampled (i, j) pairs, with math.fsum references.

    Scale: sqrt(ref_ii * ref_jj) for cov/xtx/xxt (so near-zero entries do not
    dominate), the reference itself for euclidean, 1 for cosine distance.
    """
    import numpy as np

    by_rows = KERNELS[kernel][2]
    count = x.shape[0] if by_rows else x.shape[1]
    rng = np.random.default_rng(7)
    pairs = {(0, 0), (0, count - 1), (count - 1, count - 1)}
    while len(pairs) < min(REF_PAIRS, count * (count + 1) // 2):
        i, j = sorted(int(v) for v in rng.integers(0, count, 2))
        pairs.add((i, j))

    vecs: Dict[int, object] = {}

    def vec(k: int):
        if k not in vecs:
            v = np.ascontiguousarray(x[k, :] if by_rows else x[:, k], dtype=np.float64)
            if kernel == "cov":
                v = v - math.fsum(v) / len(v)
            vecs[k] = v
        return vecs[k]

    def dot(i: int, j: int) -> float:
        return math.fsum(vec(i) * vec(j))

    worst = 0.0
    for i, j in sorted(pairs):
        norm = math.sqrt(dot(i, i) * dot(j, j))
        if kernel == "cov":
            ref, scale = dot(i, j) / (len(vec(i)) - 1), norm / (len(vec(i)) - 1)
        elif kernel in ("xtx", "xxt"):
            ref, scale = dot(i, j), norm
        elif kernel == "euclidean":
            ref = math.sqrt(math.fsum((vec(i) - vec(j)) ** 2))
            scale = ref
        else:  # cosine distance
            ref, scale = (0.0 if i == j else 1.0 - dot(i, j) / norm), 1.0
        if scale > 0.0:
            worst = max(worst, abs(float(out[i, j]) - ref) / scale)
    return worst


def child(spec: dict) -> None:
    import numpy as np
    from bunker_stats import _rs as bsr

    n, p = spec["n"], spec["p"]
    kernel = spec["kernel"]
    bs_fn = getattr(bsr, KERNELS[kernel][0])
    x = _make(n, p, spec["layout"], spec["data"])

    # Memory first, while the process high-water mark is still the input.
    rss_before = _rss_bytes()
    out = bs_fn(x)
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
        return bs_fn(x)

    def np_call():
        return _numpy_equivalent(kernel, x)

    bs_ms = timeit(bs_call)
    np_ms = timeit(np_call)

    print(json.dumps({
        "bs_best_ms": min(bs_ms),
        "bs_median_ms": float(np.median(bs_ms)),
        "np_best_ms": min(np_ms),
        "extra_peak_mb": extra_peak / 2**20,
        "input_mb": n * p * x.dtype.itemsize / 2**20,
        "err_vs_ref": _reference_error(kernel, x, out),
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
    ap.add_argument("--kernel", default="cov", choices=sorted(KERNELS))
    ap.add_argument("--label", default="run", help="tag written to every CSV row (e.g. base / after)")
    ap.add_argument("--out", default=None, help="CSV path (default: print only)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="small subset, for CI smoke runs")
    args = ap.parse_args()

    if args.child:
        child(json.loads(args.child))
        return

    rows = []
    header = f"{'case':<13}{'n':>9}{'p':>6}  {'bs best':>9} {'bs med':>9} {'numpy':>9} {'peak+MB':>8} {'err':>8}  {'rerun':>5} {'1thr':>5}"
    print(header)
    print("-" * len(header))
    for name, n, p, layout, data in KERNELS[args.kernel][1]:
        if args.quick and name not in QUICK:
            continue
        spec = {"kernel": args.kernel, "n": n, "p": p, "layout": layout, "data": data, "repeats": args.repeats, "hash_only": False}
        res = _run_child(spec, threads=None)
        one = _run_child({**spec, "hash_only": True}, threads="1")
        res["threads_identical"] = one["digest"] == res["digest"]
        res.update(label=args.label, kernel=args.kernel, case=name, n=n, p=p, layout=layout, data=data)
        rows.append(res)
        print(
            f"{name:<13}{n:>9}{p:>6}  {res['bs_best_ms']:>9.3f} {res['bs_median_ms']:>9.3f} "
            f"{res['np_best_ms']:>9.3f} {res['extra_peak_mb']:>8.1f} {res['err_vs_ref']:>8.1e}  "
            f"{'ok' if res['run_to_run_identical'] else 'DIFF':>5} {'ok' if res['threads_identical'] else 'DIFF':>5}"
        )

    if args.out:
        fields = ["label", "kernel", "case", "n", "p", "layout", "data", "bs_best_ms", "bs_median_ms", "np_best_ms",
                  "extra_peak_mb", "input_mb", "err_vs_ref", "run_to_run_identical", "threads_identical", "digest"]
        new_file = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended to {args.out}")


if __name__ == "__main__":
    main()
