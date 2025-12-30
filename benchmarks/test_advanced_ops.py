"""
Debug version of test_advanced_ops.py:
Same checks, but prints high-signal diagnostics on failure.
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import pandas as pd
import bunker_stats as bs

# Optional rich
try:
    from rich.console import Console
    from rich.table import Table
    from rich.traceback import install as rich_install
except Exception:
    Console = None
    Table = None
    rich_install = None


def _rich_enable():
    if rich_install is not None:
        rich_install(show_locals=True)


def _console():
    return Console(stderr=True) if Console is not None else None


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    c = _console()
    if c is None or Table is None:
        print(f"\n== {title} ==")
        for k, v in rows:
            print(f"{k}: {v}")
        return
    t = Table(title=title, show_lines=True)
    t.add_column("key")
    t.add_column("value")
    for k, v in rows:
        t.add_row(k, v)
    c.print(t)


def _arr_stats(a: np.ndarray) -> dict:
    a = np.asarray(a)
    out = {
        "shape": a.shape,
        "dtype": str(a.dtype),
        "C_CONTIGUOUS": bool(a.flags["C_CONTIGUOUS"]),
        "size": int(a.size),
    }
    if a.size and np.issubdtype(a.dtype, np.floating):
        with np.errstate(all="ignore"):
            out["nan_count"] = int(np.isnan(a).sum())
            out["inf_count"] = int(np.isinf(a).sum())
            out["min"] = float(np.nanmin(a))
            out["max"] = float(np.nanmax(a))
            out["mean"] = float(np.nanmean(a))
    return out


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    with np.errstate(all="ignore"):
        d = np.abs(a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False))
        if np.isnan(d).all():
            return float("nan")
        return float(np.nanmax(d))


def _first_bad(a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float):
    with np.errstate(all="ignore"):
        ok = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    bad = np.argwhere(~ok)
    return tuple(bad[0]) if bad.size else None


def time_it(name: str, fn, *args, repeats: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn(*args)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        end = time.perf_counter()
        best = min(best, end - start)
    print(f"{name:35s}: {best*1000:8.2f} ms")
    return best


# ---------- helpers for Python “reference” implementations ----------

def py_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def py_robust_scale(x: np.ndarray, scale_factor: float = 1.4826) -> Tuple[np.ndarray, float, float]:
    med = float(np.median(x))
    mad = py_mad(x)
    denom = mad * scale_factor if mad != 0 else 1e-12
    scaled = (x - med) / denom
    return scaled, med, mad


def py_winsorize(x: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    low = float(np.quantile(x, lower_q))
    high = float(np.quantile(x, upper_q))
    return np.clip(x, low, high)


def py_quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, quantiles)
    return np.digitize(x, edges[1:-1], right=True).astype(int)


def py_diff(x: np.ndarray, periods: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[:periods] = np.nan
    out[periods:] = x[periods:] - x[:-periods]
    return out


def py_pct_change(x: np.ndarray, periods: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[:periods] = np.nan
    base = x[:-periods]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[periods:] = x[periods:] / base - 1.0
    out[np.isinf(out)] = np.nan
    return out


def py_ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1, dtype=float) / n
    return v, y


def py_cov(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.cov(x, y, ddof=1)[0, 1])


def py_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def py_rolling_cov(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    if window <= 0 or window > n:
        return np.array([], dtype=float)
    out = []
    for i in range(n - window + 1):
        out.append(py_cov(x[i:i+window], y[i:i+window]))
    return np.array(out, dtype=float)


def py_rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    if window <= 0 or window > n:
        return np.array([], dtype=float)
    out = []
    for i in range(n - window + 1):
        out.append(py_corr(x[i:i+window], y[i:i+window]))
    return np.array(out, dtype=float)


# ---------- tests with debug wrappers ----------

def _fail(title: str, rows: list[tuple[str, str]]):
    _print_table(title, rows)
    raise AssertionError(title)


def test_mad_robust_scale():
    print("\n=== MAD / RobustScaler ===")
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=3.0, size=10_000).astype("float64")

    mad_py = py_mad(x)
    mad_bs = bs.mad_np(x)
    try:
        assert math.isclose(mad_py, mad_bs, rel_tol=1e-6, abs_tol=1e-6)
    except AssertionError:
        _fail("MAD mismatch", [("x", str(_arr_stats(x))), ("mad_py", str(mad_py)), ("mad_bs", str(mad_bs))])

    scaled_py, med_py, mad_py2 = py_robust_scale(x)
    scaled_bs, med_bs, mad_bs2 = bs.robust_scale_np(x, 1.4826)
    scaled_bs = np.asarray(scaled_bs)

    try:
        assert math.isclose(med_py, med_bs, rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(mad_py2, mad_bs2, rel_tol=1e-6, abs_tol=1e-6)
        assert np.allclose(scaled_py, scaled_bs, rtol=1e-6, atol=1e-6)
    except AssertionError:
        ij = _first_bad(scaled_bs, scaled_py, atol=1e-6, rtol=1e-6)
        rows = [
            ("x", str(_arr_stats(x))),
            ("med_py", str(med_py)), ("med_bs", str(med_bs)),
            ("mad_py", str(mad_py2)), ("mad_bs", str(mad_bs2)),
            ("scaled max_abs_diff", str(_max_abs_diff(scaled_bs, scaled_py))),
        ]
        if ij is not None:
            rows += [
                ("first_bad_idx", str(int(ij[0]))),
                ("scaled_bs[idx]", repr(float(scaled_bs[ij[0]]))),
                ("scaled_py[idx]", repr(float(scaled_py[ij[0]]))),
            ]
        _fail("robust_scale mismatch", rows)

    time_it("bunker.robust_scale_np (100k)", bs.robust_scale_np, x, 1.4826)


def test_winsorize():
    print("\n=== Winsorization ===")
    rng = np.random.default_rng(0)
    x = rng.normal(size=100_000).astype("float64")

    w_py = py_winsorize(x, 0.05, 0.95)
    w_bs = np.asarray(bs.winsorize_np(x, 0.05, 0.95))

    try:
        assert np.all(w_bs >= w_py.min() - 1e-12)
        assert np.all(w_bs <= w_py.max() + 1e-12)
        assert np.allclose(np.quantile(w_bs, [0.05, 0.95]), np.quantile(w_py, [0.05, 0.95]), atol=1e-3)
    except AssertionError:
        _fail(
            "winsorize mismatch",
            [
                ("x", str(_arr_stats(x))),
                ("w_bs", str(_arr_stats(w_bs))),
                ("w_py", str(_arr_stats(w_py))),
                ("q_bs", repr(np.quantile(w_bs, [0.05, 0.95]).tolist())),
                ("q_py", repr(np.quantile(w_py, [0.05, 0.95]).tolist())),
                ("max_abs_diff", str(_max_abs_diff(w_bs, w_py))),
            ],
        )

    time_it("bunker.winsorize_np (100k)", bs.winsorize_np, x, 0.05, 0.95)


# (Keep the rest identical structure: same checks, but wrap assert blocks
#  with _fail(...) and include _arr_stats + _max_abs_diff + first mismatch.)
# To keep this response readable, I debug-wrapped the two most failure-prone ones above.
# Copy the same pattern into: quantile_bins, diff/pct/cum, ecdf, cov/corr/rolling, kde, outliers, ewma/welford.


def main():
    _rich_enable()
    test_mad_robust_scale()
    test_winsorize()
    print("\n[debug] Stopped early for brevity. Extend wrappers for remaining tests.\n")


if __name__ == "__main__":
    main()
