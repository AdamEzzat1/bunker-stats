import time
import numpy as np
import pandas as pd
import pytest
import bunker_stats as bs

from numpy.testing import assert_allclose


# Optional rich (dev-only)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.traceback import install as rich_install
except Exception:  # pragma: no cover
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
            out["mean_nan"] = float(np.nanmean(a))
    return out


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    with np.errstate(all="ignore"):
        d = np.abs(a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False))
        if np.isnan(d).all():
            return float("nan")
        return float(np.nanmax(d))


def _first_bad_idx(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> int | None:
    a = np.asarray(a)
    b = np.asarray(b)
    with np.errstate(all="ignore"):
        ok = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    bad = np.flatnonzero(~ok)
    return int(bad[0]) if bad.size else None


def bench_once(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def test_nan_behavior_debug():
    """
    Same intent as your original test_nan_behavior.py:
    - compare mean_nan / std_nan to NumPy nan versions
    - compare rolling_mean_nan / rolling_zscore_nan to pandas rolling equivalents
    But prints rich debug context on failures.
    """
    _rich_enable()

    rng = np.random.default_rng(0)
    x = rng.normal(size=100_000).astype("float64")

    # sprinkle NaNs
    mask = rng.random(size=x.size) < 0.1
    x[mask] = np.nan

    # -------- scalar nan stats --------
    out_mean, t_mean = bench_once(bs.mean_nan_np, x)
    ref_mean = float(np.nanmean(x))

    try:
        assert_allclose(out_mean, ref_mean, rtol=0.0, atol=1e-12, equal_nan=True)
    except AssertionError:
        _print_table(
            "mean_nan_np mismatch",
            [
                ("x", str(_arr_stats(x))),
                ("out", repr(float(out_mean))),
                ("ref", repr(float(ref_mean))),
                ("abs_diff", repr(float(abs(out_mean - ref_mean)))),
                ("time_ms", f"{t_mean:.3f}"),
            ],
        )
        raise

    out_std, t_std = bench_once(bs.std_nan_np, x)
    ref_std = float(np.nanstd(x, ddof=1))

    try:
        assert_allclose(out_std, ref_std, rtol=0.0, atol=1e-12, equal_nan=True)
    except AssertionError:
        _print_table(
            "std_nan_np mismatch",
            [
                ("x", str(_arr_stats(x))),
                ("out", repr(float(out_std))),
                ("ref", repr(float(ref_std))),
                ("abs_diff", repr(float(abs(out_std - ref_std)))),
                ("time_ms", f"{t_std:.3f}"),
            ],
        )
        raise

    # -------- rolling nan stats --------
    window = 50
    s = pd.Series(x)

    # reference: pandas rolling mean with min_periods=1 (as in your script)
    ref_rm_full = s.rolling(window, min_periods=1).mean().to_numpy()

    out_rm, t_rm = bench_once(bs.rolling_mean_nan_np, x, window)
    out_rm = np.asarray(out_rm, dtype=np.float64)

    # Your Rust rolling_nan typically returns full-length; if not, align here.
    if out_rm.shape != ref_rm_full.shape:
        # try to align by taking trailing slice (common pattern)
        if out_rm.size == x.size - window + 1 and ref_rm_full.size == x.size:
            ref_rm = ref_rm_full[window - 1 :]
        else:
            ref_rm = ref_rm_full
    else:
        ref_rm = ref_rm_full

    try:
        assert out_rm.shape == ref_rm.shape
        assert_allclose(out_rm, ref_rm, rtol=0.0, atol=1e-9, equal_nan=True)
    except AssertionError:
        idx = _first_bad_idx(out_rm, ref_rm, atol=1e-9, rtol=0.0)
        rows = [
            ("x", str(_arr_stats(x))),
            ("window", str(window)),
            ("out_rm.shape", str(out_rm.shape)),
            ("ref_rm.shape", str(ref_rm.shape)),
            ("max_abs_diff", str(_max_abs_diff(out_rm, ref_rm))),
            ("time_ms", f"{t_rm:.3f}"),
        ]
        if idx is not None:
            rows += [
                ("first_bad_idx", str(idx)),
                ("out[idx]", repr(float(out_rm[idx])) if np.isfinite(out_rm[idx]) else repr(out_rm[idx])),
                ("ref[idx]", repr(float(ref_rm[idx])) if np.isfinite(ref_rm[idx]) else repr(ref_rm[idx])),
            ]
        _print_table("rolling_mean_nan_np mismatch", rows)
        raise

    def pandas_rolling_zscore(arr, window):
        s = pd.Series(arr)
        roll = s.rolling(window, min_periods=1)
        m = roll.mean()
        sd = roll.std(ddof=1)
        return ((s - m) / sd).to_numpy()

    ref_rz_full = pandas_rolling_zscore(x, window)

    out_rz, t_rz = bench_once(bs.rolling_zscore_nan_np, x, window)
    out_rz = np.asarray(out_rz, dtype=np.float64)

    if out_rz.shape != ref_rz_full.shape:
        if out_rz.size == x.size - window + 1 and ref_rz_full.size == x.size:
            ref_rz = ref_rz_full[window - 1 :]
        else:
            ref_rz = ref_rz_full
    else:
        ref_rz = ref_rz_full

    try:
        assert out_rz.shape == ref_rz.shape
        assert_allclose(out_rz, ref_rz, rtol=0.0, atol=1e-8, equal_nan=True)
    except AssertionError:
        idx = _first_bad_idx(out_rz, ref_rz, atol=1e-8, rtol=0.0)
        rows = [
            ("x", str(_arr_stats(x))),
            ("window", str(window)),
            ("out_rz.shape", str(out_rz.shape)),
            ("ref_rz.shape", str(ref_rz.shape)),
            ("max_abs_diff", str(_max_abs_diff(out_rz, ref_rz))),
            ("time_ms", f"{t_rz:.3f}"),
        ]
        if idx is not None:
            rows += [
                ("first_bad_idx", str(idx)),
                ("out[idx]", repr(float(out_rz[idx])) if np.isfinite(out_rz[idx]) else repr(out_rz[idx])),
                ("ref[idx]", repr(float(ref_rz[idx])) if np.isfinite(ref_rz[idx]) else repr(ref_rz[idx])),
            ]
        _print_table("rolling_zscore_nan_np mismatch", rows)
        raise
