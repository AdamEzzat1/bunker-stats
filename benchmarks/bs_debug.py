from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# rich is optional (dev-only)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.traceback import install as rich_install
except Exception:  # pragma: no cover
    Console = None
    Table = None
    Panel = None
    rich_install = None


def rich_enable() -> None:
    """Enable rich tracebacks if rich is installed."""
    if rich_install is not None:
        rich_install(show_locals=True)


def _console():
    return Console(stderr=True) if Console is not None else None


def debug_array(x: Any, name: str = "x") -> None:
    """Pretty print array diagnostics (shape/dtype/nans/min/max/etc)."""
    c = _console()
    if c is None or Table is None:
        return

    a = np.asarray(x)
    t = Table(title=f"Debug: {name}", show_lines=True)
    t.add_column("Property")
    t.add_column("Value")

    t.add_row("shape", str(a.shape))
    t.add_row("dtype", str(a.dtype))
    t.add_row("contiguous(C)", str(a.flags["C_CONTIGUOUS"]))
    t.add_row("contiguous(F)", str(a.flags["F_CONTIGUOUS"]))
    t.add_row("size", str(a.size))

    if a.size:
        if np.issubdtype(a.dtype, np.number):
            with np.errstate(all="ignore"):
                t.add_row("nan count", str(int(np.isnan(a).sum())) if a.dtype != np.bool_ else "n/a")
                t.add_row("inf count", str(int(np.isinf(a).sum())) if a.dtype != np.bool_ else "n/a")
                t.add_row("min", str(np.nanmin(a)) if a.dtype != np.bool_ else str(a.min()))
                t.add_row("max", str(np.nanmax(a)) if a.dtype != np.bool_ else str(a.max()))
                t.add_row("mean", str(np.nanmean(a)) if a.dtype != np.bool_ else "n/a")
        else:
            t.add_row("unique (<=10)", str(np.unique(a)[:10]))

    c.print(t)


def debug_pair(a: Any, b: Any, name_a="out", name_b="ref", max_rows: int = 8) -> None:
    """Show a quick diff summary (especially useful when a test fails)."""
    c = _console()
    if c is None or Panel is None:
        return

    aa = np.asarray(a)
    bb = np.asarray(b)
    msg = []
    msg.append(f"{name_a}.shape={aa.shape}, dtype={aa.dtype}, C={aa.flags['C_CONTIGUOUS']}")
    msg.append(f"{name_b}.shape={bb.shape}, dtype={bb.dtype}, C={bb.flags['C_CONTIGUOUS']}")

    if aa.shape == bb.shape and aa.size and np.issubdtype(aa.dtype, np.number) and np.issubdtype(bb.dtype, np.number):
        with np.errstate(all="ignore"):
            d = np.abs(aa.astype(np.float64, copy=False) - bb.astype(np.float64, copy=False))
            msg.append(f"max_abs_diff={np.nanmax(d)}")
            msg.append(f"mean_abs_diff={np.nanmean(d)}")
            msg.append(f"nan(out)={int(np.isnan(aa).sum())} nan(ref)={int(np.isnan(bb).sum())}")

    # sample
    if aa.ndim == 1 and bb.ndim == 1 and aa.size:
        msg.append(f"sample {name_a}[:{max_rows}]={aa[:max_rows]!r}")
        msg.append(f"sample {name_b}[:{max_rows}]={bb[:max_rows]!r}")

    c.print(Panel("\n".join(msg), title="bunker-stats debug diff"))


def max_abs_diff(a: Any, b: Any) -> float:
    """Robust max abs diff for numeric + bool."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    if a.dtype == np.bool_ or b.dtype == np.bool_:
        return float(np.mean(np.logical_xor(a, b)))
    with np.errstate(all="ignore"):
        d = np.abs(a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False))
        if np.isnan(d).all():
            return float("nan")
        return float(np.nanmax(d))


def assert_close(a: Any, b: Any, *, rtol=1e-7, atol=1e-9, name_a="out", name_b="ref") -> None:
    """
    Debug-friendly assertion:
    - tuples handled recursively
    - bool arrays use exact equality
    - numeric arrays use numpy.testing.assert_allclose for rich diffs
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        assert len(a) == len(b), f"Tuple length mismatch: {len(a)} vs {len(b)}"
        for i, (ai, bi) in enumerate(zip(a, b)):
            assert_close(ai, bi, rtol=rtol, atol=atol, name_a=f"{name_a}[{i}]", name_b=f"{name_b}[{i}]")
        return

    aa = np.asarray(a)
    bb = np.asarray(b)

    if aa.shape != bb.shape:
        debug_pair(aa, bb, name_a=name_a, name_b=name_b)
        raise AssertionError(f"shape mismatch: {aa.shape} vs {bb.shape}")

    if aa.dtype == np.bool_ or bb.dtype == np.bool_:
        try:
            assert_array_equal(aa, bb)
        except AssertionError:
            debug_pair(aa, bb, name_a=name_a, name_b=name_b)
            raise
        return

    try:
        assert_allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError:
        debug_pair(aa, bb, name_a=name_a, name_b=name_b)
        raise


@dataclass
class Timing:
    median_ms: float
    mean_ms: float
    std_ms: float
    cv: float


def timeit(fn: Callable[[], Any], warmup: int, repeats: int) -> Tuple[np.ndarray, Any]:
    out = None
    for _ in range(warmup):
        out = fn()
    times = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = time.perf_counter_ns()
        out = fn()
        t1 = time.perf_counter_ns()
        times[i] = (t1 - t0) / 1e6
    return times, out


def summarize_times(times_ms: np.ndarray) -> Timing:
    mu = float(np.mean(times_ms))
    sd = float(np.std(times_ms, ddof=1)) if times_ms.size > 1 else 0.0
    med = float(np.median(times_ms))
    cv = float(sd / mu) if mu > 0 and times_ms.size > 1 else 0.0
    return Timing(median_ms=med, mean_ms=mu, std_ms=sd, cv=cv)


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")
