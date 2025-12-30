import numpy as np
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


def _arr_stats(a: np.ndarray) -> dict:
    a = np.asarray(a)
    stats = {
        "shape": a.shape,
        "dtype": str(a.dtype),
        "C_CONTIGUOUS": bool(a.flags["C_CONTIGUOUS"]),
        "F_CONTIGUOUS": bool(a.flags["F_CONTIGUOUS"]),
        "size": int(a.size),
    }
    if np.issubdtype(a.dtype, np.floating) and a.size:
        with np.errstate(all="ignore"):
            stats["nan_count"] = int(np.isnan(a).sum())
            stats["inf_count"] = int(np.isinf(a).sum())
            stats["min"] = float(np.nanmin(a))
            stats["max"] = float(np.nanmax(a))
            stats["mean"] = float(np.nanmean(a))
    return stats


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


def _first_mismatch(a: np.ndarray, b: np.ndarray, *, rtol: float, atol: float):
    """
    Returns (row, col) of first mismatch, or None if all close.
    """
    with np.errstate(all="ignore"):
        ok = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    bad = np.argwhere(~ok)
    if bad.size == 0:
        return None
    return tuple(bad[0])


def test_rolling_mean_std_axis0_matches_numpy_debug():
    """
    Debug-friendly version of your original test.

    Same reference computation as:
    - m_ref[i] = window.mean(axis=0)
    - s_ref[i] = window.std(axis=0, ddof=1)

    On failure, prints enough context to reproduce the mismatch instantly.
    """
    _rich_enable()

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 7)).astype(np.float64)
    w = 10

    # bunker-stats output
    m, s = bs.rolling_mean_std_axis0_np(X, w)
    m = np.asarray(m)
    s = np.asarray(s)

    # numpy reference (same as your file)
    out_rows = X.shape[0] - w + 1
    m_ref = np.empty((out_rows, X.shape[1]), dtype=np.float64)
    s_ref = np.empty((out_rows, X.shape[1]), dtype=np.float64)
    for i in range(out_rows):
        window = X[i : i + w]
        m_ref[i] = window.mean(axis=0)
        s_ref[i] = window.std(axis=0, ddof=1)

    rtol = 1e-12
    atol = 1e-12

    # Shape sanity first
    if m.shape != m_ref.shape or s.shape != s_ref.shape:
        _print_table(
            "shape mismatch (axis0 rolling)",
            [
                ("X stats", str(_arr_stats(X))),
                ("window", str(w)),
                ("m.shape", str(m.shape)),
                ("m_ref.shape", str(m_ref.shape)),
                ("s.shape", str(s.shape)),
                ("s_ref.shape", str(s_ref.shape)),
            ],
        )
        raise AssertionError("shape mismatch")

    # Mean check with debug
    try:
        assert_allclose(m, m_ref, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError:
        ij = _first_mismatch(m, m_ref, rtol=rtol, atol=atol)
        rows = [
            ("X stats", str(_arr_stats(X))),
            ("m stats", str(_arr_stats(m))),
            ("m_ref stats", str(_arr_stats(m_ref))),
            ("window", str(w)),
            ("max_abs_diff(m)", str(_max_abs_diff(m, m_ref))),
        ]
        if ij is not None:
            i, j = ij
            win = X[i : i + w, j]
            rows += [
                ("first mismatch (row,col)", str((int(i), int(j)))),
                ("m[i,j]", repr(float(m[i, j]))),
                ("m_ref[i,j]", repr(float(m_ref[i, j]))),
                ("abs diff", repr(float(abs(m[i, j] - m_ref[i, j])))),
                ("X window col slice", repr(win.tolist())),
                ("window mean", repr(float(win.mean()))),
            ]
        _print_table("rolling_mean_std_axis0 mean mismatch", rows)
        raise

    # Std check with debug
    try:
        assert_allclose(s, s_ref, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError:
        ij = _first_mismatch(s, s_ref, rtol=rtol, atol=atol)
        rows = [
            ("X stats", str(_arr_stats(X))),
            ("s stats", str(_arr_stats(s))),
            ("s_ref stats", str(_arr_stats(s_ref))),
            ("window", str(w)),
            ("ddof", "1"),
            ("max_abs_diff(s)", str(_max_abs_diff(s, s_ref))),
        ]
        if ij is not None:
            i, j = ij
            win = X[i : i + w, j]
            rows += [
                ("first mismatch (row,col)", str((int(i), int(j)))),
                ("s[i,j]", repr(float(s[i, j]))),
                ("s_ref[i,j]", repr(float(s_ref[i, j]))),
                ("abs diff", repr(float(abs(s[i, j] - s_ref[i, j])))),
                ("X window col slice", repr(win.tolist())),
                ("window std(ddof=1)", repr(float(win.std(ddof=1)))),
            ]
        _print_table("rolling_mean_std_axis0 std mismatch", rows)
        raise
