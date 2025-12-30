import numpy as np
import pandas as pd
import bunker_stats as bs

from numpy.testing import assert_allclose

# Optional rich (dev-only)
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


def _console():
    return Console(stderr=True) if Console is not None else None


def _rich_enable():
    if rich_install is not None:
        rich_install(show_locals=True)


def _arr_stats(a: np.ndarray) -> dict:
    a = np.asarray(a)
    stats = {
        "shape": a.shape,
        "dtype": str(a.dtype),
        "size": int(a.size),
        "nan_count": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else None,
        "inf_count": int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "contig_C": bool(a.flags["C_CONTIGUOUS"]),
    }
    if a.size and np.issubdtype(a.dtype, np.number) and a.dtype != np.bool_:
        with np.errstate(all="ignore"):
            stats["min"] = float(np.nanmin(a))
            stats["max"] = float(np.nanmax(a))
            stats["mean"] = float(np.nanmean(a))
            stats["std"] = float(np.nanstd(a))
    return stats


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    c = _console()
    if c is None or Table is None:
        # fallback plain print
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


def _debug_cov_corr_context(x: np.ndarray, y: np.ndarray, cov_ref: float, corr_ref: float, cov_bs: float, corr_bs: float) -> None:
    # Pairwise deletion mask
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_f = x[mask]
    y_f = y[mask]

    _print_table(
        "NaN coverage debug (scalar)",
        [
            ("x stats", str(_arr_stats(x))),
            ("y stats", str(_arr_stats(y))),
            ("pairwise mask true", f"{int(mask.sum())} / {mask.size}"),
            ("x_f.size", str(x_f.size)),
            ("y_f.size", str(y_f.size)),
            ("cov_ref", str(cov_ref)),
            ("cov_bs", str(cov_bs)),
            ("corr_ref", str(corr_ref)),
            ("corr_bs", str(corr_bs)),
        ],
    )


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


def _find_mismatch_indices(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    """
    Returns indices where a and b are not close, excluding places where both are NaN.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(all="ignore"):
        close = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    bad = ~close & ~both_nan
    return np.flatnonzero(bad)


def rand_with_nans(n=1000, p_nan=0.1, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mask = rng.random(n) < p_nan
    x[mask] = np.nan
    return x


def test_cov_corr_nan_debug():
    """
    Debug-friendly version of test_cov_corr_nan.
    Same semantics as your original test, but prints context on failure.
    """
    _rich_enable()

    x = rand_with_nans(seed=123)
    y = rand_with_nans(seed=456)

    # reference (pairwise deletion)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_f = x[mask]
    y_f = y[mask]

    if x_f.size >= 2:
        cov_ref = np.cov(x_f, y_f, ddof=1)[0, 1]
        corr_ref = np.corrcoef(x_f, y_f)[0, 1]
    else:
        cov_ref = np.nan
        corr_ref = np.nan

    cov_bs = bs.cov_nan_np(x, y)
    corr_bs = bs.corr_nan_np(x, y)

    try:
        if np.isnan(cov_ref):
            assert np.isnan(cov_bs)
        else:
            assert_allclose(cov_bs, cov_ref, atol=1e-12, rtol=0.0, equal_nan=True)

        if np.isnan(corr_ref):
            assert np.isnan(corr_bs)
        else:
            assert_allclose(corr_bs, corr_ref, atol=1e-12, rtol=0.0, equal_nan=True)
    except AssertionError:
        _debug_cov_corr_context(x, y, cov_ref, corr_ref, cov_bs, corr_bs)
        raise


def test_rolling_cov_corr_nan_debug():
    """
    Debug-friendly version of test_rolling_cov_corr_nan.
    Prints window stats and mismatch indices on failure.
    """
    _rich_enable()

    x = rand_with_nans(seed=123)
    y = rand_with_nans(seed=456)
    window = 20

    s_x = pd.Series(x)
    s_y = pd.Series(y)

    cov_ref_full = s_x.rolling(window).cov(s_y, ddof=1).to_numpy()
    corr_ref_full = s_x.rolling(window).corr(s_y).to_numpy()

    cov_bs = np.asarray(bs.rolling_cov_nan_np(x, y, window), dtype=np.float64)
    corr_bs = np.asarray(bs.rolling_corr_nan_np(x, y, window), dtype=np.float64)

    # IMPORTANT:
    # Your Rust rolling_*_nan functions typically return the truncated length (n-window+1)
    # while pandas returns full-length with leading NaNs.
    # We'll align by taking the trailing slice of pandas outputs.
    cov_ref = cov_ref_full[window - 1 :]
    corr_ref = corr_ref_full[window - 1 :]

    # Basic shape sanity
    if cov_bs.shape != cov_ref.shape:
        _print_table(
            "rolling cov shape mismatch",
            [
                ("window", str(window)),
                ("x stats", str(_arr_stats(x))),
                ("y stats", str(_arr_stats(y))),
                ("cov_bs.shape", str(cov_bs.shape)),
                ("cov_ref.shape", str(cov_ref.shape)),
            ],
        )
        raise AssertionError(f"shape mismatch: cov_bs {cov_bs.shape} vs cov_ref {cov_ref.shape}")

    if corr_bs.shape != corr_ref.shape:
        _print_table(
            "rolling corr shape mismatch",
            [
                ("window", str(window)),
                ("x stats", str(_arr_stats(x))),
                ("y stats", str(_arr_stats(y))),
                ("corr_bs.shape", str(corr_bs.shape)),
                ("corr_ref.shape", str(corr_ref.shape)),
            ],
        )
        raise AssertionError(f"shape mismatch: corr_bs {corr_bs.shape} vs corr_ref {corr_ref.shape}")

    # Compare cov (ignore positions where both are NaN)
    atol_cov = 1e-9
    rtol_cov = 0.0
    try:
        mask_cov = ~(np.isnan(cov_ref) & np.isnan(cov_bs))
        assert_allclose(cov_bs[mask_cov], cov_ref[mask_cov], atol=atol_cov, rtol=rtol_cov, equal_nan=True)
    except AssertionError:
        bad = _find_mismatch_indices(cov_bs, cov_ref, atol=atol_cov, rtol=rtol_cov)
        _print_table(
            "rolling cov mismatch debug",
            [
                ("window", str(window)),
                ("len(out)", str(cov_bs.size)),
                ("nan(cov_bs)", str(int(np.isnan(cov_bs).sum()))),
                ("nan(cov_ref)", str(int(np.isnan(cov_ref).sum()))),
                ("max_abs_diff", str(_max_abs_diff(cov_bs, cov_ref))),
                ("bad_count", str(int(bad.size))),
                ("bad_indices[:20]", str(bad[:20].tolist())),
                ("cov_bs[bad[:5]]", str(cov_bs[bad[:5]].tolist() if bad.size else [])),
                ("cov_ref[bad[:5]]", str(cov_ref[bad[:5]].tolist() if bad.size else [])),
            ],
        )
        raise

    # Compare corr (ignore positions where both are NaN)
    atol_corr = 1e-9
    rtol_corr = 0.0
    try:
        mask_corr = ~(np.isnan(corr_ref) & np.isnan(corr_bs))
        assert_allclose(corr_bs[mask_corr], corr_ref[mask_corr], atol=atol_corr, rtol=rtol_corr, equal_nan=True)
    except AssertionError:
        bad = _find_mismatch_indices(corr_bs, corr_ref, atol=atol_corr, rtol=rtol_corr)
        _print_table(
            "rolling corr mismatch debug",
            [
                ("window", str(window)),
                ("len(out)", str(corr_bs.size)),
                ("nan(corr_bs)", str(int(np.isnan(corr_bs).sum()))),
                ("nan(corr_ref)", str(int(np.isnan(corr_ref).sum()))),
                ("max_abs_diff", str(_max_abs_diff(corr_bs, corr_ref))),
                ("bad_count", str(int(bad.size))),
                ("bad_indices[:20]", str(bad[:20].tolist())),
                ("corr_bs[bad[:5]]", str(corr_bs[bad[:5]].tolist() if bad.size else [])),
                ("corr_ref[bad[:5]]", str(corr_ref[bad[:5]].tolist() if bad.size else [])),
            ],
        )
        raise
