import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import stats

import bunker_stats as bs

# Locked tolerances (v0.3 contract)
RTOL_STAT = 1e-12
ATOL_STAT = 1e-12
RTOL_P = 1e-10
ATOL_P = 1e-12


# -----------------------------
# Debug helpers (local to file)
# -----------------------------

def _arr_stats(x: np.ndarray) -> dict:
    x = np.asarray(x)
    out = {
        "shape": x.shape,
        "dtype": str(x.dtype),
        "C_CONTIGUOUS": bool(x.flags["C_CONTIGUOUS"]),
        "size": int(x.size),
    }
    if x.size and np.issubdtype(x.dtype, np.floating):
        with np.errstate(all="ignore"):
            out["nan_count"] = int(np.isnan(x).sum())
            out["inf_count"] = int(np.isinf(x).sum())
            out["min"] = float(np.nanmin(x))
            out["max"] = float(np.nanmax(x))
            out["mean"] = float(np.nanmean(x))
            out["std"] = float(np.nanstd(x, ddof=1)) if x.size > 1 else float("nan")
    return out


def _diff(a: float, b: float) -> dict:
    # report abs + rel difference (safe for near-zero)
    absd = float(abs(a - b))
    denom = max(float(abs(b)), 1e-300)
    reld = float(absd / denom)
    return {"abs": absd, "rel": reld}


def _print_debug(title: str, rows: list[tuple[str, str]]) -> None:
    # Optional rich if installed; fallback to plain prints.
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.traceback import install as rich_install
        rich_install(show_locals=True)
        c = Console(stderr=True)
        t = Table(title=title, show_lines=True)
        t.add_column("key")
        t.add_column("value")
        for k, v in rows:
            t.add_row(k, v)
        c.print(t)
    except Exception:
        print(f"\n== {title} ==")
        for k, v in rows:
            print(f"{k}: {v}")


def _assert_close_scalar(
    *,
    name: str,
    out_val: float,
    ref_val: float,
    rtol: float,
    atol: float,
    context_rows: list[tuple[str, str]],
) -> None:
    ok = np.isclose(out_val, ref_val, rtol=rtol, atol=atol)
    if not bool(ok):
        d = _diff(out_val, ref_val)
        rows = context_rows + [
            (f"{name}.out", repr(float(out_val))),
            (f"{name}.ref", repr(float(ref_val))),
            (f"{name}.diff_abs", repr(d["abs"])),
            (f"{name}.diff_rel", repr(d["rel"])),
            (f"{name}.rtol", repr(rtol)),
            (f"{name}.atol", repr(atol)),
        ]
        _print_debug("SciPy parity mismatch", rows)
        raise AssertionError(f"{name} mismatch: out={out_val} ref={ref_val} abs={d['abs']} rel={d['rel']}")


def _repro_snippet_1samp(seed: int, loc: float, scale: float, size: int, popmean: float) -> str:
    return (
        "import numpy as np\n"
        "from scipy import stats\n"
        "import bunker_stats as bs\n"
        f"rng = np.random.default_rng({seed})\n"
        f"x = rng.normal(loc={loc}, scale={scale}, size={size}).astype(np.float64)\n"
        f"popmean = {popmean}\n"
        "out = bs.t_test_1samp_np(x, popmean, 'two-sided')\n"
        "ref = stats.ttest_1samp(x, popmean, alternative='two-sided')\n"
        "print('out', out)\n"
        "print('ref', ref.statistic, ref.pvalue)\n"
    )


def _repro_snippet_2samp(seed: int, x_params: tuple[float, float, int], y_params: tuple[float, float, int], equal_var: bool) -> str:
    xloc, xscale, xn = x_params
    yloc, yscale, yn = y_params
    return (
        "import numpy as np\n"
        "from scipy import stats\n"
        "import bunker_stats as bs\n"
        f"rng = np.random.default_rng({seed})\n"
        f"x = rng.normal(loc={xloc}, scale={xscale}, size={xn}).astype(np.float64)\n"
        f"y = rng.normal(loc={yloc}, scale={yscale}, size={yn}).astype(np.float64)\n"
        f"out = bs.t_test_2samp_np(x, y, {bool(equal_var)}, 'two-sided')\n"
        f"ref = stats.ttest_ind(x, y, equal_var={bool(equal_var)}, alternative='two-sided')\n"
        "print('out', out)\n"
        "print('ref', ref.statistic, ref.pvalue)\n"
    )


# -----------------------------
# Tests (same intent as original)
# -----------------------------

def test_ttest_1samp_matches_scipy_debug():
    seed = 0
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=1.2, scale=2.5, size=500).astype(np.float64)
    popmean = 0.7

    out = bs.t_test_1samp_np(x, popmean, "two-sided")
    ref = stats.ttest_1samp(x, popmean, alternative="two-sided")

    ctx = [
        ("test", "ttest_1samp two-sided"),
        ("x", str(_arr_stats(x))),
        ("popmean", repr(popmean)),
        ("out", repr(out)),
        ("ref.statistic", repr(float(ref.statistic))),
        ("ref.pvalue", repr(float(ref.pvalue))),
        ("repro", _repro_snippet_1samp(seed, 1.2, 2.5, 500, popmean)),
    ]

    _assert_close_scalar(name="statistic", out_val=float(out["statistic"]), ref_val=float(ref.statistic),
                         rtol=RTOL_STAT, atol=ATOL_STAT, context_rows=ctx)
    _assert_close_scalar(name="pvalue", out_val=float(out["pvalue"]), ref_val=float(ref.pvalue),
                         rtol=RTOL_P, atol=ATOL_P, context_rows=ctx)


def test_ttest_ind_equal_var_matches_scipy_debug():
    seed = 1
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=400).astype(np.float64)
    y = rng.normal(loc=0.3, scale=1.0, size=350).astype(np.float64)

    out = bs.t_test_2samp_np(x, y, True, "two-sided")
    ref = stats.ttest_ind(x, y, equal_var=True, alternative="two-sided")

    ctx = [
        ("test", "ttest_ind pooled equal_var=True two-sided"),
        ("x", str(_arr_stats(x))),
        ("y", str(_arr_stats(y))),
        ("out", repr(out)),
        ("ref.statistic", repr(float(ref.statistic))),
        ("ref.pvalue", repr(float(ref.pvalue))),
        ("repro", _repro_snippet_2samp(seed, (0.0, 1.0, 400), (0.3, 1.0, 350), True)),
    ]

    _assert_close_scalar(name="statistic", out_val=float(out["statistic"]), ref_val=float(ref.statistic),
                         rtol=RTOL_STAT, atol=ATOL_STAT, context_rows=ctx)
    _assert_close_scalar(name="pvalue", out_val=float(out["pvalue"]), ref_val=float(ref.pvalue),
                         rtol=RTOL_P, atol=ATOL_P, context_rows=ctx)


def test_ttest_ind_welch_matches_scipy_debug():
    seed = 2
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=300).astype(np.float64)
    y = rng.normal(loc=0.1, scale=2.0, size=250).astype(np.float64)

    out = bs.t_test_2samp_np(x, y, False, "two-sided")
    ref = stats.ttest_ind(x, y, equal_var=False, alternative="two-sided")

    ctx = [
        ("test", "ttest_ind Welch equal_var=False two-sided"),
        ("x", str(_arr_stats(x))),
        ("y", str(_arr_stats(y))),
        ("out", repr(out)),
        ("ref.statistic", repr(float(ref.statistic))),
        ("ref.pvalue", repr(float(ref.pvalue))),
        ("repro", _repro_snippet_2samp(seed, (0.0, 1.0, 300), (0.1, 2.0, 250), False)),
    ]

    _assert_close_scalar(name="statistic", out_val=float(out["statistic"]), ref_val=float(ref.statistic),
                         rtol=RTOL_STAT, atol=ATOL_STAT, context_rows=ctx)
    _assert_close_scalar(name="pvalue", out_val=float(out["pvalue"]), ref_val=float(ref.pvalue),
                         rtol=RTOL_P, atol=ATOL_P, context_rows=ctx)


def test_chisquare_matches_scipy_debug():
    observed = np.array([10, 12, 9, 11, 8], dtype=np.float64)
    expected = np.array([10, 10, 10, 10, 10], dtype=np.float64)

    out = bs.chi2_gof_np(observed, expected)
    ref = stats.chisquare(observed, expected)

    ctx = [
        ("test", "chisquare GOF"),
        ("observed", str(_arr_stats(observed))),
        ("expected", str(_arr_stats(expected))),
        ("out", repr(out)),
        ("ref.statistic", repr(float(ref.statistic))),
        ("ref.pvalue", repr(float(ref.pvalue))),
        ("repro", "import numpy as np\nfrom scipy import stats\nimport bunker_stats as bs\n"
                 "observed=np.array([10,12,9,11,8],dtype=np.float64)\n"
                 "expected=np.array([10,10,10,10,10],dtype=np.float64)\n"
                 "out=bs.chi2_gof_np(observed,expected)\n"
                 "ref=stats.chisquare(observed,expected)\n"
                 "print(out)\nprint(ref.statistic, ref.pvalue)\n"),
    ]

    _assert_close_scalar(name="statistic", out_val=float(out["statistic"]), ref_val=float(ref.statistic),
                         rtol=RTOL_STAT, atol=ATOL_STAT, context_rows=ctx)
    _assert_close_scalar(name="pvalue", out_val=float(out["pvalue"]), ref_val=float(ref.pvalue),
                         rtol=RTOL_P, atol=ATOL_P, context_rows=ctx)


def test_chi2_contingency_matches_scipy_debug():
    table = np.array([[10, 20, 30], [6,  9,  17]], dtype=np.float64)

    out = bs.chi2_independence_np(table)
    ref = stats.chi2_contingency(table, correction=False)

    ctx = [
        ("test", "chi2_contingency independence correction=False"),
        ("table", str(_arr_stats(table))),
        ("out", repr(out)),
        ("ref.statistic", repr(float(ref.statistic))),
        ("ref.pvalue", repr(float(ref.pvalue))),
        ("ref.df", repr(int(ref.dof))),
        ("repro", "import numpy as np\nfrom scipy import stats\nimport bunker_stats as bs\n"
                 "table=np.array([[10,20,30],[6,9,17]],dtype=np.float64)\n"
                 "out=bs.chi2_independence_np(table)\n"
                 "ref=stats.chi2_contingency(table, correction=False)\n"
                 "print(out)\nprint(ref.statistic, ref.pvalue, ref.dof)\n"),
    ]

    _assert_close_scalar(name="statistic", out_val=float(out["statistic"]), ref_val=float(ref.statistic),
                         rtol=RTOL_STAT, atol=ATOL_STAT, context_rows=ctx)
    _assert_close_scalar(name="pvalue", out_val=float(out["pvalue"]), ref_val=float(ref.pvalue),
                         rtol=RTOL_P, atol=ATOL_P, context_rows=ctx)


@pytest.mark.xfail(reason="Pillar C (MWU) not implemented yet")
def test_mannwhitneyu_matches_scipy_debug():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200).astype(np.float64)
    y = rng.normal(loc=0.2, size=180).astype(np.float64)

    out = bs.mann_whitney_u_np(x, y, "two-sided")
    ref = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")

    # If it ever stops xfail, the debug path below helps immediately.
    ctx = [
        ("test", "mannwhitneyu two-sided asymptotic"),
        ("x", str(_arr_stats(x))),
        ("y", str(_arr_stats(y))),
        ("out", repr(out)),
        ("ref.statistic", repr(float(ref.statistic))),
        ("ref.pvalue", repr(float(ref.pvalue))),
    ]

    _assert_close_scalar(name="statistic", out_val=float(out["statistic"]), ref_val=float(ref.statistic),
                         rtol=RTOL_STAT, atol=ATOL_STAT, context_rows=ctx)
    _assert_close_scalar(name="pvalue", out_val=float(out["pvalue"]), ref_val=float(ref.pvalue),
                         rtol=RTOL_P, atol=ATOL_P, context_rows=ctx)


@pytest.mark.xfail(reason="Pillar C (KS) not implemented yet")
def test_ks_1samp_matches_scipy_norm_debug():
    rng = np.random.default_rng(4)
    x = rng.normal(size=300).astype(np.float64)

    out = bs.ks_1samp_np(x, "norm", [0.0, 1.0], "two-sided")
    ref = stats.kstest(x, "norm", args=(0.0, 1.0), alternative="two-sided")

    ctx = [
        ("test", "kstest norm two-sided"),
        ("x", str(_arr_stats(x))),
        ("out", repr(out)),
        ("ref.statistic", repr(float(ref.statistic))),
        ("ref.pvalue", repr(float(ref.pvalue))),
    ]

    # KS p-values can differ slightly; keep your looser tolerance here.
    _assert_close_scalar(name="pvalue", out_val=float(out["pvalue"]), ref_val=float(ref.pvalue),
                         rtol=1e-6, atol=1e-10, context_rows=ctx)
