"""Notebook / reporting UX layer for bunker-stats (optional, pandas-backed).

This module is the ergonomic bridge between the Rust statistics kernels and
pandas/Jupyter workflows. Every helper here is a thin, validated wrapper: the
numbers always come from the Rust kernels exported by :mod:`bunker_stats`.

Install the optional extra::

    pip install "bunker-stats-rs[notebook]"

``import bunker_stats`` never imports pandas. Even ``import
bunker_stats.notebook`` succeeds without pandas installed -- pandas is imported
lazily on first call, and a missing install raises a clear :class:`ImportError`
with the command above.

Naming convention
-----------------
================  ==========================================================
Suffix            Returns
================  ==========================================================
``*_report``      a :class:`pandas.DataFrame` of results (one row per column)
``*_style``       a :class:`pandas.io.formats.style.Styler`
``*_columns``     a :class:`pandas.DataFrame` copy with added/transformed cols
================  ==========================================================

NaN and infinity policy
-----------------------
The Rust kernels are *strict*: a single NaN anywhere in the input poisons the
whole result. That is the right default for a numerical library and the wrong
default for exploratory data analysis, so this layer makes the choice explicit:

* **Reports** (``*_report``, :func:`robust_summary`, :func:`describe_fast`)
  drop non-finite values (NaN and +/-inf) **per column** before calling a
  kernel, and report how many were dropped in the ``n_missing`` column. A
  column with no finite values yields an all-NaN row rather than an exception.
* **Column transforms** (``*_columns``) fit the transform on the finite values
  only, then scatter the results back into their original positions. Non-finite
  inputs stay non-finite in the output, so row alignment is always preserved.
* **Stylers** never raise on NaN; NaN cells simply receive no background color.
* :func:`missingness_report` is the exception -- it exists to *count* the
  non-finite values, and separates NaN from +/-inf.

Correlation is the one place where dropping is *pairwise*: each pair of columns
uses the rows where both are finite.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import bunker_stats as _bs

__all__ = [
    # --- reports (-> DataFrame) --------------------------------------------
    "robust_summary",
    "describe_fast",
    "outlier_report",
    "normality_report",
    "correlation_report",
    "missingness_report",
    "rolling_report",
    "bootstrap_ci_report",
    # --- transforms (-> DataFrame) -----------------------------------------
    "scale_columns",
    "winsorize_columns",
    "robust_scale_column",
    # --- stylers (-> Styler) -----------------------------------------------
    "outlier_style",
    "corr_heatmap",
    "style_significance",
    "style_effect_size",
    "demean_style",
    "zscore_style",
    "iqr_outlier_style",
]

_INSTALL_HINT = (
    'pandas is required for the bunker_stats notebook layer. '
    'Install it with: pip install "bunker-stats-rs[notebook]"'
)

_OUTLIER_METHODS = ("iqr", "zscore", "robust_zscore")
_SCALE_METHODS = ("robust", "zscore", "minmax")
_CORR_METHODS = ("pearson", "spearman")
_ROLLING_STATS = ("mean", "std", "var", "count", "min", "max")

# Consistent with bunker_stats.mad_std: MAD * 1.4826 estimates sigma for a
# normal distribution.
_MAD_TO_SIGMA = 1.4826


# ======================================================================
# Lazy pandas access + shared validation
# ======================================================================

def _pd():
    """Import pandas on demand, with an actionable error if it is missing."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_INSTALL_HINT) from exc
    return pd


def _check_frame(df: Any) -> None:
    pd = _pd()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")


def _resolve_columns(
    df,
    columns: Sequence[Any] | None,
    *,
    numeric_only: bool = True,
) -> list:
    """Validate/resolve a column selection.

    ``columns=None`` selects every numeric column. An explicit selection is
    checked for existence first (KeyError) and dtype second (TypeError), so the
    user always sees the most actionable error.
    """
    _check_frame(df)
    pd = _pd()

    if columns is None:
        if not numeric_only:
            return list(df.columns)
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            raise ValueError(
                "DataFrame has no numeric columns; pass `columns=[...]` explicitly."
            )
        return cols

    if isinstance(columns, str) or not isinstance(columns, Iterable):
        raise TypeError(
            "`columns` must be None or a sequence of column labels "
            f"(got {type(columns).__name__}); use `columns=['name']` for one column."
        )

    cols = list(columns)
    if not cols:
        raise ValueError("`columns` is empty; pass None to select all numeric columns.")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) not in DataFrame: {missing}. "
            f"Available: {list(df.columns)}"
        )

    duplicates = [c for c in dict.fromkeys(cols) if cols.count(c) > 1]
    if duplicates:
        raise ValueError(f"`columns` contains duplicates: {duplicates}")

    if numeric_only:
        bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise TypeError(
                f"Column(s) are not numeric: {bad}. "
                "Only numeric columns can be summarized; drop them or cast first."
            )
    return cols


def _as_float(series) -> np.ndarray:
    """1-D float64 view of a numeric Series (nullable dtypes -> NaN)."""
    return np.ascontiguousarray(series.astype("float64").to_numpy(), dtype=np.float64)


def _finite(values: np.ndarray) -> np.ndarray:
    """Contiguous float64 array of only the finite entries."""
    return np.ascontiguousarray(values[np.isfinite(values)], dtype=np.float64)


def _safe(fn: Callable[..., Any], *args, default: float = np.nan) -> Any:
    """Call a kernel, degrading to `default` when the sample is too small.

    Kernels signal "undefined for this n" either by returning NaN (most of
    them) or by raising ValueError (e.g. ``jarque_bera`` below n=4). In a
    per-column report a short column should produce a NaN cell rather than
    abort the whole table.

    Only ValueError/ArithmeticError are caught. A TypeError or AttributeError
    means *this module* called the kernel wrongly, and must not be silently
    rendered as NaN.
    """
    try:
        return fn(*args)
    except (ValueError, ArithmeticError):
        return default


def _fit_scatter(values: np.ndarray, fn: Callable[[np.ndarray], Any]) -> np.ndarray:
    """Run a strict kernel on the finite subset and scatter results back.

    This is what lets `scale_columns`/`winsorize_columns` accept NaN without
    either poisoning the column or silently shifting rows: positions are
    preserved exactly, and non-finite inputs map to NaN outputs.
    """
    mask = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if mask.any():
        result = fn(np.ascontiguousarray(values[mask], dtype=np.float64))
        if isinstance(result, tuple):  # kernels that also return fitted params
            result = result[0]
        out[mask] = np.asarray(result, dtype=np.float64)
    return out


def _check_choice(value: str, valid: Sequence[str], name: str) -> str:
    if value not in valid:
        raise ValueError(f"{name} must be one of {list(valid)}, got {value!r}")
    return value


# ======================================================================
# Reports  ->  DataFrame
# ======================================================================

def robust_summary(df, columns: Sequence[Any] | None = None):
    """Robust + classical descriptive statistics, one row per column.

    Columns of the result: ``n`` (finite count), ``n_missing`` (non-finite
    count), ``mean``, ``std``, ``min``, ``median``, ``max``, ``mad``,
    ``mad_std``, ``iqr``, ``qn_scale``, ``trimmed_mean`` (10% each tail),
    ``skew``, ``kurtosis`` (excess/Fisher).

    Every statistic is computed by a Rust kernel on the finite values only.
    Statistics undefined for the sample size (e.g. ``kurtosis`` with n < 4)
    come back as NaN.

    Examples
    --------
    >>> robust_summary(df)                       # doctest: +SKIP
    >>> robust_summary(df, columns=["price"])    # doctest: +SKIP
    """
    pd = _pd()
    cols = _resolve_columns(df, columns)

    rows = []
    for col in cols:
        x = _finite(_as_float(df[col]))
        n = int(x.size)
        n_missing = int(len(df[col])) - n
        if n == 0:
            rows.append({"n": 0, "n_missing": n_missing})
            continue
        rows.append(
            {
                "n": n,
                "n_missing": n_missing,
                "mean": _safe(_bs.mean, x),
                "std": _safe(_bs.std, x),
                "min": float(np.min(x)),
                "median": _safe(_bs.median, x),
                "max": float(np.max(x)),
                "mad": _safe(_bs.mad, x),
                "mad_std": _safe(_bs.mad_std, x),
                "iqr": _safe(_bs.iqr, x),
                "qn_scale": _safe(_bs.qn_scale, x),
                "trimmed_mean": _safe(_bs.trimmed_mean, x, 0.1),
                "skew": _safe(_bs.skew, x),
                "kurtosis": _safe(_bs.kurtosis, x),
            }
        )

    order = [
        "n", "n_missing", "mean", "std", "min", "median", "max",
        "mad", "mad_std", "iqr", "qn_scale", "trimmed_mean", "skew", "kurtosis",
    ]
    out = pd.DataFrame(rows, index=pd.Index(cols, name="column"))
    return out.reindex(columns=order)


def describe_fast(df, columns: Sequence[Any] | None = None, *, robust: bool = True):
    """A faster, richer ``df.describe().T`` backed by the Rust kernels.

    Always returns ``n``, ``n_missing``, ``mean``, ``std``, ``min``, ``25%``,
    ``50%``, ``75%``, ``max``. With ``robust=True`` (the default) it also adds
    ``mad``, ``iqr``, ``qn_scale``, ``trimmed_mean``, ``skew`` and ``kurtosis``.

    Unlike ``df.describe()``, non-finite values are dropped per column and
    counted in ``n_missing`` instead of being silently ignored.
    """
    pd = _pd()
    cols = _resolve_columns(df, columns)

    rows = []
    for col in cols:
        x = _finite(_as_float(df[col]))
        n = int(x.size)
        row: dict[str, Any] = {"n": n, "n_missing": int(len(df[col])) - n}
        if n:
            row.update(
                {
                    "mean": _safe(_bs.mean, x),
                    "std": _safe(_bs.std, x),
                    "min": float(np.min(x)),
                    "25%": _safe(_bs.percentile, x, 25.0),
                    "50%": _safe(_bs.percentile, x, 50.0),
                    "75%": _safe(_bs.percentile, x, 75.0),
                    "max": float(np.max(x)),
                }
            )
            if robust:
                row.update(
                    {
                        "mad": _safe(_bs.mad, x),
                        "iqr": _safe(_bs.iqr, x),
                        "qn_scale": _safe(_bs.qn_scale, x),
                        "trimmed_mean": _safe(_bs.trimmed_mean, x, 0.1),
                        "skew": _safe(_bs.skew, x),
                        "kurtosis": _safe(_bs.kurtosis, x),
                    }
                )
        rows.append(row)

    order = ["n", "n_missing", "mean", "std", "min", "25%", "50%", "75%", "max"]
    if robust:
        order += ["mad", "iqr", "qn_scale", "trimmed_mean", "skew", "kurtosis"]
    out = pd.DataFrame(rows, index=pd.Index(cols, name="column"))
    return out.reindex(columns=order)


def _outlier_bounds(
    x: np.ndarray, method: str, k: float, z_threshold: float
) -> tuple[float, float]:
    """Lower/upper cutoffs for one finite sample under the chosen method."""
    if x.size == 0:
        return (np.nan, np.nan)

    if method == "iqr":
        q1 = _safe(_bs.percentile, x, 25.0)
        q3 = _safe(_bs.percentile, x, 75.0)
        width = q3 - q1
        return (q1 - k * width, q3 + k * width)

    if method == "zscore":
        center = _safe(_bs.mean, x)
        spread = _safe(_bs.std, x)
        return (center - z_threshold * spread, center + z_threshold * spread)

    # robust_zscore: median +/- z * (1.4826 * MAD) -- the breakdown-resistant
    # analogue of the mean/std rule, so a few extreme points cannot inflate the
    # cutoffs enough to hide themselves.
    center = _safe(_bs.median, x)
    spread = _safe(_bs.mad, x) * _MAD_TO_SIGMA
    return (center - z_threshold * spread, center + z_threshold * spread)


def _outlier_mask(
    values: np.ndarray, method: str, k: float, z_threshold: float
) -> tuple[np.ndarray, float, float]:
    """Boolean outlier mask aligned to `values`, plus the bounds used."""
    finite = np.isfinite(values)
    lower, upper = _outlier_bounds(_finite(values), method, k, z_threshold)
    mask = np.zeros(values.shape, dtype=bool)
    if np.isfinite(lower) and np.isfinite(upper):
        with np.errstate(invalid="ignore"):
            mask = finite & ((values < lower) | (values > upper))
    return mask, lower, upper


def outlier_report(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0,
):
    """Per-column outlier counts and cutoffs.

    Parameters
    ----------
    method : {"iqr", "zscore", "robust_zscore"}
        ``iqr`` flags points outside ``Q1 - k*IQR`` / ``Q3 + k*IQR``.
        ``zscore`` uses ``mean +/- z_threshold*std``.
        ``robust_zscore`` uses ``median +/- z_threshold*1.4826*MAD``.
    k : float
        Multiplier for the IQR fence (ignored by the z-score methods).
    z_threshold : float
        Multiplier for the z-score fences (ignored by ``iqr``).

    Returns
    -------
    pandas.DataFrame
        Indexed by column, with ``method``, ``n``, ``n_missing``,
        ``n_outliers``, ``pct_outliers``, ``lower_bound``, ``upper_bound``,
        ``min``, ``max``.
    """
    pd = _pd()
    _check_choice(method, _OUTLIER_METHODS, "method")
    cols = _resolve_columns(df, columns)
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be > 0, got {z_threshold}")

    rows = []
    for col in cols:
        values = _as_float(df[col])
        mask, lower, upper = _outlier_mask(values, method, k, z_threshold)
        finite = _finite(values)
        n = int(finite.size)
        rows.append(
            {
                "method": method,
                "n": n,
                "n_missing": int(values.size) - n,
                "n_outliers": int(mask.sum()),
                "pct_outliers": (100.0 * mask.sum() / n) if n else np.nan,
                "lower_bound": lower,
                "upper_bound": upper,
                "min": float(np.min(finite)) if n else np.nan,
                "max": float(np.max(finite)) if n else np.nan,
            }
        )

    return pd.DataFrame(rows, index=pd.Index(cols, name="column"))


def normality_report(df, columns: Sequence[Any] | None = None, *, alpha: float = 0.05):
    """Jarque-Bera and Anderson-Darling normality diagnostics per column.

    Returns ``n``, ``skewness``, ``kurtosis``, ``jb_statistic``, ``jb_pvalue``,
    ``ad_statistic``, ``normal`` and ``conclusion``.

    .. note::
       ``normal`` / ``conclusion`` are decided by the **Jarque-Bera p-value**
       only. The Rust ``anderson_darling`` kernel returns the A-squared
       statistic without a p-value (its critical values need table
       interpolation), so it is reported for reference but not used in the
       verdict. As a rule of thumb A* > 0.787 rejects normality at alpha=0.05.
    """
    pd = _pd()
    cols = _resolve_columns(df, columns)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rows = []
    for col in cols:
        x = _finite(_as_float(df[col]))
        n = int(x.size)
        row: dict[str, Any] = {"n": n}
        jb = _safe(_bs.jarque_bera, x, default={}) or {}
        ad = _safe(_bs.anderson_darling, x, default={}) or {}
        row["skewness"] = jb.get("skewness", np.nan)
        row["kurtosis"] = jb.get("kurtosis", np.nan)
        row["jb_statistic"] = jb.get("statistic", np.nan)
        row["jb_pvalue"] = jb.get("pvalue", np.nan)
        row["ad_statistic"] = ad.get("statistic", np.nan)

        p = row["jb_pvalue"]
        if p is None or not np.isfinite(p):
            row["normal"] = None
            row["conclusion"] = "inconclusive (sample too small)"
        elif p < alpha:
            row["normal"] = False
            row["conclusion"] = f"reject normality (p < {alpha})"
        else:
            row["normal"] = True
            row["conclusion"] = f"cannot reject normality (p >= {alpha})"
        rows.append(row)

    order = [
        "n", "skewness", "kurtosis", "jb_statistic", "jb_pvalue",
        "ad_statistic", "normal", "conclusion",
    ]
    out = pd.DataFrame(rows, index=pd.Index(cols, name="column"))
    return out.reindex(columns=order)


def correlation_report(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "pearson",
    pvalues: bool = False,
):
    """Correlation between numeric columns, using pairwise-complete rows.

    Parameters
    ----------
    method : {"pearson", "spearman"}
    pvalues : bool
        ``False`` (default) returns a square correlation matrix DataFrame.
        ``True`` returns a long-form DataFrame with one row per unique column
        pair: ``x``, ``y``, ``n``, ``correlation``, ``statistic``, ``pvalue``.

    Both shapes are DataFrames, so the ``*_report -> DataFrame`` convention
    holds either way. For a colored matrix use :func:`corr_heatmap`.
    """
    pd = _pd()
    _check_choice(method, _CORR_METHODS, "method")
    cols = _resolve_columns(df, columns)
    if len(cols) < 2:
        raise ValueError(
            f"Correlation needs at least 2 columns, got {len(cols)}: {cols}"
        )

    test = _bs.pearson_corr_test if method == "pearson" else _bs.spearman_corr_test

    if not pvalues:
        mat = np.full((len(cols), len(cols)), np.nan)
        np.fill_diagonal(mat, 1.0)
        for i in range(len(cols)):
            xi = _as_float(df[cols[i]])
            for j in range(i + 1, len(cols)):
                xj = _as_float(df[cols[j]])
                r = _pair_corr(test, xi, xj)[0]
                mat[i, j] = mat[j, i] = r
        return pd.DataFrame(mat, index=pd.Index(cols, name="column"), columns=cols)

    rows = []
    for i in range(len(cols)):
        xi = _as_float(df[cols[i]])
        for j in range(i + 1, len(cols)):
            xj = _as_float(df[cols[j]])
            r, stat, p, n = _pair_corr(test, xi, xj)
            rows.append(
                {
                    "x": cols[i], "y": cols[j], "n": n,
                    "correlation": r, "statistic": stat, "pvalue": p,
                }
            )
    return pd.DataFrame(rows, columns=["x", "y", "n", "correlation", "statistic", "pvalue"])


def _pair_corr(test, xi: np.ndarray, xj: np.ndarray):
    """Run a correlation test on the rows where both inputs are finite."""
    both = np.isfinite(xi) & np.isfinite(xj)
    n = int(both.sum())
    if n < 3:
        return (np.nan, np.nan, np.nan, n)
    a = np.ascontiguousarray(xi[both], dtype=np.float64)
    b = np.ascontiguousarray(xj[both], dtype=np.float64)
    res = _safe(test, a, b, default={}) or {}
    return (
        res.get("correlation", np.nan),
        res.get("statistic", np.nan),
        res.get("pvalue", np.nan),
        n,
    )


def missingness_report(df):
    """Missing / non-finite audit for **every** column, numeric or not.

    Returns ``dtype``, ``n_rows``, ``n_missing`` (null per pandas),
    ``pct_missing``, ``n_finite`` and ``n_infinite``. The finite/infinite
    counts are NaN for non-numeric columns, where they are not meaningful.

    This is the one helper that counts rather than drops non-finite values.
    """
    pd = _pd()
    _check_frame(df)

    n_rows = int(len(df))
    rows = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        row = {
            "dtype": str(series.dtype),
            "n_rows": n_rows,
            "n_missing": n_missing,
            "pct_missing": (100.0 * n_missing / n_rows) if n_rows else np.nan,
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = _as_float(series)
            row["n_finite"] = int(np.isfinite(values).sum())
            row["n_infinite"] = int(np.isinf(values).sum())
        else:
            row["n_finite"] = np.nan
            row["n_infinite"] = np.nan
        rows.append(row)

    return pd.DataFrame(
        rows,
        index=pd.Index(list(df.columns), name="column"),
        columns=["dtype", "n_rows", "n_missing", "pct_missing", "n_finite", "n_infinite"],
    )


def rolling_report(
    df,
    column: Any,
    window: int,
    *,
    stats: Sequence[str] = ("mean", "std", "min", "max"),
    min_periods: int | None = None,
    alignment: str = "trailing",
    nan_policy: str = "propagate",
):
    """Rolling-window features for one column, via the fused Rust kernel.

    Uses :class:`bunker_stats.Rolling`, which computes all requested statistics
    in a single pass instead of one pass per statistic.

    Parameters
    ----------
    stats : sequence of {"mean", "std", "var", "count", "min", "max"}
    min_periods : int or None
        Minimum valid observations per window. Defaults to ``window``.
    nan_policy : {"propagate", "ignore", "require_min_periods"}
        ``"propagate"`` (the default, matching the kernel) makes any window
        containing a NaN produce NaN.

        .. warning::
           ``nan_policy="ignore"`` has **no effect on its own**: with the
           default ``min_periods == window`` a window holding a NaN still
           fails the count check. To actually skip NaNs you must also pass
           ``min_periods`` below ``window``, e.g.
           ``rolling_report(df, "x", 5, min_periods=3, nan_policy="ignore")``.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``df``, with one ``{column}_roll{window}_{stat}`` column
        per requested statistic. If the kernel returns the shortened
        ``n - window + 1`` form, results are right-aligned and left-padded with
        NaN so the index always lines up with the input.
    """
    pd = _pd()
    _resolve_columns(df, [column])

    if not isinstance(window, (int, np.integer)) or isinstance(window, bool):
        raise TypeError(f"window must be an int, got {type(window).__name__}")
    window = int(window)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > len(df):
        raise ValueError(f"window ({window}) exceeds the number of rows ({len(df)})")

    stats = tuple(stats)
    if not stats:
        raise ValueError("`stats` is empty; request at least one statistic.")
    bad = [s for s in stats if s not in _ROLLING_STATS]
    if bad:
        raise ValueError(f"Unsupported stat(s) {bad}; valid: {list(_ROLLING_STATS)}")

    values = _as_float(df[column])
    roller = _bs.Rolling(
        values,
        window=window,
        min_periods=min_periods,
        alignment=alignment,
        nan_policy=nan_policy,
    )
    computed = roller.aggregate(*stats)

    n = len(df)
    out = {}
    for stat in stats:
        arr = np.asarray(computed[stat], dtype=np.float64)
        if arr.size != n:
            padded = np.full(n, np.nan, dtype=np.float64)
            padded[n - arr.size:] = arr
            arr = padded
        out[f"{column}_roll{window}_{stat}"] = arr

    return pd.DataFrame(out, index=df.index)


def bootstrap_ci_report(
    df,
    columns: Sequence[Any] | None = None,
    *,
    stat: str = "mean",
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: int | None = None,
):
    """Bootstrap point estimate and confidence interval per column.

    Wraps :class:`bunker_stats.BootstrapConfig` (Rust resampling kernels).
    Non-finite values are dropped per column before resampling.

    Parameters
    ----------
    stat : {"mean", "median", "std"}
    random_state : int or None
        Pass an int for reproducible intervals. ``None`` uses the kernel's
        deterministic default seed, so repeated calls still agree.

    Returns
    -------
    pandas.DataFrame
        Indexed by column: ``stat``, ``n``, ``n_missing``, ``estimate``,
        ``ci_lower``, ``ci_upper``, ``conf``.
    """
    pd = _pd()
    cols = _resolve_columns(df, columns)

    config = _bs.BootstrapConfig(
        n_resamples=n_resamples,
        conf=conf,
        stat=stat,
        random_state=random_state,
        nan_policy="omit",
    )

    rows = []
    for col in cols:
        values = _as_float(df[col])
        x = _finite(values)
        n = int(x.size)
        row: dict[str, Any] = {
            "stat": stat,
            "n": n,
            "n_missing": int(values.size) - n,
            "conf": conf,
        }
        if n < 2:
            row.update({"estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
        else:
            estimate, lower, upper = config.run(x)
            row.update({"estimate": estimate, "ci_lower": lower, "ci_upper": upper})
        rows.append(row)

    order = ["stat", "n", "n_missing", "estimate", "ci_lower", "ci_upper", "conf"]
    out = pd.DataFrame(rows, index=pd.Index(cols, name="column"))
    return out.reindex(columns=order)


# ======================================================================
# Column transforms  ->  DataFrame
# ======================================================================

def scale_columns(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "robust",
    suffix: str | None = None,
    scale_factor: float = _MAD_TO_SIGMA,
    replace: bool = False,
):
    """Batch-scale numeric columns with the Rust scaling kernels.

    Parameters
    ----------
    method : {"robust", "zscore", "minmax"}
        ``robust`` -> ``(x - median) / (scale_factor * MAD)``;
        ``zscore`` -> ``(x - mean) / std`` (ddof=1);
        ``minmax`` -> ``(x - min) / (max - min)``.
    suffix : str or None
        Appended to the source name for the new column. Defaults to
        ``"_" + method``. Ignored when ``replace=True``.
    replace : bool
        Overwrite the source columns in place of adding new ones.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df``. Scaling parameters are fit on the finite values only;
        non-finite inputs stay NaN in the output and row order is preserved.
    """
    _check_choice(method, _SCALE_METHODS, "method")
    cols = _resolve_columns(df, columns)
    suffix = f"_{method}" if suffix is None else suffix
    if not replace and not suffix:
        raise ValueError("`suffix` cannot be empty unless replace=True")

    if method == "robust":
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError(f"scale_factor must be a positive float, got {scale_factor}")
        kernel = lambda x: _bs.robust_scale(x, scale_factor)  # noqa: E731
    elif method == "zscore":
        kernel = _bs.zscore
    else:
        kernel = _bs.minmax_scale

    out = df.copy()
    for col in cols:
        scaled = _fit_scatter(_as_float(df[col]), kernel)
        out[col if replace else f"{col}{suffix}"] = scaled
    return out


def winsorize_columns(
    df,
    columns: Sequence[Any] | None = None,
    *,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    suffix: str = "_winsor",
    replace: bool = False,
):
    """Batch-winsorize numeric columns (clip tails at the given quantiles).

    ``lower_q``/``upper_q`` are quantile **fractions** in [0, 1], matching
    :func:`bunker_stats.winsorize` rather than percentile units.

    Quantiles are computed on the finite values only; non-finite inputs stay
    NaN in the output and row order is preserved.
    """
    cols = _resolve_columns(df, columns)
    for name, q in (("lower_q", lower_q), ("upper_q", upper_q)):
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {q}")
    if lower_q >= upper_q:
        raise ValueError(f"lower_q ({lower_q}) must be < upper_q ({upper_q})")
    if not replace and not suffix:
        raise ValueError("`suffix` cannot be empty unless replace=True")

    kernel = lambda x: _bs.winsorize(x, lower_q=lower_q, upper_q=upper_q)  # noqa: E731

    out = df.copy()
    for col in cols:
        clipped = _fit_scatter(_as_float(df[col]), kernel)
        out[col if replace else f"{col}{suffix}"] = clipped
    return out


def robust_scale_column(
    df,
    column: Any,
    *,
    scale_factor: float = _MAD_TO_SIGMA,
    add_suffix: str = "_robust",
):
    """Single-column robust scaling. Kept for backwards compatibility.

    Prefer :func:`scale_columns` for new code -- it handles many columns, other
    methods, and NaN-preserving scatter-back in one call.
    """
    return scale_columns(
        df,
        [column],
        method="robust",
        suffix=add_suffix,
        scale_factor=scale_factor,
    )


# ======================================================================
# Stylers  ->  Styler
# ======================================================================

def _blank_css(df):
    """An all-empty CSS frame shaped like `df` -- the base for axis=None styling."""
    pd = _pd()
    return pd.DataFrame("", index=df.index, columns=df.columns)


def outlier_style(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0,
    outlier_color: str = "#ff8a80",
):
    """Highlight outlier cells across **many** numeric columns.

    Same detection rules as :func:`outlier_report`. Non-flagged and non-finite
    cells are left unstyled.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styles the full frame; only cells in ``columns`` can be highlighted.
    """
    _check_choice(method, _OUTLIER_METHODS, "method")
    cols = _resolve_columns(df, columns)

    css = _blank_css(df)
    for col in cols:
        mask, _, _ = _outlier_mask(_as_float(df[col]), method, k, z_threshold)
        css.loc[:, col] = np.where(mask, f"background-color: {outlier_color}", "")

    return df.style.apply(lambda _: css, axis=None)


def corr_heatmap(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "pearson",
    cmap: str = "coolwarm",
):
    """Correlation matrix rendered as a background-gradient Styler.

    Uses :func:`correlation_report` for the numbers, so NaN handling is the
    same pairwise-complete rule.

    .. note::
       ``Styler.background_gradient`` requires ``matplotlib``, which ships in
       the ``[notebook]`` extra. Use :func:`correlation_report` if you only
       need the values.
    """
    corr = correlation_report(df, columns, method=method, pvalues=False)
    return corr.style.background_gradient(cmap=cmap, vmin=-1.0, vmax=1.0)


def style_significance(
    df,
    *,
    pvalue_column: Any = "pvalue",
    alpha: float = 0.05,
    highlight_row: bool = True,
):
    """Shade a results table by statistical significance.

    Tiers (darkest first): ``p < alpha/50``, ``p < alpha/5``, ``p < alpha``,
    then non-significant. With ``highlight_row=True`` the whole row is shaded;
    otherwise only the p-value cell is.

    NaN p-values are left unstyled, so a table mixing computable and
    non-computable rows renders without error.
    """
    pd = _pd()
    _check_frame(df)
    if pvalue_column not in df.columns:
        raise KeyError(
            f"p-value column {pvalue_column!r} not in DataFrame. "
            f"Available: {list(df.columns)}"
        )
    if not pd.api.types.is_numeric_dtype(df[pvalue_column]):
        raise TypeError(f"Column {pvalue_column!r} is not numeric")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    tiers = ((alpha / 50.0, "#66bb6a"), (alpha / 5.0, "#a5d6a7"), (alpha, "#e8f5e9"))
    ns_color = "#f5f5f5"

    def _css_for(p: float) -> str:
        if p is None or not np.isfinite(p):
            return ""
        for cutoff, color in tiers:
            if p < cutoff:
                return f"background-color: {color}"
        return f"background-color: {ns_color}"

    css = _blank_css(df)
    for idx, p in df[pvalue_column].items():
        style = _css_for(p)
        if highlight_row:
            css.loc[idx, :] = style
        else:
            css.loc[idx, pvalue_column] = style

    return df.style.apply(lambda _: css, axis=None)


def style_effect_size(
    df,
    effect_column: Any,
    *,
    thresholds: Sequence[float] = (0.2, 0.5, 0.8),
    highlight_row: bool = False,
):
    """Shade an effect-size column by magnitude.

    Buckets ``|effect|`` against ``thresholds`` (default Cohen's small/medium/
    large: 0.2 / 0.5 / 0.8) into negligible, small, medium, large. Pass your own
    ascending thresholds for other measures -- e.g. ``(0.147, 0.33, 0.474)`` for
    Cliff's delta, or ``(0.1, 0.3, 0.5)`` for rank-biserial.

    NaN effects are left unstyled.
    """
    pd = _pd()
    _check_frame(df)
    if effect_column not in df.columns:
        raise KeyError(
            f"Effect column {effect_column!r} not in DataFrame. "
            f"Available: {list(df.columns)}"
        )
    if not pd.api.types.is_numeric_dtype(df[effect_column]):
        raise TypeError(f"Column {effect_column!r} is not numeric")

    cuts = [float(t) for t in thresholds]
    if len(cuts) != 3:
        raise ValueError(f"`thresholds` must contain exactly 3 values, got {len(cuts)}")
    if not all(a < b for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"`thresholds` must be strictly ascending, got {cuts}")

    palette = ("#f5f5f5", "#fff9c4", "#ffe082", "#ffab91")

    def _css_for(v: float) -> str:
        if v is None or not np.isfinite(v):
            return ""
        magnitude = abs(v)
        for cutoff, color in zip(cuts, palette):
            if magnitude < cutoff:
                return f"background-color: {color}"
        return f"background-color: {palette[3]}"

    css = _blank_css(df)
    for idx, v in df[effect_column].items():
        style = _css_for(v)
        if highlight_row:
            css.loc[idx, :] = style
        else:
            css.loc[idx, effect_column] = style

    return df.style.apply(lambda _: css, axis=None)


def demean_style(
    df,
    column: Any,
    *,
    above_color: str = "#c8e6c9",
    below_color: str = "#ffcdd2",
    zero_color: str = "#e0e0e0",
    add_suffix: str = "_demeaned",
):
    """Add a demeaned column and color it above / below / at the mean.

    Non-finite inputs are excluded from the mean and left unstyled.
    """
    _resolve_columns(df, [column])
    values = _as_float(df[column])

    # One fused kernel call returns both outputs; scatter each back separately
    # rather than paying for demean_with_signs twice.
    mask = np.isfinite(values)
    demeaned = np.full(values.shape, np.nan, dtype=np.float64)
    signs = np.full(values.shape, np.nan, dtype=np.float64)
    if mask.any():
        d, s = _bs.demean_with_signs(np.ascontiguousarray(values[mask], dtype=np.float64))
        demeaned[mask] = np.asarray(d, dtype=np.float64)
        signs[mask] = np.asarray(s, dtype=np.float64)

    target = f"{column}{add_suffix}"
    out = df.copy()
    out[target] = demeaned

    colors = np.where(
        ~np.isfinite(signs), "",
        np.where(
            signs > 0, f"background-color: {above_color}",
            np.where(signs < 0, f"background-color: {below_color}",
                     f"background-color: {zero_color}"),
        ),
    )
    css = _blank_css(out)
    css.loc[:, target] = colors
    return out.style.apply(lambda _: css, axis=None)


def zscore_style(
    df,
    column: Any,
    *,
    threshold: float = 2.0,
    high_color: str = "#ffcc80",
    low_color: str = "#bbdefb",
    zero_color: str = "#f5f5f5",
    add_suffix: str = "_zscore",
):
    """Add a z-score column and highlight scores beyond +/- ``threshold``.

    Non-finite inputs are excluded from the mean/std and left unstyled.
    """
    _resolve_columns(df, [column])
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")

    z = _fit_scatter(_as_float(df[column]), _bs.zscore)

    target = f"{column}{add_suffix}"
    out = df.copy()
    out[target] = z

    with np.errstate(invalid="ignore"):
        extreme = np.isfinite(z) & (np.abs(z) >= threshold)
    colors = np.where(
        ~np.isfinite(z), "",
        np.where(
            extreme & (z > 0), f"background-color: {high_color}",
            np.where(extreme, f"background-color: {low_color}",
                     f"background-color: {zero_color}"),
        ),
    )
    css = _blank_css(out)
    css.loc[:, target] = colors
    return out.style.apply(lambda _: css, axis=None)


def iqr_outlier_style(
    df,
    column: Any,
    *,
    k: float = 1.5,
    outlier_color: str = "#ff8a80",
):
    """Highlight IQR outliers in a single column.

    Thin wrapper over :func:`outlier_style` with ``method="iqr"``.
    """
    return outlier_style(df, [column], method="iqr", k=k, outlier_color=outlier_color)
