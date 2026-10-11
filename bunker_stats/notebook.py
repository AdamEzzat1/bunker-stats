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

Every helper also takes ``rich=True``, which keeps the same numbers but wraps
them with the context needed to describe or reproduce them: every styler ->
:class:`StyleResult` (unpacks as ``styler, frame, metadata``), every transform
-> :class:`TransformResult` (``frame, metadata``), and every report ->
:class:`NotebookReport` (``data, meta``). All of them have ``.to_dict()``,
``.to_json()``, ``.info()`` and ``.warnings``, and answer to both spellings of
their parts: ``.frame`` / ``.data`` and ``.metadata`` / ``.meta``.

Warnings
--------
Conditions that make a result *misleading* are collected in ``.warnings`` on a
rich result. A few mean the returned numbers are not what they appear to be,
and those are also raised as a :class:`UserWarning` on the plain path: integer
values too large for float64, a derived column replacing an existing one,
robust scaling with a zero MAD, too few bootstrap resamples, a bootstrap
interval from fewer than 30 values, and a table too large for pandas to render
in full.

Copy semantics
--------------
No helper ever modifies the DataFrame it is given. Helpers that add columns
(``*_columns``, :func:`demean_style`, :func:`zscore_style`,
:func:`scale_compare_style`) work on a copy. Helpers that only color existing
cells style the input frame directly without copying it. There is no
``inplace`` option.

NaN and infinity policy
-----------------------
The Rust kernels are *strict*: a single NaN anywhere in the input poisons the
whole result. That is the right default for a numerical library and the wrong
default for exploratory data analysis, so this layer makes the choice explicit:

* **Reports** (``*_report``, :func:`robust_summary`, :func:`describe_fast`)
  drop non-finite values (NaN and +/-inf) **per column** before calling a
  kernel, and report how many were dropped in the ``n_missing`` column. A
  column with no finite values yields an all-NaN row rather than an exception.
  (:func:`correlation_report` and :func:`rolling_report` have no such column:
  the first reports the pairwise ``n``, the second keeps one row per input row.)
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

import math
import sys
import warnings as _warnings
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
    "missingness_style",
    "scale_compare_style",
    "available_themes",
    # --- report objects (returned by *_report(..., rich=True)) -------------
    "NotebookReport",
    "SummaryReport",
    "OutlierReport",
    "CorrelationReport",
    "BootstrapCIReport",
    "DescribeReport",
    "NormalityReport",
    "MissingnessReport",
    "RollingReport",
    # --- style / transform objects (returned by <helper>(..., rich=True)) --
    "StyleResult",
    "OutlierStyleResult",
    "IQROutlierStyleResult",
    "DemeanStyleResult",
    "ZScoreStyleResult",
    "ScaleCompareStyleResult",
    "CorrelationHeatmapResult",
    "MissingnessStyleResult",
    "SignificanceStyleResult",
    "EffectSizeStyleResult",
    "TransformResult",
    "RobustScaleResult",
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


def _check_unique_labels(df, cols: Sequence[Any]) -> None:
    """Reject column labels that select more than one column.

    With duplicate labels ``df[label]`` is a DataFrame, not a Series, and the
    float conversion below would silently pool several columns into one
    sample. ``get_loc`` returns an int only for a unique, fully-specified
    label (it returns a slice/mask for duplicates and partial MultiIndex keys).
    """
    ambiguous = [
        c for c in dict.fromkeys(cols)
        if not isinstance(df.columns.get_loc(c), (int, np.integer))
    ]
    if ambiguous:
        raise ValueError(
            f"Column label(s) {ambiguous} match more than one column. "
            "Make the column labels unique (or pass full MultiIndex keys) first."
        )


def _column_index(df, cols: Sequence[Any]):
    """Row index for a one-row-per-column report.

    ``pd.Index(tuples, name="column")`` raises for MultiIndex columns, which
    made every report unusable on such frames. Keep the levels instead, so a
    row is addressed the same way as its source column.
    """
    pd = _pd()
    if isinstance(df.columns, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(list(cols), names=df.columns.names)
    return pd.Index(list(cols), name="column")


def _is_real_numeric(series) -> bool:
    """Numeric and safely representable as float64 (so: not complex)."""
    pd = _pd()
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_complex_dtype(
        series
    )


def _check_positive(value: Any, name: str) -> float:
    """A finite float > 0, or a clear error (NaN slips past a bare ``<= 0``)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _check_number(value: Any, name: str) -> float:
    """A real number as float, or a TypeError naming the argument."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _check_open_unit(value: Any, name: str) -> float:
    """A float strictly inside (0, 1) (written so that NaN fails too)."""
    value = _check_number(value, name)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return value


def _warn(message: str) -> None:
    """Emit a UserWarning attributed to the user's own line.

    Helpers call each other (``iqr_outlier_style`` -> ``outlier_style``), so a
    fixed ``stacklevel`` points into this file for some of them. Walk out of
    this module instead.
    """
    level, caller = 1, sys._getframe(1)
    while caller is not None and caller.f_code.co_filename == __file__:
        level, caller = level + 1, caller.f_back
    _warnings.warn(message, UserWarning, stacklevel=level + 1)


def _suffixed(column: Any, suffix: str) -> Any:
    """Label for a column derived from `column`.

    A MultiIndex label is a tuple; the suffix goes on its last level so the
    derived column stays under the same group (``("g", "a")`` ->
    ``("g", "a_zscore")``) instead of becoming ``"('g', 'a')_zscore"``.
    """
    if isinstance(column, tuple) and column:
        return column[:-1] + (f"{column[-1]}{suffix}",)
    return f"{column}{suffix}"


_FLOAT64_EXACT_INT = 2 ** 53


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
        # Positional scan rather than select_dtypes(np.number): that selector
        # also returns timedelta64 and complex columns, neither of which can
        # be cast to float64, and it mishandles duplicate labels.
        cols = [
            label
            for i, label in enumerate(df.columns)
            if _is_real_numeric(df.iloc[:, i])
            and not pd.api.types.is_bool_dtype(df.iloc[:, i])
        ]
        if not cols:
            raise ValueError(
                "DataFrame has no numeric columns; pass `columns=[...]` explicitly."
            )
        _check_unique_labels(df, cols)
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

    _check_unique_labels(df, cols)

    if numeric_only:
        bad = [c for c in cols if not _is_real_numeric(df[c])]
        if bad:
            raise TypeError(
                f"Column(s) are not numeric: {bad}. "
                "Only real-valued numeric columns can be summarized; "
                "drop them or cast first."
            )
    return cols


def _as_float(series) -> np.ndarray:
    """1-D float64 copy of a numeric Series (nullable dtypes: pd.NA -> NaN).

    ``to_numpy(na_value=...)`` rather than ``astype("float64")``: the latter
    raises on masked ``Int64``/``boolean`` columns holding ``pd.NA`` in older
    pandas releases still inside the supported range. And on the *array*,
    not the Series: pandas 1.3's ``Series.to_numpy(na_value=...)`` fills the
    missing entries by label, which raises ``KeyError`` when the index has
    duplicates.
    """
    pd = _pd()
    if pd.api.types.is_integer_dtype(series) and len(series):
        # float64 has 53 bits of mantissa. Beyond that neighbouring integers
        # collapse onto the same float, and every statistic of the *spread*
        # (std, MAD, z-scores, ranks) is computed from the collapsed values.
        lo, hi = series.min(), series.max()
        if not (pd.isna(lo) or pd.isna(hi)) and max(-int(lo), int(hi)) > _FLOAT64_EXACT_INT:
            _warn(
                f"column {series.name!r} holds integers beyond 2**53, which "
                "float64 cannot represent exactly: nearby values collapse "
                "together, so the standard deviation, z-scores and other "
                "spread statistics can be wrong. Subtract a constant offset "
                "from the column first"
            )
    return np.ascontiguousarray(
        series.array.to_numpy(dtype="float64", na_value=np.nan), dtype=np.float64
    )


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

def robust_summary(df, columns: Sequence[Any] | None = None, *, rich: bool = False):
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
    out = pd.DataFrame(rows, index=_column_index(df, cols))
    out = out.reindex(columns=order)
    if rich:
        warnings = []
        empty = [str(c) for c in out.index[out["n"] == 0]]
        if empty:
            warnings.append(f"no finite values in column(s): {', '.join(empty)}")
        constant = [
            str(c) for c in out.index[(out["n"] > 0) & (out["std"] == 0.0)]
        ]
        if constant:
            warnings.append(
                f"zero variance in column(s): {', '.join(constant)} — "
                "scale statistics are 0 and z-scores are undefined"
            )
        return SummaryReport(
            out,
            meta={
                "columns": list(cols),
                "nan_policy": "non-finite dropped per column (see n_missing)",
                "trimmed_mean": "10% per tail",
                "kurtosis": "excess (Fisher)",
            },
            warnings=warnings,
        )
    return out


def describe_fast(
    df,
    columns: Sequence[Any] | None = None,
    *,
    robust: bool = True,
    rich: bool = False,
):
    """A richer ``df.describe().T`` backed by the Rust kernels.

    Always returns ``n``, ``n_missing``, ``mean``, ``std``, ``min``, ``25%``,
    ``50%``, ``75%``, ``max``. With ``robust=True`` (the default) it also adds
    ``mad``, ``iqr``, ``qn_scale``, ``trimmed_mean``, ``skew`` and ``kurtosis``.

    Unlike ``df.describe()``, non-finite values are dropped per column and
    counted in ``n_missing`` instead of being silently ignored.

    On speed: ``robust=False`` computes the same columns as ``describe()`` and
    is faster than pandas. The default adds six robust statistics and is
    slower than ``describe()`` on large frames; most of that is ``qn_scale``.

    With ``rich=True`` returns a :class:`DescribeReport`.
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
    out = pd.DataFrame(rows, index=_column_index(df, cols))
    out = out.reindex(columns=order)
    if not rich:
        return out
    warnings = []
    empty = [str(c) for c in out.index[out["n"] == 0]]
    if empty:
        warnings.append(f"no finite values in column(s): {', '.join(empty)}")
    constant = [str(c) for c in out.index[(out["n"] > 0) & (out["std"] == 0.0)]]
    if constant:
        warnings.append(f"zero variance in column(s): {', '.join(constant)}")
    meta = {
        "columns": list(cols),
        "robust": bool(robust),
        "nan_policy": "non-finite dropped per column (see n_missing)",
        "quantile_method": "linear interpolation",
    }
    if robust:
        meta.update({"trimmed_mean": "10% per tail", "kurtosis": "excess (Fisher)"})
    return DescribeReport(out, meta=meta, warnings=warnings)


def _unreachable_z_warning(sizes: Mapping[Any, int], z_threshold: float) -> str | None:
    """Columns too short for any classical z-score to reach the threshold.

    In a sample of size n no z-score (ddof=1) can exceed (n - 1) / sqrt(n),
    so e.g. n = 5 can never be flagged at 3 however extreme a value is.
    """
    short = [
        str(c) for c, n in sizes.items()
        if n >= 2 and (n - 1) / math.sqrt(n) < z_threshold
    ]
    if not short:
        return None
    return (
        f"z_threshold {z_threshold:g} cannot be reached in column(s): "
        f"{', '.join(short)} — a sample of size n has no z-score beyond "
        "(n - 1) / sqrt(n), so nothing can be flagged; use "
        "method='robust_zscore' or 'iqr'"
    )


def _outlier_bounds(
    x: np.ndarray, method: str, k: float, z_threshold: float
) -> tuple[float, float]:
    """Lower/upper cutoffs for one finite sample under the chosen method."""
    if x.size == 0:
        return (np.nan, np.nan)

    p = _outlier_params(x, method)
    if method == "iqr":
        return (p["q1"] - k * p["iqr"], p["q3"] + k * p["iqr"])
    return (
        p["center"] - z_threshold * p["scale"],
        p["center"] + z_threshold * p["scale"],
    )


def _outlier_params(x: np.ndarray, method: str) -> dict[str, float]:
    """The fitted quantities each outlier rule builds its fences from."""
    if method == "iqr":
        q1 = _safe(_bs.percentile, x, 25.0)
        q3 = _safe(_bs.percentile, x, 75.0)
        return {"q1": q1, "q3": q3, "iqr": q3 - q1}

    if method == "zscore":
        return {"center": _safe(_bs.mean, x), "scale": _safe(_bs.std, x)}

    # robust_zscore: median +/- z * (1.4826 * MAD) -- the breakdown-resistant
    # analogue of the mean/std rule, so a few extreme points cannot inflate the
    # cutoffs enough to hide themselves.
    return {
        "center": _safe(_bs.median, x),
        "scale": _safe(_bs.mad, x) * _MAD_TO_SIGMA,
    }


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
    rich: bool = False,
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
    k = _check_positive(k, "k")
    z_threshold = _check_positive(z_threshold, "z_threshold")

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

    out = pd.DataFrame(rows, index=_column_index(df, cols))
    if rich:
        warnings = []
        empty = [str(c) for c in out.index[out["n"] == 0]]
        if empty:
            warnings.append(f"no finite values in column(s): {', '.join(empty)}")
        degenerate = [
            str(c)
            for c in out.index[
                (out["n"] > 0) & (out["lower_bound"] == out["upper_bound"])
            ]
        ]
        if degenerate:
            warnings.append(
                f"zero spread in column(s): {', '.join(degenerate)} — the fences "
                "collapse to a point, so every value that differs from it is flagged"
            )
        if method == "zscore":
            note = _unreachable_z_warning(dict(zip(out.index, out["n"])), z_threshold)
            if note:
                warnings.append(note)
        return OutlierReport(
            out,
            meta={
                "method": method,
                "k": k if method == "iqr" else None,
                "z_threshold": z_threshold if method != "iqr" else None,
                "columns": list(cols),
                "nan_policy": "non-finite never flagged; counted in n_missing",
            },
            warnings=warnings,
        )
    return out


def _ad_pvalue(a_star: float, n: int) -> float:
    """Approximate p-value for the Anderson-Darling normality statistic.

    D'Agostino & Stephens (1986), Table 4.9, for mean and variance both
    estimated; valid for n >= 8. Their formulas are in terms of
    ``A^2 (1 + 0.75/n + 2.25/n^2)``, whereas the kernel returns
    ``A^2 (1 + 4/n - 25/n^2)``, so convert first.

    The fit covers statistics up to 13 (p ~ 5e-31). Its quadratic term turns
    the curve back upward further out (past 1 eventually), so beyond 13 the
    tail is continued log-linearly at the slope it has there. That keeps the
    p-value decreasing in the statistic, so stronger departures still rank as
    stronger, and it reaches 0 only by underflow. Out there it is an
    extrapolation: read it as an order of magnitude, not as digits.
    """
    if n < 8 or not np.isfinite(a_star):
        return float("nan")
    a2 = a_star / (1.0 + 4.0 / n - 25.0 / (n * n))
    z = a2 * (1.0 + 0.75 / n + 2.25 / (n * n))
    if z > _AD_FIT_LIMIT:
        edge = 1.2937 - 5.709 * _AD_FIT_LIMIT + 0.0186 * _AD_FIT_LIMIT ** 2
        slope = -5.709 + 2.0 * 0.0186 * _AD_FIT_LIMIT
        log_p = edge + slope * (z - _AD_FIT_LIMIT)
        return math.exp(log_p) if log_p > -745.0 else 0.0
    if z < 0.2:
        p = 1.0 - math.exp(-13.436 + 101.14 * z - 223.73 * z * z)
    elif z < 0.34:
        p = 1.0 - math.exp(-8.318 + 42.796 * z - 59.938 * z * z)
    elif z < 0.6:
        p = math.exp(0.9177 - 4.279 * z - 1.38 * z * z)
    else:
        p = math.exp(1.2937 - 5.709 * z + 0.0186 * z * z)
    return min(max(p, 0.0), 1.0)


_AD_FIT_LIMIT = 13.0
_NORMALITY_MIN_N = 8
_NORMALITY_LARGE_N = 5000


def normality_report(
    df,
    columns: Sequence[Any] | None = None,
    *,
    alpha: float = 0.05,
    rich: bool = False,
):
    """Anderson-Darling and Jarque-Bera normality diagnostics per column.

    Returns ``n``, ``n_missing``, ``skewness``, ``kurtosis``,
    ``jb_statistic``, ``jb_pvalue``, ``ad_statistic``, ``ad_pvalue``,
    ``normal`` and ``conclusion``. ``kurtosis`` is *excess* kurtosis (0 for a
    normal distribution), the same convention as :func:`robust_summary`.

    ``normal`` / ``conclusion`` are decided by the **Anderson-Darling**
    p-value. It responds to any departure from normality, where Jarque-Bera
    only sees skewness and kurtosis and its chi-square approximation is poor
    below a few hundred observations; the Jarque-Bera columns are reported for
    reference. ``ad_statistic`` is A-squared times the small-sample factor
    ``1 + 4/n - 25/n^2`` and ``ad_pvalue`` is the D'Agostino-Stephens
    approximation (extrapolated below about 1e-30).

    ``normal`` is ``None`` when no verdict is possible: fewer than 8 finite
    values, or a constant column. With thousands of observations the tests
    reject for departures far too small to matter, so read the skewness and
    kurtosis as well; ``rich=True`` returns a :class:`NormalityReport` that
    warns about both situations.
    """
    pd = _pd()
    cols = _resolve_columns(df, columns)
    alpha = _check_open_unit(alpha, "alpha")

    rows = []
    for col in cols:
        values = _as_float(df[col])
        x = _finite(values)
        n = int(x.size)
        row: dict[str, Any] = {"n": n, "n_missing": int(values.size) - n}
        constant = n > 0 and float(np.min(x)) == float(np.max(x))
        jb = _safe(_bs.jarque_bera, x, default={}) or {}
        ad = ({} if constant else _safe(_bs.anderson_darling, x, default={})) or {}
        row["skewness"] = jb.get("skewness", np.nan)
        # The kernel reports Pearson kurtosis (normal = 3). Every other
        # "kurtosis" in this layer is excess kurtosis (normal = 0), so convert
        # rather than show two conventions under one column name.
        row["kurtosis"] = jb.get("kurtosis", np.nan) - 3.0
        row["jb_statistic"] = jb.get("statistic", np.nan)
        row["jb_pvalue"] = jb.get("pvalue", np.nan)
        row["ad_statistic"] = ad.get("statistic", np.nan)
        row["ad_pvalue"] = _ad_pvalue(row["ad_statistic"], n)

        p = row["ad_pvalue"]
        if n == 0:
            row["normal"], row["conclusion"] = None, "no finite values"
        elif constant:
            row["normal"], row["conclusion"] = None, "constant column (no variance)"
        elif not np.isfinite(p):
            row["normal"] = None
            row["conclusion"] = f"inconclusive (n < {_NORMALITY_MIN_N})"
        elif p < alpha:
            row["normal"] = False
            row["conclusion"] = f"reject normality (Anderson-Darling p < {alpha:g})"
        else:
            row["normal"] = True
            row["conclusion"] = (
                f"cannot reject normality (Anderson-Darling p >= {alpha:g})"
            )
        rows.append(row)

    order = [
        "n", "n_missing", "skewness", "kurtosis", "jb_statistic", "jb_pvalue",
        "ad_statistic", "ad_pvalue", "normal", "conclusion",
    ]
    out = pd.DataFrame(rows, index=_column_index(df, cols))
    out = out.reindex(columns=order)
    if not rich:
        return out

    warnings = []
    empty = [str(c) for c in out.index[out["n"] == 0]]
    if empty:
        warnings.append(f"no finite values in column(s): {', '.join(empty)}")
    constant_cols = [
        str(c) for c in out.index[out["conclusion"] == "constant column (no variance)"]
    ]
    if constant_cols:
        warnings.append(
            f"constant column(s): {', '.join(constant_cols)} — normality is "
            "not testable"
        )
    small = [
        str(c) for c in out.index[(out["n"] > 0) & (out["n"] < _NORMALITY_MIN_N)]
        if str(c) not in constant_cols
    ]
    if small:
        warnings.append(
            f"fewer than {_NORMALITY_MIN_N} finite values in column(s): "
            f"{', '.join(small)} — too few for a verdict"
        )
    large = [str(c) for c in out.index[out["n"] > _NORMALITY_LARGE_N]]
    if large:
        warnings.append(
            f"large sample (n > {_NORMALITY_LARGE_N}) in column(s): "
            f"{', '.join(large)} — the tests reject for departures from "
            "normality too small to matter in practice; judge by skewness "
            "and kurtosis as well"
        )
    return NormalityReport(
        out,
        meta={
            "alpha": alpha,
            "columns": list(cols),
            "decided_by": "Anderson-Darling p-value (D'Agostino-Stephens approximation)",
            "min_n_for_verdict": _NORMALITY_MIN_N,
            "kurtosis": "excess (Fisher)",
            "nan_policy": "non-finite dropped per column (see n_missing)",
            "caveat": (
                "failing to reject is not evidence of normality, and no "
                "multiple-comparison correction is applied across columns"
            ),
        },
        warnings=warnings,
    )


def _corr_diagnostics(df, cols: Sequence[Any]) -> dict:
    """Pairwise sample sizes and the warnings they imply, for a set of columns."""
    values = np.column_stack([_as_float(df[c]) for c in cols])
    finite = np.isfinite(values)
    pair_n = finite.astype(np.int64).T @ finite.astype(np.int64)
    upper = np.triu_indices(len(cols), k=1)
    pair_counts = pair_n[upper]

    all_missing, constant = [], []
    for j, col in enumerate(cols):
        x = values[finite[:, j], j]
        if x.size == 0:
            all_missing.append(str(col))
        elif np.min(x) == np.max(x):
            constant.append(str(col))

    warnings = []
    if all_missing:
        warnings.append(f"no finite values in column(s): {', '.join(all_missing)}")
    if constant:
        warnings.append(
            f"constant column(s): {', '.join(constant)} — correlation with a "
            "constant is undefined (shown as missing)"
        )
    n_short = int((pair_counts < 3).sum())
    if n_short:
        warnings.append(
            f"{n_short} pair(s) have fewer than 3 overlapping rows — their "
            "correlation is undefined (shown as missing)"
        )
    if pair_counts.size and pair_counts.min() != pair_counts.max():
        warnings.append(
            "pairwise deletion: cells are computed from different row subsets "
            f"(n from {int(pair_counts.min())} to {int(pair_counts.max())}), so "
            "they are not directly comparable"
        )
    usable = pair_counts[pair_counts >= 3]
    if usable.size and usable.min() < 10:
        warnings.append(
            f"small overlap (min pairwise n = {int(usable.min())}): "
            "correlations from few rows are noisy"
        )
    return {
        "pair_n": pair_n,
        "pair_counts": pair_counts,
        "all_missing": all_missing,
        "constant": constant,
        "warnings": warnings,
    }


def correlation_report(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "pearson",
    pvalues: bool = False,
    rich: bool = False,
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

    meta = {
        "method": method,
        "columns": list(cols),
        "nan_policy": "pairwise-complete (each pair uses rows finite in both)",
        "form": "long" if pvalues else "matrix",
    }
    notes: list = []
    if rich:
        diag = _corr_diagnostics(df, cols)
        notes = diag["warnings"]
        counts = diag["pair_counts"]
        meta["pairwise_n_min"] = int(counts.min()) if counts.size else None
        meta["pairwise_n_max"] = int(counts.max()) if counts.size else None
        if pvalues and len(cols) > 2:
            notes = notes + [
                f"{len(cols) * (len(cols) - 1) // 2} tests with no "
                "multiple-comparison correction — consider bunker_stats.p_adjust"
            ]

    if not pvalues:
        mat = np.full((len(cols), len(cols)), np.nan)
        np.fill_diagonal(mat, 1.0)
        for i in range(len(cols)):
            xi = _as_float(df[cols[i]])
            for j in range(i + 1, len(cols)):
                xj = _as_float(df[cols[j]])
                r = _pair_corr(test, xi, xj)[0]
                mat[i, j] = mat[j, i] = r
        index = _column_index(df, cols)
        out = pd.DataFrame(
            mat, index=index,
            # Same levels on both axes for MultiIndex columns; a plain list
            # otherwise, so flat frames keep their unnamed column axis.
            columns=index if isinstance(index, pd.MultiIndex) else cols,
        )
        return CorrelationReport(out, meta=meta, warnings=notes) if rich else out

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
    out = pd.DataFrame(rows, columns=["x", "y", "n", "correlation", "statistic", "pvalue"])
    return CorrelationReport(out, meta=meta, warnings=notes) if rich else out


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


def _gap_masks(series) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(is_missing, is_infinite, numeric)`` for one column of any dtype.

    Shared by :func:`missingness_report` and :func:`missingness_style` so the
    two can never disagree. ``numeric`` says whether finite/infinite is a
    meaningful question for the column: real and complex dtypes, and object
    columns whose non-null entries are all real numbers (a column of floats
    that picked up ``object`` dtype still has its infinities counted).
    """
    pd = _pd()
    is_missing = series.isna().to_numpy(dtype=bool)
    none = np.zeros(len(series), dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return is_missing, none, False
    if pd.api.types.is_complex_dtype(series):
        # Infinite if either part is; a float cast would drop the imaginary
        # part. A value can be both (NaN in one part, inf in the other): it
        # counts as missing, so that the three counts still add up to n_rows.
        return is_missing, np.isinf(series.to_numpy()) & ~is_missing, True
    if pd.api.types.is_numeric_dtype(series):
        return is_missing, np.isinf(_as_float(series)), True
    if series.dtype == object:
        items = series.to_numpy(dtype=object)
        real = np.array(
            [
                isinstance(v, (int, float, np.integer, np.floating))
                and not isinstance(v, (bool, np.bool_))
                for v in items
            ],
            dtype=bool,
        )
        if real.any() and bool((real | is_missing).all()):
            infinite = np.array(
                [bool(r) and math.isinf(v) for r, v in zip(real, items)], dtype=bool
            )
            return is_missing, infinite & ~is_missing, True
    return is_missing, none, False


def missingness_report(df, *, rich: bool = False):
    """Missing / non-finite audit for **every** column, numeric or not.

    Returns ``dtype``, ``n_rows``, ``n_missing`` (null per pandas),
    ``pct_missing``, ``n_finite`` and ``n_infinite``. The finite/infinite
    counts are NaN for columns where they are not meaningful (text, dates,
    booleans). Complex columns count a value as infinite if either part is,
    and an ``object`` column holding only real numbers is treated as numeric.

    This is the one helper that counts rather than drops non-finite values.
    With ``rich=True`` returns a :class:`MissingnessReport`.
    """
    pd = _pd()
    _check_frame(df)

    n_rows = int(len(df))
    rows = []
    incomplete = np.zeros(n_rows, dtype=bool)
    for i in range(df.shape[1]):
        series = df.iloc[:, i]  # positional: survives duplicate column labels
        is_missing, is_infinite, numeric = _gap_masks(series)
        incomplete |= is_missing | is_infinite
        n_missing = int(is_missing.sum())
        row = {
            "dtype": str(series.dtype),
            "n_rows": n_rows,
            "n_missing": n_missing,
            "pct_missing": (100.0 * n_missing / n_rows) if n_rows else np.nan,
        }
        if numeric:
            row["n_infinite"] = int(is_infinite.sum())
            row["n_finite"] = n_rows - n_missing - row["n_infinite"]
        else:
            row["n_finite"] = np.nan
            row["n_infinite"] = np.nan
        rows.append(row)

    out = pd.DataFrame(
        rows,
        index=_column_index(df, list(df.columns)),
        columns=["dtype", "n_rows", "n_missing", "pct_missing", "n_finite", "n_infinite"],
    )
    if not rich:
        return out

    warnings = []
    hollow = [
        str(c) for c, m in zip(out.index, out["n_missing"]) if n_rows and m == n_rows
    ]
    if hollow:
        warnings.append(f"entirely missing column(s): {', '.join(hollow)}")
    if n_rows and df.shape[1] and not (~incomplete).any():
        warnings.append(
            "no row is complete across all columns — listwise deletion would "
            "discard every row"
        )
    return MissingnessReport(
        out,
        meta={
            "n_rows": n_rows,
            "n_columns": int(df.shape[1]),
            "n_complete_rows": int((~incomplete).sum()),
            "n_incomplete_rows": int(incomplete.sum()),
            "definition": (
                "missing = null per pandas (NaN/None/NA/NaT); infinite = +/-inf "
                "in numeric columns; a complete row has neither"
            ),
        },
        warnings=warnings,
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
    rich: bool = False,
):
    """Rolling-window features for one column, via the fused Rust kernel.

    Uses :class:`bunker_stats.Rolling`, which computes all requested statistics
    in a single pass instead of one pass per statistic.

    Parameters
    ----------
    stats : sequence of {"mean", "std", "var", "count", "min", "max"}
    min_periods : int or None
        Minimum valid observations per window. Defaults to ``window``, so
        every reported value comes from a full window. Lowering it lets a
        window with NaNs produce a value (together with
        ``nan_policy="ignore"``) and lets the rows that cannot have a full
        window use a partial one: the first ``window - 1`` rows for
        ``alignment="trailing"`` (as pandas ``rolling(window, min_periods=m)``
        does), the rows near either end for ``"centered"``.
    alignment : {"trailing", "centered"}
        ``"trailing"`` (default): the value at row ``i`` covers rows
        ``i - window + 1 .. i``. ``"centered"``: it covers ``window`` rows
        around ``i``, and an even window takes its extra row on the left
        (``i - window//2 .. i + (window-1)//2``), matching pandas
        ``rolling(window, center=True)``. Rows too close to either end for a
        full window are NaN unless ``min_periods`` is lowered. The ``count``
        statistic is the number of valid (non-NaN) values in each window. It
        is reported under every policy, including where the other statistics
        are NaN, since it is what explains them.
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
        per requested statistic (for a MultiIndex column the suffix goes on
        the last level). If the kernel returns the shortened
        ``n - window + 1`` form, results are right-aligned and left-padded with
        NaN so the index always lines up with the input. With ``rich=True``
        returns a :class:`RollingReport`.
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

    _check_choice(alignment, ("trailing", "centered"), "alignment")
    # The centered kernel accepts truncated windows at the edges when
    # min_periods is left unset. A report column named "..._roll{window}_mean"
    # must not silently hold 3-point means, so ask for full windows.
    if alignment == "centered" and min_periods is None:
        min_periods = window

    values = _as_float(df[column])
    # full_length: one value per row, with partial leading windows once
    # min_periods is met (a no-op for centered, which is already full length).
    roller = _bs.Rolling(
        values,
        window=window,
        min_periods=min_periods,
        alignment=alignment,
        nan_policy=nan_policy,
        full_length=True,
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
        out[_suffixed(column, f"_roll{window}_{stat}")] = arr

    frame = pd.DataFrame(out, index=df.index)
    if isinstance(df.columns, pd.MultiIndex):
        frame.columns = pd.MultiIndex.from_tuples(list(out), names=df.columns.names)
    if not rich:
        return frame

    n_valid = int(np.isfinite(values).sum())
    warnings = []
    if n_valid < n and nan_policy == "ignore" and (min_periods is None or min_periods >= window):
        warnings.append(
            "nan_policy='ignore' has no effect while min_periods equals the "
            "window: a window holding a NaN still fails the count check. "
            "Pass min_periods below the window"
        )
    first = next(iter(out.values()))
    return RollingReport(
        frame,
        meta={
            "column": column,
            "window": window,
            "stats": list(stats),
            "alignment": alignment,
            "min_periods": window if min_periods is None else int(min_periods),
            "nan_policy": nan_policy,
            "n_rows": int(n),
            "n_missing": int(n) - n_valid,
            "n_rows_without_a_value": int((~np.isfinite(first)).sum()),
            "derived_columns": list(out),
        },
        warnings=warnings,
    )


def bootstrap_ci_report(
    df,
    columns: Sequence[Any] | None = None,
    *,
    stat: str = "mean",
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: int | None = None,
    rich: bool = False,
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
        ``estimate`` is the statistic of the sample itself; the interval is
        a percentile bootstrap interval, which is too narrow for small
        samples; a column with n < 30 raises a ``UserWarning`` saying so (or,
        with ``rich=True``, a note in ``.warnings``).
        For ``stat="std"`` both the estimate and the draws are the ddof=0
        (population) standard deviation, which is smaller than
        :func:`bunker_stats.std` by a factor ``sqrt((n - 1) / n)``.
    """
    pd = _pd()
    cols = _resolve_columns(df, columns)
    conf = _check_open_unit(conf, "conf")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)):
        raise TypeError(
            f"n_resamples must be an int, got {type(n_resamples).__name__}"
        )
    n_resamples = int(n_resamples)
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    few_resamples = None
    if n_resamples < 100:
        # A percentile interval reads the alpha/2 and 1 - alpha/2 quantiles of
        # the draws; with a handful of draws those are just the extremes (and
        # with one draw the interval has zero width).
        few_resamples = (
            f"n_resamples={n_resamples} is too few for a percentile interval: "
            "its endpoints are little more than the most extreme draws. Use "
            "at least 1000"
        )
        if not rich:
            _warn(few_resamples)

    config = _bs.BootstrapConfig(
        n_resamples=n_resamples,
        conf=conf,
        stat=stat,
        random_state=random_state,
        nan_policy="omit",
    )
    # BootstrapConfig has validated `stat` by now. The estimate must use the
    # same estimator as the bootstrap draws or it can fall outside its own
    # interval: the resampling kernel computes std with ddof=0, so the "std"
    # estimate is the ddof=0 value too (the ddof=1 kernel, rescaled).
    point = {
        "mean": _bs.mean,
        "median": _bs.median,
        "std": lambda v: _bs.std(v) * math.sqrt((v.size - 1) / v.size),
    }[stat]

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
            # config.run's first value is the mean of the bootstrap draws,
            # which differs from the statistic of the sample itself (visibly
            # so for the median). Report the sample statistic as the estimate.
            _, lower, upper = config.run(x)
            row.update(
                {"estimate": _safe(point, x), "ci_lower": lower, "ci_upper": upper}
            )
        rows.append(row)

    order = ["stat", "n", "n_missing", "estimate", "ci_lower", "ci_upper", "conf"]
    out = pd.DataFrame(rows, index=_column_index(df, cols))
    out = out.reindex(columns=order)
    tiny = [str(c) for c in out.index[out["n"] < 30]]
    small_sample = (
        f"small sample (n < 30) in column(s): {', '.join(tiny)} — "
        "percentile bootstrap intervals are too narrow at small n "
        "(a nominal 95% interval for a mean covers roughly 83% at "
        "n = 5 and 93% at n = 30; intervals for a std are narrower "
        "still)"
        if tiny else None
    )
    if small_sample and not rich:
        # The interval looks like any other but under-covers, so say so
        # without requiring rich=True.
        _warn(small_sample)
    if rich:
        warnings = []
        if few_resamples:
            warnings.append(few_resamples)
        elif n_resamples < 1000:
            warnings.append(
                f"n_resamples={n_resamples} is low; CI endpoints may be "
                "unstable (>= 1000 recommended)"
            )
        if small_sample:
            warnings.append(small_sample)
        return BootstrapCIReport(
            out,
            meta={
                "stat": stat,
                "n_resamples": n_resamples,
                "conf": conf,
                "random_state": random_state,
                "columns": list(cols),
                "estimate": "the statistic of the sample (not the bootstrap mean)",
                "std_ddof": 0 if stat == "std" else None,
                "interval": "percentile bootstrap",
                "nan_policy": "omit (non-finite dropped per column)",
            },
            warnings=warnings,
        )
    return out


# ======================================================================
# Column transforms  ->  DataFrame
# ======================================================================

_COPY_POLICY = "input DataFrame is never modified; the result is a copy"


def _scale_params(x: np.ndarray, method: str, scale_factor: float) -> dict[str, Any]:
    """Fitted parameters of one scaling method on a finite sample."""
    if x.size == 0:
        return {}
    if method == "robust":
        mad = float(_safe(_bs.mad, x))
        return {
            "median": float(_safe(_bs.median, x)),
            "mad": mad,
            "scale": scale_factor * mad,
            "zero_mad": mad == 0.0,
        }
    if method == "zscore":
        return {"mean": float(_safe(_bs.mean, x)), "std": float(_safe(_bs.std, x)), "ddof": 1}
    lo, hi = float(np.min(x)), float(np.max(x))
    return {"min": lo, "max": hi, "range": hi - lo}


_ZERO_MAD_BEHAVIOR = (
    "when MAD is 0 the kernel divides by a ~1e-12 floor: values equal to the "
    "median map to 0 and every other value to a huge magnitude, so the output "
    "is not a usable robust score"
)


def scale_columns(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "robust",
    suffix: str | None = None,
    scale_factor: float = _MAD_TO_SIGMA,
    replace: bool = False,
    rich: bool = False,
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
        Overwrite the source columns *in the returned copy* instead of adding
        new ones. The input DataFrame is never modified either way.
    rich : bool
        Return a :class:`TransformResult` (``frame, metadata``) carrying the
        fitted parameters per column, so the transform can be reported or
        re-applied to new data.

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
        scale_factor = _check_positive(scale_factor, "scale_factor")
        kernel = lambda x: _bs.robust_scale(x, scale_factor)  # noqa: E731
    elif method == "zscore":
        kernel = _bs.zscore
    else:
        kernel = _bs.minmax_scale

    out = df.copy()
    targets = {col: col if replace else _suffixed(col, suffix) for col in cols}
    overwrote = [t for t in targets.values() if not replace and t in df.columns]
    per_column: dict[Any, dict] = {}
    zero_mad = []
    for col in cols:
        values = _as_float(df[col])
        out[targets[col]] = _fit_scatter(values, kernel)
        x = _finite(values)
        if method == "robust" and x.size and float(_safe(_bs.mad, x)) == 0.0:
            zero_mad.append(str(col))
        if rich:
            per_column[col] = {
                "derived_column": targets[col],
                "n": int(x.size),
                "n_missing": int(values.size - x.size),
                **_scale_params(x, method, scale_factor),
            }
    zero_mad_note = (
        f"MAD is 0 in column(s): {', '.join(zero_mad)} — " + _ZERO_MAD_BEHAVIOR
        if zero_mad else None
    )
    if not rich:
        # Both of these change what the returned numbers mean, so they are
        # not left to the rich result alone.
        if zero_mad_note:
            _warn(zero_mad_note)
        if overwrote:
            _warn(_overwrite_warning(overwrote))
        return out

    warnings = []
    empty = [str(c) for c, d in per_column.items() if d["n"] == 0]
    if empty:
        warnings.append(f"no finite values in column(s): {', '.join(empty)}")
    if zero_mad_note:
        warnings.append(zero_mad_note)
    zero_std = [str(c) for c, d in per_column.items() if d.get("std") == 0.0]
    if zero_std:
        warnings.append(
            f"zero variance in column(s): {', '.join(zero_std)} — z-scores are NaN"
        )
    zero_range = [str(c) for c, d in per_column.items() if d.get("range") == 0.0]
    if zero_range:
        warnings.append(
            f"zero range in column(s): {', '.join(zero_range)} — min-max output is all 0"
        )
    if overwrote:
        warnings.append(_overwrite_warning(overwrote))
    meta = {
        "method": method,
        "columns": list(cols),
        "replace": bool(replace),
        "scale_factor": scale_factor if method == "robust" else None,
        "per_column": per_column,
        "zero_mad_behavior": _ZERO_MAD_BEHAVIOR if method == "robust" else None,
        "nan_policy": "fit on finite values; non-finite inputs stay NaN in place",
        "copy_policy": _COPY_POLICY,
        "frame_is_copy": True,
        "overwrote_existing_column": bool(overwrote),
    }
    return TransformResult(out, meta, warnings)


def winsorize_columns(
    df,
    columns: Sequence[Any] | None = None,
    *,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    suffix: str = "_winsor",
    replace: bool = False,
    rich: bool = False,
):
    """Batch-winsorize numeric columns (clip tails at the given quantiles).

    ``lower_q``/``upper_q`` are quantile **fractions** in [0, 1], matching
    :func:`bunker_stats.winsorize` rather than percentile units.

    Quantiles are computed on the finite values only; non-finite inputs stay
    NaN in the output and row order is preserved. The input DataFrame is never
    modified; ``replace=True`` overwrites the source columns in the copy.

    With ``rich=True`` returns a :class:`TransformResult` whose metadata gives,
    per column, the clipping bounds and how many values each tail changed.
    """
    cols = _resolve_columns(df, columns)
    lower_q = _check_number(lower_q, "lower_q")
    upper_q = _check_number(upper_q, "upper_q")
    for name, q in (("lower_q", lower_q), ("upper_q", upper_q)):
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {q}")
    if lower_q >= upper_q:
        raise ValueError(f"lower_q ({lower_q}) must be < upper_q ({upper_q})")
    if not replace and not suffix:
        raise ValueError("`suffix` cannot be empty unless replace=True")

    kernel = lambda x: _bs.winsorize(x, lower_q=lower_q, upper_q=upper_q)  # noqa: E731

    out = df.copy()
    targets = {col: col if replace else _suffixed(col, suffix) for col in cols}
    overwrote = [t for t in targets.values() if not replace and t in df.columns]
    if overwrote and not rich:
        _warn(_overwrite_warning(overwrote))
    per_column: dict[Any, dict] = {}
    for col in cols:
        values = _as_float(df[col])
        clipped = _fit_scatter(values, kernel)
        out[targets[col]] = clipped
        if rich:
            finite = np.isfinite(values)
            n = int(finite.sum())
            with np.errstate(invalid="ignore"):
                raised = int((finite & (clipped > values)).sum())
                lowered = int((finite & (clipped < values)).sum())
            per_column[col] = {
                "derived_column": targets[col],
                "n": n,
                "n_missing": int(values.size) - n,
                "lower_bound": float(np.min(clipped[finite])) if n else float("nan"),
                "upper_bound": float(np.max(clipped[finite])) if n else float("nan"),
                "n_clipped_low": raised,
                "n_clipped_high": lowered,
                "pct_clipped": (100.0 * (raised + lowered) / n) if n else float("nan"),
            }
    if not rich:
        return out

    warnings = []
    empty = [str(c) for c, d in per_column.items() if d["n"] == 0]
    if empty:
        warnings.append(f"no finite values in column(s): {', '.join(empty)}")
    if overwrote:
        warnings.append(_overwrite_warning(overwrote))
    meta = {
        "method": "winsorize",
        "columns": list(cols),
        "lower_q": float(lower_q),
        "upper_q": float(upper_q),
        "replace": bool(replace),
        "per_column": per_column,
        "bounds": "smallest / largest value remaining after clipping",
        "nan_policy": "quantiles from finite values; non-finite inputs stay NaN in place",
        "caveat": (
            "winsorizing changes the data: report it, and note that variances "
            "and tests computed afterwards understate tail variability"
        ),
        "copy_policy": _COPY_POLICY,
        "frame_is_copy": True,
        "overwrote_existing_column": bool(overwrote),
    }
    return TransformResult(out, meta, warnings)


def robust_scale_column(
    df,
    column: Any,
    *,
    scale_factor: float = _MAD_TO_SIGMA,
    suffix: str | None = None,
    add_suffix: str | None = None,
    rich: bool = False,
):
    """Single-column robust scaling. Kept for backwards compatibility.

    Prefer :func:`scale_columns` for new code -- it handles many columns, other
    methods, and NaN-preserving scatter-back in one call.

    Returns a copy of ``df`` with ``{column}{suffix}`` added (default
    ``"_robust"``; ``add_suffix`` is the older name for ``suffix``); the input
    is never modified. With ``rich=True`` returns a :class:`RobustScaleResult`
    (``frame, metadata``) whose flat metadata has ``median``, ``mad``,
    ``scale_factor``, ``scale`` and ``zero_mad``.
    """
    result = scale_columns(
        df,
        [column],
        method="robust",
        suffix=_one_suffix(suffix, add_suffix, "_robust"),
        scale_factor=scale_factor,
        rich=rich,
    )
    if not rich:
        return result
    multi = result.metadata
    meta = {
        "source_column": column,
        **multi["per_column"][column],
        "scale_factor": multi["scale_factor"],
        "zero_mad_behavior": multi["zero_mad_behavior"],
        "nan_policy": multi["nan_policy"],
        "copy_policy": multi["copy_policy"],
        "frame_is_copy": True,
        "overwrote_existing_column": multi["overwrote_existing_column"],
    }
    return RobustScaleResult(result.frame, meta, result.warnings)


# ======================================================================
# Themes
# ======================================================================
# Every styled cell sets BOTH a background and a text color. Setting only the
# background leaves the notebook's own text color in place, which is light in
# dark-mode Jupyter/VS Code and unreadable on a pastel cell.
#
# ``default`` keeps the colors these helpers have always used. ``colorblind``
# swaps the red/green above/below pairing for blue/orange tints and uses
# lightness-ordered single-hue ramps for the tiered roles.
_THEMES: dict[str, dict[str, Any]] = {
    "default": {
        "text": "#1a1a1a",
        "above": "#c8e6c9",
        "below": "#ffcdd2",
        "at_center": "#e0e0e0",
        "high": "#ffcc80",
        "low": "#bbdefb",
        "inlier": "#f5f5f5",
        "outlier": "#ff8a80",
        "missing": "#cfd8dc",
        "infinite": "#ffe082",
        "significance": ("#66bb6a", "#a5d6a7", "#e8f5e9"),
        "not_significant": "#f5f5f5",
        "effect": ("#f5f5f5", "#fff9c4", "#ffe082", "#ffab91"),
    },
    "colorblind": {
        "text": "#1a1a1a",
        "above": "#b3d9f2",
        "below": "#f7cf9a",
        "at_center": "#e6e6e6",
        "high": "#f7cf9a",
        "low": "#b3d9f2",
        "inlier": "#f5f5f5",
        "outlier": "#f4b6a0",
        "missing": "#d9d9d9",
        "infinite": "#f5e58c",
        "significance": ("#6baed6", "#bdd7e7", "#eff3ff"),
        "not_significant": "#f5f5f5",
        "effect": ("#f5f5f5", "#fee8c8", "#fdbb84", "#ef8a62"),
    },
}
_THEME_ROLES = tuple(_THEMES["default"])
_TIERED_ROLES = {"significance": 3, "effect": 4}


def available_themes() -> dict:
    """Named color themes accepted by every styler's ``theme=`` argument.

    Returns a copy of ``{theme_name: {role: color}}``. ``theme=`` also accepts
    a mapping of ``role -> color`` to override individual roles of the default
    theme, e.g. ``theme={"outlier": "#ffd54f"}``.
    """
    return {name: dict(roles) for name, roles in _THEMES.items()}


def _check_color(value: Any, role: str) -> str:
    # Colors are interpolated into CSS that ends up in exported HTML reports,
    # so refuse anything that could close the declaration or the style block.
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ch in value for ch in ";{}<>\"'")
    ):
        raise ValueError(f"Invalid CSS color for {role!r}: {value!r}")
    return value.strip()


def _resolve_theme(theme: Any, **overrides: Any) -> tuple[str, dict[str, Any]]:
    """Resolve ``theme`` (name or role mapping) plus explicit color overrides."""
    if isinstance(theme, str):
        if theme not in _THEMES:
            raise ValueError(
                f"Unknown theme {theme!r}; choose from {sorted(_THEMES)} "
                "or pass a mapping of role -> color."
            )
        name, roles = theme, dict(_THEMES[theme])
    elif isinstance(theme, Mapping):
        unknown = [k for k in theme if k not in _THEME_ROLES]
        if unknown:
            raise ValueError(
                f"Unknown theme role(s) {unknown}; valid roles: {list(_THEME_ROLES)}"
            )
        name, roles = "custom", {**_THEMES["default"], **theme}
    else:
        raise TypeError(
            f"theme must be a theme name or a mapping, got {type(theme).__name__}"
        )

    for role, value in overrides.items():
        if value is not None:
            roles[role] = value
            name = "custom"

    for role, value in roles.items():
        if role in _TIERED_ROLES:
            tiers = tuple(value) if not isinstance(value, str) else (value,)
            if len(tiers) != _TIERED_ROLES[role]:
                raise ValueError(
                    f"Theme role {role!r} needs {_TIERED_ROLES[role]} colors, "
                    f"got {len(tiers)}"
                )
            roles[role] = tuple(_check_color(c, role) for c in tiers)
        else:
            roles[role] = _check_color(value, role)
    return name, roles


def _theme_meta(name: str, roles: Mapping[str, Any], used: Sequence[str]) -> dict:
    """The part of a theme a helper actually painted with, for metadata."""
    colors = {role: roles[role] for role in ("text", *used)}
    return {
        "name": name,
        "colors": {k: list(v) if isinstance(v, tuple) else v for k, v in colors.items()},
    }


def _cell(background: str, text: str) -> str:
    return f"background-color: {background}; color: {text}"


# ======================================================================
# Stylers  ->  Styler   (or a StyleResult with rich=True)
# ======================================================================

_STYLER_HINT = (
    "pandas Styler needs jinja2. "
    'Install it with: pip install "bunker-stats-rs[notebook]"'
)

_COLOR_CAVEAT = (
    "highlighting marks values that are unusual under the stated rule; "
    "it is not evidence that a value is an error"
)


def _blank_css(df):
    """An all-empty CSS frame shaped like `df` -- the base for axis=None styling."""
    pd = _pd()
    # object dtype on purpose: pandas 3 infers its new `str` dtype for an
    # all-string frame, and assigning a length-1 array into such a column
    # with .loc raises -- which broke every styler on a one-row DataFrame.
    return pd.DataFrame("", index=df.index, columns=df.columns, dtype=object)


def _check_stylable(df) -> None:
    """Fail early, with a fix, on frames pandas Styler cannot render.

    ``Styler.apply`` raises a bare KeyError for non-unique labels -- and only
    at *render* time, i.e. when the notebook cell displays, far from the call.
    """
    if not df.index.is_unique:
        raise ValueError(
            "Styling needs a unique index, but this DataFrame's index has "
            "duplicate labels. Use df.reset_index(drop=True) first."
        )
    if not df.columns.is_unique:
        raise ValueError(
            "Styling needs unique column labels, but this DataFrame has duplicates."
        )


def _base_styler(frame):
    try:
        return frame.style
    except (ImportError, AttributeError) as exc:
        raise ImportError(_STYLER_HINT) from exc


def _render_limit_note(frame) -> str | None:
    """Explain pandas' render cap when `frame` is too big to show in full.

    Above ``styler.render.max_elements`` cells pandas renders only the leading
    rows/columns. Highlights further down are then simply absent from the
    output, while the counts in a rich result still cover the whole frame.
    """
    pd = _pd()
    try:
        limit = int(pd.get_option("styler.render.max_elements"))
    except (KeyError, TypeError, ValueError):  # option absent in old pandas
        return None
    if frame.size <= limit:
        return None
    return (
        f"this table has {frame.size:,} cells but pandas renders at most "
        f"{limit:,} (styler.render.max_elements), so the display is truncated "
        "and highlighted cells beyond it are not shown. Style a filtered "
        "subset of rows, or raise the option"
    )


def _warn_render_limit(frame) -> None:
    """Emit the render-limit note as a UserWarning attributed to the caller."""
    note = _render_limit_note(frame)
    if note:
        _warn(note)


def _apply_css(frame, css):
    _warn_render_limit(frame)
    return _base_styler(frame).apply(lambda _: css, axis=None)


def _one_suffix(suffix: Any, add_suffix: Any, default: str) -> Any:
    """Resolve ``suffix=`` against its older spelling ``add_suffix=``.

    ``suffix`` is the keyword every helper takes; ``add_suffix`` is kept so
    existing calls keep working. Passing both with different values is a
    mistake rather than something to guess at.
    """
    if suffix is None:
        return default if add_suffix is None else add_suffix
    if add_suffix is not None and add_suffix != suffix:
        raise TypeError(
            f"got both suffix={suffix!r} and add_suffix={add_suffix!r}; "
            "they are the same option, pass only `suffix`"
        )
    return suffix


def _derived_name(column: Any, suffix: Any, name: str = "suffix") -> Any:
    if not isinstance(suffix, str) or not suffix:
        raise ValueError(
            f"`{name}` must be a non-empty string so the derived column does "
            "not overwrite its source"
        )
    return _suffixed(column, suffix)


def _normal_two_sided_pct(z: float) -> float:
    """Percent of a normal distribution beyond +/- z standard deviations."""
    return 100.0 * math.erfc(z / math.sqrt(2.0))


def _outlier_detail(
    values: np.ndarray, method: str, k: float, z_threshold: float
) -> tuple[np.ndarray, dict]:
    """Outlier mask plus the per-column numbers a rich result reports."""
    mask, lower, upper = _outlier_mask(values, method, k, z_threshold)
    x = _finite(values)
    n = int(x.size)
    params = _outlier_params(x, method) if n else {}
    with np.errstate(invalid="ignore"):
        n_low = int((mask & (values < lower)).sum())
        n_high = int((mask & (values > upper)).sum())
    detail: dict[str, Any] = {"n": n, "n_missing": int(values.size) - n}
    if method == "iqr":
        detail.update(
            q1=float(params.get("q1", np.nan)),
            q3=float(params.get("q3", np.nan)),
            iqr=float(params.get("iqr", np.nan)),
        )
    else:
        detail.update(
            center=float(params.get("center", np.nan)),
            scale=float(params.get("scale", np.nan)),
        )
    detail.update(
        lower_fence=float(lower),
        upper_fence=float(upper),
        # the names outlier_report uses for the same two numbers
        lower_bound=float(lower),
        upper_bound=float(upper),
        n_outliers=n_low + n_high,
        n_low=n_low,
        n_high=n_high,
        pct_outliers=(100.0 * (n_low + n_high) / n) if n else float("nan"),
    )
    return mask, detail


def _expected_outlier_pct(method: str, k: float, z_threshold: float) -> float:
    """Share of an exactly normal sample the rule would flag, in percent.

    For Tukey fences the quartiles sit at +/-0.6745 sigma and the IQR is
    1.349 sigma, so the fences are at +/-(0.6745 + 1.349 k) sigma.
    """
    z = (0.6745 + 1.349 * k) if method == "iqr" else z_threshold
    return _normal_two_sided_pct(z)


def outlier_style(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0,
    outlier_color: str | None = None,
    theme: Any = "default",
    rich: bool = False,
):
    """Highlight outlier cells across **many** numeric columns.

    Same detection rules as :func:`outlier_report`. Non-flagged and non-finite
    cells are left unstyled. The input frame is styled as-is: nothing is added
    to it and it is not copied.

    Parameters
    ----------
    outlier_color : str, optional
        Overrides the theme's ``outlier`` color.
    theme : str or mapping
        A name from :func:`available_themes` or a ``role -> color`` mapping.
    rich : bool
        Return an :class:`OutlierStyleResult` (``styler, frame, metadata``)
        with per-column fences and counts instead of the bare Styler.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styles the full frame; only cells in ``columns`` can be highlighted.
    """
    _check_choice(method, _OUTLIER_METHODS, "method")
    cols = _resolve_columns(df, columns)
    _check_stylable(df)
    k = _check_positive(k, "k")
    z_threshold = _check_positive(z_threshold, "z_threshold")
    theme_name, roles = _resolve_theme(theme, outlier=outlier_color)
    flagged = _cell(roles["outlier"], roles["text"])

    css = _blank_css(df)
    per_column: dict[Any, dict] = {}
    for col in cols:
        mask, detail = _outlier_detail(_as_float(df[col]), method, k, z_threshold)
        css.loc[:, col] = np.where(mask, flagged, "")
        per_column[col] = detail

    styler = _apply_css(df, css)
    if not rich:
        return styler

    warnings = []
    empty = [str(c) for c, d in per_column.items() if d["n"] == 0]
    if empty:
        warnings.append(f"no finite values in column(s): {', '.join(empty)}")
    flat = [
        str(c) for c, d in per_column.items()
        if d["n"] > 0 and d["lower_fence"] == d["upper_fence"]
    ]
    if flat:
        warnings.append(
            f"zero spread in column(s): {', '.join(flat)} — the fences collapse "
            "to a point, so every value that differs from it is flagged"
        )
    if method == "zscore":
        note = _unreachable_z_warning(
            {c: d["n"] for c, d in per_column.items()}, z_threshold
        )
        if note:
            warnings.append(note)
    meta = {
        "method": method,
        "k": k if method == "iqr" else None,
        "z_threshold": z_threshold if method != "iqr" else None,
        "columns": list(cols),
        "n_rows": int(len(df)),
        "n_outliers_total": int(sum(d["n_outliers"] for d in per_column.values())),
        "expected_pct_if_normal": _expected_outlier_pct(method, k, z_threshold),
        "per_column": per_column,
        "quantile_method": "linear interpolation" if method == "iqr" else None,
        "nan_policy": "non-finite never flagged; counted in n_missing",
        "caveat": (
            "mean/std fences are inflated by the extreme values they are meant "
            "to catch (masking); compare with method='robust_zscore'. "
            if method == "zscore" else ""
        ) + _COLOR_CAVEAT,
        "theme": _theme_meta(theme_name, roles, ["outlier"]),
        "frame_is_copy": False,
    }
    return OutlierStyleResult(styler, df, meta, warnings)


def iqr_outlier_style(
    df,
    column: Any,
    *,
    k: float = 1.5,
    outlier_color: str | None = None,
    theme: Any = "default",
    rich: bool = False,
):
    """Highlight IQR (Tukey fence) outliers in a single column.

    Thin wrapper over :func:`outlier_style` with ``method="iqr"``. With
    ``rich=True`` returns an :class:`IQROutlierStyleResult` whose metadata is
    flat: ``q1``, ``q3``, ``iqr``, ``k``, ``lower_fence``, ``upper_fence``
    (also as ``lower_bound`` / ``upper_bound``, the names
    :func:`outlier_report` uses), ``n_outliers``, ``n_low``, ``n_high``,
    ``n``, ``n_missing``.
    """
    result = outlier_style(
        df, [column], method="iqr", k=k,
        outlier_color=outlier_color, theme=theme, rich=rich,
    )
    if not rich:
        return result
    multi = result.metadata
    detail = dict(multi["per_column"][column])
    meta = {
        "source_column": column,
        "n": detail.pop("n"),
        "n_missing": detail.pop("n_missing"),
        "k": multi["k"],
        **detail,
        "expected_pct_if_normal": multi["expected_pct_if_normal"],
        "quantile_method": multi["quantile_method"],
        "nan_policy": multi["nan_policy"],
        "caveat": multi["caveat"],
        "theme": multi["theme"],
        "frame_is_copy": False,
    }
    return IQROutlierStyleResult(result.styler, result.frame, meta, result.warnings)


def corr_heatmap(
    df,
    columns: Sequence[Any] | None = None,
    *,
    method: str = "pearson",
    cmap: str = "coolwarm",
    theme: Any = "default",
    rich: bool = False,
):
    """Correlation matrix rendered as a background-gradient Styler.

    Uses :func:`correlation_report` for the numbers, so NaN handling is the
    same pairwise-complete rule. The color scale is pinned to [-1, 1], so the
    same color means the same correlation in every heatmap. Undefined cells
    (a constant column, or fewer than 3 overlapping rows) are painted with the
    theme's ``missing`` color instead of a point on the gradient.

    With ``rich=True`` returns a :class:`CorrelationHeatmapResult`; ``frame``
    is the correlation matrix and the metadata lists the strongest pairs, the
    pairwise sample sizes, and any constant / all-missing columns.

    .. note::
       ``Styler.background_gradient`` requires ``matplotlib``, which ships in
       the ``[notebook]`` extra. Use :func:`correlation_report` if you only
       need the values.
    """
    corr = correlation_report(df, columns, method=method, pvalues=False)
    theme_name, roles = _resolve_theme(theme)
    undefined = _cell(roles["missing"], roles["text"])
    matrix = corr.to_numpy(dtype=float)

    # Applied after the gradient so it wins: matplotlib maps NaN to its "bad"
    # color, which pandas renders as a solid black cell.
    css = _pd().DataFrame(
        np.where(np.isfinite(matrix), "", undefined),
        index=corr.index, columns=corr.columns,
    )
    _warn_render_limit(corr)
    styler = (
        _base_styler(corr)
        .background_gradient(cmap=cmap, vmin=-1.0, vmax=1.0)
        .apply(lambda _: css, axis=None)
    )
    if not rich:
        return styler

    cols = list(corr.columns)
    diag = _corr_diagnostics(df, cols)
    pair_n, pair_counts = diag["pair_n"], diag["pair_counts"]
    all_missing, constant, warnings = diag["all_missing"], diag["constant"], diag["warnings"]

    upper = np.triu_indices(len(cols), k=1)
    off = matrix[upper]
    defined = np.isfinite(off)
    order = np.argsort(-np.abs(off[defined]), kind="stable")[:5]
    rows_i, rows_j = upper[0][defined], upper[1][defined]
    strongest = [
        {
            "x": cols[rows_i[o]],
            "y": cols[rows_j[o]],
            "correlation": float(off[defined][o]),
            "n": int(pair_n[rows_i[o], rows_j[o]]),
        }
        for o in order
    ]

    meta = {
        "method": method,
        "columns": cols,
        "shape": tuple(corr.shape),
        "n_rows": int(len(df)),
        "nan_policy": "pairwise-complete (each pair uses rows finite in both)",
        # same text under the key this helper has always used
        "missing_policy": "pairwise-complete (each pair uses rows finite in both)",
        "pairwise_n_min": int(pair_counts.min()) if pair_counts.size else None,
        "pairwise_n_max": int(pair_counts.max()) if pair_counts.size else None,
        "n_pairs": int(off.size),
        "n_undefined_pairs": int((~defined).sum()),
        "min_correlation": float(off[defined].min()) if defined.any() else float("nan"),
        "max_offdiag_correlation": (
            float(off[defined].max()) if defined.any() else float("nan")
        ),
        "max_abs_offdiag_correlation": (
            float(np.abs(off[defined]).max()) if defined.any() else float("nan")
        ),
        "strongest_pairs": strongest,
        "constant_columns": constant,
        "all_missing_columns": all_missing,
        "caveat": (
            "color encodes the sign and size of r only, not its statistical "
            "significance; use correlation_report(pvalues=True) for tests"
        ),
        "theme": {
            "name": theme_name,
            "cmap": cmap,
            "vmin": -1.0,
            "vmax": 1.0,
            "colors": {"text": roles["text"], "missing": roles["missing"]},
        },
        "frame_is_copy": True,
    }
    return CorrelationHeatmapResult(styler, corr, meta, warnings)


def missingness_style(
    df,
    columns: Sequence[Any] | None = None,
    *,
    missing_color: str | None = None,
    infinite_color: str | None = None,
    theme: Any = "default",
    rich: bool = False,
):
    """Highlight missing and infinite cells, to see *where* the gaps are.

    The styled companion to :func:`missingness_report`, and counted by the
    same rule. Missing cells (NaN, ``None``, ``pd.NA``, ``NaT``) get the
    theme's ``missing`` color; ``+/-inf`` in numeric columns (including complex
    ones and ``object`` columns holding only numbers) get the ``infinite``
    color. ``columns=None`` covers every column, numeric or not. The input
    frame is styled as-is.

    With ``rich=True`` returns a :class:`MissingnessStyleResult` with per-column
    counts, the number of fully complete rows, and warnings for columns that
    are entirely missing.
    """
    cols = _resolve_columns(df, columns, numeric_only=False)
    _check_stylable(df)
    theme_name, roles = _resolve_theme(
        theme, missing=missing_color, infinite=infinite_color
    )
    missing_css = _cell(roles["missing"], roles["text"])
    infinite_css = _cell(roles["infinite"], roles["text"])

    n_rows = int(len(df))
    css = _blank_css(df)
    incomplete = np.zeros(n_rows, dtype=bool)
    per_column: dict[Any, dict] = {}
    for col in cols:
        series = df[col]
        is_missing, is_infinite, _ = _gap_masks(series)
        css.loc[:, col] = np.where(
            is_missing, missing_css, np.where(is_infinite, infinite_css, "")
        )
        incomplete |= is_missing | is_infinite
        per_column[col] = {
            "dtype": str(series.dtype),
            "n_missing": int(is_missing.sum()),
            "n_infinite": int(is_infinite.sum()),
            "pct_missing": (100.0 * is_missing.sum() / n_rows) if n_rows else float("nan"),
        }

    styler = _apply_css(df, css)
    if not rich:
        return styler

    n_missing = int(sum(d["n_missing"] for d in per_column.values()))
    n_infinite = int(sum(d["n_infinite"] for d in per_column.values()))
    n_cells = n_rows * len(cols)
    warnings = []
    hollow = [
        str(c) for c, d in per_column.items() if n_rows and d["n_missing"] == n_rows
    ]
    if hollow:
        warnings.append(f"entirely missing column(s): {', '.join(hollow)}")
    if n_rows and not (~incomplete).any():
        warnings.append(
            "no row is complete across the selected columns — listwise "
            "deletion would discard every row"
        )
    meta = {
        "columns": list(cols),
        "n_rows": n_rows,
        "n_cells": n_cells,
        "n_missing_cells": n_missing,
        "n_infinite_cells": n_infinite,
        "pct_missing_cells": (100.0 * n_missing / n_cells) if n_cells else float("nan"),
        "n_complete_rows": int((~incomplete).sum()),
        "n_incomplete_rows": int(incomplete.sum()),
        "per_column": per_column,
        "definition": (
            "missing = null per pandas (NaN/None/NA/NaT); infinite = +/-inf in "
            "numeric columns; a complete row has neither in the selected columns"
        ),
        "theme": _theme_meta(theme_name, roles, ["missing", "infinite"]),
        "frame_is_copy": False,
    }
    return MissingnessStyleResult(styler, df, meta, warnings)


def _tiered_css(df, column: Any, cell_css: np.ndarray, highlight_row: bool):
    """CSS frame painting one column, or whole rows, from per-row styles."""
    pd = _pd()
    if highlight_row:
        return pd.DataFrame(
            np.repeat(cell_css[:, None], df.shape[1], axis=1),
            index=df.index, columns=df.columns,
        )
    css = _blank_css(df)
    css.loc[:, column] = cell_css
    return css


def _check_result_column(df, column: Any, what: str) -> np.ndarray:
    _check_frame(df)
    if column not in df.columns:
        raise KeyError(
            f"{what} column {column!r} not in DataFrame. "
            f"Available: {list(df.columns)}"
        )
    _check_unique_labels(df, [column])
    if not _is_real_numeric(df[column]):
        raise TypeError(f"Column {column!r} is not numeric")
    _check_stylable(df)
    return _as_float(df[column])


def style_significance(
    df,
    *,
    pvalue_column: Any = "pvalue",
    alpha: float = 0.05,
    highlight_row: bool = True,
    adjusted: bool = False,
    theme: Any = "default",
    rich: bool = False,
):
    """Shade a results table by statistical significance.

    Tiers (darkest first): ``p < alpha/50``, ``p < alpha/5``, ``p < alpha``,
    then non-significant. With ``highlight_row=True`` the whole row is shaded;
    otherwise only the p-value cell is.

    NaN p-values are left unstyled, so a table mixing computable and
    non-computable rows renders without error. P-values outside [0, 1] raise:
    that is almost always the wrong column.

    .. warning::
       The p-values are shaded exactly as given -- **no multiple-comparison
       correction is applied**. With ``m`` true nulls you expect about
       ``alpha * m`` rows shaded as significant by chance. Adjust first with
       :func:`bunker_stats.p_adjust` and style the adjusted column. That
       function rejects NaN, so pass it the finite p-values only::

           p = df["pvalue"].to_numpy(dtype=float)
           ok = np.isfinite(p)
           df["p_adj"] = np.nan
           df.loc[ok, "p_adj"] = bs.p_adjust(p[ok], "holm")

       Then pass ``adjusted=True`` to say the column is already corrected;
       the "no correction" warning and the expected-false-positive count are
       dropped. It is a declaration only: the values are shaded as given
       either way.

    With ``rich=True`` returns a :class:`SignificanceStyleResult` whose
    metadata reports the number of tests, per-tier counts and that expected
    false-positive count.
    """
    p = _check_result_column(df, pvalue_column, "p-value")
    # np.number covers numpy scalars such as float32, which are not `float`.
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float, np.number))
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    alpha = float(alpha)
    finite = np.isfinite(p)
    with np.errstate(invalid="ignore"):
        out_of_range = finite & ((p < 0.0) | (p > 1.0))
    if out_of_range.any():
        raise ValueError(
            f"Column {pvalue_column!r} has {int(out_of_range.sum())} value(s) "
            "outside [0, 1]; it does not look like a p-value column."
        )

    theme_name, roles = _resolve_theme(theme)
    cutoffs = (alpha / 50.0, alpha / 5.0, alpha)
    with np.errstate(invalid="ignore"):
        conditions = [finite & (p < c) for c in cutoffs] + [finite]
    styles = [_cell(c, roles["text"]) for c in roles["significance"]]
    styles.append(_cell(roles["not_significant"], roles["text"]))
    cell_css = np.select(conditions, styles, default="")

    styler = _apply_css(df, _tiered_css(df, pvalue_column, cell_css, highlight_row))
    if not rich:
        return styler

    n_tests = int(finite.sum())
    tier = np.select(conditions, [0, 1, 2, 3], default=-1)
    n_significant = int((tier >= 0).sum() - (tier == 3).sum())
    warnings = []
    if n_tests > 1 and not adjusted:
        warnings.append(
            f"{n_tests} tests shaded with no multiple-comparison correction; "
            f"about {alpha * n_tests:.3g} would fall below alpha={alpha:g} by "
            "chance if every null were true — consider bunker_stats.p_adjust"
        )
    meta = {
        "pvalue_column": pvalue_column,
        "alpha": float(alpha),
        "tier_cutoffs": [float(c) for c in cutoffs],
        "n_tests": n_tests,
        "n_missing": int(p.size) - n_tests,
        "n_significant": n_significant,
        "n_per_tier": {
            f"p < {cutoffs[0]:g}": int((tier == 0).sum()),
            f"p < {cutoffs[1]:g}": int((tier == 1).sum()),
            f"p < {cutoffs[2]:g}": int((tier == 2).sum()),
            "not significant": int((tier == 3).sum()),
        },
        "multiple_comparison_correction": (
            "declared already adjusted by the caller (adjusted=True)"
            if adjusted else "none (p-values shaded as given)"
        ),
        "expected_false_positives_if_all_null": (
            None if adjusted else float(alpha * n_tests)
        ),
        "highlight_row": bool(highlight_row),
        "caveat": (
            "shading reflects p-value thresholds only; a small p-value says "
            "nothing about the size or importance of an effect"
        ),
        "theme": _theme_meta(theme_name, roles, ["significance", "not_significant"]),
        "frame_is_copy": False,
    }
    return SignificanceStyleResult(styler, df, meta, warnings)


_EFFECT_LABELS = ("negligible", "small", "medium", "large")


def style_effect_size(
    df,
    effect_column: Any,
    *,
    thresholds: Sequence[float] = (0.2, 0.5, 0.8),
    highlight_row: bool = False,
    theme: Any = "default",
    rich: bool = False,
):
    """Shade an effect-size column by magnitude.

    Buckets ``|effect|`` against ``thresholds`` (default Cohen's small/medium/
    large: 0.2 / 0.5 / 0.8) into negligible, small, medium, large. Pass your own
    ascending thresholds for other measures -- e.g. ``(0.147, 0.33, 0.474)`` for
    Cliff's delta, or ``(0.1, 0.3, 0.5)`` for rank-biserial.

    NaN effects are left unstyled. The default cutoffs are Cohen's conventions
    for standardized mean differences; they are not universal and what counts
    as "large" depends on the field.

    With ``rich=True`` returns an :class:`EffectSizeStyleResult` with the count
    in each bucket.
    """
    effect = _check_result_column(df, effect_column, "Effect")

    if isinstance(thresholds, str) or not isinstance(thresholds, Iterable):
        raise TypeError("`thresholds` must be a sequence of 3 numbers")
    cuts = [_check_number(t, "each threshold") for t in thresholds]
    if len(cuts) != 3:
        raise ValueError(f"`thresholds` must contain exactly 3 values, got {len(cuts)}")
    # The effect is compared by magnitude, so a threshold at or below zero (or
    # an infinite one) leaves a bucket that nothing can fall into.
    if not all(math.isfinite(c) and c > 0 for c in cuts):
        raise ValueError(f"`thresholds` must be finite and > 0, got {cuts}")
    if not all(a < b for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"`thresholds` must be strictly ascending, got {cuts}")

    theme_name, roles = _resolve_theme(theme)
    finite = np.isfinite(effect)
    with np.errstate(invalid="ignore"):
        magnitude = np.abs(effect)
        conditions = [finite & (magnitude < c) for c in cuts] + [finite]
    styles = [_cell(c, roles["text"]) for c in roles["effect"]]
    cell_css = np.select(conditions, styles, default="")

    styler = _apply_css(df, _tiered_css(df, effect_column, cell_css, highlight_row))
    if not rich:
        return styler

    bucket = np.select(conditions, [0, 1, 2, 3], default=-1)
    meta = {
        "effect_column": effect_column,
        "thresholds": cuts,
        "n": int(finite.sum()),
        "n_missing": int(effect.size - finite.sum()),
        "n_per_bucket": {
            label: int((bucket == i).sum()) for i, label in enumerate(_EFFECT_LABELS)
        },
        "bucketing": "absolute value; lower bound inclusive",
        "highlight_row": bool(highlight_row),
        "caveat": (
            "negligible/small/medium/large are conventions for the chosen "
            "thresholds, not properties of the data"
        ),
        "theme": _theme_meta(theme_name, roles, ["effect"]),
        "frame_is_copy": False,
    }
    return EffectSizeStyleResult(styler, df, meta, [])


def _attach(df, target: Any, values: np.ndarray):
    """Copy `df` and set one derived column; report whether it replaced one."""
    overwrote = target in df.columns
    out = df.copy()
    out[target] = values
    return out, overwrote


def _overwrite_warning(targets: Sequence[Any]) -> str:
    return (
        f"derived column(s) {', '.join(map(str, targets))} already existed in "
        "the input and were replaced in the returned frame"
    )


def demean_style(
    df,
    column: Any,
    *,
    above_color: str | None = None,
    below_color: str | None = None,
    zero_color: str | None = None,
    suffix: str | None = None,
    add_suffix: str | None = None,
    theme: Any = "default",
    rich: bool = False,
):
    """Add a demeaned column and color it above / below / at the mean.

    Non-finite inputs are excluded from the mean and left unstyled. The input
    DataFrame is never modified: the styled frame is a copy with
    ``{column}{suffix}`` added (default ``"_demeaned"``; ``add_suffix`` is the
    older name for ``suffix``).

    With ``rich=True`` returns a :class:`DemeanStyleResult`; its metadata has
    ``mean``, ``n``, ``n_missing`` and the ``n_above`` / ``n_below`` /
    ``n_at_mean`` counts (``n_at_mean`` counts exact equality with the mean).
    """
    _resolve_columns(df, [column])
    _check_stylable(df)
    target = _derived_name(column, _one_suffix(suffix, add_suffix, "_demeaned"))
    theme_name, roles = _resolve_theme(
        theme, above=above_color, below=below_color, at_center=zero_color
    )
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

    out, overwrote = _attach(df, target, demeaned)
    if overwrote and not rich:
        _warn(_overwrite_warning([target]))

    text = roles["text"]
    with np.errstate(invalid="ignore"):
        colors = np.select(
            [signs > 0, signs < 0, signs == 0],
            [_cell(roles["above"], text), _cell(roles["below"], text),
             _cell(roles["at_center"], text)],
            default="",
        )
    css = _blank_css(out)
    css.loc[:, target] = colors
    styler = _apply_css(out, css)
    if not rich:
        return styler

    n = int(mask.sum())
    with np.errstate(invalid="ignore"):
        n_above, n_below, n_at = (
            int((signs > 0).sum()), int((signs < 0).sum()), int((signs == 0).sum())
        )
    warnings = []
    if n == 0:
        warnings.append(f"no finite values in column: {column}")
    if overwrote:
        warnings.append(_overwrite_warning([target]))
    meta = {
        "source_column": column,
        "derived_column": target,
        "n": n,
        "n_missing": int(values.size) - n,
        "mean": float(_safe(_bs.mean, values[mask])) if n else float("nan"),
        "n_above": n_above,
        "n_below": n_below,
        "n_at_mean": n_at,
        "nan_policy": "non-finite excluded from the mean; left NaN and unstyled",
        "caveat": (
            "the mean is not robust: a few extreme values move it, and with "
            "it which rows read as above or below"
        ),
        "theme": _theme_meta(theme_name, roles, ["above", "below", "at_center"]),
        "frame_is_copy": True,
        "overwrote_existing_column": overwrote,
    }
    return DemeanStyleResult(styler, out, meta, warnings)


def _zscore_cells(z: np.ndarray, threshold: float, roles: Mapping[str, Any]):
    """CSS per z-score plus the high/low flag masks (|z| >= threshold)."""
    text = roles["text"]
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(z)
        high = finite & (z >= threshold)
        low = finite & (z <= -threshold)
    css = np.select(
        [high, low, finite],
        [_cell(roles["high"], text), _cell(roles["low"], text),
         _cell(roles["inlier"], text)],
        default="",
    )
    return css, high, low


def zscore_style(
    df,
    column: Any,
    *,
    threshold: float = 2.0,
    high_color: str | None = None,
    low_color: str | None = None,
    zero_color: str | None = None,
    suffix: str | None = None,
    add_suffix: str | None = None,
    theme: Any = "default",
    rich: bool = False,
):
    """Add a z-score column and highlight scores beyond +/- ``threshold``.

    Z-scores use the sample mean and sample standard deviation (ddof=1).
    Non-finite inputs are excluded from both and left unstyled. The input
    DataFrame is never modified: the styled frame is a copy with
    ``{column}{suffix}`` added (default ``"_zscore"``; ``add_suffix`` is the
    older name for ``suffix``).

    With ``rich=True`` returns a :class:`ZScoreStyleResult`; its metadata has
    ``mean``, ``std``, ``threshold``, ``n_high`` / ``n_low`` / ``n_flagged``
    and, for calibration, the share a normal sample would flag.

    .. note::
       The mean and std are themselves inflated by extreme values, which can
       hide exactly the points you are looking for (masking). In a sample of
       size ``n`` no z-score can exceed ``(n - 1) / sqrt(n)``, so small samples
       cannot reach large thresholds at all. :func:`scale_compare_style` puts
       the classical and robust scores side by side.
    """
    _resolve_columns(df, [column])
    _check_stylable(df)
    threshold = _check_positive(threshold, "threshold")
    target = _derived_name(column, _one_suffix(suffix, add_suffix, "_zscore"))
    theme_name, roles = _resolve_theme(
        theme, high=high_color, low=low_color, inlier=zero_color
    )
    values = _as_float(df[column])

    z = _fit_scatter(values, _bs.zscore)
    out, overwrote = _attach(df, target, z)
    if overwrote and not rich:
        _warn(_overwrite_warning([target]))

    colors, high, low = _zscore_cells(z, threshold, roles)
    css = _blank_css(out)
    css.loc[:, target] = colors
    styler = _apply_css(out, css)
    if not rich:
        return styler

    x = _finite(values)
    n = int(x.size)
    std = float(_safe(_bs.std, x)) if n else float("nan")
    n_high, n_low = int(high.sum()), int(low.sum())
    warnings = []
    if n == 0:
        warnings.append(f"no finite values in column: {column}")
    elif n == 1 or std == 0.0:
        warnings.append(
            f"zero variance in column: {column} — z-scores are undefined (NaN)"
        )
    elif (n - 1) / math.sqrt(n) < threshold:
        warnings.append(
            f"with n = {n} no z-score can reach {threshold:g} "
            f"(maximum possible is {(n - 1) / math.sqrt(n):.3g}), so nothing "
            "can be flagged"
        )
    if overwrote:
        warnings.append(_overwrite_warning([target]))
    meta = {
        "source_column": column,
        "derived_column": target,
        "n": n,
        "n_missing": int(values.size) - n,
        "mean": float(_safe(_bs.mean, x)) if n else float("nan"),
        "std": std,
        "ddof": 1,
        "threshold": threshold,
        "n_high": n_high,
        "n_low": n_low,
        "n_flagged": n_high + n_low,
        "pct_flagged": (100.0 * (n_high + n_low) / n) if n else float("nan"),
        "expected_pct_if_normal": _normal_two_sided_pct(threshold),
        "flag_rule": "|z| >= threshold",
        "nan_policy": "non-finite excluded from mean/std; left NaN and unstyled",
        "caveat": (
            "mean and std are not robust; extreme values inflate both and can "
            "mask themselves. " + _COLOR_CAVEAT
        ),
        "theme": _theme_meta(theme_name, roles, ["high", "low", "inlier"]),
        "frame_is_copy": True,
        "overwrote_existing_column": overwrote,
    }
    return ZScoreStyleResult(styler, out, meta, warnings)


def scale_compare_style(
    df,
    column: Any,
    *,
    threshold: float = 3.0,
    scale_factor: float = _MAD_TO_SIGMA,
    suffixes: Sequence[str] | None = None,
    suffix: Sequence[str] | None = None,
    theme: Any = "default",
    rich: bool = False,
):
    """Classical and robust standard scores side by side, flagged alike.

    Adds ``{column}_zscore`` = ``(x - mean) / std`` and ``{column}_robust_z`` =
    ``(x - median) / (scale_factor * MAD)``, and highlights scores at or beyond
    +/- ``threshold`` in each. Rows flagged by the robust score but not the
    classical one are the signature of *masking*: extreme values inflating the
    mean and std enough to hide themselves.

    This helper adds two columns, so the suffix is a pair, default
    ``("_zscore", "_robust_z")``. It is accepted as ``suffix=`` like on every
    other helper, and as ``suffixes=`` (the original name, as in
    ``pandas.merge``).

    The input DataFrame is never modified. When the MAD is 0 (more than half
    the values are identical) the robust score is undefined and its column is
    all NaN, rather than the huge finite values a division by ~0 would give.

    With ``rich=True`` returns a :class:`ScaleCompareStyleResult` with both
    fits and the agreement counts (``n_flagged_both``, ``n_robust_only``,
    ``n_classical_only``).
    """
    _resolve_columns(df, [column])
    _check_stylable(df)
    threshold = _check_positive(threshold, "threshold")
    scale_factor = _check_positive(scale_factor, "scale_factor")
    if suffix is not None and suffixes is not None and tuple(suffix) != tuple(suffixes):
        raise TypeError(
            f"got both suffix={tuple(suffix)!r} and suffixes={tuple(suffixes)!r}; "
            "they are the same option, pass only `suffix`"
        )
    chosen = suffix if suffix is not None else suffixes
    if isinstance(chosen, str):
        raise TypeError(
            "this helper adds two columns, so the suffix must be a pair of "
            f"strings, got the single string {chosen!r}"
        )
    suffixes = ("_zscore", "_robust_z") if chosen is None else tuple(chosen)
    if len(suffixes) != 2 or suffixes[0] == suffixes[1]:
        raise ValueError(
            f"`suffix` must be two different strings, got {suffixes!r}"
        )
    z_col = _derived_name(column, suffixes[0], "suffixes")
    r_col = _derived_name(column, suffixes[1], "suffixes")
    theme_name, roles = _resolve_theme(theme)

    values = _as_float(df[column])
    x = _finite(values)
    n = int(x.size)
    median = float(_safe(_bs.median, x)) if n else float("nan")
    mad = float(_safe(_bs.mad, x)) if n else float("nan")
    zero_mad = n > 0 and mad == 0.0

    z = _fit_scatter(values, _bs.zscore)
    if zero_mad:
        robust = np.full(values.shape, np.nan, dtype=np.float64)
    else:
        robust = _fit_scatter(values, lambda v: _bs.robust_scale(v, scale_factor))

    overwrote = [c for c in (z_col, r_col) if c in df.columns]
    if overwrote and not rich:
        _warn(_overwrite_warning(overwrote))
    out = df.copy()
    out[z_col] = z
    out[r_col] = robust

    z_css, z_high, z_low = _zscore_cells(z, threshold, roles)
    r_css, r_high, r_low = _zscore_cells(robust, threshold, roles)
    css = _blank_css(out)
    css.loc[:, z_col] = z_css
    css.loc[:, r_col] = r_css
    styler = _apply_css(out, css)
    if not rich:
        return styler

    classical, robust_flag = z_high | z_low, r_high | r_low
    n_robust_only = int((robust_flag & ~classical).sum())
    std = float(_safe(_bs.std, x)) if n else float("nan")
    warnings = []
    if n == 0:
        warnings.append(f"no finite values in column: {column}")
    if zero_mad:
        warnings.append(
            f"MAD is 0 in column: {column} (more than half the values equal "
            "the median) — the robust score is undefined and left NaN"
        )
    if n > 1 and std == 0.0:
        warnings.append(
            f"zero variance in column: {column} — z-scores are undefined (NaN)"
        )
    if n_robust_only:
        warnings.append(
            f"{n_robust_only} value(s) flagged by the robust score but not the "
            "classical z-score — consistent with masking by extreme values"
        )
    if overwrote:
        warnings.append(_overwrite_warning(overwrote))
    meta = {
        "source_column": column,
        "derived_columns": {"zscore": z_col, "robust_z": r_col},
        "n": n,
        "n_missing": int(values.size) - n,
        "threshold": threshold,
        "mean": float(_safe(_bs.mean, x)) if n else float("nan"),
        "std": std,
        "median": median,
        "mad": mad,
        "scale_factor": scale_factor,
        "robust_scale": scale_factor * mad if n else float("nan"),
        "zero_mad": bool(zero_mad),
        "n_flagged_classical": int(classical.sum()),
        "n_flagged_robust": int(robust_flag.sum()),
        "n_flagged_both": int((classical & robust_flag).sum()),
        "n_robust_only": n_robust_only,
        "n_classical_only": int((classical & ~robust_flag).sum()),
        "flag_rule": "|score| >= threshold",
        "nan_policy": "non-finite excluded from both fits; left NaN and unstyled",
        "caveat": (
            "the two scores share a scale only for roughly normal data "
            "(scale_factor 1.4826 makes MAD estimate sigma under normality); "
            "for skewed data the robust score flags the long tail by design"
        ),
        "theme": _theme_meta(theme_name, roles, ["high", "low", "inlier"]),
        "frame_is_copy": True,
        "overwrote_existing_column": bool(overwrote),
    }
    return ScaleCompareStyleResult(styler, out, meta, warnings)


# ======================================================================
# Style / transform result objects  (returned with rich=True)
# ======================================================================

def _json_key(key: Any) -> Any:
    """A dict key ``json.dumps`` accepts (it takes str, int, float, bool, None)."""
    if key is None or isinstance(key, (str, bool)):
        return key
    if isinstance(key, (int, np.integer)):
        return int(key)
    if isinstance(key, (float, np.floating)) and math.isfinite(key):
        return float(key)
    if isinstance(key, tuple):  # a MultiIndex label
        return str(tuple(_jsonable(k) for k in key))
    return str(_jsonable(key))


def _jsonable(value: Any) -> Any:
    """Strict-JSON copy of a value: what ``json.dumps(..., allow_nan=False)``
    accepts.

    numpy scalars and arrays become Python numbers and lists; NaN, +/-inf,
    ``pd.NA`` and ``NaT`` become ``None`` (JSON has no spelling for them);
    timestamps and dates become ISO strings; tuple keys become strings.
    """
    if isinstance(value, Mapping):
        return {_json_key(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    pd = sys.modules.get("pandas")
    if pd is not None and (value is pd.NA or value is pd.NaT):
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return str(value)


def _fmt_meta(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={_fmt_meta(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_meta(v) for v in value)
    return str(value)


def _meta_lines(meta: Mapping[Any, Any], indent: int = 0) -> list[str]:
    """Render metadata for ``info()``: nested mappings indent, lists of
    mappings (e.g. ``strongest_pairs``) get one line per entry."""
    pad = " " * indent
    width = max((len(str(k)) for k in meta), default=0) + 3
    lines = []
    for key, value in meta.items():
        label = f"{key}:"
        if isinstance(value, Mapping) and any(isinstance(v, Mapping) for v in value.values()):
            lines.append(f"{pad}{label}")
            lines += _meta_lines(value, indent + 2)
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(v, Mapping) for v in value)
        ):
            lines.append(f"{pad}{label}")
            lines += [f"{pad}  {_fmt_meta(v)}" for v in value]
        else:
            lines.append(f"{pad}{label:<{width}}{_fmt_meta(value)}")
    return lines


class StyleResult:
    """A styled table together with the numbers behind the styling.

    Returned by the ``*_style`` / ``style_*`` helpers and :func:`corr_heatmap`
    when called with ``rich=True`` (the default return stays a bare Styler).

    Attributes
    ----------
    styler : pandas.io.formats.style.Styler
        What the helper returns without ``rich=True``.
    frame : pandas.DataFrame
        The table being styled. For helpers that add columns this is a copy of
        the input with the derived columns; otherwise it is the input itself.
        ``metadata["frame_is_copy"]`` says which.
    metadata : dict
        Sample sizes, fitted values, thresholds, counts and the colors used --
        everything needed to describe or reproduce the highlighting in prose.

    Unpacks as ``styler, frame, metadata = result``. Displays as the styled
    table in Jupyter.
    """

    _title = "Style Result"
    _fields: tuple[str, ...] = ("styler", "frame", "metadata")

    def __init__(self, styler, frame, metadata: Mapping[str, Any],
                 warnings: Sequence[str] | None = None):
        self._styler = styler
        self.frame = frame
        self.metadata = dict(metadata)
        self._warnings = list(warnings or [])
        # Counts in `metadata` describe the whole frame; say so when the
        # rendered table cannot show all of it.
        note = _render_limit_note(frame) if styler is not None else None
        if note and note not in self._warnings:
            self._warnings.append(note)

    @property
    def styler(self):
        return self._styler

    @property
    def meta(self) -> dict:
        """Alias of ``metadata`` (the name the ``*_report`` objects use)."""
        return self.metadata

    @property
    def data(self):
        """Alias of ``frame`` (the name the ``*_report`` objects use)."""
        return self.frame

    def to_frame(self):
        """Return a copy of the table."""
        return self.frame.copy()

    @property
    def warnings(self) -> list:
        """Misuse-prevention warnings attached to this result (may be empty)."""
        return list(self._warnings)

    def add_warning(self, message: str) -> None:
        """Attach a warning shown by ``info()`` and included in ``to_dict()``."""
        self._warnings.append(message)

    # -- tuple-like access ------------------------------------------------
    def __iter__(self):
        return (getattr(self, name) for name in self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, index):
        return [getattr(self, name) for name in self._fields][index]

    # -- serialization ----------------------------------------------------
    def to_dict(self, *, frame: bool = False) -> dict:
        """Stable dict schema: ``{"title", "metadata", "warnings"}``.

        The result is strict JSON: plain Python values, ``None`` for NaN and
        +/-inf, ISO strings for dates, and string keys for tuple (MultiIndex)
        labels -- so it goes straight into ``json.dumps(..., allow_nan=False)``.
        Use ``.metadata`` when you need the original values. The table itself
        is left out by default because it can be the whole input frame; pass
        ``frame=True`` to add it as ``DataFrame.to_dict("list")``.
        """
        out = {
            "title": self._title,
            "metadata": _jsonable(self.metadata),
            "warnings": list(self._warnings),
        }
        if frame:
            out["frame"] = _jsonable(self.frame.to_dict(orient="list"))
        return out

    def to_json(self, *, frame: bool = False, **kwargs: Any) -> str:
        """``to_dict()`` serialized with :func:`json.dumps` (strict JSON)."""
        import json

        return json.dumps(self.to_dict(frame=frame), allow_nan=False, **kwargs)

    def info(self) -> str:
        """Pretty, multi-line human summary of the metadata and warnings."""
        lines = [self._title, "=" * 60]
        lines += _meta_lines(self.metadata)
        if self._warnings:
            lines.append("")
            lines += [f"Warning: {w}" for w in self._warnings]
        return "\n".join(lines)

    def _repr_html_(self):
        return self.styler._repr_html_()

    def __repr__(self) -> str:
        scalars = [
            f"{k}={_fmt_meta(v)}"
            for k, v in self.metadata.items()
            # counts and fitted values only; prose fields belong in info()
            if not isinstance(v, (Mapping, list, tuple))
            and not (isinstance(v, str) and len(v) > 24)
        ]
        return f"{type(self).__name__}({', '.join(scalars[:6])})"


class OutlierStyleResult(StyleResult):
    """Rich result of :func:`outlier_style` (per-column fences and counts)."""

    _title = "Outlier Style"


class IQROutlierStyleResult(OutlierStyleResult):
    """Rich result of :func:`iqr_outlier_style` (flat, single-column metadata)."""

    _title = "IQR Outlier Style"


class DemeanStyleResult(StyleResult):
    """Rich result of :func:`demean_style`."""

    _title = "Demean Style"


class ZScoreStyleResult(StyleResult):
    """Rich result of :func:`zscore_style`."""

    _title = "Z-Score Style"


class ScaleCompareStyleResult(StyleResult):
    """Rich result of :func:`scale_compare_style`."""

    _title = "Classical vs Robust Scores"


class CorrelationHeatmapResult(StyleResult):
    """Rich result of :func:`corr_heatmap`; ``frame`` is the correlation matrix."""

    _title = "Correlation Heatmap"


class MissingnessStyleResult(StyleResult):
    """Rich result of :func:`missingness_style`."""

    _title = "Missingness Style"


class SignificanceStyleResult(StyleResult):
    """Rich result of :func:`style_significance`."""

    _title = "Significance Style"


class EffectSizeStyleResult(StyleResult):
    """Rich result of :func:`style_effect_size`."""

    _title = "Effect Size Style"


class TransformResult(StyleResult):
    """A transformed frame together with the fitted parameters.

    Returned by :func:`scale_columns` / :func:`winsorize_columns` with
    ``rich=True``. Unpacks as ``frame, metadata = result``. ``.styler`` is a
    plain Styler over the frame, built on first access (it needs jinja2; the
    rest of the object does not).
    """

    _title = "Column Transform"
    _fields = ("frame", "metadata")

    def __init__(self, frame, metadata: Mapping[str, Any],
                 warnings: Sequence[str] | None = None):
        super().__init__(None, frame, metadata, warnings)

    @property
    def styler(self):
        if self._styler is None:
            self._styler = _base_styler(self.frame)
        return self._styler

    def _repr_html_(self):
        return self.frame._repr_html_()


class RobustScaleResult(TransformResult):
    """Rich result of :func:`robust_scale_column` (flat, single-column metadata)."""

    _title = "Robust Scale"


# ======================================================================
# Report objects  (returned by *_report(..., rich=True))
# ======================================================================

class NotebookReport:
    """Base class for rich notebook reports.

    Wraps the report DataFrame (``.data``) together with ``.meta`` — the
    method context an analyst needs to interpret or reproduce the numbers
    (method names, parameters, NaN policy, seeds). Reports are returned by the
    ``*_report`` helpers when called with ``rich=True``; the default return
    stays a plain DataFrame.

    All reports support ``.to_frame()``, ``.to_dict()``, ``.to_json()``,
    ``.style()``, ``.info()`` and render as a table in Jupyter via
    ``_repr_html_``. A report unpacks as ``data, meta = report``, and answers
    to the names the style results use as well (``.frame``, ``.metadata``).
    """

    _title = "Report"
    _fields: tuple[str, ...] = ("data", "meta")

    def __init__(self, data, meta: dict, warnings: Sequence[str] | None = None):
        self._data = data
        self.meta = dict(meta)
        self._warnings = list(warnings or [])

    # -- tuple-like access, as on StyleResult -------------------------------
    def __iter__(self):
        return (getattr(self, name) for name in self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, index):
        return [getattr(self, name) for name in self._fields][index]

    @property
    def frame(self):
        """Alias of ``data`` (the name the style results use)."""
        return self._data

    @property
    def metadata(self) -> dict:
        """Alias of ``meta`` (the name the style results use)."""
        return self.meta

    @property
    def warnings(self) -> list:
        """Misuse-prevention warnings attached to this report (may be empty)."""
        return list(self._warnings)

    def add_warning(self, message: str) -> None:
        """Attach a warning shown by ``info()`` and included in ``to_dict()``."""
        self._warnings.append(message)

    @property
    def data(self):
        """The underlying report DataFrame (not a copy)."""
        return self._data

    def to_frame(self):
        """Return a copy of the report DataFrame."""
        return self._data.copy()

    def to_dict(self) -> dict:
        """Stable dict schema: ``{"title", "data", "meta", "warnings"}``.

        ``data`` is ``DataFrame.to_dict(orient="index")`` — one entry per
        report row, keyed by the row label. The result is strict JSON: plain
        Python values, ``None`` for NaN and +/-inf, ISO strings for dates and
        string keys for tuple (MultiIndex) labels, so it goes straight into
        ``json.dumps(..., allow_nan=False)``. Use ``.data`` / ``.meta`` when
        you need the original values. ``warnings`` is always present
        (possibly empty).
        """
        return {
            "title": self._title,
            "data": _jsonable(self._data.to_dict(orient="index")),
            "meta": _jsonable(self.meta),
            "warnings": list(self._warnings),
        }

    def to_json(self, **kwargs: Any) -> str:
        """``to_dict()`` serialized with :func:`json.dumps` (strict JSON)."""
        import json

        return json.dumps(self.to_dict(), allow_nan=False, **kwargs)

    def style(self):
        """A basic pandas Styler over the report table (float formatting)."""
        numeric = self._data.select_dtypes(include="number").columns
        return self._data.style.format({c: "{:.4g}" for c in numeric}, na_rep="·")

    def info(self) -> str:
        lines = [self._title, "=" * 60]
        lines.append(f"{'Rows:':<18}{len(self._data)}")
        for key, value in self.meta.items():
            lines.append(f"{key + ':':<18}{value}")
        if self._warnings:
            lines.append("")
            lines += [f"Warning: {w}" for w in self._warnings]
        return "\n".join(lines)

    def _repr_html_(self):
        return self._data._repr_html_()

    def __repr__(self) -> str:
        meta = ", ".join(f"{k}={v!r}" for k, v in list(self.meta.items())[:4])
        return f"{type(self).__name__}(rows={len(self._data)}, {meta})"


class SummaryReport(NotebookReport):
    """Rich wrapper over :func:`robust_summary`."""

    _title = "Robust Summary"


class DescribeReport(SummaryReport):
    """Rich wrapper over :func:`describe_fast`."""

    _title = "Describe"


class NormalityReport(NotebookReport):
    """Rich wrapper over :func:`normality_report`."""

    _title = "Normality Report"


class MissingnessReport(NotebookReport):
    """Rich wrapper over :func:`missingness_report`."""

    _title = "Missingness Report"


class RollingReport(NotebookReport):
    """Rich wrapper over :func:`rolling_report`; ``data`` has one row per input row."""

    _title = "Rolling Report"


class OutlierReport(NotebookReport):
    """Rich wrapper over :func:`outlier_report`, with a Plotly counts chart."""

    _title = "Outlier Report"

    def plot_counts(self):
        """Bar chart of outlier counts per column (Plotly Figure).

        Hover shows the percentage and detection method. Never calls
        ``.show()``; returns the Figure.
        """
        from ._plotly import require_go

        go = require_go()
        df = self._data
        hover = [
            f"{idx}: {int(row['n_outliers'])} outliers "
            f"({row['pct_outliers']:.2f}% of {int(row['n'])}) — {row['method']}"
            if np.isfinite(row["pct_outliers"])
            else f"{idx}: no finite values"
            for idx, row in df.iterrows()
        ]
        fig = go.Figure(
            go.Bar(
                x=[str(i) for i in df.index],
                y=df["n_outliers"].to_numpy(dtype=float),
                hovertext=hover,
                hoverinfo="text",
            )
        )
        fig.update_layout(
            title=f"Outliers per column (method={self.meta.get('method')})",
            xaxis_title="column",
            yaxis_title="n_outliers",
        )
        return fig


class CorrelationReport(NotebookReport):
    """Rich wrapper over :func:`correlation_report` (matrix or long form)."""

    _title = "Correlation Report"

    def style(self):
        """Gradient-styled table (requires matplotlib via pandas Styler).

        The [-1, 1] color scale applies to correlations only. In the long
        form that is the ``correlation`` column; ``n``, ``statistic`` and
        ``pvalue`` are formatted but not colored, since a gradient pinned to
        [-1, 1] says nothing about them.
        """
        if self.meta.get("form") == "long":
            return self._data.style.background_gradient(
                cmap="coolwarm", vmin=-1.0, vmax=1.0, subset=["correlation"]
            ).format(
                {"correlation": "{:.4g}", "statistic": "{:.4g}", "pvalue": "{:.3g}"},
                na_rep="·",
            )
        return self._data.style.background_gradient(
            cmap="coolwarm", vmin=-1.0, vmax=1.0
        )

    def plot_heatmap(self):
        """Correlation matrix heatmap (Plotly Figure).

        Hover shows the variable pair and the correlation value; NaN cells
        (insufficient pairwise overlap) render as gaps. Only available for the
        matrix form (``pvalues=False``).
        """
        from ._plotly import require_go

        if self.meta.get("form") == "long":
            raise ValueError(
                "plot_heatmap() needs the matrix form; call "
                "correlation_report(df, pvalues=False, rich=True)"
            )
        go = require_go()
        df = self._data
        labels = [str(c) for c in df.columns]
        z = df.to_numpy(dtype=float)
        hover = [
            [
                f"{labels[i]} vs {labels[j]}: r = {z[i, j]:.4f}"
                if np.isfinite(z[i, j])
                else f"{labels[i]} vs {labels[j]}: insufficient overlap"
                for j in range(len(labels))
            ]
            for i in range(len(labels))
        ]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                zmin=-1.0,
                zmax=1.0,
                colorscale="RdBu",
                reversescale=True,
                hovertext=hover,
                hoverinfo="text",
            )
        )
        fig.update_layout(
            title=f"Correlation ({self.meta.get('method', 'pearson')})",
            yaxis_autorange="reversed",
        )
        return fig


class BootstrapCIReport(NotebookReport):
    """Rich wrapper over :func:`bootstrap_ci_report`."""

    _title = "Bootstrap CI Report"

    def plot_intervals(self):
        """Point estimates with CI error bars per column (Plotly Figure)."""
        from ._plotly import require_go

        go = require_go()
        df = self._data
        est = df["estimate"].to_numpy(dtype=float)
        lo = df["ci_lower"].to_numpy(dtype=float)
        hi = df["ci_upper"].to_numpy(dtype=float)
        labels = [str(i) for i in df.index]
        hover = [
            f"{labels[i]}: {est[i]:.4g} [{lo[i]:.4g}, {hi[i]:.4g}] "
            f"({self.meta.get('stat')}, conf={self.meta.get('conf')}, "
            f"n_resamples={self.meta.get('n_resamples')})"
            for i in range(len(labels))
        ]
        fig = go.Figure(
            go.Scatter(
                x=labels,
                y=est,
                mode="markers",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": hi - est,
                    "arrayminus": est - lo,
                },
                hovertext=hover,
                hoverinfo="text",
            )
        )
        fig.update_layout(
            title=(
                f"Bootstrap {self.meta.get('stat', 'mean')} with "
                f"{int(float(self.meta.get('conf', 0.95)) * 100)}% CI"
            ),
            xaxis_title="column",
            yaxis_title="estimate",
        )
        return fig
