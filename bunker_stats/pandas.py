"""pandas integration layer (optional).

Labeled matrix results plus the full notebook helper surface, re-exported so
that ``bunker_stats.pandas`` stays a valid entry point.

- Labeled matrices: :func:`cov_df` / :func:`corr_df` return DataFrames indexed
  by the input's column names instead of bare ndarrays.
- Everything from :mod:`bunker_stats.notebook`: reports, stylers, transforms.

Usage::

    import bunker_stats.pandas as bsp
    C = bsp.corr_df(df)            # labeled DataFrame
    bsp.corr_heatmap(df)           # pandas Styler

pandas is imported lazily on first call, so importing this module without
pandas installed succeeds; the call then raises a clear ImportError pointing at
``pip install "bunker-stats-rs[notebook]"``.
"""
from __future__ import annotations

import numpy as np

import bunker_stats as _bs

from . import notebook as _nb
from .notebook import *  # noqa: F401,F403 - re-export the whole notebook surface
from .notebook import _as_float, _pd, _resolve_columns

__all__ = ["cov_df", "corr_df", *_nb.__all__]


def _numeric_frame(df, columns=None, *, skipna: bool = True):
    """Float64 matrix of the selected numeric columns, plus their labels.

    With ``skipna`` every non-finite entry becomes NaN, so that +/-inf is
    skipped along with NaN -- the same "drop non-finite" rule as the rest of
    the notebook layer (the kernels themselves skip NaN only).
    """
    cols = _resolve_columns(df, columns)
    matrix = np.column_stack([_as_float(df[c]) for c in cols])
    if skipna:
        matrix[~np.isfinite(matrix)] = np.nan
    return np.ascontiguousarray(matrix, dtype=np.float64), cols


def cov_df(df, columns=None, *, skipna: bool = True):
    """Covariance matrix of a DataFrame's numeric columns, labeled by column.

    Same kernel as :func:`bunker_stats.cov_matrix` (ddof=1, columns as
    variables); the result is a DataFrame with the column names on both axes.

    ``skipna=True`` (the default) gives pairwise-complete covariances: each
    pair uses the rows where both columns are finite, as ``DataFrame.cov()``
    and :func:`bunker_stats.notebook.correlation_report` do. ``skipna=False``
    is the strict kernel, where one NaN anywhere makes the whole matrix NaN.
    """
    pd = _pd()
    matrix, cols = _numeric_frame(df, columns, skipna=skipna)
    m = np.asarray(_bs.cov_matrix(matrix, skipna=skipna))
    return pd.DataFrame(m, index=pd.Index(cols, name="column"), columns=cols)


def corr_df(df, columns=None, *, skipna: bool = True):
    """Correlation matrix of a DataFrame's numeric columns, labeled by column.

    ``skipna=True`` (the default) gives pairwise-complete Pearson
    correlations: each pair uses the rows where both columns are finite, and
    both standard deviations come from those same rows, as in
    ``DataFrame.corr()`` and
    :func:`bunker_stats.notebook.correlation_report` (which also returns
    p-values and supports Spearman). A pair with fewer than 2 such rows is
    NaN. ``skipna=False`` is the strict :func:`bunker_stats.corr_matrix`
    kernel, where one NaN anywhere makes the whole matrix NaN.
    """
    pd = _pd()
    matrix, cols = _numeric_frame(df, columns, skipna=skipna)
    m = np.asarray(_bs.corr_matrix(matrix, skipna=skipna))
    return pd.DataFrame(m, index=pd.Index(cols, name="column"), columns=cols)
