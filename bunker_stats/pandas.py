"""pandas integration layer (optional).

This module is the ONLY part of bunker-stats that imports pandas; the core
package stays numpy-only. It provides:

- Labeled matrix results: ``cov_df`` / ``corr_df`` return DataFrames indexed by
  the input's column names instead of bare ndarrays.
- Styler helpers (re-exported from the original helpers module): visual EDA on
  DataFrames backed by the Rust kernels.

Usage::

    import bunker_stats.pandas as bsp
    C = bsp.corr_df(df)            # labeled DataFrame
    bsp.corr_heatmap(df)           # pandas Styler
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bunker_stats as _bs

# Styler helpers keep living in pandas_helpers.py; re-export them here so the
# documented namespace is `bunker_stats.pandas`.
from .pandas_helpers import (  # noqa: F401
    demean_style,
    zscore_style,
    iqr_outlier_style,
    corr_heatmap,
    robust_scale_column,
)

__all__ = [
    "cov_df",
    "corr_df",
    "demean_style",
    "zscore_style",
    "iqr_outlier_style",
    "corr_heatmap",
    "robust_scale_column",
]


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError("DataFrame has no numeric columns")
    return num


def cov_df(df: pd.DataFrame, *, skipna: bool = False) -> pd.DataFrame:
    """Covariance matrix of a DataFrame's numeric columns, labeled by column.

    Same kernel as ``bunker_stats.cov_matrix`` (ddof=1, columns as variables);
    the result is a DataFrame with the numeric column names on both axes.
    """
    num = _numeric_frame(df)
    m = np.asarray(_bs.cov_matrix(num.to_numpy(dtype=np.float64), skipna=skipna))
    return pd.DataFrame(m, index=num.columns, columns=num.columns)


def corr_df(df: pd.DataFrame, *, skipna: bool = False) -> pd.DataFrame:
    """Correlation matrix of a DataFrame's numeric columns, labeled by column.

    Same kernel as ``bunker_stats.corr_matrix``; result is a labeled DataFrame.
    """
    num = _numeric_frame(df)
    m = np.asarray(_bs.corr_matrix(num.to_numpy(dtype=np.float64), skipna=skipna))
    return pd.DataFrame(m, index=num.columns, columns=num.columns)
