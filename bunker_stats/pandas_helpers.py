"""Backwards-compatibility shim for the original pandas Styler helpers.

These helpers now live in :mod:`bunker_stats.notebook`, which hardens them
(explicit NaN policy, dtype validation, multi-column support) and adds the
full report/style/transform surface. This module re-exports the original five
names so existing notebooks keep working unchanged.

New code should import from the notebook layer instead::

    from bunker_stats.notebook import outlier_style, robust_summary

Unlike the original module, importing this one does **not** import pandas;
that happens lazily on first call.
"""
from __future__ import annotations

from .notebook import (  # noqa: F401
    corr_heatmap,
    demean_style,
    iqr_outlier_style,
    robust_scale_column,
    zscore_style,
)

__all__ = [
    "demean_style",
    "zscore_style",
    "iqr_outlier_style",
    "corr_heatmap",
    "robust_scale_column",
]
