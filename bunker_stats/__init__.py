# bunker_stats/__init__.py

"""
Python facade for the Rust extension.

It supports two layouts:
1. Extension packaged inside the Python package  -> .bunker_stats_rs
2. Extension installed as a top-level module    -> bunker_stats_rs
"""

from __future__ import annotations

try:
    # Case 1: extension lives inside the bunker_stats package
    from .bunker_stats_rs import *  # type: ignore[attr-defined]
except ImportError:
    # Case 2: extension is installed as top-level module
    from bunker_stats_rs import *  # type: ignore[attr-defined]

# Explicit export list
__all__ = [
    # basic stats
    "mean_np",
    "std_np",
    "var_np",
    "zscore_np",
    "percentile_np",
    "iqr_np",
    "mad_np",

    # NaN-aware scalar stats
    "mean_nan_np",
    "std_nan_np",
    "var_nan_np",

    # rolling stats
    "rolling_mean_np",
    "rolling_std_np",
    "rolling_zscore_np",
    "ewma_np",

    # NaN-aware rolling stats
    "rolling_mean_nan_np",
    "rolling_std_nan_np",
    "rolling_zscore_nan_np",

    # Welford + masks
    "welford_np",
    "sign_mask_np",
    "demean_with_signs_np",

    # outliers & scaling
    "iqr_outliers_np",
    "zscore_outliers_np",
    "minmax_scale_np",
    "robust_scale_np",
    "winsorize_np",
    "quantile_bins_np",

    # diffs / cumulatives / ECDF
    "diff_np",
    "pct_change_np",
    "cumsum_np",
    "cummean_np",
    "ecdf_np",

    # covariance / correlation
    "cov_np",
    "corr_np",
    "cov_matrix_np",
    "corr_matrix_np",
    "rolling_cov_np",
    "rolling_corr_np",

    # NaN-aware covariance / correlation
    "cov_nan_np",
    "corr_nan_np",
    "rolling_cov_nan_np",
    "rolling_corr_nan_np",

    # KDE
    "kde_gaussian_np",
]
