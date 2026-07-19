"""Rolling window statistics module for bunker-stats v0.2.9+.

This module provides high-level Rolling class and low-level wrapper functions
for computing rolling window statistics with flexible configuration.
"""

from .rolling_class import Rolling, RollingConfig

# Low-level wrapper functions that call Rust extension directly
from bunker_stats_rs import rolling_multi_np, rolling_multi_axis0_np


def rolling_multi(
    x,
    window,
    min_periods=None,
    alignment="trailing",
    nan_policy="propagate",
    stats=None,
):
    """
    Compute multiple rolling statistics on a 1D array.
    
    This is a low-level function that directly wraps the Rust extension.
    For a higher-level interface, use the Rolling class.
    
    Parameters
    ----------
    x : array_like
        1D input array.
    window : int
        Window size.
    min_periods : int, optional
        Minimum number of valid observations required.
    alignment : {"trailing", "centered"}, default "trailing"
        Window alignment strategy.
    nan_policy : {"propagate", "ignore", "require_min_periods"}, default "propagate"
        NaN handling policy.
    stats : list of str, optional
        Statistics to compute. Default is ["mean"].
        Options: "mean", "std", "var", "count", "min", "max"
    
    Returns
    -------
    tuple of ndarray
        Tuple of result arrays, one per requested statistic.
        Order matches the order in `stats`.
    
    Examples
    --------
    >>> x = np.arange(10.0)
    >>> result = rolling_multi(x, window=3, stats=["mean", "std"])
    >>> means, stds = result
    >>> means.shape
    (8,)
    """
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("rolling_multi requires 1D array")
    
    if stats is None:
        stats = ["mean"]
    
    return rolling_multi_np(
        x,
        window=window,
        min_periods=min_periods,
        alignment=alignment,
        nan_policy=nan_policy,
        stats=stats,
    )


def rolling_multi_axis0(
    x,
    window,
    min_periods=None,
    alignment="trailing",
    nan_policy="propagate",
    stats=None,
):
    """
    Compute multiple rolling statistics along axis 0 (column-wise) of a 2D array.
    
    This is a low-level function that directly wraps the Rust extension.
    For a higher-level interface, use the Rolling class.
    
    Parameters
    ----------
    x : array_like
        2D input array.
    window : int
        Window size.
    min_periods : int, optional
        Minimum number of valid observations required.
    alignment : {"trailing", "centered"}, default "trailing"
        Window alignment strategy.
    nan_policy : {"propagate", "ignore", "require_min_periods"}, default "propagate"
        NaN handling policy.
    stats : list of str, optional
        Statistics to compute. Default is ["mean"].
        Options: "mean", "std", "var", "count", "min", "max"
    
    Returns
    -------
    tuple of ndarray
        Tuple of 2D result arrays, one per requested statistic.
        Order matches the order in `stats`.
    
    Examples
    --------
    >>> x = np.random.randn(10, 3)
    >>> result = rolling_multi_axis0(x, window=5, stats=["mean", "std"])
    >>> means, stds = result
    >>> means.shape
    (6, 3)
    """
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("rolling_multi_axis0 requires 2D array")
    
    if stats is None:
        stats = ["mean"]
    
    return rolling_multi_axis0_np(
        x,
        window=window,
        min_periods=min_periods,
        alignment=alignment,
        nan_policy=nan_policy,
        stats=stats,
    )


__all__ = [
    "Rolling",
    "RollingConfig",
    "rolling_multi",
    "rolling_multi_axis0",
]
