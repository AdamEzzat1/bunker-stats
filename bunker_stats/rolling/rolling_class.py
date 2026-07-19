"""Rolling window statistics with composable configuration.

This module provides a high-level Rolling class for computing rolling window
statistics with flexible configuration options.
"""

import numpy as np
from typing import Optional, Literal, Tuple, Union, Dict
from dataclasses import dataclass

# Import from Rust extension
try:
    from bunker_stats_rs import rolling_multi_np, rolling_multi_axis0_np
except ImportError:
    # Fallback for development/testing
    rolling_multi_np = None
    rolling_multi_axis0_np = None

Alignment = Literal["trailing", "centered"]
NanPolicy = Literal["propagate", "ignore", "require_min_periods"]

# Valid values for validation
VALID_ALIGNMENTS = {"trailing", "centered"}
VALID_NAN_POLICIES = {"propagate", "ignore", "require_min_periods"}


@dataclass
class RollingConfig:
    """Configuration for rolling window operations.
    
    Parameters
    ----------
    window : int
        Window size (number of observations).
    min_periods : int, optional
        Minimum number of valid (non-NaN) observations required.
        If None, defaults to window size.
    alignment : {"trailing", "centered"}, default "trailing"
        Window alignment strategy.
        - "trailing": window ends at current position
        - "centered": window is centered at current position
    nan_policy : {"propagate", "ignore", "require_min_periods"}, default "propagate"
        NaN handling policy.
        - "propagate": any NaN in window -> result is NaN
        - "ignore": skip NaNs when computing statistics
        - "require_min_periods": like "ignore" but explicit about min_periods
    
    Raises
    ------
    ValueError
        If validation fails (e.g., window < 1, min_periods > window).
    """
    
    window: int
    min_periods: Optional[int] = None
    alignment: Alignment = "trailing"
    nan_policy: NanPolicy = "propagate"
    
    def __post_init__(self):
        # Validate alignment
        if self.alignment not in VALID_ALIGNMENTS:
            raise ValueError(
                f"Invalid alignment: '{self.alignment}'. "
                f"Must be one of: {', '.join(sorted(VALID_ALIGNMENTS))}"
            )
        
        # Validate nan_policy
        if self.nan_policy not in VALID_NAN_POLICIES:
            raise ValueError(
                f"Invalid nan_policy: '{self.nan_policy}'. "
                f"Must be one of: {', '.join(sorted(VALID_NAN_POLICIES))}"
            )
        
        # Validate window
        if self.window < 1:
            raise ValueError("window must be >= 1")
        
        # Validate min_periods
        if self.min_periods is not None:
            if self.min_periods < 1:
                raise ValueError("min_periods must be >= 1 or None")
            if self.min_periods > self.window:
                raise ValueError(
                    f"min_periods ({self.min_periods}) cannot exceed "
                    f"window ({self.window})"
                )


class Rolling:
    """
    Rolling window statistics with composable configuration.
    
    This class provides a high-level interface for computing rolling window
    statistics on 1D or 2D arrays. It supports flexible window alignment,
    NaN handling policies, and efficient fused multi-statistic computation.
    
    Parameters
    ----------
    data : array_like
        Input data (1D or 2D numpy array).
    window : int
        Window size (number of observations).
    min_periods : int, optional
        Minimum number of valid (non-NaN) observations required.
        If None, defaults to window size.
    alignment : {"trailing", "centered"}, default "trailing"
        Window alignment strategy.
    nan_policy : {"propagate", "ignore", "require_min_periods"}, default "propagate"
        NaN handling policy.
    axis : int, optional
        Axis along which to compute statistics (for 2D arrays).
        Currently only axis=0 (column-wise) is supported.
    
    Examples
    --------
    Basic trailing window:
    
    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> r = Rolling(x, window=3)
    >>> r.mean()
    array([2., 3., 4.])
    
    Centered window (pandas-like output shape):
    
    >>> r = Rolling(x, window=3, alignment='centered')
    >>> r.mean()
    array([1.5, 2., 3., 4., 4.5])
    
    Ignore NaNs with custom min_periods:
    
    >>> x = np.array([1., np.nan, 3., 4., 5.])
    >>> r = Rolling(x, window=3, nan_policy='ignore', min_periods=2)
    >>> r.mean()
    array([2., 3.5, 4.])
    
    Compute multiple statistics efficiently:
    
    >>> r = Rolling(x, window=3)
    >>> result = r.aggregate("mean", "std", "max")
    >>> result["mean"]
    array([...])
    """
    
    def __init__(
        self,
        data: np.ndarray,
        window: int,
        min_periods: Optional[int] = None,
        alignment: Alignment = "trailing",
        nan_policy: NanPolicy = "propagate",
        axis: Optional[int] = None,
    ):
        self.data = np.asarray(data, dtype=np.float64)
        
        # Validate alignment and nan_policy before creating config
        # (RollingConfig will also validate, but we want clear error messages)
        if alignment not in VALID_ALIGNMENTS:
            raise ValueError(
                f"Invalid alignment: '{alignment}'. "
                f"Must be 'trailing' or 'centered'"
            )
        
        if nan_policy not in VALID_NAN_POLICIES:
            raise ValueError(
                f"Invalid nan_policy: '{nan_policy}'. "
                f"Must be 'propagate', 'ignore', or 'require_min_periods'"
            )
        
        self.config = RollingConfig(window, min_periods, alignment, nan_policy)
        self.axis = axis
        
        # Validate dimensions
        if self.data.ndim == 1:
            if axis is not None and axis != 0:
                raise ValueError("For 1D arrays, axis must be None or 0")
        elif self.data.ndim == 2:
            if axis is None:
                self.axis = 0  # Default to column-wise
            elif axis not in (0, 1):
                raise ValueError("axis must be 0 or 1 for 2D arrays")
            if axis == 1:
                raise NotImplementedError("axis=1 (row-wise) not yet implemented")
        else:
            raise ValueError("Only 1D and 2D arrays are supported")
    
    def _compute_1d(self, *stats: str) -> Tuple[np.ndarray, ...]:
        """Compute statistics for 1D array."""
        if rolling_multi_np is None:
            raise RuntimeError("Rust extension not loaded")
        
        return rolling_multi_np(
            self.data,
            window=self.config.window,
            min_periods=self.config.min_periods,
            alignment=self.config.alignment,
            nan_policy=self.config.nan_policy,
            stats=list(stats),
        )
    
    def _compute_2d(self, *stats: str) -> Tuple[np.ndarray, ...]:
        """Compute statistics for 2D array (axis=0)."""
        if rolling_multi_axis0_np is None:
            raise RuntimeError("Rust extension not loaded")
        
        return rolling_multi_axis0_np(
            self.data,
            window=self.config.window,
            min_periods=self.config.min_periods,
            alignment=self.config.alignment,
            nan_policy=self.config.nan_policy,
            stats=list(stats),
        )
    
    def _compute(self, *stats: str) -> Tuple[np.ndarray, ...]:
        """Compute multiple statistics at once."""
        if self.data.ndim == 1:
            return self._compute_1d(*stats)
        else:
            return self._compute_2d(*stats)
    
    def mean(self) -> np.ndarray:
        """Compute rolling mean.
        
        Returns
        -------
        np.ndarray
            Rolling mean values.
        """
        return self._compute("mean")[0]
    
    def std(self) -> np.ndarray:
        """Compute rolling standard deviation (sample, ddof=1).
        
        Returns
        -------
        np.ndarray
            Rolling standard deviation values.
        """
        return self._compute("std")[0]
    
    def var(self) -> np.ndarray:
        """Compute rolling variance (sample, ddof=1).
        
        Returns
        -------
        np.ndarray
            Rolling variance values.
        """
        return self._compute("var")[0]
    
    def count(self) -> np.ndarray:
        """Compute rolling valid count (non-NaN observations).
        
        Returns
        -------
        np.ndarray
            Rolling valid count values.
        """
        return self._compute("count")[0]
    
    def min(self) -> np.ndarray:
        """Compute rolling minimum.
        
        Returns
        -------
        np.ndarray
            Rolling minimum values.
        """
        return self._compute("min")[0]
    
    def max(self) -> np.ndarray:
        """Compute rolling maximum.
        
        Returns
        -------
        np.ndarray
            Rolling maximum values.
        """
        return self._compute("max")[0]
    
    def aggregate(self, *stats: str) -> Dict[str, np.ndarray]:
        """
        Compute multiple statistics efficiently.
        
        This method computes multiple statistics in a single pass through
        the data, which is more efficient than calling individual methods.
        
        Parameters
        ----------
        *stats : str
            Statistic names: "mean", "std", "var", "count", "min", "max"
        
        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping stat name to result array.
        
        Examples
        --------
        >>> r = Rolling(x, window=5)
        >>> result = r.aggregate("mean", "std", "max")
        >>> result["mean"]
        array([...])
        >>> result["std"]
        array([...])
        """
        if not stats:
            raise ValueError("At least one statistic must be specified")
        
        results = self._compute(*stats)
        return dict(zip(stats, results))
    
    @classmethod
    def trailing(
        cls,
        data: np.ndarray,
        window: int,
        min_periods: Optional[int] = None,
        nan_policy: NanPolicy = "propagate",
    ) -> "Rolling":
        """Create a Rolling instance with trailing window alignment.
        
        This is a convenience method for the most common use case.
        
        Parameters
        ----------
        data : array_like
            Input data.
        window : int
            Window size.
        min_periods : int, optional
            Minimum valid observations.
        nan_policy : {"propagate", "ignore", "require_min_periods"}
            NaN handling policy.
        
        Returns
        -------
        Rolling
            Rolling instance with trailing alignment.
        """
        return cls(data, window, min_periods, "trailing", nan_policy)
    
    @classmethod
    def centered(
        cls,
        data: np.ndarray,
        window: int,
        min_periods: Optional[int] = None,
        nan_policy: NanPolicy = "propagate",
    ) -> "Rolling":
        """Create a Rolling instance with centered window alignment.
        
        This produces pandas-like output (same length as input).
        
        Parameters
        ----------
        data : array_like
            Input data.
        window : int
            Window size.
        min_periods : int, optional
            Minimum valid observations.
        nan_policy : {"propagate", "ignore", "require_min_periods"}
            NaN handling policy.
        
        Returns
        -------
        Rolling
            Rolling instance with centered alignment.
        """
        return cls(data, window, min_periods, "centered", nan_policy)


__all__ = [
    "Rolling",
    "RollingConfig",
    "Alignment",
    "NanPolicy",
]