# bunker_stats/resampling/config.py
"""
Ergonomic config objects for resampling methods.

These are thin Python wrappers around the Rust kernels that provide:
- Input validation with helpful error messages
- Consistent defaults and parameter naming
- Optional NaN policy handling (pre-filter in Python)
- No performance overhead in "propagate" mode (direct passthrough)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union
import numpy as np
import warnings

# Import the Rust extension (same pattern as main __init__.py)
try:
    from bunker_stats import _rs
except ImportError:
    import importlib
    _rs = None
    for module_path in [
        "bunker_stats.bunker_stats_rs",
        "bunker_stats_rs.bunker_stats_rs",
        "bunker_stats_rs"
    ]:
        try:
            _rs = importlib.import_module(module_path)
            break
        except ImportError:
            continue
    
    if _rs is None:
        raise ImportError("Could not import Rust extension for resampling")


# ======================================================================================
# VALIDATION HELPERS
# ======================================================================================

def _validate_array(arr: np.ndarray, name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate and convert to float64 array."""
    arr = np.asarray(arr, dtype=np.float64)
    
    if arr.ndim != 1:
        raise ValueError(
            f"{name}: expected 1D array, got shape {arr.shape}. "
            f"Hint: flatten with .ravel() or select a single column."
        )
    
    if not allow_empty and arr.size == 0:
        raise ValueError(
            f"{name}: array is empty. "
            f"Hint: ensure your data is non-empty before resampling."
        )
    
    return arr


def _validate_conf(conf: float, function_name: str) -> None:
    """Validate confidence level."""
    if not (0.0 < conf < 1.0):
        raise ValueError(
            f"{function_name}: conf must be in (0, 1), got {conf}. "
            f"Hint: use conf=0.95 for a 95% confidence interval."
        )


def _validate_n_resamples(n: int, function_name: str) -> None:
    """Validate number of resamples."""
    if n < 1:
        raise ValueError(
            f"{function_name}: n_resamples must be >= 1, got {n}. "
            f"Hint: typical values are 1000-10000 for bootstrap."
        )


def _validate_n_permutations(n: int, function_name: str) -> None:
    """Validate number of permutations."""
    if n < 1:
        raise ValueError(
            f"{function_name}: n_permutations must be >= 1, got {n}. "
            f"Hint: typical values are 1000-10000 for permutation tests."
        )


def _validate_stat(stat: str, function_name: str, supported: list[str]) -> None:
    """Validate statistic name."""
    if stat not in supported:
        raise ValueError(
            f"{function_name}: stat must be one of {supported}, got '{stat}'. "
            f"Hint: currently supported statistics are: {', '.join(supported)}."
        )


def _validate_alternative(alt: str, function_name: str) -> None:
    """Validate alternative hypothesis."""
    valid = ["two-sided", "greater", "less"]
    if alt not in valid:
        raise ValueError(
            f"{function_name}: alternative must be one of {valid}, got '{alt}'. "
            f"Hint: use 'two-sided' for ≠, 'greater' for >, 'less' for <."
        )


def _validate_random_state(rs: Optional[int], function_name: str) -> Optional[int]:
    """Validate random_state parameter."""
    if rs is None:
        return None
    
    if not isinstance(rs, (int, np.integer)):
        raise TypeError(
            f"{function_name}: random_state must be int or None, got {type(rs).__name__}. "
            f"Hint: use random_state=42 for reproducible results."
        )
    
    # Convert to u64 range
    rs_int = int(rs)
    if rs_int < 0:
        # Allow negative seeds by wrapping to unsigned
        rs_int = rs_int % (2**64)
    
    return rs_int


# ======================================================================================
# NaN FILTERING HELPERS
# ======================================================================================

def _filter_nans_single(x: np.ndarray, function_name: str) -> np.ndarray:
    """
    Filter NaNs from a single array.
    
    Returns a copy with NaNs removed. Raises if result is empty.
    """
    mask = np.isfinite(x)
    n_nan = (~mask).sum()
    
    if n_nan == len(x):
        raise ValueError(
            f"{function_name}: all values are NaN after filtering. "
            f"Hint: check your input data."
        )
    
    x_clean = x[mask]
    
    if n_nan > 0:
        warnings.warn(
            f"{function_name}: removed {n_nan} NaN value(s) from input array "
            f"({100 * n_nan / len(x):.1f}% of data).",
            UserWarning,
            stacklevel=3
        )
    
    return x_clean


def _filter_nans_paired(
    x: np.ndarray, 
    y: np.ndarray, 
    function_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter NaNs from paired arrays (pairwise deletion).
    
    Removes pairs where either x[i] or y[i] is NaN.
    Returns copies with valid pairs only. Raises if result is empty.
    """
    if len(x) != len(y):
        raise ValueError(
            f"{function_name}: x and y must have same length. "
            f"Got len(x)={len(x)}, len(y)={len(y)}."
        )
    
    mask = np.isfinite(x) & np.isfinite(y)
    n_removed = (~mask).sum()
    
    if n_removed == len(x):
        raise ValueError(
            f"{function_name}: all pairs have NaN after filtering. "
            f"Hint: check your input data."
        )
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    if n_removed > 0:
        warnings.warn(
            f"{function_name}: removed {n_removed} pair(s) with NaN "
            f"({100 * n_removed / len(x):.1f}% of data).",
            UserWarning,
            stacklevel=3
        )
    
    return x_clean, y_clean


def _filter_nans_two_sample(
    x: np.ndarray,
    y: np.ndarray,
    function_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter NaNs from two independent samples.
    
    Removes NaNs within each sample independently (not pairwise).
    Returns copies with NaNs removed from each. Raises if either is empty.
    """
    mask_x = np.isfinite(x)
    mask_y = np.isfinite(y)
    
    n_nan_x = (~mask_x).sum()
    n_nan_y = (~mask_y).sum()
    
    if n_nan_x == len(x) or n_nan_y == len(y):
        raise ValueError(
            f"{function_name}: one or both samples are entirely NaN after filtering. "
            f"Hint: check your input data."
        )
    
    x_clean = x[mask_x]
    y_clean = y[mask_y]
    
    if n_nan_x > 0 or n_nan_y > 0:
        warnings.warn(
            f"{function_name}: removed {n_nan_x} NaN(s) from x, {n_nan_y} from y "
            f"({100 * n_nan_x / len(x):.1f}% and {100 * n_nan_y / len(y):.1f}% respectively).",
            UserWarning,
            stacklevel=3
        )
    
    return x_clean, y_clean


# ======================================================================================
# BOOTSTRAP CONFIG
# ======================================================================================

@dataclass
class BootstrapConfig:
    """
    Configuration for bootstrap resampling.
    
    This is a thin wrapper around Rust bootstrap functions that adds:
    - Input validation with helpful error messages
    - Consistent defaults
    - Optional NaN handling (pre-filter in Python, no kernel changes)
    
    Parameters
    ----------
    n_resamples : int, default=1000
        Number of bootstrap resamples to generate.
        Typical values: 1000-10000 for CI estimation.
    
    conf : float, default=0.95
        Confidence level for interval (0, 1).
        Examples: 0.95 for 95% CI, 0.99 for 99% CI.
    
    stat : str, default="mean"
        Statistic to compute. Supported: "mean", "median", "std".
        Note: "median" and "std" are slower than "mean".
    
    random_state : int or None, default=None
        Seed for reproducible results. If None, uses 0 (deterministic default).
        Use same seed for reproducible results across calls.
    
    nan_policy : {"propagate", "omit"}, default="propagate"
        How to handle NaNs:
        - "propagate": pass NaNs to Rust (fast, returns NaN if any NaN present)
        - "omit": filter NaNs in Python before calling Rust (slower, creates copy)
    
    parallel : bool, default=True
        Whether to use parallel execution (currently always True in Rust).
        Future: may add sequential mode for debugging.
    
    Examples
    --------
    >>> config = BootstrapConfig(n_resamples=5000, conf=0.99, random_state=42)
    >>> result = config.run(data)  # Returns (estimate, lower, upper)
    >>> 
    >>> # Equivalent shorthand:
    >>> result = config(data)
    """
    
    n_resamples: int = 1000
    conf: float = 0.95
    stat: Literal["mean", "median", "std"] = "mean"
    random_state: Optional[int] = None
    nan_policy: Literal["propagate", "omit"] = "propagate"
    parallel: bool = True  # Currently not exposed to Rust, documented for future
    
    def __post_init__(self):
        """Validate config parameters on construction."""
        _validate_n_resamples(self.n_resamples, "BootstrapConfig")
        _validate_conf(self.conf, "BootstrapConfig")
        _validate_stat(self.stat, "BootstrapConfig", ["mean", "median", "std"])
        self.random_state = _validate_random_state(self.random_state, "BootstrapConfig")
        
        if self.nan_policy not in ["propagate", "omit"]:
            raise ValueError(
                f"BootstrapConfig: nan_policy must be 'propagate' or 'omit', got '{self.nan_policy}'."
            )
    
    def run(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Run bootstrap on data.
        
        Parameters
        ----------
        x : array-like
            1D data array to bootstrap.
        
        Returns
        -------
        estimate : float
            Bootstrap estimate of the statistic (mean of bootstrap distribution).
        lower : float
            Lower confidence bound.
        upper : float
            Upper confidence bound.
        
        Raises
        ------
        ValueError
            If input validation fails or results are invalid.
        """
        # Validate and convert input
        x = _validate_array(x, "BootstrapConfig.run(x)")
        
        # Apply NaN policy
        if self.nan_policy == "omit":
            x = _filter_nans_single(x, "BootstrapConfig")
        
        # Call Rust kernel
        # Note: bootstrap_ci returns (estimate, lower, upper)
        result = _rs.bootstrap_ci(
            x,
            stat=self.stat,
            n_resamples=self.n_resamples,
            conf=self.conf,
            random_state=self.random_state
        )
        
        return result
    
    def __call__(self, x: np.ndarray) -> Tuple[float, float, float]:
        """Shorthand for .run()"""
        return self.run(x)


@dataclass
class BootstrapCorrConfig:
    """
    Configuration for bootstrap correlation with confidence interval.
    
    Parameters
    ----------
    n_resamples : int, default=1000
        Number of bootstrap resamples.
    
    conf : float, default=0.95
        Confidence level (0, 1).
    
    random_state : int or None, default=None
        Seed for reproducibility.
    
    nan_policy : {"propagate", "omit"}, default="propagate"
        How to handle NaNs:
        - "propagate": pass to Rust (returns NaN if any NaN)
        - "omit": pairwise deletion in Python (slower)
    
    parallel : bool, default=True
        Use parallel execution (not yet exposed to Rust).
    
    Examples
    --------
    >>> config = BootstrapCorrConfig(n_resamples=5000, random_state=42)
    >>> r, lower, upper = config.run(x, y)
    """
    
    n_resamples: int = 1000
    conf: float = 0.95
    random_state: Optional[int] = None
    nan_policy: Literal["propagate", "omit"] = "propagate"
    parallel: bool = True
    
    def __post_init__(self):
        _validate_n_resamples(self.n_resamples, "BootstrapCorrConfig")
        _validate_conf(self.conf, "BootstrapCorrConfig")
        self.random_state = _validate_random_state(self.random_state, "BootstrapCorrConfig")
        
        if self.nan_policy not in ["propagate", "omit"]:
            raise ValueError(
                f"BootstrapCorrConfig: nan_policy must be 'propagate' or 'omit', "
                f"got '{self.nan_policy}'."
            )
    
    def run(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute bootstrap correlation CI.
        
        Parameters
        ----------
        x, y : array-like
            Paired 1D data arrays.
        
        Returns
        -------
        corr : float
            Bootstrap estimate of correlation.
        lower : float
            Lower confidence bound.
        upper : float
            Upper confidence bound.
        """
        x = _validate_array(x, "BootstrapCorrConfig.run(x)")
        y = _validate_array(y, "BootstrapCorrConfig.run(y)")
        
        if len(x) != len(y):
            raise ValueError(
                f"BootstrapCorrConfig: x and y must have same length. "
                f"Got len(x)={len(x)}, len(y)={len(y)}."
            )
        
        # Apply NaN policy
        if self.nan_policy == "omit":
            x, y = _filter_nans_paired(x, y, "BootstrapCorrConfig")
        
        result = _rs.bootstrap_corr(
            x, y,
            n_resamples=self.n_resamples,
            conf=self.conf,
            random_state=self.random_state
        )
        
        return result
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Shorthand for .run()"""
        return self.run(x, y)


# ======================================================================================
# PERMUTATION TEST CONFIG
# ======================================================================================

@dataclass
class PermutationConfig:
    """
    Configuration for permutation tests.
    
    Parameters
    ----------
    n_permutations : int, default=1000
        Number of random permutations to generate.
        Typical values: 1000-10000 for accurate p-values.
    
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis:
        - "two-sided": test if statistic ≠ null
        - "greater": test if statistic > null
        - "less": test if statistic < null
    
    random_state : int or None, default=None
        Seed for reproducibility.
    
    nan_policy : {"propagate", "omit"}, default="propagate"
        How to handle NaNs:
        - "propagate": pass to Rust
        - "omit": pre-filter in Python (semantics depend on test type)
    
    parallel : bool, default=True
        Use parallel execution.
    
    Examples
    --------
    >>> config = PermutationConfig(n_permutations=5000, alternative="greater")
    >>> statistic, pvalue = config.run_mean_diff(group1, group2)
    """
    
    n_permutations: int = 1000
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    random_state: Optional[int] = None
    nan_policy: Literal["propagate", "omit"] = "propagate"
    parallel: bool = True
    
    def __post_init__(self):
        _validate_n_permutations(self.n_permutations, "PermutationConfig")
        _validate_alternative(self.alternative, "PermutationConfig")
        self.random_state = _validate_random_state(self.random_state, "PermutationConfig")
        
        if self.nan_policy not in ["propagate", "omit"]:
            raise ValueError(
                f"PermutationConfig: nan_policy must be 'propagate' or 'omit', "
                f"got '{self.nan_policy}'."
            )
    
    def run_corr(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Permutation test for correlation.
        
        Parameters
        ----------
        x, y : array-like
            Paired 1D arrays to test for correlation.
        
        Returns
        -------
        observed : float
            Observed correlation.
        pvalue : float
            Two-sided p-value (or one-sided if alternative != "two-sided").
        """
        x = _validate_array(x, "PermutationConfig.run_corr(x)")
        y = _validate_array(y, "PermutationConfig.run_corr(y)")
        
        if len(x) != len(y):
            raise ValueError(
                f"PermutationConfig.run_corr: x and y must have same length. "
                f"Got len(x)={len(x)}, len(y)={len(y)}."
            )
        
        # Apply NaN policy (pairwise for correlation)
        if self.nan_policy == "omit":
            x, y = _filter_nans_paired(x, y, "PermutationConfig.run_corr")
        
        result = _rs.permutation_corr_test(
            x, y,
            n_permutations=self.n_permutations,
            alternative=self.alternative,
            random_state=self.random_state
        )
        
        return result
    
    def run_mean_diff(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Permutation test for mean difference (two independent samples).
        
        Parameters
        ----------
        x, y : array-like
            Independent samples to compare.
        
        Returns
        -------
        observed_diff : float
            Observed mean(x) - mean(y).
        pvalue : float
            P-value under permutation null.
        """
        x = _validate_array(x, "PermutationConfig.run_mean_diff(x)")
        y = _validate_array(y, "PermutationConfig.run_mean_diff(y)")
        
        # Apply NaN policy (independent samples: filter each separately)
        if self.nan_policy == "omit":
            x, y = _filter_nans_two_sample(x, y, "PermutationConfig.run_mean_diff")
        
        result = _rs.permutation_mean_diff_test(
            x, y,
            n_permutations=self.n_permutations,
            alternative=self.alternative,
            random_state=self.random_state
        )
        
        return result


# ======================================================================================
# JACKKNIFE CONFIG
# ======================================================================================

@dataclass
class JackknifeConfig:
    """
    Configuration for jackknife resampling.
    
    Jackknife has no random component, so no random_state parameter.
    NaN policy is supported for consistency.
    
    Parameters
    ----------
    conf : float, default=0.95
        Confidence level for CI methods (0, 1).
    
    nan_policy : {"propagate", "omit"}, default="propagate"
        How to handle NaNs.
    
    Examples
    --------
    >>> config = JackknifeConfig(conf=0.99)
    >>> estimate, lower, upper = config.run_mean_ci(data)
    """
    
    conf: float = 0.95
    nan_policy: Literal["propagate", "omit"] = "propagate"
    
    def __post_init__(self):
        _validate_conf(self.conf, "JackknifeConfig")
        
        if self.nan_policy not in ["propagate", "omit"]:
            raise ValueError(
                f"JackknifeConfig: nan_policy must be 'propagate' or 'omit', "
                f"got '{self.nan_policy}'."
            )
    
    def run_mean(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Jackknife estimate for the mean.
        
        Returns
        -------
        estimate : float
            Jackknife estimate.
        bias : float
            Estimated bias.
        std_error : float
            Standard error.
        """
        x = _validate_array(x, "JackknifeConfig.run_mean(x)")
        
        if self.nan_policy == "omit":
            x = _filter_nans_single(x, "JackknifeConfig.run_mean")
        
        result = _rs.jackknife_mean(x)
        return result
    
    def run_mean_ci(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Jackknife estimate with percentile CI.
        
        Returns
        -------
        estimate : float
            Jackknife estimate.
        lower : float
            Lower confidence bound.
        upper : float
            Upper confidence bound.
        """
        x = _validate_array(x, "JackknifeConfig.run_mean_ci(x)")
        
        if self.nan_policy == "omit":
            x = _filter_nans_single(x, "JackknifeConfig.run_mean_ci")
        
        result = _rs.jackknife_mean_ci(x, conf=self.conf)
        return result


# ======================================================================================
# STRATIFIED BOOTSTRAP CONFIG (v0.3)
# ======================================================================================

def _filter_nans_stratified(
    x: np.ndarray,
    strata: np.ndarray,
    k: int,
    min_size: int,
    function_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter NaNs within each stratum independently.
    
    Returns clean arrays with NaN pairs removed.
    Raises if any stratum becomes too small.
    """
    mask = np.isfinite(x)
    x_clean = x[mask]
    strata_clean = strata[mask]
    
    # Check post-filter stratum sizes
    stratum_counts = np.bincount(strata_clean, minlength=k)
    n_removed_total = len(x) - len(x_clean)
    
    if n_removed_total > 0:
        warnings.warn(
            f"{function_name}: removed {n_removed_total} NaN value(s) "
            f"({100 * n_removed_total / len(x):.1f}% of data).",
            UserWarning,
            stacklevel=4
        )
    
    # Check if any stratum is now too small or empty
    small_strata = np.where(stratum_counts < min_size)[0]
    if len(small_strata) > 0:
        raise ValueError(
            f"{function_name}: after NaN filtering, {len(small_strata)} "
            f"stratum/strata have < {min_size} observations: {small_strata.tolist()}. "
            f"Post-filter counts: {stratum_counts[small_strata].tolist()}."
        )
    
    return x_clean, strata_clean


@dataclass
class StratifiedBootstrapConfig:
    """
    Stratified bootstrap preserving stratum proportions.
    
    Parameters
    ----------
    n_resamples : int, default=1000
        Number of bootstrap resamples.
    
    conf : float, default=0.95
        Confidence level for interval.
    
    stat : str, default="mean"
        Statistic to compute: "mean", "median", "std".
    
    method : str, default="percentile"
        CI method. v0.3 supports "percentile" only.
        "basic" and "bca" planned for v0.4.
    
    min_stratum_size : int, default=2
        Minimum observations per stratum after NaN filtering.
        Strata below this threshold trigger ValueError.
    
    random_state : int | None, default=None
        Seed for reproducibility.
    
    nan_policy : {"propagate", "omit"}, default="propagate"
        NaN handling (applied per-stratum).
    
    Examples
    --------
    >>> config = StratifiedBootstrapConfig(n_resamples=5000, conf=0.99)
    >>> strata = np.array([0, 0, 1, 1, 1, 2, 2])
    >>> data = np.array([1.2, 1.5, 2.1, 2.3, 2.0, 3.5, 3.8])
    >>> estimate, lower, upper = config.run(data, strata)
    
    Notes
    -----
    Stratum proportions are preserved using deterministic rounding:
    - Each stratum i gets floor(n_i / n * n_resample) samples
    - Remainder distributed to largest strata (by original size)
    - Guarantees bit-for-bit reproducibility given same seed
    """
    
    n_resamples: int = 1000
    conf: float = 0.95
    stat: Literal["mean", "median", "std"] = "mean"
    method: Literal["percentile"] = "percentile"  # v0.3: percentile only
    min_stratum_size: int = 2
    random_state: Optional[int] = None
    nan_policy: Literal["propagate", "omit"] = "propagate"
    
    def __post_init__(self):
        _validate_n_resamples(self.n_resamples, "StratifiedBootstrapConfig")
        _validate_conf(self.conf, "StratifiedBootstrapConfig")
        _validate_stat(self.stat, "StratifiedBootstrapConfig", ["mean", "median", "std"])
        
        if self.method != "percentile":
            raise ValueError(
                f"StratifiedBootstrapConfig: v0.3 only supports method='percentile', "
                f"got '{self.method}'. 'basic' and 'bca' planned for v0.4."
            )
        
        if self.min_stratum_size < 1:
            raise ValueError(
                f"StratifiedBootstrapConfig: min_stratum_size must be >= 1, "
                f"got {self.min_stratum_size}."
            )
        
        if self.nan_policy not in ["propagate", "omit"]:
            raise ValueError(
                f"StratifiedBootstrapConfig: nan_policy must be 'propagate' or 'omit', "
                f"got '{self.nan_policy}'."
            )
        
        self.random_state = _validate_random_state(self.random_state, "StratifiedBootstrapConfig")
    
    def run(
        self, 
        x: np.ndarray, 
        strata: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Run stratified bootstrap.
        
        Parameters
        ----------
        x : array-like, shape (n,)
            Data values.
        
        strata : array-like, shape (n,)
            Stratum labels (int or categorical).
            Will be mapped to contiguous 0..k-1 internally.
        
        Returns
        -------
        estimate : float
            Point estimate (mean of bootstrap distribution).
        lower : float
            Lower confidence bound.
        upper : float
            Upper confidence bound.
        
        Raises
        ------
        ValueError
            - If x and strata have different lengths
            - If any stratum has < min_stratum_size observations
            - If strata has only 1 unique value (no stratification)
            - If all values in a stratum are NaN (with nan_policy="omit")
        """
        # Validate inputs
        x = _validate_array(x, "StratifiedBootstrapConfig.run(x)")
        strata = np.asarray(strata)
        
        if len(x) != len(strata):
            raise ValueError(
                f"StratifiedBootstrapConfig.run: x and strata must have same length. "
                f"Got len(x)={len(x)}, len(strata)={len(strata)}."
            )
        
        # Map strata to contiguous 0..k-1
        unique_strata, strata_codes = np.unique(strata, return_inverse=True)
        k = len(unique_strata)
        
        if k == 1:
            raise ValueError(
                f"StratifiedBootstrapConfig.run: strata has only 1 unique value. "
                f"Use regular bootstrap instead."
            )
        
        # NaN handling per-stratum
        if self.nan_policy == "omit":
            x, strata_codes = _filter_nans_stratified(
                x, strata_codes, k, self.min_stratum_size, 
                "StratifiedBootstrapConfig.run"
            )
        
        # Check stratum sizes
        stratum_counts = np.bincount(strata_codes, minlength=k)
        small_strata = np.where(stratum_counts < self.min_stratum_size)[0]
        
        if len(small_strata) > 0:
            raise ValueError(
                f"StratifiedBootstrapConfig.run: {len(small_strata)} stratum/strata "
                f"have < {self.min_stratum_size} observations: {small_strata.tolist()}. "
                f"Counts: {stratum_counts[small_strata].tolist()}. "
                f"Hint: increase min_stratum_size or combine small strata."
            )
        
        # Convert to int64 for Rust
        strata_codes = strata_codes.astype(np.int64)
        
        # Call Rust kernel
        result = _rs.bootstrap_ci_stratified(
            x,
            strata_codes,
            stat=self.stat,
            n_resamples=self.n_resamples,
            conf=self.conf,
            random_state=self.random_state,
            min_stratum_size=self.min_stratum_size,
        )
        
        return result


# ======================================================================================
# CLUSTERED BOOTSTRAP CONFIG (v0.3)
# ======================================================================================

def _filter_nans_clustered(
    x: np.ndarray,
    clusters: np.ndarray,
    k: int,
    min_size: int,
    function_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter NaNs within each cluster independently.
    
    Returns clean arrays with NaN pairs removed.
    Raises if any cluster becomes too small.
    """
    mask = np.isfinite(x)
    x_clean = x[mask]
    clusters_clean = clusters[mask]
    
    # Check post-filter cluster sizes
    cluster_counts = np.bincount(clusters_clean, minlength=k)
    n_removed_total = len(x) - len(x_clean)
    
    if n_removed_total > 0:
        warnings.warn(
            f"{function_name}: removed {n_removed_total} NaN value(s) "
            f"({100 * n_removed_total / len(x):.1f}% of data).",
            UserWarning,
            stacklevel=4
        )
    
    # Check if any cluster is now too small or empty
    small_clusters = np.where(cluster_counts < min_size)[0]
    if len(small_clusters) > 0:
        raise ValueError(
            f"{function_name}: after NaN filtering, {len(small_clusters)} "
            f"cluster(s) have < {min_size} observations: {small_clusters.tolist()}. "
            f"Post-filter counts: {cluster_counts[small_clusters].tolist()}."
        )
    
    return x_clean, clusters_clean


@dataclass
class ClusteredBootstrapConfig:
    """
    Clustered bootstrap resampling entire clusters.
    
    Use when data has hierarchical structure where entire groups
    should be resampled together (e.g., students in schools,
    repeated measures on subjects, patients in hospitals).
    
    **Key Difference from Stratified**:
    - Stratified: Samples *within* each stratum (preserves proportions)
    - Clustered: Samples *entire clusters* with replacement
    
    Parameters
    ----------
    n_resamples : int, default=1000
        Number of bootstrap resamples.
    
    conf : float, default=0.95
        Confidence level for interval.
    
    stat : str, default="mean"
        Statistic to compute: "mean", "median", "std".
    
    method : str, default="percentile"
        CI method. v0.3 supports "percentile" only.
    
    min_cluster_size : int, default=2
        Minimum observations per cluster.
        Clusters below this threshold trigger ValueError.
    
    random_state : int | None, default=None
        Seed for reproducibility.
    
    nan_policy : {"propagate", "omit"}, default="propagate"
        NaN handling (applied per-cluster).
    
    Examples
    --------
    >>> config = ClusteredBootstrapConfig(n_resamples=5000, random_state=42)
    >>> clusters = np.array([0,0,0,1,1,1,2,2,2])  # 3 clusters, 3 obs each
    >>> data = np.array([1.0, 1.2, 1.1, 2.0, 2.1, 2.2, 3.0, 3.1, 2.9])
    >>> estimate, lower, upper = config.run(data, clusters)
    
    Notes
    -----
    Clustered bootstrap CIs are typically **wider** than standard bootstrap
    due to intra-cluster correlation. This correctly accounts for the
    hierarchical structure of the data.
    
    One bootstrap resample:
    - Draws k clusters with replacement
    - Includes ALL observations from each sampled cluster
    - Total observations may vary if cluster sizes differ
    """
    
    n_resamples: int = 1000
    conf: float = 0.95
    stat: Literal["mean", "median", "std"] = "mean"
    method: Literal["percentile"] = "percentile"  # v0.3: percentile only
    min_cluster_size: int = 2
    random_state: Optional[int] = None
    nan_policy: Literal["propagate", "omit"] = "propagate"
    
    def __post_init__(self):
        _validate_n_resamples(self.n_resamples, "ClusteredBootstrapConfig")
        _validate_conf(self.conf, "ClusteredBootstrapConfig")
        _validate_stat(self.stat, "ClusteredBootstrapConfig", ["mean", "median", "std"])
        
        if self.method != "percentile":
            raise ValueError(
                f"ClusteredBootstrapConfig: v0.3 only supports method='percentile', "
                f"got '{self.method}'. 'basic' and 'bca' planned for v0.4."
            )
        
        if self.min_cluster_size < 1:
            raise ValueError(
                f"ClusteredBootstrapConfig: min_cluster_size must be >= 1, "
                f"got {self.min_cluster_size}."
            )
        
        if self.nan_policy not in ["propagate", "omit"]:
            raise ValueError(
                f"ClusteredBootstrapConfig: nan_policy must be 'propagate' or 'omit', "
                f"got '{self.nan_policy}'."
            )
        
        self.random_state = _validate_random_state(self.random_state, "ClusteredBootstrapConfig")
    
    def run(
        self, 
        x: np.ndarray, 
        clusters: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Run clustered bootstrap.
        
        Parameters
        ----------
        x : array-like, shape (n,)
            Data values.
        
        clusters : array-like, shape (n,)
            Cluster labels (int or categorical).
            Will be mapped to contiguous 0..k-1 internally.
        
        Returns
        -------
        estimate : float
            Point estimate (mean of bootstrap distribution).
        lower : float
            Lower confidence bound.
        upper : float
            Upper confidence bound.
        
        Raises
        ------
        ValueError
            - If x and clusters have different lengths
            - If any cluster has < min_cluster_size observations
            - If only 1 unique cluster (need at least 2)
            - If all values in a cluster are NaN (with nan_policy="omit")
        """
        # Validate inputs
        x = _validate_array(x, "ClusteredBootstrapConfig.run(x)")
        clusters = np.asarray(clusters)
        
        if len(x) != len(clusters):
            raise ValueError(
                f"ClusteredBootstrapConfig.run: x and clusters must have same length. "
                f"Got len(x)={len(x)}, len(clusters)={len(clusters)}."
            )
        
        # Map clusters to contiguous 0..k-1
        unique_clusters, cluster_codes = np.unique(clusters, return_inverse=True)
        k = len(unique_clusters)
        
        if k == 1:
            raise ValueError(
                f"ClusteredBootstrapConfig.run: only 1 unique cluster found. "
                f"Clustered bootstrap requires at least 2 clusters. "
                f"Use regular bootstrap instead."
            )
        
        # NaN handling per-cluster
        if self.nan_policy == "omit":
            x, cluster_codes = _filter_nans_clustered(
                x, cluster_codes, k, self.min_cluster_size, 
                "ClusteredBootstrapConfig.run"
            )
        
        # Check cluster sizes
        cluster_counts = np.bincount(cluster_codes, minlength=k)
        small_clusters = np.where(cluster_counts < self.min_cluster_size)[0]
        
        if len(small_clusters) > 0:
            raise ValueError(
                f"ClusteredBootstrapConfig.run: {len(small_clusters)} cluster(s) "
                f"have < {self.min_cluster_size} observations: {small_clusters.tolist()}. "
                f"Counts: {cluster_counts[small_clusters].tolist()}. "
                f"Hint: reduce min_cluster_size or remove small clusters."
            )
        
        # Convert to int64 for Rust
        cluster_codes = cluster_codes.astype(np.int64)
        
        # Call Rust kernel
        result = _rs.bootstrap_ci_clustered(
            x,
            cluster_codes,
            stat=self.stat,
            n_resamples=self.n_resamples,
            conf=self.conf,
            random_state=self.random_state,
            min_cluster_size=self.min_cluster_size,
        )
        
        return result


# ======================================================================================
# CONVENIENCE FUNCTIONS (optional - for users who prefer functional API)
# ======================================================================================

def bootstrap(
    x: np.ndarray,
    *,
    stat: str = "mean",
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: Optional[int] = None,
    nan_policy: Literal["propagate", "omit"] = "propagate",
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval (convenience wrapper).
    
    Equivalent to: BootstrapConfig(...).run(x)
    
    See BootstrapConfig for parameter documentation.
    """
    config = BootstrapConfig(
        n_resamples=n_resamples,
        conf=conf,
        stat=stat,
        random_state=random_state,
        nan_policy=nan_policy,
    )
    return config.run(x)


def bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: Optional[int] = None,
    nan_policy: Literal["propagate", "omit"] = "propagate",
) -> Tuple[float, float, float]:
    """
    Bootstrap correlation CI (convenience wrapper).
    
    Equivalent to: BootstrapCorrConfig(...).run(x, y)
    """
    config = BootstrapCorrConfig(
        n_resamples=n_resamples,
        conf=conf,
        random_state=random_state,
        nan_policy=nan_policy,
    )
    return config.run(x, y)


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test: Literal["corr", "mean_diff"] = "mean_diff",
    n_permutations: int = 1000,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_state: Optional[int] = None,
    nan_policy: Literal["propagate", "omit"] = "propagate",
) -> Tuple[float, float]:
    """
    Permutation test (convenience wrapper).
    
    Parameters
    ----------
    test : {"corr", "mean_diff"}
        Which test to run:
        - "corr": correlation test (paired data)
        - "mean_diff": mean difference test (independent samples)
    
    See PermutationConfig for other parameter documentation.
    """
    config = PermutationConfig(
        n_permutations=n_permutations,
        alternative=alternative,
        random_state=random_state,
        nan_policy=nan_policy,
    )
    
    if test == "corr":
        return config.run_corr(x, y)
    elif test == "mean_diff":
        return config.run_mean_diff(x, y)
    else:
        raise ValueError(
            f"permutation_test: test must be 'corr' or 'mean_diff', got '{test}'."
        )


def jackknife(
    x: np.ndarray,
    *,
    method: Literal["mean", "mean_ci"] = "mean_ci",
    conf: float = 0.95,
    nan_policy: Literal["propagate", "omit"] = "propagate",
) -> Tuple[float, ...]:
    """
    Jackknife resampling (convenience wrapper).
    
    Parameters
    ----------
    method : {"mean", "mean_ci"}
        Which jackknife method:
        - "mean": returns (estimate, bias, std_error)
        - "mean_ci": returns (estimate, lower, upper)
    
    See JackknifeConfig for other parameter documentation.
    """
    config = JackknifeConfig(conf=conf, nan_policy=nan_policy)
    
    if method == "mean":
        return config.run_mean(x)
    elif method == "mean_ci":
        return config.run_mean_ci(x)
    else:
        raise ValueError(
            f"jackknife: method must be 'mean' or 'mean_ci', got '{method}'."
        )

def bootstrap_stratified(
    x: np.ndarray,
    strata: np.ndarray,
    *,
    stat: str = "mean",
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: Optional[int] = None,
    min_stratum_size: int = 2,
    nan_policy: Literal["propagate", "omit"] = "propagate",
) -> Tuple[float, float, float]:
    """
    Stratified bootstrap CI (convenience wrapper).
    
    Equivalent to: StratifiedBootstrapConfig(...).run(x, strata)
    
    See StratifiedBootstrapConfig for parameter documentation.
    """
    config = StratifiedBootstrapConfig(
        n_resamples=n_resamples,
        conf=conf,
        stat=stat,
        min_stratum_size=min_stratum_size,
        random_state=random_state,
        nan_policy=nan_policy,
    )
    return config.run(x, strata)


def bootstrap_clustered(
    x: np.ndarray,
    clusters: np.ndarray,
    *,
    stat: str = "mean",
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: Optional[int] = None,
    min_cluster_size: int = 2,
    nan_policy: Literal["propagate", "omit"] = "propagate",
) -> Tuple[float, float, float]:
    """
    Clustered bootstrap CI (convenience wrapper).
    
    Equivalent to: ClusteredBootstrapConfig(...).run(x, clusters)
    
    See ClusteredBootstrapConfig for parameter documentation.
    """
    config = ClusteredBootstrapConfig(
        n_resamples=n_resamples,
        conf=conf,
        stat=stat,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
        nan_policy=nan_policy,
    )
    return config.run(x, clusters)


# ======================================================================================
# WILD BOOTSTRAP OLS CONFIG (v0.3)
# ======================================================================================

@dataclass
class WildBootstrapOLSConfig:
    """
    Wild bootstrap for OLS regression with fixed regressors.
    
    Provides heteroskedasticity-robust inference by resampling
    residuals with random transformations.
    
    **Use When**:
    - Regressors are fixed by design (experimental settings)
    - Heteroskedasticity is present or suspected
    - Sample size is moderate (asymptotic theory may be unreliable)
    - You need robust standard errors
    
    **Algorithm**:
    1. Fit OLS: β̂, compute residuals ε̂ = y - Xβ̂
    2. For each bootstrap replicate:
       - Generate random weights w_i from specified distribution
       - Create y* = Xβ̂ + w_i × ε̂_i
       - Refit OLS on (X, y*) → β*
    3. Compute percentile CI from bootstrap distribution
    
    Parameters
    ----------
    n_resamples : int, default=1000
        Number of bootstrap resamples.
    
    conf : float, default=0.95
        Confidence level for intervals.
    
    variant : {"rademacher", "mammen"}, default="rademacher"
        Wild bootstrap variant:
        
        - "rademacher": w_i = ±1 with equal probability
          Simple, conservative, works well in practice
          
        - "mammen": Two-point distribution matching higher moments
          w_i = (√5+1)/2 ≈ 1.618 with prob ≈ 0.724
          w_i = (√5-1)/2 ≈ 0.618 with prob ≈ 0.276
          Better finite-sample properties, matches E[w³]=1
    
    random_state : int | None, default=None
        Seed for reproducibility.
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import WildBootstrapOLSConfig
    >>> 
    >>> # Generate data with heteroskedasticity
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> true_beta = np.array([2.0, 3.0])
    >>> errors = np.random.randn(n) * (1 + X[:, 1]**2)  # Heteroskedastic
    >>> y = X @ true_beta + errors
    >>> 
    >>> # Wild bootstrap inference
    >>> config = WildBootstrapOLSConfig(variant="mammen", random_state=42)
    >>> coef, lower, upper = config.run(X, y)
    >>> 
    >>> print("Coefficients:", coef.ravel())
    >>> print("95% CI:")
    >>> for i in range(len(coef)):
    ...     print(f"  β{i}: [{lower[i, 0]:.3f}, {upper[i, 0]:.3f}]")
    
    Notes
    -----
    Wild bootstrap is particularly useful when:
    - Standard errors from OLS are unreliable due to heteroskedasticity
    - You want robust inference without assuming error distribution
    - Sample size is too small for asymptotic approximations
    - Regressors are fixed (e.g., experimental design, time trend)
    
    For clustered standard errors, use ClusteredBootstrapConfig or wait
    for clustered wild bootstrap in v0.4+.
    
    References
    ----------
    - Liu, R.Y. (1988). Bootstrap procedures under some non-i.i.d. models.
      Annals of Statistics, 16(4), 1696-1708.
    - Mammen, E. (1993). Bootstrap and wild bootstrap for high dimensional
      linear models. Annals of Statistics, 21(1), 255-285.
    """
    
    n_resamples: int = 1000
    conf: float = 0.95
    variant: Literal["rademacher", "mammen"] = "rademacher"
    random_state: Optional[int] = None
    
    def __post_init__(self):
        _validate_n_resamples(self.n_resamples, "WildBootstrapOLSConfig")
        _validate_conf(self.conf, "WildBootstrapOLSConfig")
        
        if self.variant not in ["rademacher", "mammen"]:
            raise ValueError(
                f"WildBootstrapOLSConfig: variant must be 'rademacher' or 'mammen', "
                f"got '{self.variant}'."
            )
        
        self.random_state = _validate_random_state(self.random_state, "WildBootstrapOLSConfig")
    
    def run(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run wild bootstrap for OLS.
        
        Parameters
        ----------
        X : array-like, shape (n, p)
            Design matrix (regressors).
            
        y : array-like, shape (n,)
            Response variable.
        
        Returns
        -------
        coefficients : ndarray, shape (p, 1)
            Point estimates (mean of bootstrap distribution).
            
        lower : ndarray, shape (p, 1)
            Lower confidence bounds for each coefficient.
            
        upper : ndarray, shape (p, 1)
            Upper confidence bounds for each coefficient.
        
        Raises
        ------
        ValueError
            - If X and y have incompatible shapes
            - If X is rank deficient (singular)
            - If n < p (underdetermined system)
            - If X or y contain NaN/inf values
        """
        # Validate inputs
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(
                f"WildBootstrapOLSConfig.run: X must be 2-dimensional. "
                f"Got shape {X.shape}."
            )
        
        if y.ndim != 1:
            raise ValueError(
                f"WildBootstrapOLSConfig.run: y must be 1-dimensional. "
                f"Got shape {y.shape}."
            )
        
        n, p = X.shape
        
        if len(y) != n:
            raise ValueError(
                f"WildBootstrapOLSConfig.run: X and y must have same number of rows. "
                f"Got X: ({n}, {p}), y: ({len(y)},)."
            )
        
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "WildBootstrapOLSConfig.run: X contains NaN or inf values."
            )
        
        if not np.all(np.isfinite(y)):
            raise ValueError(
                "WildBootstrapOLSConfig.run: y contains NaN or inf values."
            )
        
        if n < p:
            raise ValueError(
                f"WildBootstrapOLSConfig.run: underdetermined system (n < p). "
                f"Got n={n}, p={p}. Need n >= p for OLS."
            )
        
        # Ensure X is contiguous (required by Rust)
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        
        # Call Rust kernel
        result = _rs.wild_bootstrap_ols(
            X,
            y,
            variant=self.variant,
            n_resamples=self.n_resamples,
            conf=self.conf,
            random_state=self.random_state,
        )
        
        return result


def wild_bootstrap_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    variant: Literal["rademacher", "mammen"] = "rademacher",
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wild bootstrap for OLS regression (convenience wrapper).
    
    Equivalent to: WildBootstrapOLSConfig(...).run(X, y)
    
    See WildBootstrapOLSConfig for parameter documentation.
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import wild_bootstrap_ols
    >>> 
    >>> # Simple linear regression
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = 2 + 3*X[:, 1] + np.random.randn(n)
    >>> 
    >>> coef, lower, upper = wild_bootstrap_ols(X, y, random_state=42)
    >>> print(f"β: {coef.ravel()}")
    >>> print(f"95% CI: [{lower.ravel()}, {upper.ravel()}]")
    """
    config = WildBootstrapOLSConfig(
        n_resamples=n_resamples,
        conf=conf,
        variant=variant,
        random_state=random_state,
    )
    return config.run(X, y)

# ======================================================================================
# BCA CORRELATION (v0.3)
# ======================================================================================

@dataclass
class BcaCorrConfig:
    """
    BCa bootstrap for Pearson correlation coefficient.
    
    Provides bias-corrected and accelerated (BCa) bootstrap confidence intervals
    for the correlation between two variables. More accurate than percentile
    bootstrap for skewed distributions or small samples.
    
    Parameters
    ----------
    n_resamples : int, default=1000
        Number of bootstrap resamples. Larger values give more stable CIs.
        Typical: 1000-5000.
    conf : float, default=0.95
        Confidence level (must be in (0, 1)). E.g., 0.95 for 95% CI.
    random_state : int or None, default=None
        Random seed for reproducibility. Use same seed for identical results.
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import BcaCorrConfig
    >>> 
    >>> # Generate correlated data
    >>> np.random.seed(42)
    >>> x = np.random.randn(100)
    >>> y = 0.7 * x + 0.3 * np.random.randn(100)
    >>> 
    >>> # Compute BCa CI for correlation
    >>> config = BcaCorrConfig(n_resamples=2000, random_state=42)
    >>> est, lower, upper, accel, bias = config.run(x, y)
    >>> print(f"r = {est:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
    >>> print(f"Acceleration: {accel:.4f}, Bias: {bias:.4f}")
    
    Notes
    -----
    Returns:
        - estimate: Pearson correlation coefficient
        - lower: Lower CI bound
        - upper: Upper CI bound
        - acceleration: BCa acceleration parameter (skewness measure)
        - bias_correction: BCa bias correction (z0)
    """
    n_resamples: int = 1000
    conf: float = 0.95
    random_state: Optional[int] = None
    
    def __post_init__(self):
        _validate_conf(self.conf, "BcaCorrConfig")
        _validate_n_resamples(self.n_resamples, "BcaCorrConfig")
    
    def run(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute BCa CI for correlation coefficient.
        
        Parameters
        ----------
        x : array-like, shape (n,)
            First variable
        y : array-like, shape (n,)
            Second variable (must have same length as x)
        
        Returns
        -------
        estimate : float
            Pearson correlation coefficient
        lower : float
            Lower confidence bound
        upper : float
            Upper confidence bound
        acceleration : float
            BCa acceleration parameter
        bias_correction : float
            BCa bias correction (z0)
        """
        x = _validate_array(x, "x")
        y = _validate_array(y, "y")
        
        if len(x) != len(y):
            raise ValueError(
                f"x and y must have same length. Got x: {len(x)}, y: {len(y)}"
            )
        
        if len(x) < 3:
            raise ValueError(
                f"BCa requires at least 3 observations, got {len(x)}. "
                f"Hint: use larger sample size for reliable bootstrap inference."
            )
        
        # Call Rust kernel
        result = _rs.bca_ci_corr(
            x,
            y,
            n_resamples=self.n_resamples,
            conf=self.conf,
            random_state=self.random_state,
        )
        
        return result


def bca_corr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, float, float]:
    """
    BCa bootstrap CI for Pearson correlation (convenience wrapper).
    
    Equivalent to: BcaCorrConfig(...).run(x, y)
    
    See BcaCorrConfig for parameter documentation.
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import bca_corr
    >>> 
    >>> # Generate correlated data
    >>> np.random.seed(42)
    >>> x = np.random.randn(50)
    >>> y = 0.5 * x + np.random.randn(50)
    >>> 
    >>> # Get BCa CI
    >>> r, lower, upper, accel, bias = bca_corr(x, y, random_state=42)
    >>> print(f"r = {r:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
    """
    config = BcaCorrConfig(
        n_resamples=n_resamples,
        conf=conf,
        random_state=random_state,
    )
    return config.run(x, y)


# ======================================================================================
# BCA MEAN DIFFERENCE (v0.3)
# ======================================================================================

@dataclass
class BcaMeanDiffConfig:
    """
    BCa bootstrap for difference of means (two-sample).
    
    Provides bias-corrected and accelerated (BCa) bootstrap confidence intervals
    for the difference between two group means: mean(x) - mean(y).
    More accurate than percentile bootstrap for skewed distributions.
    
    Parameters
    ----------
    n_resamples : int, default=1000
        Number of bootstrap resamples. Larger values give more stable CIs.
        Typical: 1000-5000.
    conf : float, default=0.95
        Confidence level (must be in (0, 1)). E.g., 0.95 for 95% CI.
    random_state : int or None, default=None
        Random seed for reproducibility. Use same seed for identical results.
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import BcaMeanDiffConfig
    >>> 
    >>> # Two groups with different means
    >>> np.random.seed(42)
    >>> group1 = np.random.randn(50) + 2.0  # mean ≈ 2
    >>> group2 = np.random.randn(60) + 0.5  # mean ≈ 0.5
    >>> 
    >>> # Compute BCa CI for mean difference
    >>> config = BcaMeanDiffConfig(n_resamples=2000, random_state=42)
    >>> est, lower, upper, accel, bias = config.run(group1, group2)
    >>> print(f"Δμ = {est:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
    >>> print(f"Acceleration: {accel:.4f}, Bias: {bias:.4f}")
    
    Notes
    -----
    - x and y can have different sample sizes (this is a two-sample test)
    - The statistic is: mean(x) - mean(y)
    - BCa accounts for bias and skewness in the bootstrap distribution
    
    Returns:
        - estimate: mean(x) - mean(y)
        - lower: Lower CI bound
        - upper: Upper CI bound
        - acceleration: BCa acceleration parameter (skewness measure)
        - bias_correction: BCa bias correction (z0)
    """
    n_resamples: int = 1000
    conf: float = 0.95
    random_state: Optional[int] = None
    
    def __post_init__(self):
        _validate_conf(self.conf, "BcaMeanDiffConfig")
        _validate_n_resamples(self.n_resamples, "BcaMeanDiffConfig")
    
    def run(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute BCa CI for mean difference.
        
        Parameters
        ----------
        x : array-like, shape (n1,)
            First group
        y : array-like, shape (n2,)
            Second group (can have different size than x)
        
        Returns
        -------
        estimate : float
            Mean difference: mean(x) - mean(y)
        lower : float
            Lower confidence bound
        upper : float
            Upper confidence bound
        acceleration : float
            BCa acceleration parameter
        bias_correction : float
            BCa bias correction (z0)
        """
        x = _validate_array(x, "x")
        y = _validate_array(y, "y")
        
        if len(x) < 2:
            raise ValueError(
                f"x must have at least 2 observations for BCa, got {len(x)}. "
                f"Hint: use larger sample size for reliable bootstrap inference."
            )
        
        if len(y) < 2:
            raise ValueError(
                f"y must have at least 2 observations for BCa, got {len(y)}. "
                f"Hint: use larger sample size for reliable bootstrap inference."
            )
        
        # Call Rust kernel
        result = _rs.bca_ci_mean_diff(
            x,
            y,
            n_resamples=self.n_resamples,
            conf=self.conf,
            random_state=self.random_state,
        )
        
        return result


def bca_mean_diff(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, float, float]:
    """
    BCa bootstrap CI for mean difference (convenience wrapper).
    
    Equivalent to: BcaMeanDiffConfig(...).run(x, y)
    
    See BcaMeanDiffConfig for parameter documentation.
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import bca_mean_diff
    >>> 
    >>> # Two groups
    >>> np.random.seed(42)
    >>> treatment = np.random.randn(40) + 1.0
    >>> control = np.random.randn(40)
    >>> 
    >>> # Get BCa CI for difference
    >>> diff, lower, upper, accel, bias = bca_mean_diff(
    ...     treatment, control, random_state=42
    ... )
    >>> print(f"Δμ = {diff:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
    """
    config = BcaMeanDiffConfig(
        n_resamples=n_resamples,
        conf=conf,
        random_state=random_state,
    )
    return config.run(x, y)

# ======================================================================================
# EXTENDED PERMUTATION TESTS (v0.3)
# ======================================================================================

@dataclass
class PermutationTestAnovaConfig:
    """
    Permutation test for one-way ANOVA (F-test).
    
    Tests whether 3+ groups have different means via permutation.
    More robust than parametric ANOVA to non-normality.
    
    Parameters
    ----------
    n_permutations : int, default=1000
        Number of permutations for p-value estimation.
        Larger values give more stable p-values.
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import PermutationTestAnovaConfig
    >>> 
    >>> # Three groups with different means
    >>> group1 = np.random.randn(30)
    >>> group2 = np.random.randn(30) + 1.0
    >>> group3 = np.random.randn(30) + 2.0
    >>> 
    >>> config = PermutationTestAnovaConfig(n_permutations=5000, random_state=42)
    >>> f_stat, p_value = config.run([group1, group2, group3])
    >>> print(f"F = {f_stat:.3f}, p = {p_value:.4f}")
    
    Notes
    -----
    - Null hypothesis: all groups have same mean
    - Alternative: at least one group differs
    - P-value is one-sided (large F = more different)
    """
    n_permutations: int = 1000
    random_state: Optional[int] = None
    
    def __post_init__(self):
        _validate_n_permutations(self.n_permutations, "PermutationTestAnovaConfig")
    
    def run(self, groups: List[np.ndarray]) -> Tuple[float, float]:
        """
        Run ANOVA permutation test.
        
        Parameters
        ----------
        groups : list of arrays
            List of 2+ groups to compare
            
        Returns
        -------
        statistic : float
            F-statistic
        p_value : float
            One-sided p-value
        """
        if len(groups) < 2:
            raise ValueError("ANOVA requires at least 2 groups")
        
        # Convert to float64 arrays
        groups = [_validate_array(g, f"group_{i}") for i, g in enumerate(groups)]
        
        # Validate
        for i, g in enumerate(groups):
            if len(g) < 2:
                raise ValueError(f"Group {i} has fewer than 2 observations")
        
        return _rs.permutation_test_anova(
            groups,
            n_permutations=self.n_permutations,
            random_state=self.random_state,
        )


@dataclass
class PermutationTestMannWhitneyConfig:
    """
    Mann-Whitney U test via permutation (rank-based, two-sample).
    
    Tests whether two groups have different distributions.
    Non-parametric alternative to t-test.
    
    Parameters
    ----------
    n_permutations : int, default=1000
        Number of permutations
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis
    random_state : int or None, default=None
        Random seed
    
    Examples
    --------
    >>> from bunker_stats.resampling import PermutationTestMannWhitneyConfig
    >>> 
    >>> group1 = np.random.randn(40)
    >>> group2 = np.random.randn(40) + 0.5
    >>> 
    >>> config = PermutationTestMannWhitneyConfig(
    ...     n_permutations=5000,
    ...     alternative="two-sided",
    ...     random_state=42
    ... )
    >>> u_stat, p_value = config.run(group1, group2)
    """
    n_permutations: int = 1000
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    random_state: Optional[int] = None
    
    def __post_init__(self):
        _validate_n_permutations(self.n_permutations, "PermutationTestMannWhitneyConfig")
        _validate_alternative(self.alternative, "PermutationTestMannWhitneyConfig")
    
    def run(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Run Mann-Whitney U test.
        
        Parameters
        ----------
        x, y : arrays
            Two groups to compare
            
        Returns
        -------
        statistic : float
            Mann-Whitney U statistic
        p_value : float
            P-value for specified alternative
        """
        x = _validate_array(x, "x")
        y = _validate_array(y, "y")
        
        return _rs.permutation_test_mann_whitney(
            x, y,
            n_permutations=self.n_permutations,
            alternative=self.alternative,
            random_state=self.random_state,
        )


@dataclass
class PermutationTestKruskalWallisConfig:
    """
    Kruskal-Wallis H test via permutation (rank-based, 3+ groups).
    
    Tests whether 3+ groups have different distributions.
    Non-parametric alternative to ANOVA.
    
    Parameters
    ----------
    n_permutations : int, default=1000
        Number of permutations
    random_state : int or None, default=None
        Random seed
    
    Examples
    --------
    >>> from bunker_stats.resampling import PermutationTestKruskalWallisConfig
    >>> 
    >>> # Three groups with different medians
    >>> group1 = np.random.randn(30)
    >>> group2 = np.random.randn(30) + 1.0
    >>> group3 = np.random.randn(30) + 2.0
    >>> 
    >>> config = PermutationTestKruskalWallisConfig(
    ...     n_permutations=5000,
    ...     random_state=42
    ... )
    >>> h_stat, p_value = config.run([group1, group2, group3])
    """
    n_permutations: int = 1000
    random_state: Optional[int] = None
    
    def __post_init__(self):
        _validate_n_permutations(self.n_permutations, "PermutationTestKruskalWallisConfig")
    
    def run(self, groups: List[np.ndarray]) -> Tuple[float, float]:
        """
        Run Kruskal-Wallis H test.
        
        Parameters
        ----------
        groups : list of arrays
            List of 2+ groups to compare
            
        Returns
        -------
        statistic : float
            Kruskal-Wallis H statistic
        p_value : float
            One-sided p-value
        """
        if len(groups) < 2:
            raise ValueError("Kruskal-Wallis requires at least 2 groups")
        
        # Convert to float64 arrays
        groups = [_validate_array(g, f"group_{i}") for i, g in enumerate(groups)]
        
        return _rs.permutation_test_kruskal_wallis(
            groups,
            n_permutations=self.n_permutations,
            random_state=self.random_state,
        )


@dataclass
class PermutationTestPartialCorrConfig:
    """
    Partial correlation test via permutation.
    
    Tests correlation between x and y while controlling for confounders z.
    
    Parameters
    ----------
    n_permutations : int, default=1000
        Number of permutations
    random_state : int or None, default=None
        Random seed
    
    Examples
    --------
    >>> from bunker_stats.resampling import PermutationTestPartialCorrConfig
    >>> 
    >>> # x and y are correlated through z
    >>> z = np.random.randn(100)
    >>> x = 0.5 * z + np.random.randn(100)
    >>> y = 0.5 * z + np.random.randn(100)
    >>> 
    >>> config = PermutationTestPartialCorrConfig(
    ...     n_permutations=5000,
    ...     random_state=42
    ... )
    >>> r, p_value = config.run(x, y, z)
    >>> print(f"Partial r = {r:.3f}, p = {p_value:.4f}")
    """
    n_permutations: int = 1000
    random_state: Optional[int] = None
    
    def __post_init__(self):
        _validate_n_permutations(self.n_permutations, "PermutationTestPartialCorrConfig")
    
    def run(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Run partial correlation test.
        
        Parameters
        ----------
        x, y : arrays, shape (n,)
            Variables to correlate
        z : array, shape (n,) or (n, p)
            Confounding variable(s) to control for
            
        Returns
        -------
        statistic : float
            Partial correlation coefficient
        p_value : float
            Two-sided p-value
        """
        x = _validate_array(x, "x")
        y = _validate_array(y, "y")
        z = np.asarray(z, dtype=np.float64)
        
        # Ensure z is 2D
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        elif z.ndim != 2:
            raise ValueError("z must be 1D or 2D array")
        
        if len(x) != len(y) or len(x) != z.shape[0]:
            raise ValueError("x, y, and z must have same number of rows")
        
        return _rs.permutation_test_partial_corr(
            x, y, z,
            n_permutations=self.n_permutations,
            random_state=self.random_state,
        )


# Convenience functions
def permutation_test_anova(
    groups: List[np.ndarray],
    *,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    ANOVA permutation test (convenience function).
    
    Equivalent to: PermutationTestAnovaConfig(...).run(groups)
    
    See PermutationTestAnovaConfig for parameter documentation.
    """
    config = PermutationTestAnovaConfig(
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return config.run(groups)


def permutation_test_mann_whitney(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int = 1000,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Mann-Whitney U test (convenience function).
    
    Equivalent to: PermutationTestMannWhitneyConfig(...).run(x, y)
    
    See PermutationTestMannWhitneyConfig for parameter documentation.
    """
    config = PermutationTestMannWhitneyConfig(
        n_permutations=n_permutations,
        alternative=alternative,
        random_state=random_state,
    )
    return config.run(x, y)


def permutation_test_kruskal_wallis(
    groups: List[np.ndarray],
    *,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Kruskal-Wallis H test (convenience function).
    
    Equivalent to: PermutationTestKruskalWallisConfig(...).run(groups)
    
    See PermutationTestKruskalWallisConfig for parameter documentation.
    """
    config = PermutationTestKruskalWallisConfig(
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return config.run(groups)


def permutation_test_partial_corr(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Partial correlation test (convenience function).
    
    Equivalent to: PermutationTestPartialCorrConfig(...).run(x, y, z)
    
    See PermutationTestPartialCorrConfig for parameter documentation.
    """
    config = PermutationTestPartialCorrConfig(
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return config.run(x, y, z)


# ======================================================================================
# BAYESIAN BOOTSTRAP SE (v0.3 #6)
# ======================================================================================

@dataclass
class BayesianBootstrapSEConfig:
    """
    Bayesian bootstrap standard error estimation.
    
    Uses Dirichlet-weighted resampling for smoother uncertainty estimates
    compared to standard bootstrap. Particularly useful for small samples.
    
    The Bayesian bootstrap uses random Dirichlet(1,...,1) weights instead
    of resampling indices, which provides:
    - Smoother uncertainty estimates
    - Better small-sample properties
    - More stable SE for extreme statistics (e.g., median)
    
    Parameters
    ----------
    stat : {"mean", "std", "median"}, default="mean"
        Statistic to compute SE for
    n_resamples : int, default=1000
        Number of Bayesian bootstrap resamples
    random_state : int or None, default=None
        Random seed for reproducibility
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import BayesianBootstrapSEConfig
    >>> 
    >>> # Small sample example
    >>> np.random.seed(42)
    >>> data = np.random.randn(20)
    >>> 
    >>> config = BayesianBootstrapSEConfig(
    ...     stat="mean",
    ...     n_resamples=5000,
    ...     random_state=42
    ... )
    >>> estimate, se = config.run(data)
    >>> 
    >>> # Approximate 95% CI
    >>> lower = estimate - 1.96 * se
    >>> upper = estimate + 1.96 * se
    >>> print(f"Mean: {estimate:.3f}")
    >>> print(f"SE:   {se:.3f}")
    >>> print(f"95% CI (approx): [{lower:.3f}, {upper:.3f}]")
    
    >>> # For median (more stable than regular bootstrap)
    >>> config_median = BayesianBootstrapSEConfig(stat="median", random_state=42)
    >>> estimate, se = config_median.run(data)
    >>> print(f"Median SE: {se:.3f}")
    
    Notes
    -----
    - Uses Dirichlet(1,...,1) weights (uniform on simplex)
    - More stable than regular bootstrap for n < 30
    - SE computed as standard deviation of bootstrap distribution
    - For median, uses weighted quantile approach
    - Returns (estimate, 0.0) for single-observation data
    
    References
    ----------
    .. [1] Rubin, D.B. (1981). "The Bayesian Bootstrap."
           The Annals of Statistics, 9(1), 130-134.
    """
    stat: Literal["mean", "std", "median"] = "mean"
    n_resamples: int = 1000
    random_state: Optional[int] = None
    
    def __post_init__(self):
        _validate_stat(self.stat, "BayesianBootstrapSEConfig", 
                      supported=["mean", "std", "median"])
        _validate_n_resamples(self.n_resamples, "BayesianBootstrapSEConfig")
    
    def run(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Compute Bayesian bootstrap SE.
        
        Parameters
        ----------
        x : array-like, shape (n,)
            1D array of data
            
        Returns
        -------
        estimate : float
            Point estimate of statistic
        se : float
            Bayesian bootstrap standard error
            
        Raises
        ------
        ValueError
            If x is empty or contains NaN/Inf
        """
        x = _validate_array(x, "x")
        
        if len(x) == 0:
            raise ValueError("Input array is empty")
        
        return _rs.bayesian_bootstrap_se(
            x,
            stat=self.stat,
            n_resamples=self.n_resamples,
            random_state=self.random_state,
        )


def bayesian_bootstrap_se(
    x: np.ndarray,
    *,
    stat: Literal["mean", "std", "median"] = "mean",
    n_resamples: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Bayesian bootstrap standard error (convenience function).
    
    Equivalent to: BayesianBootstrapSEConfig(...).run(x)
    
    Uses Dirichlet-weighted resampling for smoother uncertainty estimates.
    More stable than regular bootstrap for small samples.
    
    Parameters
    ----------
    x : array-like
        1D array of data
    stat : {"mean", "std", "median"}, default="mean"
        Statistic to compute SE for
    n_resamples : int, default=1000
        Number of Bayesian bootstrap resamples
    random_state : int or None, default=None
        Random seed for reproducibility
    
    Returns
    -------
    estimate : float
        Point estimate of statistic
    se : float
        Bayesian bootstrap standard error
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import bayesian_bootstrap_se
    >>> 
    >>> np.random.seed(42)
    >>> data = np.random.randn(30)
    >>> 
    >>> # SE for mean
    >>> estimate, se = bayesian_bootstrap_se(
    ...     data,
    ...     stat="mean",
    ...     n_resamples=5000,
    ...     random_state=42
    ... )
    >>> print(f"Mean ± SE: {estimate:.3f} ± {se:.3f}")
    >>> 
    >>> # Construct approximate 95% CI
    >>> ci_lower = estimate - 1.96 * se
    >>> ci_upper = estimate + 1.96 * se
    >>> print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    >>> 
    >>> # SE for median
    >>> median_est, median_se = bayesian_bootstrap_se(
    ...     data,
    ...     stat="median",
    ...     n_resamples=5000,
    ...     random_state=42
    ... )
    >>> print(f"Median SE: {median_se:.3f}")
    
    See Also
    --------
    BayesianBootstrapSEConfig : Config object for reusable workflows
    bootstrap_se : Regular bootstrap SE (for comparison)
    
    Notes
    -----
    Bayesian bootstrap is particularly useful when:
    - Sample size is small (n < 30)
    - Estimating SE for robust statistics (median, trimmed mean)
    - Want smoother uncertainty estimates
    
    The Bayesian bootstrap uses Dirichlet(1,...,1) weights instead of
    resampling indices, which provides more stable estimates.
    """
    config = BayesianBootstrapSEConfig(
        stat=stat,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    return config.run(x)


# ======================================================================================
# MULTIPLE TESTING CORRECTIONS (v0.3 #7)
# ======================================================================================

@dataclass
class BonferroniConfig:
    """
    Bonferroni correction for multiple testing.
    
    Controls the Family-Wise Error Rate (FWER) at level alpha.
    Most conservative method - use when you need strong error control.
    
    The Bonferroni correction is the simplest and most conservative method
    for controlling the FWER. It multiplies each p-value by the number of tests.
    
    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for hypothesis testing
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import BonferroniConfig
    >>> 
    >>> # Multiple t-tests
    >>> pvalues = np.array([0.01, 0.04, 0.03, 0.50])
    >>> 
    >>> config = BonferroniConfig(alpha=0.05)
    >>> reject, pvals_adj = config.run(pvalues)
    >>> 
    >>> print("Adjusted p-values:", pvals_adj)
    >>> print("Reject H0:", reject)
    
    Notes
    -----
    Bonferroni correction: p_adj[i] = min(p[i] * m, 1.0)
    where m is the number of tests.
    
    Guarantees FWER <= alpha under any dependence structure.
    Very conservative - loses power with large m.
    
    References
    ----------
    .. [1] Bonferroni, C. E. (1936). "Teoria statistica delle classi e calcolo 
           delle probabilità." Pubblicazioni del R Istituto Superiore di Scienze
           Economiche e Commerciali di Firenze, 8, 3-62.
    """
    alpha: float = 0.05
    
    def __post_init__(self):
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
    
    def run(self, pvalues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Bonferroni correction.
        
        Parameters
        ----------
        pvalues : array-like, shape (n,)
            Array of p-values to adjust
            
        Returns
        -------
        reject : ndarray, shape (n,), dtype=bool
            True for hypotheses to reject at level alpha
        pvals_corrected : ndarray, shape (n,), dtype=float
            Bonferroni-adjusted p-values
            
        Raises
        ------
        ValueError
            If pvalues is empty, contains NaN, or values outside [0, 1]
        """
        pvalues = _validate_array(pvalues, "pvalues")
        
        if len(pvalues) == 0:
            raise ValueError("pvalues array is empty")
        
        pvals_adj = np.asarray(_rs.p_adjust_np(pvalues, "bonferroni"))
        reject = pvals_adj <= self.alpha
        return reject, pvals_adj


@dataclass
class HolmConfig:
    """
    Holm step-down correction for multiple testing.
    
    More powerful than Bonferroni while maintaining FWER control at level alpha.
    Uniformly better than Bonferroni - always use this over Bonferroni.
    
    The Holm procedure is a step-down method that adjusts p-values sequentially,
    providing more power than Bonferroni while still controlling FWER.
    
    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for hypothesis testing
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import HolmConfig
    >>> 
    >>> pvalues = np.array([0.01, 0.04, 0.03, 0.50])
    >>> 
    >>> config = HolmConfig(alpha=0.05)
    >>> reject, pvals_adj = config.run(pvalues)
    >>> 
    >>> print("Adjusted p-values:", pvals_adj)
    >>> print("Reject H0:", reject)
    
    Notes
    -----
    Holm's step-down procedure:
    1. Sort p-values: p(1) <= p(2) <= ... <= p(m)
    2. Compute: p_adj(i) = max_{j<=i} (p(j) * (m - j + 1))
    3. Reject if p_adj <= alpha
    
    Guarantees FWER <= alpha under any dependence structure.
    Uniformly more powerful than Bonferroni.
    
    References
    ----------
    .. [1] Holm, S. (1979). "A simple sequentially rejective multiple test
           procedure." Scandinavian Journal of Statistics, 6(2), 65-70.
    """
    alpha: float = 0.05
    
    def __post_init__(self):
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
    
    def run(self, pvalues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Holm step-down correction.
        
        Parameters
        ----------
        pvalues : array-like, shape (n,)
            Array of p-values to adjust
            
        Returns
        -------
        reject : ndarray, shape (n,), dtype=bool
            True for hypotheses to reject
        pvals_corrected : ndarray, shape (n,), dtype=float
            Holm-adjusted p-values
        """
        pvalues = _validate_array(pvalues, "pvalues")
        
        if len(pvalues) == 0:
            raise ValueError("pvalues array is empty")
        
        pvals_adj = np.asarray(_rs.p_adjust_np(pvalues, "holm"))
        reject = pvals_adj <= self.alpha
        return reject, pvals_adj


@dataclass
class FDRConfig:
    """
    Benjamini-Hochberg FDR correction for multiple testing.
    
    Controls the False Discovery Rate (expected proportion of false discoveries)
    at level alpha. More powerful than FWER methods for large numbers of tests.
    
    The BH procedure is a step-up method that provides more power than FWER
    methods by controlling the expected proportion of false positives among
    rejected hypotheses, rather than the probability of any false positive.
    
    Parameters
    ----------
    alpha : float, default=0.05
        FDR level (expected proportion of false discoveries)
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import FDRConfig
    >>> 
    >>> # Many tests (e.g., gene expression)
    >>> pvalues = np.random.uniform(0, 1, 1000)
    >>> pvalues[:10] = 0.001  # 10 true positives
    >>> 
    >>> config = FDRConfig(alpha=0.1)  # Allow 10% FDR
    >>> reject, pvals_adj = config.run(pvalues)
    >>> 
    >>> print(f"Rejected {reject.sum()} hypotheses")
    >>> print(f"Adjusted p-values (first 10): {pvals_adj[:10]}")
    
    Notes
    -----
    Benjamini-Hochberg procedure:
    1. Sort p-values: p(1) <= p(2) <= ... <= p(m)
    2. Compute: p_adj(i) = min(p(i) * m / i, 1.0)
    3. Enforce monotonicity (reverse order)
    4. Reject if p_adj <= alpha
    
    Controls E[FDR] <= alpha under independence or positive dependence (PRDS).
    
    For arbitrary dependence, use Benjamini-Yekutieli (not implemented).
    
    References
    ----------
    .. [1] Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing." Journal of the Royal Statistical Society: Series B, 57(1),
           289-300.
    """
    alpha: float = 0.05
    
    def __post_init__(self):
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
    
    def run(self, pvalues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Parameters
        ----------
        pvalues : array-like, shape (n,)
            Array of p-values to adjust
            
        Returns
        -------
        reject : ndarray, shape (n,), dtype=bool
            True for hypotheses to reject
        pvals_corrected : ndarray, shape (n,), dtype=float
            BH FDR-adjusted p-values
        """
        pvalues = _validate_array(pvalues, "pvalues")
        
        if len(pvalues) == 0:
            raise ValueError("pvalues array is empty")
        
        pvals_adj = np.asarray(_rs.p_adjust_np(pvalues, "bh"))
        reject = pvals_adj <= self.alpha
        return reject, pvals_adj


# Convenience functions
def multipletests(
    pvalues: np.ndarray,
    *,
    alpha: float = 0.05,
    method: Literal["bonferroni", "holm", "fdr_bh"] = "holm",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multiple testing correction (convenience function).
    
    Apply multiple testing correction and return rejection decisions
    plus adjusted p-values. Recommended default is Holm correction.
    
    Parameters
    ----------
    pvalues : array-like, shape (n,)
        Array of p-values to correct
    alpha : float, default=0.05
        Significance level
    method : {"bonferroni", "holm", "fdr_bh"}, default="holm"
        Correction method:
        - "bonferroni": Most conservative FWER control
        - "holm": Step-down FWER control (recommended for FWER)
        - "fdr_bh": Benjamini-Hochberg FDR control (for large m)
        
    Returns
    -------
    reject : ndarray, shape (n,), dtype=bool
        True for hypotheses to reject at level alpha
    pvals_corrected : ndarray, shape (n,), dtype=float
        Adjusted p-values
    
    Examples
    --------
    >>> import numpy as np
    >>> from bunker_stats.resampling import multipletests
    >>> 
    >>> # Simulate 100 t-tests
    >>> np.random.seed(42)
    >>> pvalues = np.random.uniform(0, 1, 100)
    >>> pvalues[:10] = 0.001  # 10 true positives
    >>> 
    >>> # Holm correction (recommended)
    >>> reject, pvals_adj = multipletests(pvalues, method="holm", alpha=0.05)
    >>> print(f"Rejected {reject.sum()} hypotheses with Holm")
    >>> 
    >>> # FDR correction (more power for large m)
    >>> reject_fdr, pvals_fdr = multipletests(pvalues, method="fdr_bh", alpha=0.1)
    >>> print(f"Rejected {reject_fdr.sum()} hypotheses with FDR")
    
    See Also
    --------
    BonferroniConfig : Bonferroni correction config object
    HolmConfig : Holm correction config object
    FDRConfig : FDR correction config object
    
    Notes
    -----
    Method selection guide:
    - Use "holm" for general FWER control (default, recommended)
    - Use "bonferroni" only for comparison (Holm is strictly better)
    - Use "fdr_bh" when you have many tests (m > 20) and can tolerate

      some false discoveries
    
    All methods are deterministic and return adjusted p-values in [0, 1].
    """
    pvalues = _validate_array(pvalues, "pvalues")
    
    if method not in ["bonferroni", "holm", "fdr_bh"]:
        raise ValueError(
            f"Unknown method '{method}'. Use: 'bonferroni', 'holm', or 'fdr_bh'"
        )
    
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # The Rust extension exposes a single p_adjust_np(pvalues, method) kernel
    # (methods: bonferroni / holm / bh). "fdr_bh" maps to the kernel's "bh".
    _kernel_method = {"bonferroni": "bonferroni", "holm": "holm", "fdr_bh": "bh"}[method]
    pvals_adj = np.asarray(_rs.p_adjust_np(pvalues, _kernel_method))
    reject = pvals_adj <= alpha
    return reject, pvals_adj


# ======================================================================================
# METADATA-RICH RESULT OBJECTS (v0.3 #8 - FINAL FEATURE)
# ======================================================================================

@dataclass
class BootstrapResult:
    """
    Result from bootstrap confidence interval estimation with optional metadata.
    
    This object provides access to both the primary statistical results
    and optional diagnostic information for reproducibility and visualization.
    
    Attributes
    ----------
    estimate : float
        Point estimate of the statistic
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    se : float, optional
        Standard error (if computed)
    
    Metadata (populated if return_draws=True or return_metadata=True)
    -------------------------------------------------------------------
    draws : ndarray, optional
        Bootstrap replicates, shape (n_resamples,)
        Available when return_draws=True
    method : str, optional
        CI method used ("percentile", "basic", "bca", "studentized")
    n_resamples : int, optional
        Number of bootstrap resamples performed
    random_state : int, optional
        Random seed used for reproducibility
    confidence_level : float, optional
        Confidence level (e.g., 0.95 for 95% CI)
    
    Method-specific metadata
    ------------------------
    bca_acceleration : float, optional
        BCa acceleration parameter (BCa method only)
    bca_bias_correction : float, optional
        BCa bias correction parameter (BCa method only)
    
    Examples
    --------
    >>> from bunker_stats.resampling import BootstrapConfig
    >>> import numpy as np
    >>> 
    >>> data = np.random.randn(100)
    >>> 
    >>> # Fast path (no metadata) - backward compatible
    >>> config = BootstrapConfig(n_resamples=1000, random_state=42)
    >>> estimate, ci_lower, ci_upper = config.run(data)
    >>> 
    >>> # With metadata and draws for visualization
    >>> config = BootstrapConfig(n_resamples=1000, return_draws=True, random_state=42)
    >>> result = config.run(data)
    >>> print(f"Estimate: {result.estimate:.3f}")
    >>> print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    >>> print(f"Bootstrap distribution: {result.draws.shape}")
    >>> 
    >>> # Reproducibility
    >>> print(f"Method: {result.method}, Seed: {result.random_state}")
    >>> 
    >>> # Visualization
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(result.draws, bins=50, alpha=0.7)
    >>> plt.axvline(result.estimate, color='r', label='Estimate')
    >>> plt.axvline(result.ci_lower, color='g', linestyle='--', label='95% CI')
    >>> plt.axvline(result.ci_upper, color='g', linestyle='--')
    >>> plt.legend()
    >>> plt.show()
    
    See Also
    --------
    BootstrapConfig : Configuration for bootstrap CI estimation
    BayesianBootstrapSEResult : Result from Bayesian bootstrap SE estimation
    
    Notes
    -----
    When return_draws=False and return_metadata=False (default), bootstrap
    functions return simple tuples (estimate, ci_lower, ci_upper) for maximum
    performance. Use this class only when you need access to the bootstrap
    distribution or reproducibility metadata.
    
    Performance:
    - Fast path (no metadata): 0% overhead
    - With metadata only: ~3-5% overhead
    - With draws: ~5-10% overhead
    """
    # Core results (always present)
    estimate: float
    ci_lower: float
    ci_upper: float
    se: Optional[float] = None
    
    # Optional diagnostics (populated when requested)
    draws: Optional[np.ndarray] = None
    method: Optional[str] = None
    n_resamples: Optional[int] = None
    random_state: Optional[int] = None
    confidence_level: Optional[float] = None
    
    # Method-specific metadata
    bca_acceleration: Optional[float] = None
    bca_bias_correction: Optional[float] = None
    
    def summary(self) -> str:
        """
        Return formatted summary string.
        
        Returns
        -------
        str
            Multiline summary of results
            
        Examples
        --------
        >>> result = BootstrapResult(estimate=0.5, ci_lower=0.3, ci_upper=0.7,
        ...                          method="percentile", n_resamples=1000)
        >>> print(result.summary())
        Bootstrap Result (percentile)
          Estimate: 0.5000
          95% CI: [0.3000, 0.7000]
          Resamples: 1000
        """
        level = self.confidence_level or 0.95
        ci_pct = int(level * 100)
        
        lines = [
            f"Bootstrap Result ({self.method or 'percentile'})",
            f"  Estimate: {self.estimate:.4f}",
            f"  {ci_pct}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
        ]
        
        if self.se is not None:
            lines.append(f"  SE: {self.se:.4f}")
        
        if self.n_resamples is not None:
            lines.append(f"  Resamples: {self.n_resamples}")
        
        if self.draws is not None:
            lines.append(f"  Draws available: {len(self.draws)}")
        
        if self.random_state is not None:
            lines.append(f"  Random seed: {self.random_state}")
        
        if self.bca_acceleration is not None:
            lines.append(f"  BCa acceleration: {self.bca_acceleration:.4f}")
        
        if self.bca_bias_correction is not None:
            lines.append(f"  BCa bias correction: {self.bca_bias_correction:.4f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Short representation showing key results"""
        return (f"BootstrapResult(estimate={self.estimate:.4f}, "
                f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
                f"method={self.method or 'percentile'})")


@dataclass
class BayesianBootstrapSEResult:
    """
    Result from Bayesian bootstrap standard error estimation.
    
    Attributes
    ----------
    estimate : float
        Point estimate of the statistic
    se : float
        Standard error
    
    Metadata (if return_draws=True or return_metadata=True)
    --------------------------------------------------------
    draws : ndarray, optional
        Bayesian bootstrap replicates, shape (n_resamples,)
    method : str, optional
        Always "bayesian_bootstrap"
    n_resamples : int, optional
        Number of bootstrap resamples
    random_state : int, optional
        Random seed used
    stat : str, optional
        Statistic computed ("mean", "std", "median")
    
    Examples
    --------
    >>> from bunker_stats.resampling import BayesianBootstrapSEConfig
    >>> import numpy as np
    >>> 
    >>> data = np.random.randn(50)
    >>> config = BayesianBootstrapSEConfig(n_resamples=1000, return_draws=True, random_state=42)
    >>> result = config.run(data)
    >>> 
    >>> print(f"Mean: {result.estimate:.3f} ± {result.se:.3f}")
    >>> print(f"Distribution shape: {result.draws.shape}")
    
    Notes
    -----
    The Bayesian bootstrap uses Dirichlet weights for non-parametric
    Bayesian inference. This provides a natural posterior distribution
    over the statistic of interest.
    """
    # Core results
    estimate: float
    se: float
    
    # Optional metadata
    draws: Optional[np.ndarray] = None
    method: Optional[str] = None
    n_resamples: Optional[int] = None
    random_state: Optional[int] = None
    stat: Optional[str] = None
    
    def summary(self) -> str:
        """Return formatted summary"""
        lines = [
            f"Bayesian Bootstrap SE Result",
            f"  Estimate: {self.estimate:.4f}",
            f"  SE: {self.se:.4f}",
        ]
        
        if self.stat is not None:
            lines.append(f"  Statistic: {self.stat}")
        
        if self.n_resamples is not None:
            lines.append(f"  Resamples: {self.n_resamples}")
        
        if self.draws is not None:
            lines.append(f"  Draws available: {len(self.draws)}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (f"BayesianBootstrapSEResult(estimate={self.estimate:.4f}, "
                f"se={self.se:.4f}, stat={self.stat or 'mean'})")


@dataclass
class PermutationTestResult:
    """
    Result from permutation test with optional null distribution.
    
    Attributes
    ----------
    statistic : float
        Observed test statistic
    pvalue : float
        Two-sided p-value
    
    Metadata (if return_distribution=True or return_metadata=True)
    ---------------------------------------------------------------
    permutation_distribution : ndarray, optional
        Permutation distribution, shape (n_permutations,)
    method : str, optional
        Test name (e.g., "mann_whitney", "anova", "kruskal_wallis")
    n_permutations : int, optional
        Number of permutations performed
    random_state : int, optional
        Random seed used
    exact : bool, optional
        Whether exact (exhaustive) test was used
    
    Examples
    --------
    >>> from bunker_stats.resampling import PermutationTestMannWhitneyConfig
    >>> import numpy as np
    >>> 
    >>> group1 = np.random.randn(30)
    >>> group2 = np.random.randn(30) + 0.5
    >>> 
    >>> config = PermutationTestMannWhitneyConfig(
    ...     n_permutations=10000,
    ...     return_distribution=True,
    ...     random_state=42
    ... )
    >>> result = config.run(group1, group2)
    >>> 
    >>> print(f"U statistic: {result.statistic:.2f}")
    >>> print(f"p-value: {result.pvalue:.4f}")
    >>> 
    >>> # Visualize permutation distribution
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(result.permutation_distribution, bins=50, alpha=0.7)
    >>> plt.axvline(result.statistic, color='r', label=f'Observed (p={result.pvalue:.4f})')
    >>> plt.legend()
    >>> plt.show()
    
    Notes
    -----
    The permutation distribution represents the null distribution of the
    test statistic under the null hypothesis of no group differences.
    Visualization of this distribution can provide insights into the
    strength and nature of the evidence against the null hypothesis.
    """
    # Core results
    statistic: float
    pvalue: float
    
    # Optional metadata
    permutation_distribution: Optional[np.ndarray] = None
    method: Optional[str] = None
    n_permutations: Optional[int] = None
    random_state: Optional[int] = None
    exact: Optional[bool] = None
    
    def summary(self) -> str:
        """Return formatted summary"""
        lines = [
            f"Permutation Test Result ({self.method or 'unknown'})",
            f"  Statistic: {self.statistic:.4f}",
            f"  p-value: {self.pvalue:.4f}",
        ]
        
        if self.n_permutations is not None:
            perm_type = "exact" if self.exact else "approximate"
            lines.append(f"  Permutations: {self.n_permutations} ({perm_type})")
        
        if self.permutation_distribution is not None:
            lines.append(f"  Distribution available: {len(self.permutation_distribution)}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        sig = "***" if self.pvalue < 0.001 else "**" if self.pvalue < 0.01 else "*" if self.pvalue < 0.05 else "ns"
        return (f"PermutationTestResult(statistic={self.statistic:.4f}, "
                f"pvalue={self.pvalue:.4f} {sig}, method={self.method or 'unknown'})")


@dataclass
class WildBootstrapOLSResult:
    """
    Result from wild bootstrap for OLS regression with optional coefficient matrix.
    
    Wild bootstrap is robust to heteroskedasticity and provides
    valid inference without homoskedasticity assumptions.
    
    Attributes
    ----------
    coefficients : ndarray
        OLS coefficient estimates, shape (p,)
    se : ndarray
        Standard errors, shape (p,)
    ci_lower : ndarray
        Lower confidence bounds, shape (p,)
    ci_upper : ndarray
        Upper confidence bounds, shape (p,)
    
    Metadata (if return_bootstrap_coefs=True or return_metadata=True)
    ------------------------------------------------------------------
    bootstrap_coefs : ndarray, optional
        Bootstrap coefficient replicates, shape (n_resamples, p)
    weight_type : str, optional
        Weight distribution used ("mammen", "rademacher")
    n_resamples : int, optional
        Number of bootstrap resamples
    random_state : int, optional
        Random seed used
    confidence_level : float, optional
        Confidence level (e.g., 0.95)
    
    Examples
    --------
    >>> from bunker_stats.resampling import WildBootstrapOLSConfig
    >>> import numpy as np
    >>> 
    >>> # Simulate heteroskedastic data
    >>> np.random.seed(42)
    >>> n, p = 100, 3
    >>> X = np.random.randn(n, p)
    >>> X[:, 0] = 1  # Intercept
    >>> true_beta = np.array([1.0, 2.0, -0.5])
    >>> y = X @ true_beta + np.random.randn(n) * (1 + 0.5 * X[:, 1]**2)  # Heteroskedastic
    >>> 
    >>> config = WildBootstrapOLSConfig(
    ...     weight_type="mammen",
    ...     n_resamples=1000,
    ...     return_bootstrap_coefs=True,
    ...     random_state=42
    ... )
    >>> result = config.run(X, y)
    >>> 
    >>> print("Coefficients:", result.coefficients)
    >>> print("Standard errors:", result.se)
    >>> print("Bootstrap coefs shape:", result.bootstrap_coefs.shape)
    
    Notes
    -----
    The wild bootstrap preserves heteroskedasticity structure while
    resampling, making it appropriate for regression with non-constant
    error variance. The Mammen distribution is recommended for general use,
    while Rademacher is computationally simpler but slightly less powerful.
    """
    # Core results
    coefficients: np.ndarray
    se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    
    # Optional metadata
    bootstrap_coefs: Optional[np.ndarray] = None
    weight_type: Optional[str] = None
    n_resamples: Optional[int] = None
    random_state: Optional[int] = None
    confidence_level: Optional[float] = None
    
    def summary(self) -> str:
        """Return formatted summary"""
        p = len(self.coefficients)
        level = self.confidence_level or 0.95
        ci_pct = int(level * 100)
        
        lines = [
            f"Wild Bootstrap OLS Result ({self.weight_type or 'mammen'} weights)",
            f"  Coefficients ({p} parameters):",
        ]
        
        for i, (coef, se, lower, upper) in enumerate(zip(
            self.coefficients, self.se, self.ci_lower, self.ci_upper
        )):
            sig = "***" if abs(coef/se) > 2.576 else "**" if abs(coef/se) > 1.96 else "*" if abs(coef/se) > 1.645 else ""
            lines.append(
                f"    β[{i}] = {coef:>8.4f} (SE: {se:.4f}, {ci_pct}% CI: [{lower:.4f}, {upper:.4f}]) {sig}"
            )
        
        if self.n_resamples is not None:
            lines.append(f"  Resamples: {self.n_resamples}")
        
        if self.bootstrap_coefs is not None:
            lines.append(f"  Bootstrap matrix: {self.bootstrap_coefs.shape}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (f"WildBootstrapOLSResult(p={len(self.coefficients)}, "
                f"weight_type={self.weight_type or 'mammen'}, "
                f"n_resamples={self.n_resamples})")