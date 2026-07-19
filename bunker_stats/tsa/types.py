"""
Result types for TSA functions.

Rich result objects with opt-in metadata for statistical tests.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Iterator, Tuple
import numpy as np


@dataclass
class KPSSResult:
    """Result from KPSS stationarity test with optional metadata.
    
    Attributes
    ----------
    stat : float
        KPSS test statistic
    pval : float
        P-value (interpolated from critical values)
    
    Metadata (only if metadata=True)
    --------------------------------
    bandwidth : int
        Number of lags used for long-run variance
    lrv : float
        Long-run variance estimate
    variance_components : dict
        Breakdown of LRV calculation
    critical_values : dict
        Critical values at {1%, 5%, 10%}
    regression : str
        Regression type ('c' or 'ct')
    n_obs : int
        Number of observations
    bandwidth_method : str
        Bandwidth selection method
    
    Examples
    --------
    >>> result = bs.kpss_test(x, metadata=True)
    >>> print(result.info())
    >>> stat, pval = result  # Tuple unpacking works
    """
    stat: float
    pval: float
    
    # Metadata (optional)
    bandwidth: Optional[int] = None
    lrv: Optional[float] = None
    variance_components: Optional[Dict[str, float]] = None
    critical_values: Optional[Dict[str, float]] = None
    regression: Optional[str] = None
    n_obs: Optional[int] = None
    bandwidth_method: Optional[str] = None
    
    def __iter__(self) -> Iterator[float]:
        """Enable tuple unpacking: stat, pval = result"""
        return iter([self.stat, self.pval])
    
    def __getitem__(self, idx: int) -> float:
        """Enable indexing: result[0], result[1]"""
        if idx == 0:
            return self.stat
        elif idx == 1:
            return self.pval
        else:
            raise IndexError(f"Index {idx} out of range (0, 1)")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {'stat': self.stat, 'pval': self.pval}
        
        if self.bandwidth is not None:
            result.update({
                'bandwidth': self.bandwidth,
                'lrv': self.lrv,
                'variance_components': self.variance_components,
                'critical_values': self.critical_values,
                'regression': self.regression,
                'n_obs': self.n_obs,
                'bandwidth_method': self.bandwidth_method,
            })
        
        return result
    
    def info(self) -> str:
        """Pretty-print test summary."""
        lines = [
            "KPSS Stationarity Test",
            "=" * 60,
            f"Statistic:  {self.stat:.6f}",
            f"P-value:    {self.pval:.4f}",
            ""
        ]
        
        # Interpretation
        if self.pval > 0.10:
            conclusion = "Cannot reject H0: Series appears stationary"
        elif self.pval > 0.05:
            conclusion = "Marginal evidence against stationarity"
        else:
            conclusion = "Reject H0: Series appears non-stationary"
        
        lines.append(f"Conclusion: {conclusion}")
        
        # Metadata
        if self.bandwidth is not None:
            reg_name = "Constant + trend" if self.regression == 'ct' else "Constant only"
            lines.extend([
                "",
                "Test Details:",
                "-" * 60,
                f"  Regression:   {reg_name}",
                f"  Observations: {self.n_obs}",
                f"  Bandwidth:    {self.bandwidth} lags ({self.bandwidth_method})",
                f"  LRV:          {self.lrv:.6f}",
            ])
            
            if self.critical_values:
                lines.extend([
                    "",
                    "Critical Values:",
                    f"  1%:   {self.critical_values.get('1%', float('nan')):.4f}",
                    f"  5%:   {self.critical_values.get('5%', float('nan')):.4f}",
                    f"  10%:  {self.critical_values.get('10%', float('nan')):.4f}",
                ])
        
        return "\n".join(lines)
    
    def conclusion(self, alpha: float = 0.05) -> str:
        """Get test conclusion at significance level."""
        if self.pval > alpha:
            return (f"Cannot reject H0 at {alpha*100:.0f}% level: "
                   f"series appears stationary (p={self.pval:.4f})")
        else:
            return (f"Reject H0 at {alpha*100:.0f}% level: "
                   f"series appears non-stationary (p={self.pval:.4f})")


@dataclass
class VarianceRatioResult:
    """Result from Lo-MacKinlay variance ratio test.
    
    Attributes
    ----------
    vr : float
        Variance ratio VR(q)
    z_score : float
        Test statistic (z-score)
    pval : float
        Two-tailed p-value
    
    Metadata
    --------
    lags : int
        Lag q used
    test_type : str
        Test type used
    var_1 : float
        Variance of 1-period differences
    var_q : float
        Variance of q-period differences
    n_obs : int
        Number of observations
    """
    vr: float
    z_score: float
    pval: float
    
    # Metadata
    lags: Optional[int] = None
    test_type: Optional[str] = None
    var_1: Optional[float] = None
    var_q: Optional[float] = None
    n_obs: Optional[int] = None
    
    def __iter__(self):
        return iter([self.vr, self.z_score, self.pval])
    
    def __getitem__(self, idx: int):
        return [self.vr, self.z_score, self.pval][idx]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {'vr': self.vr, 'z_score': self.z_score, 'pval': self.pval}
        
        if self.lags is not None:
            result.update({
                'lags': self.lags,
                'test_type': self.test_type,
                'var_1': self.var_1,
                'var_q': self.var_q,
                'n_obs': self.n_obs,
            })
        
        return result
    
    def interpretation(self) -> str:
        """Interpret VR value."""
        if abs(self.vr - 1.0) < 0.05:
            return "VR ≈ 1.0: Consistent with random walk"
        elif self.vr < 0.95:
            return f"VR = {self.vr:.3f} < 1: Mean reversion"
        else:
            return f"VR = {self.vr:.3f} > 1: Momentum/trending"
    
    def info(self) -> str:
        """Pretty-print summary."""
        lines = [
            "Lo-MacKinlay Variance Ratio Test",
            "=" * 60,
            f"Variance Ratio VR({self.lags if self.lags else '?'}): {self.vr:.6f}",
            f"Test statistic: {self.z_score:.6f}",
            f"P-value:        {self.pval:.6f}",
            "",
            f"Interpretation: {self.interpretation()}",
        ]
        
        if self.pval < 0.05:
            lines.append("\nConclusion: Reject random walk hypothesis")
        else:
            lines.append("\nConclusion: Cannot reject random walk hypothesis")
        
        if self.lags is not None:
            lines.extend([
                "",
                "Test Details:",
                "-" * 60,
                f"  Lag q:         {self.lags}",
                f"  Observations:  {self.n_obs}",
                f"  Var(1-diff):   {self.var_1:.6f}",
                f"  Var({self.lags}-diff): {self.var_q:.6f}",
                f"  Test type:     {self.test_type}",
            ])
        
        return "\n".join(lines)


@dataclass
class ZivotAndrewsResult:
    """Result from Zivot-Andrews structural break test.
    
    Attributes
    ----------
    stat : float
        Minimum ADF statistic (most evidence against unit root)
    breakpoint : int
        Location of detected breakpoint
    pval : float
        P-value
    
    Metadata
    --------
    stat_at_each_bp : ndarray
        Test statistics at each tested breakpoint
    tested_breakpoints : ndarray
        Array of tested breakpoint locations
    regression : str
        Regression type used
    max_lag : int
        Max lag used in ADF
    n_obs : int
        Number of observations
    timed_out : bool
        Whether computation timed out
    computation_time : float
        Time taken in seconds
    """
    stat: float
    breakpoint: int
    pval: float
    
    # Metadata
    stat_at_each_bp: Optional[np.ndarray] = None
    tested_breakpoints: Optional[np.ndarray] = None
    regression: Optional[str] = None
    max_lag: Optional[int] = None
    n_obs: Optional[int] = None
    timed_out: Optional[bool] = None
    computation_time: Optional[float] = None
    
    def __iter__(self):
        return iter([self.stat, self.breakpoint, self.pval])
    
    def __getitem__(self, idx: int):
        return [self.stat, self.breakpoint, self.pval][idx]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {'stat': self.stat, 'breakpoint': self.breakpoint, 'pval': self.pval}
        
        if self.tested_breakpoints is not None:
            result.update({
                'stat_at_each_bp': self.stat_at_each_bp.tolist() if self.stat_at_each_bp is not None else None,
                'tested_breakpoints': self.tested_breakpoints.tolist() if self.tested_breakpoints is not None else None,
                'regression': self.regression,
                'max_lag': self.max_lag,
                'n_obs': self.n_obs,
                'timed_out': self.timed_out,
                'computation_time': self.computation_time,
            })
        
        return result
    
    def plot_breakpoint_scan(self, ax=None):
        """Plot test statistic vs breakpoint location."""
        if self.stat_at_each_bp is None:
            raise ValueError("Metadata not available. Use metadata=True when calling test.")
        
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(self.tested_breakpoints, self.stat_at_each_bp, 'b-', alpha=0.7)
        ax.axvline(self.breakpoint, color='r', linestyle='--',
                   label=f'Detected break (t={self.breakpoint})')
        ax.axhline(self.stat, color='g', linestyle=':',
                   label=f'Min stat = {self.stat:.3f}')
        
        ax.set_xlabel('Breakpoint Location')
        ax.set_ylabel('Test Statistic')
        ax.set_title('Zivot-Andrews Breakpoint Scan')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def info(self) -> str:
        """Pretty-print summary."""
        lines = [
            "Zivot-Andrews Structural Break Test",
            "=" * 60,
            f"Test Statistic: {self.stat:.6f}",
            f"Breakpoint:     {self.breakpoint}",
            f"P-value:        {self.pval:.6f}",
        ]
        
        if self.pval < 0.05:
            lines.append("\nConclusion: Evidence of structural break")
        else:
            lines.append("\nConclusion: No evidence of structural break")
        
        if self.tested_breakpoints is not None:
            bp_range = (int(self.tested_breakpoints[0]), int(self.tested_breakpoints[-1]))
            lines.extend([
                "",
                "Test Details:",
                "-" * 60,
                f"  Regression:       {self.regression}",
                f"  Observations:     {self.n_obs}",
                f"  Max lag:          {self.max_lag}",
                f"  Tested range:     [{bp_range[0]}, {bp_range[1]}]",
                f"  Computation time: {self.computation_time:.2f}s",
                f"  Timed out:        {self.timed_out}",
            ])
        
        return "\n".join(lines)


@dataclass
class ADFResult:
    """Result from Augmented Dickey-Fuller test.
    
    Attributes
    ----------
    stat : float
        ADF test statistic
    pval : float
        P-value
    
    Metadata
    --------
    regression : str
        Regression type
    n_lags : int
        Number of lags used
    n_obs : int
        Number of observations
    critical_values : dict
        Critical values at {1%, 5%, 10%}
    """
    stat: float
    pval: float
    
    # Metadata
    regression: Optional[str] = None
    n_lags: Optional[int] = None
    n_obs: Optional[int] = None
    critical_values: Optional[Dict[str, float]] = None
    
    def __iter__(self):
        return iter([self.stat, self.pval])
    
    def __getitem__(self, idx: int):
        return [self.stat, self.pval][idx]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {'stat': self.stat, 'pval': self.pval}
        
        if self.regression is not None:
            result.update({
                'regression': self.regression,
                'n_lags': self.n_lags,
                'n_obs': self.n_obs,
                'critical_values': self.critical_values,
            })
        
        return result
    
    def info(self) -> str:
        """Pretty-print summary."""
        lines = [
            "Augmented Dickey-Fuller Test",
            "=" * 60,
            f"Test Statistic: {self.stat:.6f}",
            f"P-value:        {self.pval:.6f}",
        ]
        
        if self.pval < 0.05:
            lines.append("\nConclusion: Reject H0 (series is stationary)")
        else:
            lines.append("\nConclusion: Cannot reject H0 (unit root present)")
        
        if self.n_lags is not None:
            lines.extend([
                "",
                "Test Details:",
                "-" * 60,
                f"  Regression:   {self.regression}",
                f"  Lags used:    {self.n_lags}",
                f"  Observations: {self.n_obs}",
            ])
            
            if self.critical_values:
                lines.extend([
                    "",
                    "Critical Values:",
                    f"  1%:   {self.critical_values.get('1%', float('nan')):.4f}",
                    f"  5%:   {self.critical_values.get('5%', float('nan')):.4f}",
                    f"  10%:  {self.critical_values.get('10%', float('nan')):.4f}",
                ])
        
        return "\n".join(lines)


@dataclass
class PPResult:
    """Result from Phillips-Perron test.
    
    Attributes
    ----------
    stat : float
        PP test statistic
    pval : float
        P-value
    
    Metadata
    --------
    regression : str
        Regression type
    lags : int
        Number of lags used for LRV
    lrv : float
        Long-run variance
    n_obs : int
        Number of observations
    critical_values : dict
        Critical values at {1%, 5%, 10%}
    """
    stat: float
    pval: float
    
    # Metadata
    regression: Optional[str] = None
    lags: Optional[int] = None
    lrv: Optional[float] = None
    n_obs: Optional[int] = None
    critical_values: Optional[Dict[str, float]] = None
    
    def __iter__(self):
        return iter([self.stat, self.pval])
    
    def __getitem__(self, idx: int):
        return [self.stat, self.pval][idx]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {'stat': self.stat, 'pval': self.pval}
        
        if self.regression is not None:
            result.update({
                'regression': self.regression,
                'lags': self.lags,
                'lrv': self.lrv,
                'n_obs': self.n_obs,
                'critical_values': self.critical_values,
            })
        
        return result
    
    def info(self) -> str:
        """Pretty-print summary."""
        lines = [
            "Phillips-Perron Test",
            "=" * 60,
            f"Test Statistic: {self.stat:.6f}",
            f"P-value:        {self.pval:.6f}",
        ]
        
        if self.pval < 0.05:
            lines.append("\nConclusion: Reject H0 (series is stationary)")
        else:
            lines.append("\nConclusion: Cannot reject H0 (unit root present)")
        
        if self.lags is not None:
            lines.extend([
                "",
                "Test Details:",
                "-" * 60,
                f"  Regression:   {self.regression}",
                f"  Lags (LRV):   {self.lags}",
                f"  LRV:          {self.lrv:.6f}",
                f"  Observations: {self.n_obs}",
            ])
            
            if self.critical_values:
                lines.extend([
                    "",
                    "Critical Values:",
                    f"  1%:   {self.critical_values.get('1%', float('nan')):.4f}",
                    f"  5%:   {self.critical_values.get('5%', float('nan')):.4f}",
                    f"  10%:  {self.critical_values.get('10%', float('nan')):.4f}",
                ])
        
        return "\n".join(lines)


# ==============================================================================
# NEW RESULT CLASSES FOR v0.3+ TSA FUNCTIONS
# ==============================================================================

from typing import NamedTuple

class HurstResult(NamedTuple):
    """Rich result from Hurst exponent analysis with diagnostic information.
    
    Attributes
    ----------
    h : float
        The Hurst exponent (0 to 1)
    rs_values : np.ndarray
        R/S values at each lag used
    lags : np.ndarray
        Lag values used in estimation
    log_lags : np.ndarray
        Natural log of lags (for plotting)
    log_rs : np.ndarray
        Natural log of R/S values (for plotting)
    regression_slope : float
        Slope of log-log regression (equals h)
    regression_intercept : float
        Intercept of log-log regression
    n_valid_lags : int
        Number of valid lags with finite R/S
    min_lag : int
        Minimum lag used
    max_lag : int
        Maximum lag used
    
    Examples
    --------
    >>> result = bs.hurst_exponent(x, rich_output=True)
    >>> print(f"H = {result.h:.3f}")
    >>> plt.plot(result.log_lags, result.log_rs)
    """
    h: float
    rs_values: np.ndarray
    lags: np.ndarray
    log_lags: np.ndarray
    log_rs: np.ndarray
    regression_slope: float
    regression_intercept: float
    n_valid_lags: int
    min_lag: int
    max_lag: int


class FractionalDiffResult(NamedTuple):
    """Rich result from fractional differencing with diagnostic information.
    
    Attributes
    ----------
    series : np.ndarray
        The fractionally differenced series
    d : float
        Differencing order used
    coefficients : np.ndarray
        Binomial coefficients computed
    truncation_order : int
        Number of coefficients used
    memory_retained : float
        Estimated percentage of memory retained (0-100)
    original_length : int
        Length of input series
    effective_length : int
        Length of output series
    
    Examples
    --------
    >>> result = bs.fractional_diff(x, d=0.5, rich_output=True)
    >>> print(f"Memory retained: {result.memory_retained:.1f}%")
    >>> plt.plot(result.coefficients)
    """
    series: np.ndarray
    d: float
    coefficients: np.ndarray
    truncation_order: int
    memory_retained: float
    original_length: int
    effective_length: int


class BDSResult(NamedTuple):
    """Rich result from BDS nonlinearity test with diagnostic information.
    
    Attributes
    ----------
    statistic : float
        BDS test statistic (z-score)
    p_value : float
        Two-tailed p-value
    m : int
        Embedding dimension used
    epsilon : float
        Distance threshold used
    correlation_integrals : np.ndarray
        Correlation integral C_m at each dimension
    n_pairs : int
        Number of pairs compared
    effective_n : int
        Effective sample size (after subsampling)
    subsampled : bool
        Whether subsampling was applied
    independence : bool
        Reject independence? (p < 0.05)
    
    Examples
    --------
    >>> result = bs.bds_test(x, m=3, rich_output=True)
    >>> print(f"Independence: {not result.independence}")
    >>> plt.plot(result.correlation_integrals)
    """
    statistic: float
    p_value: float
    m: int
    epsilon: float
    correlation_integrals: np.ndarray
    n_pairs: int
    effective_n: int
    subsampled: bool
    independence: bool


class HPFilterResult(NamedTuple):
    """Rich result from HP filter with diagnostic information.
    
    Attributes
    ----------
    trend : np.ndarray
        Smooth trend component
    cycle : np.ndarray
        Cyclical component
    lambda_param : float
        Smoothing parameter used
    trend_roughness : float
        Sum of squared second differences
    cycle_variance : float
        Variance of cyclical component
    solver_used : str
        Solver type: "dense_lu" or "sparse_band"
    n_observations : int
        Length of input series
    reconstruction_mse : float
        MSE of original - (trend + cycle)
    
    Examples
    --------
    >>> result = bs.hp_filter(x, lambda_param=1600, rich_output=True)
    >>> print(f"Solver: {result.solver_used}")
    >>> print(f"Trend roughness: {result.trend_roughness:.2e}")
    >>> plt.plot(result.trend, label='Trend')
    """
    trend: np.ndarray
    cycle: np.ndarray
    lambda_param: float
    trend_roughness: float
    cycle_variance: float
    solver_used: str
    n_observations: int
    reconstruction_mse: float


class EngleGrangerResult(NamedTuple):
    """Rich result from Engle-Granger cointegration test with diagnostics.
    
    Attributes
    ----------
    statistic : float
        ADF test statistic on residuals
    p_value : float
        P-value for cointegration test
    cointegrated : bool
        Reject null of no cointegration?
    residuals : np.ndarray
        OLS regression residuals
    beta_coefficients : np.ndarray
        OLS coefficients from regression
    r_squared : float
        R² from cointegration regression
    n_observations : int
        Number of observations
    adf_lags : int
        Lags used in ADF test
    trend_type : str
        Trend specification used
    
    Examples
    --------
    >>> result = bs.engle_granger_test(y, x, rich_output=True)
    >>> print(f"Cointegrated: {result.cointegrated}")
    >>> print(f"R²: {result.r_squared:.3f}")
    >>> plt.plot(result.residuals)
    """
    statistic: float
    p_value: float
    cointegrated: bool
    residuals: np.ndarray
    beta_coefficients: np.ndarray
    r_squared: float
    n_observations: int
    adf_lags: int
    trend_type: str


class STLResult(NamedTuple):
    """Rich result from STL decomposition with diagnostic information.
    
    Attributes
    ----------
    seasonal : np.ndarray
        Seasonal component
    trend : np.ndarray
        Trend component
    residual : np.ndarray
        Residual component
    period : int
        Seasonal period used
    seasonal_window : int
        LOESS window for seasonal
    trend_window : int
        LOESS window for trend
    n_iterations : int
        Iterations performed
    converged : bool
        Whether algorithm converged
    robustness_weights : Optional[np.ndarray]
        Weights if robust=True
    seasonal_strength : float
        Strength of seasonal component (0-1)
    trend_strength : float
        Strength of trend component (0-1)
    
    Examples
    --------
    >>> result = bs.stl_decompose(x, period=12, rich_output=True)
    >>> print(f"Converged: {result.converged}")
    >>> print(f"Seasonal strength: {result.seasonal_strength:.3f}")
    """
    seasonal: np.ndarray
    trend: np.ndarray
    residual: np.ndarray
    period: int
    seasonal_window: int
    trend_window: int
    n_iterations: int
    converged: bool
    robustness_weights: Optional[np.ndarray]
    seasonal_strength: float
    trend_strength: float


# Export all result classes
__all__ = [
    'KPSSResult',
    'VarianceRatioResult',
    'ZivotAndrewsResult',
    'ADFResult',
    'PPResult',
    # New TSA v0.3+ result classes
    'HurstResult',
    'FractionalDiffResult',
    'BDSResult',
    'HPFilterResult',
    'EngleGrangerResult',
    'STLResult',
]