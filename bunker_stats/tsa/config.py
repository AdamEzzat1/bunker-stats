"""
Configuration classes for TSA (Time Series Analysis) functions.

These provide ergonomic, validated configuration objects for complex tests.
All config classes support both object creation and kwarg passing.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class KPSSConfig:
    """Configuration for KPSS stationarity test.
    
    The KPSS test tests the null hypothesis that a time series is stationary
    around a deterministic trend.
    
    Parameters
    ----------
    regression : {'c', 'ct'}, default='c'
        Type of regression:
        
        - 'c': Test for stationarity around level (constant only)
        - 'ct': Test for stationarity around trend (constant + trend)
    
    nlags : int or None, default=None
        Number of lags for long-run variance estimation.
        If None, uses automatic bandwidth selection based on `bandwidth_method`.
    
    bandwidth_method : {'schwert', 'conservative', 'tstat'}, default='schwert'
        Method for automatic bandwidth selection when nlags=None:
        
        - 'schwert': Schwert (1989) formula ``floor(12*(n/100)^0.25)``
          General-purpose, accounts for sample size
        - 'conservative': Fixed small bandwidth (nlags=2)
          Matches typical statsmodels 'auto' selection for white noise
        - 'tstat': Iterative t-stat based selection
          Data-dependent, matches statsmodels 'auto' behavior exactly
    
    Examples
    --------
    >>> from bunker_stats.tsa import KPSSConfig
    >>> import bunker_stats as bs
    
    >>> # Test with trend
    >>> config = KPSSConfig(regression='ct')
    >>> stat, pval = bs.kpss_test(x, config=config)
    
    >>> # Conservative bandwidth for comparison with statsmodels
    >>> config = KPSSConfig(bandwidth_method='conservative')
    >>> stat, pval = bs.kpss_test(x, config=config)
    
    >>> # Explicit bandwidth
    >>> config = KPSSConfig(nlags=10)
    >>> stat, pval = bs.kpss_test(x, config=config)
    
    Notes
    -----
    The choice of bandwidth method affects the test statistic value but
    typically does not change statistical conclusions. The Schwert formula
    is theoretically grounded and widely used in econometrics.
    
    References
    ----------
    .. [1] Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992).
           Testing the null hypothesis of stationarity against the alternative
           of a unit root. Journal of Econometrics, 54(1-3), 159-178.
    .. [2] Schwert, G. W. (1989). Tests for unit roots: A Monte Carlo investigation.
           Journal of Business & Economic Statistics, 7(2), 147-159.
    """
    regression: Literal['c', 'ct'] = 'c'
    nlags: Optional[int] = None
    bandwidth_method: Literal['schwert', 'conservative', 'tstat'] = 'schwert'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if self.regression not in ('c', 'ct'):
            raise ValueError(
                f"regression must be 'c' or 'ct', got '{self.regression}'"
            )
        
        if self.nlags is not None and self.nlags < 0:
            raise ValueError(
                f"nlags must be non-negative, got {self.nlags}"
            )
        
        if self.bandwidth_method not in ('schwert', 'conservative', 'tstat'):
            raise ValueError(
                f"bandwidth_method must be 'schwert', 'conservative', or 'tstat', "
                f"got '{self.bandwidth_method}'"
            )


@dataclass
class ZivotAndrewsConfig:
    """Configuration for Zivot-Andrews structural break test.
    
    The Zivot-Andrews test extends the Augmented Dickey-Fuller test by
    allowing for a structural break at an unknown point.
    
    Parameters
    ----------
    regression : {'c', 't', 'both'}, default='both'
        Type of structural break to test:
        
        - 'c': Break in level (intercept)
        - 't': Break in trend slope
        - 'both': Break in both level and trend
    
    max_lag : int or None, default=None
        Maximum lag for ADF regression. If None, uses ``floor(12*(n/100)^0.25)``.
    
    trim : float, default=0.15
        Fraction of sample to trim from both ends when searching for breakpoint.
        Must be in (0, 0.5). Typical values: 0.10 to 0.15.
        
        Trimming prevents breakpoints too close to endpoints where test has
        low power.
    
    timeout_secs : int, default=300
        Maximum computation time in seconds. For large datasets (n>1000),
        the test can be slow. Timeout ensures graceful termination.
    
    stride : int, default=1
        Step size when scanning breakpoints. For very large datasets (n>5000),
        use stride=2 or stride=3 to reduce computation time at cost of
        breakpoint location precision.
    
    Examples
    --------
    >>> from bunker_stats.tsa import ZivotAndrewsConfig
    >>> import bunker_stats as bs
    
    >>> # Test for break in both level and trend
    >>> config = ZivotAndrewsConfig(regression='both', max_lag=8)
    >>> stat, bp, pval = bs.zivot_andrews_test(x, config=config)
    
    >>> # Fast scan for large dataset
    >>> config = ZivotAndrewsConfig(stride=2, timeout_secs=60)
    >>> stat, bp, pval = bs.zivot_andrews_test(large_series, config=config)
    
    >>> # Wide breakpoint search (trim less)
    >>> config = ZivotAndrewsConfig(trim=0.10)
    >>> stat, bp, pval = bs.zivot_andrews_test(x, config=config)
    
    Notes
    -----
    The test searches for the breakpoint that gives the strongest evidence
    against the unit root hypothesis (minimum t-statistic). This is NOT
    necessarily the true breakpoint location - it's the point that maximizes
    evidence of stationarity.
    
    For high-persistence series (ρ near 1), detected breakpoints can be
    ±20-30 observations from the true break for n=100-200.
    
    References
    ----------
    .. [1] Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the
           great crash, the oil-price shock, and the unit-root hypothesis.
           Journal of Business & Economic Statistics, 10(3), 251-270.
    """
    regression: Literal['c', 't', 'both'] = 'both'
    max_lag: Optional[int] = None
    trim: float = 0.15
    timeout_secs: int = 300
    stride: int = 1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if self.regression not in ('c', 't', 'both'):
            raise ValueError(
                f"regression must be 'c', 't', or 'both', got '{self.regression}'"
            )
        
        if self.trim <= 0 or self.trim >= 0.5:
            raise ValueError(
                f"trim must be in (0, 0.5), got {self.trim}"
            )
        
        if self.max_lag is not None and self.max_lag < 0:
            raise ValueError(
                f"max_lag must be non-negative, got {self.max_lag}"
            )
        
        if self.timeout_secs <= 0:
            raise ValueError(
                f"timeout_secs must be positive, got {self.timeout_secs}"
            )
        
        if self.stride < 1:
            raise ValueError(
                f"stride must be >= 1, got {self.stride}"
            )


@dataclass
class VarianceRatioConfig:
    """Configuration for Lo-MacKinlay variance ratio test.
    
    The variance ratio test tests the random walk hypothesis by comparing
    variance of q-period vs 1-period differences.
    
    Parameters
    ----------
    lags : int, default=2
        Lag for variance ratio calculation (q in VR(q)).
        Must be >= 2.
        
        Common values: 2, 4, 8, 16 for different time horizons.
    
    test_type : {'homoskedastic', 'heteroskedastic', 'both'}, default='both'
        Type of test statistic:
        
        - 'homoskedastic': Assumes homoskedastic errors (classic Lo-MacKinlay)
        - 'heteroskedastic': Robust to heteroskedasticity (Wright, 2000)
        - 'both': Compute both test statistics
    
    Examples
    --------
    >>> from bunker_stats.tsa import VarianceRatioConfig
    >>> import bunker_stats as bs
    
    >>> # Test at quarterly lag
    >>> config = VarianceRatioConfig(lags=4)
    >>> vr, z, pval = bs.variance_ratio_test(x, config=config)
    
    >>> # Heteroskedasticity-robust test
    >>> config = VarianceRatioConfig(lags=2, test_type='heteroskedastic')
    >>> vr, z, pval = bs.variance_ratio_test(x, config=config)
    
    >>> # Test multiple lags
    >>> for q in [2, 4, 8, 16]:
    ...     config = VarianceRatioConfig(lags=q)
    ...     vr, z, pval = bs.variance_ratio_test(x, config=config)
    ...     print(f"VR({q}) = {vr:.3f}, p-value = {pval:.3f}")
    
    Notes
    -----
    Under the random walk hypothesis, VR(q) should be approximately 1.0.
    
    - VR(q) < 1: Suggests mean reversion (negative autocorrelation)
    - VR(q) > 1: Suggests momentum/trending (positive autocorrelation)
    
    For white noise (differenced random walk), theoretical VR(q) = 1/q.
    For example, white noise should have VR(2) ≈ 0.5, VR(4) ≈ 0.25, etc.
    
    References
    ----------
    .. [1] Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not
           follow random walks: Evidence from a simple specification test.
           Review of Financial Studies, 1(1), 41-66.
    .. [2] Wright, J. H. (2000). Alternative variance-ratio tests using ranks
           and signs. Journal of Business & Economic Statistics, 18(1), 1-9.
    """
    lags: int = 2
    test_type: Literal['homoskedastic', 'heteroskedastic', 'both'] = 'both'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if self.lags < 2:
            raise ValueError(
                f"lags must be >= 2, got {self.lags}"
            )
        
        if self.test_type not in ('homoskedastic', 'heteroskedastic', 'both'):
            raise ValueError(
                f"test_type must be 'homoskedastic', 'heteroskedastic', or 'both', "
                f"got '{self.test_type}'"
            )


@dataclass  
class ADFConfig:
    """Configuration for Augmented Dickey-Fuller unit root test.
    
    Parameters
    ----------
    regression : {'c', 'ct', 'ctt', 'n'}, default='c'
        Regression type:
        
        - 'c': Constant only
        - 'ct': Constant and trend
        - 'ctt': Constant, linear and quadratic trend
        - 'n': No constant, no trend
    
    max_lag : int or None, default=None
        Maximum lag for ADF regression. If None, uses ``floor(12*(n/100)^0.25)``.
    
    Examples
    --------
    >>> from bunker_stats.tsa import ADFConfig
    >>> import bunker_stats as bs
    
    >>> # Test with trend
    >>> config = ADFConfig(regression='ct')
    >>> stat, pval = bs.adf_test(x, config=config)
    
    >>> # Explicit max lag
    >>> config = ADFConfig(max_lag=8)
    >>> stat, pval = bs.adf_test(x, config=config)
    """
    regression: Literal['c', 'ct', 'ctt', 'n'] = 'c'
    max_lag: Optional[int] = None
    
    def __post_init__(self):
        self.validate()
    
    def validate(self) -> None:
        if self.regression not in ('c', 'ct', 'ctt', 'n'):
            raise ValueError(
                f"regression must be 'c', 'ct', 'ctt', or 'n', got '{self.regression}'"
            )
        
        if self.max_lag is not None and self.max_lag < 0:
            raise ValueError(
                f"max_lag must be non-negative, got {self.max_lag}"
            )


@dataclass
class PPConfig:
    """Configuration for Phillips-Perron unit root test.
    
    Parameters
    ----------
    regression : {'c', 'ct'}, default='c'
        Regression type:
        
        - 'c': Constant only  
        - 'ct': Constant and trend
    
    lags : int or None, default=None
        Number of lags for Newey-West long-run variance.
        If None, uses ``floor(12*(n/100)^0.25)``.
    
    Examples
    --------
    >>> from bunker_stats.tsa import PPConfig
    >>> import bunker_stats as bs
    
    >>> # Test with trend
    >>> config = PPConfig(regression='ct')
    >>> stat, pval = bs.pp_test(x, config=config)
    """
    regression: Literal['c', 'ct'] = 'c'
    lags: Optional[int] = None
    
    def __post_init__(self):
        self.validate()
    
    def validate(self) -> None:
        if self.regression not in ('c', 'ct'):
            raise ValueError(
                f"regression must be 'c' or 'ct', got '{self.regression}'"
            )
        
        if self.lags is not None and self.lags < 0:
            raise ValueError(
                f"lags must be non-negative, got {self.lags}"
            )


# Export all config classes
__all__ = [
    'KPSSConfig',
    'ZivotAndrewsConfig',
    'VarianceRatioConfig',
    'ADFConfig',
    'PPConfig',
]
