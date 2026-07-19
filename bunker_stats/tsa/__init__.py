"""
Time Series Analysis (TSA) Submodule
=====================================

Configuration classes and result types for TSA functions.

Config Classes
--------------
- KPSSConfig: Configuration for KPSS stationarity test
- ZivotAndrewsConfig: Configuration for Zivot-Andrews structural break test
- VarianceRatioConfig: Configuration for Lo-MacKinlay variance ratio test
- ADFConfig: Configuration for Augmented Dickey-Fuller test
- PPConfig: Configuration for Phillips-Perron test

Result Classes
--------------
- KPSSResult: Rich result from KPSS test (with metadata)
- ZivotAndrewsResult: Rich result from Zivot-Andrews test
- VarianceRatioResult: Rich result from variance ratio test
- ADFResult: Rich result from ADF test
- PPResult: Rich result from PP test

Usage
-----
Config classes can be used with any TSA function:

>>> from bunker_stats.tsa import KPSSConfig
>>> import bunker_stats as bs
>>>
>>> # Create config
>>> config = KPSSConfig(regression='ct', bandwidth_method='conservative')
>>>
>>> # Use with test
>>> stat, pval = bs.kpss_test(x, config=config)

Or use kwargs directly (backward compatible):

>>> stat, pval = bs.kpss_test(x, regression='ct')

With metadata (Phase 2B):

>>> result = bs.kpss_test(x, metadata=True)
>>> print(result.info())
>>> stat, pval = result  # Tuple unpacking still works
"""

# Import config classes
from .config import (
    KPSSConfig,
    ZivotAndrewsConfig,
    VarianceRatioConfig,
    ADFConfig,
    PPConfig,
)

# Import result classes
from .types import (
    KPSSResult,
    VarianceRatioResult,
    ZivotAndrewsResult,
    ADFResult,
    PPResult,
)

# Re-export the TSA computational functions from the root package, so callers
# can do `from bunker_stats.tsa import zivot_andrews_test` next to the config
# and result classes. These are the same callables as ``bunker_stats.<name>``.
# (Root does not import this submodule, so there is no circular-import risk.)
from bunker_stats import (
    # Stationarity / unit-root tests
    adf_test,
    kpss_test,
    pp_test,
    variance_ratio_test,
    zivot_andrews_test,
    trend_stationarity_test,
    integration_order_test,
    seasonal_diff_test,
    seasonal_unit_root_test,
    # Serial-correlation diagnostics
    ljung_box,
    durbin_watson,
    bg_test,
    box_pierce,
    runs_test,
    acf_zero_crossing,
    # Autocorrelation / partial autocorrelation
    acf,
    pacf,
    acovf,
    acf_with_ci,
    ccf,
    pacf_yw,
    pacf_innovations,
    pacf_burg,
    # Rolling time-series operations
    rolling_autocorr,
    rolling_correlation,
    rolling_autocorr_multi,
    # Spectral analysis
    periodogram,
    welch_psd,
    cumulative_periodogram,
    dominant_frequency,
    spectral_entropy,
    spectral_peaks,
    spectral_flatness,
    band_power,
    spectral_centroid,
    spectral_rolloff,
)

_TSA_FUNCTIONS = [
    "adf_test", "kpss_test", "pp_test", "variance_ratio_test",
    "zivot_andrews_test", "trend_stationarity_test", "integration_order_test",
    "seasonal_diff_test", "seasonal_unit_root_test",
    "ljung_box", "durbin_watson", "bg_test", "box_pierce", "runs_test",
    "acf_zero_crossing",
    "acf", "pacf", "acovf", "acf_with_ci", "ccf",
    "pacf_yw", "pacf_innovations", "pacf_burg",
    "rolling_autocorr", "rolling_correlation", "rolling_autocorr_multi",
    "periodogram", "welch_psd", "cumulative_periodogram", "dominant_frequency",
    "spectral_entropy", "spectral_peaks", "spectral_flatness", "band_power",
    "spectral_centroid", "spectral_rolloff",
]

# Public API
__all__ = [
    # Config classes
    'KPSSConfig',
    'ZivotAndrewsConfig',
    'VarianceRatioConfig',
    'ADFConfig',
    'PPConfig',

    # Result classes
    'KPSSResult',
    'VarianceRatioResult',
    'ZivotAndrewsResult',
    'ADFResult',
    'PPResult',
] + _TSA_FUNCTIONS