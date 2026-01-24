# bunker_stats/__init__.py

"""
Python facade for the Rust extension.

Key rule:
- Always bind to the *binary extension module*, not the pure-Python package wrapper.

We try (in order):
1) in-package extension: bunker_stats.bunker_stats_rs  (maturin can place it here)
2) binary inside the installed package: bunker_stats_rs.bunker_stats_rs
3) top-level binary module: bunker_stats_rs
"""

from __future__ import annotations

from typing import Any, Callable
import importlib
import numpy as _np
import warnings as _warnings

# --------------------
# Import the Rust binary module robustly
# --------------------
_rs = None

# 1) If extension is inside this package
try:
    _rs = importlib.import_module("bunker_stats.bunker_stats_rs")
except Exception:
    _rs = None

# 2) If extension is installed as a package that contains the binary module
if _rs is None:
    try:
        _rs = importlib.import_module("bunker_stats_rs.bunker_stats_rs")
    except Exception:
        _rs = None

# 3) If extension is installed as a top-level binary module
if _rs is None:
    try:
        _rs = importlib.import_module("bunker_stats_rs")
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import the Rust extension. Tried:\n"
            "  - bunker_stats.bunker_stats_rs\n"
            "  - bunker_stats_rs.bunker_stats_rs\n"
            "  - bunker_stats_rs\n"
        ) from e


def _missing(name: str) -> Callable[..., Any]:
    def _fn(*_a: Any, **_k: Any) -> Any:  # pragma: no cover
        raise AttributeError(
            f"Rust extension does not export '{name}'. "
            "You may be importing an old wheel. "
            "Run `maturin develop --release` in the repo root and verify imports."
        )
    return _fn


def _get_rs(*names: str) -> Callable[..., Any]:
    """
    Return the first attribute found in the Rust extension from `names`,
    otherwise raise a nice error mentioning all attempted names.
    """
    for n in names:
        fn = getattr(_rs, n, None)
        if fn is not None:
            return fn
    return _missing("/".join(names))


def _deprecated_alias(new_name: str, old_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap an old alias name (e.g. mean_np) so calling it emits a DeprecationWarning.
    """
    def _wrapped(*a: Any, **k: Any) -> Any:
        _warnings.warn(
            f"'{old_name}' is deprecated and will be removed in a future release. "
            f"Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return fn(*a, **k)
    _wrapped.__name__ = old_name
    return _wrapped


# --------------------
# Small Python fallbacks (only if a symbol is missing)
# --------------------
def _py_zscore(x: _np.ndarray) -> _np.ndarray:
    x = _np.asarray(x, dtype=_np.float64)
    m = _np.mean(x)
    s = _np.std(x, ddof=1)
    if not _np.isfinite(s) or s == 0.0:
        return _np.full_like(x, _np.nan, dtype=_np.float64)
    return (x - m) / s


# ======================================================================================
# Public API (clean names, no *_np)
# ======================================================================================

# --------------------
# basic stats (strict)
# --------------------
mean = _get_rs("mean", "mean_np")
std = _get_rs("std", "std_np")
var = _get_rs("var", "var_np")
zscore = _get_rs("zscore", "zscore_np") if hasattr(_rs, "zscore") or hasattr(_rs, "zscore_np") else _py_zscore

percentile = _get_rs("percentile", "percentile_np")
iqr = _get_rs("iqr_robust", "iqr_robust_np")  # Returns scalar IQR width with proper edge case handling
iqr_width = _get_rs("iqr_width", "iqr_width_np")
mad = _get_rs("mad", "mad_np")
skew = _get_rs("skew", "skew_np")
kurtosis = _get_rs("kurtosis", "kurtosis_np")
trimmed_mean = _get_rs("trimmed_mean", "trimmed_mean_np")

# --------------------
# Robust statistics - NEW policy-driven API (v0.2.9)
# --------------------
RobustStats = _get_rs("RobustStats")
robust_fit = _get_rs("robust_fit")
robust_score = _get_rs("robust_score")
rolling_median = _get_rs("rolling_median")

# --------------------
# Robust statistics - extended functions (legacy)
# --------------------
median = _get_rs("median", "median_np")
iqr_robust = _get_rs("iqr_robust", "iqr_robust_np")
winsorized_mean = _get_rs("winsorized_mean", "winsorized_mean_np")
trimmed_std = _get_rs("trimmed_std", "trimmed_std_np")
mad_std = _get_rs("mad_std", "mad_std_np")
biweight_midvariance = _get_rs("biweight_midvariance", "biweight_midvariance_np")
qn_scale = _get_rs("qn_scale", "qn_scale_np")
huber_location = _get_rs("huber_location", "huber_location_np")

# --------------------
# Robust statistics - NaN-aware variants
# --------------------
median_skipna = _get_rs("median_skipna", "median_skipna_np", "median_nan", "median_nan_np")
mad_skipna = _get_rs("mad_skipna", "mad_skipna_np", "mad_nan", "mad_nan_np")
trimmed_mean_skipna = _get_rs("trimmed_mean_skipna", "trimmed_mean_skipna_np", "trimmed_mean_nan", "trimmed_mean_nan_np")
iqr_skipna = _get_rs("iqr_skipna", "iqr_skipna_np", "iqr_nan", "iqr_nan_np")

# --------------------
# NaN-aware scalar stats (public naming: *_skipna)
# mean_skipna maps to either new export or prior conventions.
# --------------------
mean_skipna = _get_rs("mean_skipna", "mean_nan", "mean_nan_np", "mean_skipna_np")
std_skipna  = _get_rs("std_skipna",  "std_nan",  "std_nan_np",  "std_skipna_np")
var_skipna  = _get_rs("var_skipna",  "var_nan",  "var_nan_np",  "var_skipna_np")

# --------------------
# multi-dimensional operations
# --------------------
mean_axis = _get_rs("mean_axis", "mean_axis_np")
mean_over_last_axis_dyn = _get_rs("mean_over_last_axis_dyn", "mean_over_last_axis_dyn_np")

# --------------------
# rolling stats (strict)
# --------------------
rolling_mean = _get_rs("rolling_mean", "rolling_mean_np")
rolling_std = _get_rs("rolling_std", "rolling_std_np")
rolling_var = _get_rs("rolling_var", "rolling_var_np")
rolling_mean_std = _get_rs("rolling_mean_std", "rolling_mean_std_np")
rolling_zscore = _get_rs("rolling_zscore", "rolling_zscore_np")
ewma = _get_rs("ewma", "ewma_np")

# axis0 rolling
rolling_mean_axis0 = _get_rs("rolling_mean_axis0", "rolling_mean_axis0_np")
rolling_std_axis0 = _get_rs("rolling_std_axis0", "rolling_std_axis0_np")
rolling_mean_std_axis0 = _get_rs("rolling_mean_std_axis0", "rolling_mean_std_axis0_np")

# --------------------
# NaN-aware rolling (public naming: *_skipna)
# --------------------
rolling_mean_skipna = _get_rs("rolling_mean_skipna", "rolling_mean_nan", "rolling_mean_nan_np")
rolling_std_skipna  = _get_rs("rolling_std_skipna",  "rolling_std_nan",  "rolling_std_nan_np")
rolling_zscore_skipna = _get_rs("rolling_zscore_skipna", "rolling_zscore_nan", "rolling_zscore_nan_np")

# --------------------
# Welford + masks
# --------------------
welford = _get_rs("welford", "welford_np")
sign_mask = _get_rs("sign_mask", "sign_mask_np")
demean_with_signs = _get_rs("demean_with_signs", "demean_with_signs_np")

# --------------------
# outliers & scaling
# --------------------
iqr_outliers = _get_rs("iqr_outliers", "iqr_outliers_np")
zscore_outliers = _get_rs("zscore_outliers", "zscore_outliers_np")
minmax_scale = _get_rs("minmax_scale", "minmax_scale_np")
robust_scale = _get_rs("robust_scale", "robust_scale_np")
winsorize = _get_rs("winsorize", "winsorize_np")
quantile_bins = _get_rs("quantile_bins", "quantile_bins_np")

# --------------------
# diffs / cumulatives / ECDF
# --------------------
diff = _get_rs("diff", "diff_np")
pct_change = _get_rs("pct_change", "pct_change_np")
cumsum = _get_rs("cumsum", "cumsum_np")
cummean = _get_rs("cummean", "cummean_np")
ecdf = _get_rs("ecdf", "ecdf_np")

# --------------------
# covariance / correlation (already clean in your Rust exports)
# --------------------
cov = _get_rs("cov", "cov_np")
corr = _get_rs("corr", "corr_np")
cov_matrix = _get_rs("cov_matrix", "cov_matrix_np")
corr_matrix = _get_rs("corr_matrix", "corr_matrix_np")
cov_matrix_bias = _get_rs("cov_matrix_bias", "cov_matrix_bias_np")
cov_matrix_centered = _get_rs("cov_matrix_centered", "cov_matrix_centered_np")
cov_matrix_skipna = _get_rs("cov_matrix_skipna", "cov_matrix_skipna_np")
corr_matrix_skipna = _get_rs("corr_matrix_skipna", "corr_matrix_skipna_np")
corr_distance = _get_rs("corr_distance", "corr_distance_np")

xtx_matrix = _get_rs("xtx_matrix", "xtx_matrix_np")
xxt_matrix = _get_rs("xxt_matrix", "xxt_matrix_np")

pairwise_euclidean_cols = _get_rs("pairwise_euclidean_cols", "pairwise_euclidean_cols_np")
pairwise_cosine_cols = _get_rs("pairwise_cosine_cols", "pairwise_cosine_cols_np")

diag = _get_rs("diag", "diag_np")
trace = _get_rs("trace", "trace_np")
is_symmetric = _get_rs("is_symmetric", "is_symmetric_np")
rolling_cov = _get_rs("rolling_cov", "rolling_cov_np")
rolling_corr = _get_rs("rolling_corr", "rolling_corr_np")

# NaN-aware covariance / correlation (public naming: *_skipna)
cov_skipna = _get_rs("cov_skipna", "cov_nan", "cov_nan_np")
corr_skipna = _get_rs("corr_skipna", "corr_nan", "corr_nan_np")
rolling_cov_skipna = _get_rs("rolling_cov_skipna", "rolling_cov_nan", "rolling_cov_nan_np")
rolling_corr_skipna = _get_rs("rolling_corr_skipna", "rolling_corr_nan", "rolling_corr_nan_np")

# rolling linear-model primitives (skipna)
rolling_beta_skipna = _get_rs("rolling_beta_skipna", "rolling_beta_skipna_np")
rolling_linreg_skipna = _get_rs("rolling_linreg_skipna", "rolling_linreg_skipna_np")

# --------------------
# KDE
# --------------------
kde_gaussian = _get_rs("kde_gaussian", "kde_gaussian_np")

# --------------------
# INFERENCE MODULE - OPTIMIZED VERSION
# ============================================================================

# Existing tests (all with bug fixes and optimizations)
t_test_1samp = _get_rs("t_test_1samp", "t_test_1samp_np")
t_test_2samp = _get_rs("t_test_2samp", "t_test_2samp_np")
chi2_gof = _get_rs("chi2_gof", "chi2_gof_np")
chi2_independence = _get_rs("chi2_independence", "chi2_independence_np")
cohens_d_2samp = _get_rs("cohens_d_2samp", "cohens_d_2samp_np")
hedges_g_2samp = _get_rs("hedges_g_2samp", "hedges_g_2samp_np2")
mean_diff_ci = _get_rs("mean_diff_ci", "mean_diff_ci_np")
mann_whitney_u = _get_rs("mann_whitney_u", "mann_whitney_u_np")
ks_1samp = _get_rs("ks_1samp", "ks_1samp_np")

# NEW: ANOVA
f_test_oneway = _get_rs("f_test_oneway", "f_test_oneway_np")
levene_test = _get_rs("levene_test", "levene_test_np")

# NEW: Normality tests
jarque_bera = _get_rs("jarque_bera", "jarque_bera_np")
anderson_darling = _get_rs("anderson_darling", "anderson_darling_np")

# NEW: Correlation tests
pearson_corr_test = _get_rs("pearson_corr_test", "pearson_corr_test_np")
spearman_corr_test = _get_rs("spearman_corr_test", "spearman_corr_test_np")

# NEW: Variance tests
f_test_var = _get_rs("f_test_var", "f_test_var_np")
bartlett_test = _get_rs("bartlett_test", "bartlett_test_np")

# prefer new name, fallback to older wheels
hedges_g_2samp = _get_rs("hedges_g_2samp", "hedges_g_2samp_2", "hedges_g_2samp_np", "hedges_g_2samp_np2")

mean_diff_ci = _get_rs("mean_diff_ci", "mean_diff_ci_np")

# staged / optional
mann_whitney_u = _get_rs("mann_whitney_u", "mann_whitney_u_np")
ks_1samp = _get_rs("ks_1samp", "ks_1samp_np")

# utilities
pad_nan = _get_rs("pad_nan", "pad_nan_np")


# ======================================================================================
# Backward-compatible aliases (deprecated) — NOT part of the "surface API"
# ======================================================================================
mean_np = _deprecated_alias("mean", "mean_np", mean)
std_np = _deprecated_alias("std", "std_np", std)
var_np = _deprecated_alias("var", "var_np", var)
zscore_np = _deprecated_alias("zscore", "zscore_np", zscore)
percentile_np = _deprecated_alias("percentile", "percentile_np", percentile)
iqr_np = _deprecated_alias("iqr", "iqr_np", iqr)
iqr_width_np = _deprecated_alias("iqr_width", "iqr_width_np", iqr_width)
mad_np = _deprecated_alias("mad", "mad_np", mad)
skew_np = _deprecated_alias("skew", "skew_np", skew)
kurtosis_np = _deprecated_alias("kurtosis", "kurtosis_np", kurtosis)
trimmed_mean_np = _deprecated_alias("trimmed_mean", "trimmed_mean_np", trimmed_mean)

# Robust statistics - extended (deprecated _np versions)
median_np = _deprecated_alias("median", "median_np", median)
iqr_robust_np = _deprecated_alias("iqr_robust", "iqr_robust_np", iqr_robust)
winsorized_mean_np = _deprecated_alias("winsorized_mean", "winsorized_mean_np", winsorized_mean)
trimmed_std_np = _deprecated_alias("trimmed_std", "trimmed_std_np", trimmed_std)
mad_std_np = _deprecated_alias("mad_std", "mad_std_np", mad_std)
biweight_midvariance_np = _deprecated_alias("biweight_midvariance", "biweight_midvariance_np", biweight_midvariance)
qn_scale_np = _deprecated_alias("qn_scale", "qn_scale_np", qn_scale)
huber_location_np = _deprecated_alias("huber_location", "huber_location_np", huber_location)

# Robust statistics - skipna (deprecated _np versions)
median_skipna_np = _deprecated_alias("median_skipna", "median_skipna_np", median_skipna)
mad_skipna_np = _deprecated_alias("mad_skipna", "mad_skipna_np", mad_skipna)
trimmed_mean_skipna_np = _deprecated_alias("trimmed_mean_skipna", "trimmed_mean_skipna_np", trimmed_mean_skipna)
iqr_skipna_np = _deprecated_alias("iqr_skipna", "iqr_skipna_np", iqr_skipna)

mean_nan_np = _deprecated_alias("mean_skipna", "mean_nan_np", mean_skipna)
std_nan_np  = _deprecated_alias("std_skipna",  "std_nan_np",  std_skipna)
var_nan_np  = _deprecated_alias("var_skipna",  "var_nan_np",  var_skipna)

mean_axis_np = _deprecated_alias("mean_axis", "mean_axis_np", mean_axis)
mean_over_last_axis_dyn_np = _deprecated_alias("mean_over_last_axis_dyn", "mean_over_last_axis_dyn_np", mean_over_last_axis_dyn)

rolling_mean_np = _deprecated_alias("rolling_mean", "rolling_mean_np", rolling_mean)
rolling_std_np = _deprecated_alias("rolling_std", "rolling_std_np", rolling_std)
rolling_var_np = _deprecated_alias("rolling_var", "rolling_var_np", rolling_var)
rolling_mean_std_np = _deprecated_alias("rolling_mean_std", "rolling_mean_std_np", rolling_mean_std)
rolling_zscore_np = _deprecated_alias("rolling_zscore", "rolling_zscore_np", rolling_zscore)
ewma_np = _deprecated_alias("ewma", "ewma_np", ewma)

rolling_mean_axis0_np = _deprecated_alias("rolling_mean_axis0", "rolling_mean_axis0_np", rolling_mean_axis0)
rolling_std_axis0_np = _deprecated_alias("rolling_std_axis0", "rolling_std_axis0_np", rolling_std_axis0)
rolling_mean_std_axis0_np = _deprecated_alias("rolling_mean_std_axis0", "rolling_mean_std_axis0_np", rolling_mean_std_axis0)

rolling_mean_nan_np = _deprecated_alias("rolling_mean_skipna", "rolling_mean_nan_np", rolling_mean_skipna)
rolling_std_nan_np = _deprecated_alias("rolling_std_skipna", "rolling_std_nan_np", rolling_std_skipna)
rolling_zscore_nan_np = _deprecated_alias("rolling_zscore_skipna", "rolling_zscore_nan_np", rolling_zscore_skipna)

welford_np = _deprecated_alias("welford", "welford_np", welford)
sign_mask_np = _deprecated_alias("sign_mask", "sign_mask_np", sign_mask)
demean_with_signs_np = _deprecated_alias("demean_with_signs", "demean_with_signs_np", demean_with_signs)

iqr_outliers_np = _deprecated_alias("iqr_outliers", "iqr_outliers_np", iqr_outliers)
zscore_outliers_np = _deprecated_alias("zscore_outliers", "zscore_outliers_np", zscore_outliers)
minmax_scale_np = _deprecated_alias("minmax_scale", "minmax_scale_np", minmax_scale)
robust_scale_np = _deprecated_alias("robust_scale", "robust_scale_np", robust_scale)
winsorize_np = _deprecated_alias("winsorize", "winsorize_np", winsorize)
quantile_bins_np = _deprecated_alias("quantile_bins", "quantile_bins_np", quantile_bins)

diff_np = _deprecated_alias("diff", "diff_np", diff)
pct_change_np = _deprecated_alias("pct_change", "pct_change_np", pct_change)
cumsum_np = _deprecated_alias("cumsum", "cumsum_np", cumsum)
cummean_np = _deprecated_alias("cummean", "cummean_np", cummean)
ecdf_np = _deprecated_alias("ecdf", "ecdf_np", ecdf)

cov_np = _deprecated_alias("cov", "cov_np", cov)
corr_np = _deprecated_alias("corr", "corr_np", corr)
cov_matrix_np = _deprecated_alias("cov_matrix", "cov_matrix_np", cov_matrix)
corr_matrix_np = _deprecated_alias("corr_matrix", "corr_matrix_np", corr_matrix)
rolling_cov_np = _deprecated_alias("rolling_cov", "rolling_cov_np", rolling_cov)
rolling_corr_np = _deprecated_alias("rolling_corr", "rolling_corr_np", rolling_corr)

cov_nan_np = _deprecated_alias("cov_skipna", "cov_nan_np", cov_skipna)
corr_nan_np = _deprecated_alias("corr_skipna", "corr_nan_np", corr_skipna)
rolling_cov_nan_np = _deprecated_alias("rolling_cov_skipna", "rolling_cov_nan_np", rolling_cov_skipna)
rolling_corr_nan_np = _deprecated_alias("rolling_corr_skipna", "rolling_corr_nan_np", rolling_corr_skipna)

kde_gaussian_np = _deprecated_alias("kde_gaussian", "kde_gaussian_np", kde_gaussian)

t_test_1samp_np = _deprecated_alias("t_test_1samp", "t_test_1samp_np", t_test_1samp)
t_test_2samp_np = _deprecated_alias("t_test_2samp", "t_test_2samp_np", t_test_2samp)
chi2_gof_np = _deprecated_alias("chi2_gof", "chi2_gof_np", chi2_gof)
chi2_independence_np = _deprecated_alias("chi2_independence", "chi2_independence_np", chi2_independence)
cohens_d_2samp_np = _deprecated_alias("cohens_d_2samp", "cohens_d_2samp_np", cohens_d_2samp)
hedges_g_2samp_np = _deprecated_alias("hedges_g_2samp", "hedges_g_2samp_np", hedges_g_2samp)
mean_diff_ci_np = _deprecated_alias("mean_diff_ci", "mean_diff_ci_np", mean_diff_ci)

mann_whitney_u_np = _deprecated_alias("mann_whitney_u", "mann_whitney_u_np", mann_whitney_u)
ks_1samp_np = _deprecated_alias("ks_1samp", "ks_1samp_np", ks_1samp)

pad_nan_np = _deprecated_alias("pad_nan", "pad_nan_np", pad_nan)

## separate kernel math from PyO3 wrappers, Move algorithm bodies into *_core or *_impl functions that take &[f64], usize, etc. ,Keep #[pyfunction] wrappers either in:, src/py/*.rs wrappers, or, src/lib.rs wrapper section
bootstrap_mean = _get_rs("bootstrap_mean")
bootstrap_mean_ci = _get_rs("bootstrap_mean_ci")
bootstrap_ci = _get_rs("bootstrap_ci")
bootstrap_corr = _get_rs("bootstrap_corr")
jackknife_mean = _get_rs("jackknife_mean")
jackknife_mean_ci = _get_rs("jackknife_mean_ci")
permutation_test_corr = _get_rs("permutation_corr_test")
permutation_mean_diff_test = _get_rs("permutation_mean_diff_test")

# Time-series analysis - Stationarity Tests
# --------------------
adf_test = _get_rs("adf_test")
kpss_test = _get_rs("kpss_test")
pp_test = _get_rs("pp_test")
variance_ratio_test = _get_rs("variance_ratio_test")
zivot_andrews_test = _get_rs("zivot_andrews_test")
trend_stationarity_test = _get_rs("trend_stationarity_test")
integration_order_test = _get_rs("integration_order_test")
seasonal_diff_test = _get_rs("seasonal_diff_test")
seasonal_unit_root_test = _get_rs("seasonal_unit_root_test")

# --------------------
# Time-series analysis - Diagnostics
# --------------------
ljung_box = _get_rs("ljung_box")
durbin_watson = _get_rs("durbin_watson")
bg_test = _get_rs("bg_test")  # NOW FIXED - correct TSS calculation
box_pierce = _get_rs("box_pierce")
runs_test = _get_rs("runs_test")
acf_zero_crossing = _get_rs("acf_zero_crossing")

# --------------------
# Time-series analysis - ACF/PACF
# --------------------
acf = _get_rs("acf")
pacf = _get_rs("pacf")  # NOW uses Levinson-Durbin (10-50× faster)
acovf = _get_rs("acovf")
acf_with_ci = _get_rs("acf_with_ci")
ccf = _get_rs("ccf")

# Alternative PACF methods
pacf_yw = _get_rs("pacf_yw")
pacf_innovations = _get_rs("pacf_innovations")
pacf_burg = _get_rs("pacf_burg")

# --------------------
# Time-series analysis - Spectral (FFT-based)
# --------------------
periodogram = _get_rs("periodogram")  # NOW uses FFT (100× faster)
welch_psd = _get_rs("welch_psd")
cumulative_periodogram = _get_rs("cumulative_periodogram")
dominant_frequency = _get_rs("dominant_frequency")
spectral_entropy = _get_rs("spectral_entropy")
bartlett_psd = _get_rs("bartlett_psd")
spectral_peaks = _get_rs("spectral_peaks")
spectral_flatness = _get_rs("spectral_flatness")
band_power = _get_rs("band_power")
spectral_centroid = _get_rs("spectral_centroid")
spectral_rolloff = _get_rs("spectral_rolloff")

# --------------------
# Time-series analysis - Rolling Operations
# --------------------
rolling_autocorr = _get_rs("rolling_autocorr")
rolling_correlation = _get_rs("rolling_correlation")
rolling_autocorr_multi = _get_rs("rolling_autocorr_multi")

# O(1) rolling operations (these override existing rolling functions)
# NOTE: These are TSA-specific O(1) versions. Original rolling functions still available.
# rolling_mean = _get_rs("rolling_mean")  # CONFLICT - keep existing
# rolling_sum = _get_rs("rolling_sum")
# rolling_var = _get_rs("rolling_var")  # CONFLICT - keep existing  
# rolling_std = _get_rs("rolling_std")  # CONFLICT - keep existing
rolling_min = _get_rs("rolling_min")
rolling_max = _get_rs("rolling_max")
rolling_range = _get_rs("rolling_range")
rolling_count_above = _get_rs("rolling_count_above")
rolling_pct_above = _get_rs("rolling_pct_above")
# rolling_zscore = _get_rs("rolling_zscore")  # CONFLICT - keep existing
rolling_cv = _get_rs("rolling_cv")


# --------------------
# Distribution functions - Normal
# --------------------
norm_pdf = _get_rs("norm_pdf")
norm_logpdf = _get_rs("norm_logpdf")
norm_cdf = _get_rs("norm_cdf")
norm_sf = _get_rs("norm_sf")
norm_logsf = _get_rs("norm_logsf")
norm_cumhazard = _get_rs("norm_cumhazard")
norm_ppf = _get_rs("norm_ppf")

# --------------------
# Distribution functions - Exponential
# --------------------
exp_pdf = _get_rs("exp_pdf")
exp_logpdf = _get_rs("exp_logpdf")
exp_cdf = _get_rs("exp_cdf")
exp_sf = _get_rs("exp_sf")
exp_logsf = _get_rs("exp_logsf")
exp_cumhazard = _get_rs("exp_cumhazard")
exp_ppf = _get_rs("exp_ppf")

# --------------------
# Distribution functions - Uniform
# --------------------
unif_pdf = _get_rs("unif_pdf")
unif_logpdf = _get_rs("unif_logpdf")
unif_cdf = _get_rs("unif_cdf")
unif_sf = _get_rs("unif_sf")
unif_logsf = _get_rs("unif_logsf")
unif_cumhazard = _get_rs("unif_cumhazard")
unif_ppf = _get_rs("unif_ppf")


# ======================================================================================
# Resampling config objects (v0.2.9 ergonomics layer)
# ======================================================================================

# These provide ergonomic wrappers around the flat Rust resampling functions
# with validation, NaN handling, and helpful error messages.

try:
    from bunker_stats.resampling import (
        # Config dataclasses
        BootstrapConfig,
        BootstrapCorrConfig,
        PermutationConfig,
        JackknifeConfig,
        
        # Convenience functions
        bootstrap,
        bootstrap_corr,
        permutation_test,
        jackknife,
    )
    
    _resampling_config_exports = [
        "BootstrapConfig",
        "BootstrapCorrConfig",
        "PermutationConfig",
        "JackknifeConfig",
        "bootstrap",
        "bootstrap_corr",
        "permutation_test",
        "jackknife",
    ]
    
except ImportError:
    # If resampling module not available, provide empty list
    # (This can happen during development or if files haven't been created yet)
    _resampling_config_exports = []


# ======================================================================================
# Rolling statistics config (v0.2.9 ergonomics layer)
# ======================================================================================

# NEW v0.2.9: Policy-driven rolling statistics with composable configuration
# This provides an ergonomic wrapper around the fused Rust rolling kernels
try:
    from bunker_stats.rolling import (
        # Config dataclasses
        RollingConfig,
        
        # Main user-facing class
        Rolling,
        
        # Type hints
        Alignment,
        NanPolicy,
    )
    
    _rolling_config_exports = [
        "Rolling",
        "RollingConfig",
        "Alignment",
        "NanPolicy",
    ]
    
except ImportError:
    # If rolling module not available, provide empty list
    # (This can happen during development or if files haven't been created yet)
    _rolling_config_exports = []

# NEW v0.2.9: Low-level multi-stat functions (for advanced users)
rolling_multi = _get_rs("rolling_multi", "rolling_multi_np")
rolling_multi_axis0 = _get_rs("rolling_multi_axis0", "rolling_multi_axis0_np")


# ======================================================================================
# Public surface exports (clean names only)
# ======================================================================================
__all__ = [
    # scalar
    "mean", "std", "var", "zscore",
    "percentile", "iqr", "iqr_width", "mad", "skew", "kurtosis", "trimmed_mean",
    "mean_skipna", "std_skipna", "var_skipna",

    # Robust statistics - NEW policy-driven API (v0.2.9)
    "RobustStats", "robust_fit", "robust_score", "rolling_median",

    # Robust statistics - extended (legacy)
    "median", "iqr_robust", "winsorized_mean", "trimmed_std", "mad_std",
    "biweight_midvariance", "qn_scale", "huber_location",
    
    # Robust statistics - skipna
    "median_skipna", "mad_skipna", "trimmed_mean_skipna", "iqr_skipna",

    # multi-d
    "mean_axis", "mean_over_last_axis_dyn",

    # rolling
    "rolling_mean", "rolling_std", "rolling_var", "rolling_mean_std", "rolling_zscore", "ewma",
    "rolling_mean_axis0", "rolling_std_axis0", "rolling_mean_std_axis0",
    "rolling_mean_skipna", "rolling_std_skipna", "rolling_zscore_skipna",
    
    # NEW v0.2.9: Multi-stat fused functions
    "rolling_multi", "rolling_multi_axis0",
    
    # NEW v0.2.9: Composable rolling API
    *_rolling_config_exports,

    # welford/masks
    "welford", "sign_mask", "demean_with_signs",

    # outliers/scaling
    "iqr_outliers", "zscore_outliers", "minmax_scale", "robust_scale", "winsorize", "quantile_bins",

    # diffs/cums/ecdf
    "diff", "pct_change", "cumsum", "cummean", "ecdf",

    # cov/corr
    "cov", "corr", "cov_matrix", "cov_matrix_bias", "cov_matrix_centered", "cov_matrix_skipna",
    "corr_matrix", "corr_matrix_skipna", "corr_distance",
    "xtx_matrix", "xxt_matrix",
    "pairwise_euclidean_cols", "pairwise_cosine_cols",
    "diag", "trace", "is_symmetric",
    "rolling_cov", "rolling_corr",
    "cov_skipna", "corr_skipna", "rolling_cov_skipna", "rolling_corr_skipna",
    "rolling_beta_skipna", "rolling_linreg_skipna",

    # kde
    "kde_gaussian",

    # inference
    # Inference - existing (now with bug fixes)
    "t_test_1samp", "t_test_2samp", 
    "chi2_gof", "chi2_independence",
    "cohens_d_2samp", "hedges_g_2samp", "mean_diff_ci",
    "mann_whitney_u", "ks_1samp",
    
    # Inference - NEW: ANOVA
    "f_test_oneway", "levene_test",
    
    # Inference - NEW: Normality
    "jarque_bera", "anderson_darling",
    
    # Inference - NEW: Correlation tests
    "pearson_corr_test", "spearman_corr_test",
    
    # Inference - NEW: Variance tests
    "f_test_var", "bartlett_test",


    # utilities
    "pad_nan",
    
    # NEW: resampling (bootstrap & jackknife) - flat functions
    "bootstrap_mean", "bootstrap_mean_ci", "bootstrap_ci", "bootstrap_corr",
    "jackknife_mean", "jackknife_mean_ci",
    "permutation_test_corr", "permutation_mean_diff_test",
    
    # NEW: resampling config objects (v0.2.9)
    *_resampling_config_exports,
    
    # NEW: time series analysis - stationarity tests
    "adf_test", "kpss_test", "pp_test",
    "variance_ratio_test", "zivot_andrews_test", "trend_stationarity_test",
    "integration_order_test", "seasonal_diff_test", "seasonal_unit_root_test",
    
    # NEW: time series analysis - diagnostics
    "ljung_box", "durbin_watson", "bg_test",  # bg_test NOW FIXED
    "box_pierce", "runs_test", "acf_zero_crossing",
    
    # NEW: time series analysis - autocorrelation
    "acf", "pacf", "acovf", "acf_with_ci", "ccf",
    "pacf_yw", "pacf_innovations", "pacf_burg",
    "rolling_autocorr", "rolling_correlation", "rolling_autocorr_multi",
    
    # NEW: time series analysis - spectral (FFT-based)
    "periodogram", "welch_psd", "cumulative_periodogram",
    "dominant_frequency", "spectral_entropy", "bartlett_psd",
    "spectral_peaks", "spectral_flatness", "band_power",
    "spectral_centroid", "spectral_rolloff",
    
    # NEW: time series analysis - rolling helpers
    "rolling_min", "rolling_max", "rolling_range",
    "rolling_count_above", "rolling_pct_above", "rolling_cv",
    
    # NEW: distribution functions - normal
    "norm_pdf", "norm_logpdf", "norm_cdf", "norm_sf", "norm_logsf", "norm_cumhazard", "norm_ppf",
    
    # NEW: distribution functions - exponential
    "exp_pdf", "exp_logpdf", "exp_cdf", "exp_sf", "exp_logsf", "exp_cumhazard", "exp_ppf",
    
    # NEW: distribution functions - uniform
    "unif_pdf", "unif_logpdf", "unif_cdf", "unif_sf", "unif_logsf", "unif_cumhazard", "unif_ppf",

]