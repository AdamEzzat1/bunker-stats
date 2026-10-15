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

# v0.3.1: additional inference functions
t_test_paired = _get_rs("t_test_paired", "t_test_paired_np")
p_adjust = _get_rs("p_adjust", "p_adjust_np")
proportion_ztest = _get_rs("proportion_ztest", "proportion_ztest_np")
two_proportions_ztest = _get_rs("two_proportions_ztest", "two_proportions_ztest_np")
corr_ci = _get_rs("corr_ci", "corr_ci_np")
var_ci = _get_rs("var_ci", "var_ci_np")
odds_ratio = _get_rs("odds_ratio", "odds_ratio_np")
rank_biserial = _get_rs("rank_biserial", "rank_biserial_np")
cliffs_delta = _get_rs("cliffs_delta", "cliffs_delta_np")
anova_effect_sizes = _get_rs("anova_effect_sizes", "anova_effect_sizes_np")
normality_summary = _get_rs("normality_summary", "normality_summary_np")

# prefer new name, fallback to older wheels
hedges_g_2samp = _get_rs("hedges_g_2samp", "hedges_g_2samp_2", "hedges_g_2samp_np", "hedges_g_2samp_np2")

mean_diff_ci = _get_rs("mean_diff_ci", "mean_diff_ci_np")

# staged / optional
mann_whitney_u = _get_rs("mann_whitney_u", "mann_whitney_u_np")
ks_1samp = _get_rs("ks_1samp", "ks_1samp_np")

# utilities
pad_nan = _get_rs("pad_nan", "pad_nan_np")


# ======================================================================================
# Rich result objects (opt-in) — bunker_stats.infer.* via rich=True
# ======================================================================================
# The hypothesis tests above return plain dicts by default. Re-bind each to a thin
# facade that ALSO accepts `rich=True`, returning a rich result object (tuple-
# unpackable, with .to_dict() / .info() / .conclusion()). The facade forwards every
# other argument untouched, so the default dict return is byte-for-byte unchanged.
#
# NOTE: the actual re-binding happens further down (see `_apply_rich_inference`),
# AFTER the hand-written "modern facade" defs (e.g. `def t_test_2samp`) so that we
# wrap the final public callables rather than the raw kernels they shadow.
from bunker_stats.infer.facade import wrap_inference as _wrap_inference

from bunker_stats.infer import (  # noqa: E402  (result classes for direct use)
    TTestResult,
    ChiSquareResult,
    MannWhitneyResult,
    KSResult,
    ANOVAResult,
    CorrelationTestResult,
    NormalityResult,
)


# ======================================================================================
# Backward-compatible aliases (deprecated) — NOT part of the "surface API"
# ======================================================================================
mean_np = _deprecated_alias("mean", "mean_np", mean)
std_np = _deprecated_alias("std", "std_np", std)
var_np = _deprecated_alias("var", "var_np", var)
zscore_np = _deprecated_alias("zscore", "zscore_np", zscore)
percentile_np = _deprecated_alias("percentile", "percentile_np", percentile)
# `iqr_np` historically returned the (q1, q3, width) tuple from the Rust kernel.
# Preserve that return shape so existing callers (and parity tests) keep working;
# aliasing it to the scalar `iqr` was a silent breaking change. New code should
# use `iqr` (scalar width) or `iqr_width`.
iqr_np = _deprecated_alias("iqr", "iqr_np", _get_rs("iqr_np"))
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

        # Rich result objects
        BootstrapResult,
        PermutationTestResult,

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
        "BootstrapResult",
        "PermutationTestResult",
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

        # Rich result object
        RollingResult,

        # Type hints
        Alignment,
        NanPolicy,
    )

    _rolling_config_exports = [
        "Rolling",
        "RollingConfig",
        "RollingResult",
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
# Modern keyword layer (v0.3 API)
#
# Design rules:
#   1. One public name per statistic; NaN handling is a `skipna=` keyword, not a
#      twin function. The strict and skip-NaN Rust kernels stay separate under
#      the hood (no branch in the hot loop) — dispatch happens once, here.
#   2. Options are keyword-only with sensible defaults (`equal_var=True`,
#      `lower_q=0.05`), so `bs.t_test_2samp(x, y)` just works.
#   3. Keyword names carry their unit (`lower_q` is a quantile in [0, 1];
#      `q` in `percentile` is in [0, 100], matching numpy).
# The *_skipna names remain available and are NOT deprecated; they are the same
# kernels the `skipna=True` path dispatches to.
# ======================================================================================

zscore_skipna = _get_rs("zscore_skipna", "zscore_skipna_np")


def _check_window(window) -> int:
    """Validate an integer window argument at the facade boundary.

    Converts index-like values (Python ints, numpy integer scalars) and raises
    ValueError for window < 1 — including negative values, which would
    otherwise surface as OverflowError from the Rust usize conversion.
    """
    import operator

    w = operator.index(window)
    if w < 1:
        raise ValueError("window must be >= 1")
    return w


_VALID_ALTERNATIVES = ("two-sided", "less", "greater")


def _check_alternative(alternative: str) -> str:
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {list(_VALID_ALTERNATIVES)}, got {alternative!r}"
        )
    return alternative


# Preserve raw kernel bindings before rebinding the public names to wrappers.
_strict = {
    "mean": mean, "std": std, "var": var, "median": median, "mad": mad,
    "zscore": zscore, "iqr": iqr, "trimmed_mean": trimmed_mean,
    "cov": cov, "corr": corr, "cov_matrix": cov_matrix, "corr_matrix": corr_matrix,
    "rolling_mean": rolling_mean, "rolling_std": rolling_std,
    "rolling_zscore": rolling_zscore, "rolling_cov": rolling_cov,
    "rolling_corr": rolling_corr, "winsorize": winsorize,
    "t_test_2samp": t_test_2samp, "cohens_d_2samp": cohens_d_2samp,
    "hedges_g_2samp": hedges_g_2samp,
}
_skipna_kernel = {
    "mean": mean_skipna, "std": std_skipna, "var": var_skipna,
    "median": median_skipna, "mad": mad_skipna, "zscore": zscore_skipna,
    "iqr": iqr_skipna, "trimmed_mean": trimmed_mean_skipna,
    "cov": cov_skipna, "corr": corr_skipna,
    "cov_matrix": cov_matrix_skipna, "corr_matrix": corr_matrix_skipna,
    "rolling_cov": rolling_cov_skipna, "rolling_corr": rolling_corr_skipna,
    "rolling_mean": rolling_mean_skipna, "rolling_std": rolling_std_skipna,
    "rolling_zscore": rolling_zscore_skipna,
}


def mean(x, *, skipna: bool = False):
    """Arithmetic mean. `skipna=True` ignores NaNs (numpy.nanmean semantics)."""
    return _skipna_kernel["mean"](x) if skipna else _strict["mean"](x)

def std(x, *, skipna: bool = False):
    """Sample standard deviation (ddof=1). `skipna=True` ignores NaNs."""
    return _skipna_kernel["std"](x) if skipna else _strict["std"](x)

def var(x, *, skipna: bool = False):
    """Sample variance (ddof=1). `skipna=True` ignores NaNs."""
    return _skipna_kernel["var"](x) if skipna else _strict["var"](x)

def median(x, *, skipna: bool = False):
    """Median. `skipna=True` ignores NaNs (numpy.nanmedian semantics)."""
    return _skipna_kernel["median"](x) if skipna else _strict["median"](x)

def mad(x, *, skipna: bool = False):
    """Median absolute deviation (unscaled). `skipna=True` ignores NaNs."""
    return _skipna_kernel["mad"](x) if skipna else _strict["mad"](x)

def zscore(x, *, skipna: bool = False):
    """Standard scores using the sample std (ddof=1). `skipna=True` ignores NaNs."""
    return _skipna_kernel["zscore"](x) if skipna else _strict["zscore"](x)

def iqr(x, *, skipna: bool = False):
    """Interquartile range width (Q3 - Q1). `skipna=True` ignores NaNs."""
    return _skipna_kernel["iqr"](x) if skipna else _strict["iqr"](x)

def trimmed_mean(x, proportion_to_cut: float = 0.1, *, skipna: bool = False):
    """Mean after trimming `proportion_to_cut` from each tail (scipy.trim_mean)."""
    k = _skipna_kernel["trimmed_mean"] if skipna else _strict["trimmed_mean"]
    return k(x, proportion_to_cut)

def cov(x, y, *, skipna: bool = False):
    """Sample covariance (ddof=1). `skipna=True` uses pairwise-complete observations."""
    k = _skipna_kernel["cov"] if skipna else _strict["cov"]
    return k(x, y)

def corr(x, y, *, skipna: bool = False):
    """Pearson correlation. `skipna=True` uses pairwise-complete observations."""
    k = _skipna_kernel["corr"] if skipna else _strict["corr"]
    return k(x, y)

def cov_matrix(X, *, skipna: bool = False):
    """Covariance matrix, columns as variables (numpy.cov(rowvar=False, ddof=1))."""
    k = _skipna_kernel["cov_matrix"] if skipna else _strict["cov_matrix"]
    return k(X)

def corr_matrix(X, *, skipna: bool = False):
    """Correlation matrix, columns as variables (numpy.corrcoef(rowvar=False))."""
    k = _skipna_kernel["corr_matrix"] if skipna else _strict["corr_matrix"]
    return k(X)

def rolling_mean(x, window: int, *, skipna: bool = False):
    """Trailing rolling mean. Strict: length n-window+1; `skipna=True`: length n,
    pandas min_periods=1 semantics."""
    window = _check_window(window)
    k = _skipna_kernel["rolling_mean"] if skipna else _strict["rolling_mean"]
    return k(x, window)

def rolling_std(x, window: int, *, skipna: bool = False):
    """Trailing rolling sample std (ddof=1). See `rolling_mean` for shape rules."""
    window = _check_window(window)
    k = _skipna_kernel["rolling_std"] if skipna else _strict["rolling_std"]
    return k(x, window)

def rolling_zscore(x, window: int, *, skipna: bool = False):
    """Rolling standard score within each trailing window."""
    window = _check_window(window)
    k = _skipna_kernel["rolling_zscore"] if skipna else _strict["rolling_zscore"]
    return k(x, window)

def rolling_cov(x, y, window: int, *, skipna: bool = False):
    """Trailing rolling covariance. `skipna=True` matches pandas
    rolling(window).cov default semantics (NaN unless the window is complete)."""
    window = _check_window(window)
    k = _skipna_kernel["rolling_cov"] if skipna else _strict["rolling_cov"]
    return k(x, y, window)

def rolling_corr(x, y, window: int, *, skipna: bool = False):
    """Trailing rolling Pearson correlation. See `rolling_cov` for NaN rules."""
    window = _check_window(window)
    k = _skipna_kernel["rolling_corr"] if skipna else _strict["rolling_corr"]
    return k(x, y, window)

def winsorize(x, *, lower_q: float = 0.05, upper_q: float = 0.95):
    """Clip tails at the `lower_q` / `upper_q` quantiles (both in [0, 1])."""
    return _strict["winsorize"](x, lower_q, upper_q)

_winsorized_mean_kernel = winsorized_mean

def winsorized_mean(x, lower_q: float = 0.05, upper_q: float = 0.95):
    """Winsorized mean with tail bounds given as quantile fractions in [0, 1].

    The underlying kernel works in percentile units [0, 100]; this wrapper
    validates fraction inputs and converts, so `winsorize` and
    `winsorized_mean` share one unit convention.
    """
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError(
            "winsorized_mean quantiles must satisfy 0 <= lower_q < upper_q <= 1"
        )
    return _winsorized_mean_kernel(x, lower_q * 100.0, upper_q * 100.0)

_quantile_bins_kernel = quantile_bins

def quantile_bins(x, n_bins: int):
    """Assign each value to one of `n_bins` quantile bins (n_bins >= 1)."""
    import operator

    n = operator.index(n_bins)
    if n < 1:
        raise ValueError("n_bins must be >= 1")
    return _quantile_bins_kernel(x, n)

def t_test_2samp(x, y, *, equal_var: bool = True, alternative: str = "two-sided"):
    """Two-sample t-test. `equal_var=False` gives Welch's t (scipy semantics)."""
    return _strict["t_test_2samp"](x, y, equal_var, _check_alternative(alternative))

def cohens_d_2samp(x, y, *, pooled: bool = True):
    """Cohen's d effect size for two samples."""
    return _strict["cohens_d_2samp"](x, y, pooled)

def hedges_g_2samp(x, y, *, pooled: bool = True):
    """Hedges' g (small-sample corrected Cohen's d)."""
    return _strict["hedges_g_2samp"](x, y, pooled)


# Apply the opt-in `rich=True` wrappers now that every public inference callable
# (raw kernels AND the modern facades above, e.g. `t_test_2samp`) is in its final
# form. Wrapping here means the facade wraps the real public function, not a raw
# kernel it shadows.
globals().update(_wrap_inference({
    "t_test_1samp": t_test_1samp,
    "t_test_2samp": t_test_2samp,
    "t_test_paired": t_test_paired,
    "chi2_gof": chi2_gof,
    "chi2_independence": chi2_independence,
    "mann_whitney_u": mann_whitney_u,
    "ks_1samp": ks_1samp,
    "f_test_oneway": f_test_oneway,
    "pearson_corr_test": pearson_corr_test,
    "spearman_corr_test": spearman_corr_test,
    "jarque_bera": jarque_bera,
    "anderson_darling": anderson_darling,
}))

# Matrix (corr_matrix / cov_matrix) and robust (robust_fit / *_outliers) rich
# wrappers, applied after their modern facades are defined, same as inference.
from bunker_stats.matrix.facade import wrap_matrix as _wrap_matrix
from bunker_stats.robust.facade import wrap_robust as _wrap_robust

globals().update(_wrap_matrix({
    "corr_matrix": corr_matrix,
    "cov_matrix": cov_matrix,
}))
globals().update(_wrap_robust({
    "robust_fit": robust_fit,
    "iqr_outliers": iqr_outliers,
    "zscore_outliers": zscore_outliers,
}))

from bunker_stats.matrix import (  # noqa: E402
    CorrelationMatrixResult,
    CovarianceMatrixResult,
)
from bunker_stats.robust import (  # noqa: E402
    RobustFitResult,
    OutlierResult,
)


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

    # Inference - v0.3.1: additional functions
    "t_test_paired", "p_adjust", "proportion_ztest", "two_proportions_ztest",
    "corr_ci", "var_ci", "odds_ratio",
    "rank_biserial", "cliffs_delta", "anova_effect_sizes", "normality_summary",

    # Rich result objects (returned via rich=True, or from Rolling.result())
    "TTestResult", "ChiSquareResult", "MannWhitneyResult", "KSResult",
    "ANOVAResult", "CorrelationTestResult", "NormalityResult",
    "CorrelationMatrixResult", "CovarianceMatrixResult",
    "RobustFitResult", "OutlierResult",
    "RollingResult",


    # utilities
    "pad_nan",
    
    # NEW: resampling (bootstrap & jackknife) - flat functions
    # NOTE: "bootstrap_corr" is intentionally NOT listed here — the resampling
    # config layer rebinds that name and exports it via
    # _resampling_config_exports; listing it twice duplicated __all__.
    "bootstrap_mean", "bootstrap_mean_ci", "bootstrap_ci",
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

# ======================================================================================
# Optional notebook / reporting layer  (pandas-backed, lazily imported)
# ======================================================================================
# `bunker_stats.notebook` is the ergonomic pandas/Jupyter bridge. pandas is NOT a
# runtime dependency of bunker-stats, so the submodule is resolved through PEP 562
# `__getattr__` rather than imported at package import time. This keeps
# `import bunker_stats` numpy-only while still allowing `bs.notebook.robust_summary(df)`
# after `pip install "bunker-stats-rs[notebook]"`.
#
# Deliberately NOT added to __all__: this facade's __all__ is a list of callables
# (enforced by tests/test_hardening_v030.py) and drives `from bunker_stats import *`.
# The submodules stay reachable via attribute access and are advertised in __dir__
# so tab-completion still finds them.
_LAZY_SUBMODULES = ("notebook", "pandas", "pandas_helpers")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache so subsequent lookups skip __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
