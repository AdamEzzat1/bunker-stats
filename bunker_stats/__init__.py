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


# --------------------
# Bindings (prefer Rust)
# --------------------
# basic stats
mean_np = getattr(_rs, "mean_np", _missing("mean_np"))
std_np = getattr(_rs, "std_np", _missing("std_np"))
var_np = getattr(_rs, "var_np", _missing("var_np"))
zscore_np = getattr(_rs, "zscore_np", _py_zscore)
percentile_np = getattr(_rs, "percentile_np", _missing("percentile_np"))
iqr_np = getattr(_rs, "iqr_np", _missing("iqr_np"))
iqr_width_np = getattr(_rs, "iqr_width_np", _missing("iqr_width_np"))
mad_np = getattr(_rs, "mad_np", _missing("mad_np"))
skew_np = getattr(_rs, "skew_np", _missing("skew_np"))
kurtosis_np = getattr(_rs, "kurtosis_np", _missing("kurtosis_np"))
trimmed_mean_np = getattr(_rs, "trimmed_mean_np", _missing("trimmed_mean_np"))

# NaN-aware scalar stats (support either naming convention)
mean_nan_np = getattr(_rs, "mean_nan_np", getattr(_rs, "mean_skipna_np", _missing("mean_nan_np/mean_skipna_np")))
std_nan_np  = getattr(_rs, "std_nan_np",  getattr(_rs, "std_skipna_np",  _missing("std_nan_np/std_skipna_np")))
var_nan_np  = getattr(_rs, "var_nan_np",  getattr(_rs, "var_skipna_np",  _missing("var_nan_np/var_skipna_np")))

# multi-dimensional operations
mean_axis_np = getattr(_rs, "mean_axis_np", _missing("mean_axis_np"))
mean_over_last_axis_dyn_np = getattr(_rs, "mean_over_last_axis_dyn_np", _missing("mean_over_last_axis_dyn_np"))

# rolling stats
rolling_mean_np = getattr(_rs, "rolling_mean_np", _missing("rolling_mean_np"))
rolling_std_np = getattr(_rs, "rolling_std_np", _missing("rolling_std_np"))
rolling_var_np = getattr(_rs, "rolling_var_np", _missing("rolling_var_np"))
rolling_mean_std_np = getattr(_rs, "rolling_mean_std_np", _missing("rolling_mean_std_np"))
rolling_zscore_np = getattr(_rs, "rolling_zscore_np", _missing("rolling_zscore_np"))
ewma_np = getattr(_rs, "ewma_np", _missing("ewma_np"))

# axis0 rolling
rolling_mean_axis0_np = getattr(_rs, "rolling_mean_axis0_np", _missing("rolling_mean_axis0_np"))
rolling_std_axis0_np = getattr(_rs, "rolling_std_axis0_np", _missing("rolling_std_axis0_np"))
rolling_mean_std_axis0_np = getattr(_rs, "rolling_mean_std_axis0_np", _missing("rolling_mean_std_axis0_np"))

# NaN-aware rolling (these MUST exist in Rust for your tests)
rolling_mean_nan_np = getattr(_rs, "rolling_mean_nan_np", _missing("rolling_mean_nan_np"))
rolling_std_nan_np = getattr(_rs, "rolling_std_nan_np", _missing("rolling_std_nan_np"))
rolling_zscore_nan_np = getattr(_rs, "rolling_zscore_nan_np", _missing("rolling_zscore_nan_np"))

# Welford + masks
welford_np = getattr(_rs, "welford_np", _missing("welford_np"))
sign_mask_np = getattr(_rs, "sign_mask_np", _missing("sign_mask_np"))
demean_with_signs_np = getattr(_rs, "demean_with_signs_np", _missing("demean_with_signs_np"))

# outliers & scaling
iqr_outliers_np = getattr(_rs, "iqr_outliers_np", _missing("iqr_outliers_np"))
zscore_outliers_np = getattr(_rs, "zscore_outliers_np", _missing("zscore_outliers_np"))
minmax_scale_np = getattr(_rs, "minmax_scale_np", _missing("minmax_scale_np"))
robust_scale_np = getattr(_rs, "robust_scale_np", _missing("robust_scale_np"))
winsorize_np = getattr(_rs, "winsorize_np", _missing("winsorize_np"))
quantile_bins_np = getattr(_rs, "quantile_bins_np", _missing("quantile_bins_np"))

# diffs / cumulatives / ECDF
diff_np = getattr(_rs, "diff_np", _missing("diff_np"))
pct_change_np = getattr(_rs, "pct_change_np", _missing("pct_change_np"))
cumsum_np = getattr(_rs, "cumsum_np", _missing("cumsum_np"))
cummean_np = getattr(_rs, "cummean_np", _missing("cummean_np"))
ecdf_np = getattr(_rs, "ecdf_np", _missing("ecdf_np"))

# covariance / correlation
cov_np = getattr(_rs, "cov_np", _missing("cov_np"))
corr_np = getattr(_rs, "corr_np", _missing("corr_np"))
cov_matrix_np = getattr(_rs, "cov_matrix_np", _missing("cov_matrix_np"))
corr_matrix_np = getattr(_rs, "corr_matrix_np", _missing("corr_matrix_np"))
rolling_cov_np = getattr(_rs, "rolling_cov_np", _missing("rolling_cov_np"))
rolling_corr_np = getattr(_rs, "rolling_corr_np", _missing("rolling_corr_np"))

# NaN-aware covariance / correlation
cov_nan_np = getattr(_rs, "cov_nan_np", _missing("cov_nan_np"))
corr_nan_np = getattr(_rs, "corr_nan_np", _missing("corr_nan_np"))
rolling_cov_nan_np = getattr(_rs, "rolling_cov_nan_np", _missing("rolling_cov_nan_np"))
rolling_corr_nan_np = getattr(_rs, "rolling_corr_nan_np", _missing("rolling_corr_nan_np"))

# KDE
kde_gaussian_np = getattr(_rs, "kde_gaussian_np", _missing("kde_gaussian_np"))

# Inference (Pillar C)
t_test_1samp_np = getattr(_rs, "t_test_1samp_np", _missing("t_test_1samp_np"))
t_test_2samp_np = getattr(_rs, "t_test_2samp_np", _missing("t_test_2samp_np"))
chi2_gof_np = getattr(_rs, "chi2_gof_np", _missing("chi2_gof_np"))
chi2_independence_np = getattr(_rs, "chi2_independence_np", _missing("chi2_independence_np"))
cohens_d_2samp_np = getattr(_rs, "cohens_d_2samp_np", _missing("cohens_d_2samp_np"))

# UPDATED: prefer hedges_g_2samp_np, fallback to hedges_g_2samp_np2 (older wheels)
hedges_g_2samp_np = getattr(
    _rs,
    "hedges_g_2samp_np",
    getattr(_rs, "hedges_g_2samp_np2", _missing("hedges_g_2samp_np/hedges_g_2samp_np2")),
)

mean_diff_ci_np = getattr(_rs, "mean_diff_ci_np", _missing("mean_diff_ci_np"))

# staged / optional
mann_whitney_u_np = getattr(_rs, "mann_whitney_u_np", _missing("mann_whitney_u_np"))
ks_1samp_np = getattr(_rs, "ks_1samp_np", _missing("ks_1samp_np"))

# utilities
pad_nan_np = getattr(_rs, "pad_nan_np", _missing("pad_nan_np"))

__all__ = [
    "mean_np", "std_np", "var_np", "zscore_np", "percentile_np", "iqr_np", "iqr_width_np", "mad_np",
    "skew_np", "kurtosis_np", "trimmed_mean_np",
    "mean_nan_np", "std_nan_np", "var_nan_np",
    "mean_axis_np", "mean_over_last_axis_dyn_np",
    "rolling_mean_np", "rolling_std_np", "rolling_var_np", "rolling_mean_std_np", "rolling_zscore_np", "ewma_np",
    "rolling_mean_axis0_np", "rolling_std_axis0_np", "rolling_mean_std_axis0_np",
    "rolling_mean_nan_np", "rolling_std_nan_np", "rolling_zscore_nan_np",
    "welford_np", "sign_mask_np", "demean_with_signs_np",
    "iqr_outliers_np", "zscore_outliers_np", "minmax_scale_np", "robust_scale_np",
    "winsorize_np", "quantile_bins_np",
    "diff_np", "pct_change_np", "cumsum_np", "cummean_np", "ecdf_np",
    "cov_np", "corr_np", "cov_matrix_np", "corr_matrix_np", "rolling_cov_np", "rolling_corr_np",
    "cov_nan_np", "corr_nan_np", "rolling_cov_nan_np", "rolling_corr_nan_np",
    "kde_gaussian_np",
    "t_test_1samp_np", "t_test_2samp_np", "chi2_gof_np", "chi2_independence_np",
    "cohens_d_2samp_np", "hedges_g_2samp_np", "mean_diff_ci_np",
    "mann_whitney_u_np", "ks_1samp_np",
    "pad_nan_np",
]