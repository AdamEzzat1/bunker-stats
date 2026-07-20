# bunker-stats

**Production-grade statistical computing library combining Rust performance with Python ergonomics**

Version: 0.3.0  
Status: Release candidate  
License: See LICENSE file

---

## Overview

`bunker-stats` is a high-performance statistical computing library that delivers production-grade functionality through Rust backend kernels with Python bindings via PyO3. The library emphasizes **deterministic results**, **numerical stability**, and **minimal allocations** while maintaining an intuitive, Pythonic API.

### Core Principles

🎯 **Deterministic** - Bit-exact given the same seed: every randomized routine is reproducible, and `random_state=None` behaves as seed 0  
⚡ **High-Performance** - 2-5× faster than SciPy/pandas/numpy on its hot paths (rolling windows, robust estimators, bootstrap); see [Performance](#performance-highlights) for measured, per-function numbers  
🔢 **Numerically Stable** - Kahan summation, Welford's algorithm, careful conditioning  
🧪 **Thoroughly Tested** - 53 Rust unit/property tests + 600+ Python tests (numpy/scipy/pandas/statsmodels parity, edge cases, and never-panic properties); ~99% of public functions have a direct reference test  
🔒 **Type-Safe** - Rust implementation with full input validation  
📦 **Zero Dependencies** - Core functionality requires only NumPy

---

## Release Notes — 0.3.0 consistency changes

A correctness-and-consistency pass. Every change below is pinned by a
regression test (see `tests/test_hardening_v030.py`).

### Numeric bug fixes (behavior changes)

- **Catastrophic cancellation eliminated in one-pass covariance/correlation
  kernels.** `cov`/`corr`, `cov_skipna`/`corr_skipna`,
  `cov_matrix_skipna`/`corr_matrix_skipna`, the NaN-aware rolling
  `std`/`zscore` paths, and the strict `rolling_cov`/`rolling_corr` kernels
  now shift data by the first finite value (per series / per column) before
  accumulating. Second moments are translation-invariant, so results are
  mathematically identical but no longer lose all precision on large-offset
  data (e.g. series centered near 1e8 or 1e12). `rolling_beta_skipna` and
  `rolling_linreg_skipna` use the same shift (the linreg intercept is
  un-shifted exactly). Verified against two-pass numpy references at offsets
  1e8 and 1e12 — note that pandas' own one-pass rolling kernels fail this test.
- **Correlation outputs are clamped to [-1, 1]** across scalar, matrix and
  rolling correlation kernels; round-off can no longer produce |r| slightly
  above 1.
- **`welch_psd` / `bartlett_psd` now produce true density-scaled one-sided
  spectra**: scale is `1/(fs·Σw²)` with interior bins doubled (the Nyquist bin
  is not doubled for even `nperseg`). Matches `scipy.signal.welch` to ~1e-15
  relative. Previously the output was neither 'density' nor 'spectrum' scaled
  and interior bins were not doubled.
- **`ks_1samp` one-sided p-values are now exact** (Birnbaum–Tingey / Smirnov
  formula, evaluated in log space), matching
  `scipy.stats.ks_1samp(alternative="greater"/"less")`. Previously a rough
  `exp(-2nD²)` asymptotic was used.
- **`exp_cdf` uses `-expm1(-λx)`**, keeping full precision for tiny arguments
  (`exp_cdf(1e-18) == 1e-18` instead of `0.0`).

### Previously unregistered functions now exported

`rolling_min`, `rolling_max`, `rolling_range`, `rolling_cv`,
`rolling_count_above`, `rolling_pct_above` (O(1) sliding kernels) and
`kde_gaussian` (Gaussian KDE with Scott's-rule bandwidth, parity with
`scipy.stats.gaussian_kde`) are now registered in the extension module. Every
name in the facade's `__all__` now resolves to a callable.

### API consistency changes

One shared edge/validation contract across the library. Each item below is a
deliberate behavior change:

- **Rolling edge rule** (strict, truncated-output kernels): `window < 1` now
  raises `ValueError` everywhere (several kernels silently returned an empty
  array); `window > len(x)` now returns an **empty array** everywhere
  (`rolling_cov`, `rolling_corr`, `rolling_autocorr`, `rolling_correlation`,
  `rolling_autocorr_multi`, `rolling_min/max/range/cv/count_above/pct_above`
  previously raised). Exception: `rolling_median` intentionally keeps its
  pandas-like full-length output (length n, NaN head).
- **Facade rolling wrappers validate `window` as an integer >= 1**, so a
  negative window raises `ValueError` instead of `OverflowError`.
- **`winsorize` accepts quantile fractions in [0, 1] only** with
  `0 <= lower_q < upper_q <= 1`; the old dual-unit auto-detection (which made
  `1.0` ambiguous and silently clamped out-of-range arguments) now raises
  `ValueError`. **`winsorized_mean` (facade) uses the same fraction units**,
  validating and converting to the kernel's percentile units; the raw
  `winsorized_mean_np` binding keeps percentile units for backwards
  compatibility.
- **`percentile` raises `ValueError` for q outside [0, 100]** (previously
  silently clamped).
- **`trimmed_mean` / `trimmed_mean_skipna` raise `ValueError` for
  `proportion_to_cut` outside [0, 0.5)** (previously returned NaN).
- **`ewma` requires `0 < alpha <= 1`**, raising `ValueError` otherwise.
- **`quantile_bins` requires `n_bins >= 1`**; the facade also rejects negative
  `n_bins` with `ValueError` (previously `OverflowError` or empty output).
- **Outlier detectors have keyword defaults**: `iqr_outliers(x, k=1.5)`,
  `zscore_outliers(x, threshold=3.0)`.
- **`alternative` is validated** against `{"two-sided", "less", "greater"}` in
  the t-tests, `mann_whitney_u`, and the facade `t_test_2samp` wrapper
  (`ValueError` otherwise).
- **`cov` / `corr` raise `ValueError` on length mismatch** (previously the
  longer input was silently truncated).
- **`acf`/`pacf` (all methods) on constant input return 1.0 at lag 0 and NaN
  at lags >= 1** (statsmodels semantics; previously reported spurious
  autocorrelation). `pacf_innovations` reports NaN instead of impossible
  coefficients with |value| > 1.
- **`welford([])` returns `(nan, nan, 0)`** — the mean of no observations is
  NaN, not 0.0.

### Determinism

- **`random_state=None` now means seed 0 for every randomized routine.** The
  three block bootstraps (`moving_block_bootstrap_mean_ci`,
  `circular_block_bootstrap_mean_ci`, `stationary_bootstrap_mean_ci`)
  previously drew OS entropy when unseeded, and
  `jackknife_after_bootstrap_se_mean` used a different fixed default seed.
  All 15 resamplers now return bit-identical results for `None` vs `0` and
  across repeated calls.
- **`bootstrap_mean` is bit-identical across thread counts**: per-resample
  means are collected in index order and averaged with a serial sum, removing
  the last parallel floating-point reduction whose rounding could depend on
  rayon's work-stealing schedule.

---

## Release Notes — 0.2.9 Hardening

A full-codebase review and hardening pass. Every fix below is pinned by a
regression test; the suite now stands at **458 Python tests + 53 Rust tests
passing, 0 compiler warnings**.

### Crash fixes (highest impact)

- **NaN inputs no longer kill the Python process.** The release build uses
  `panic = "abort"`, so any Rust panic terminated the whole interpreter rather
  than raising an exception. ~24 sort/select code paths (`median`, `mad`,
  `iqr`, `percentile`, `robust_scale`, `ecdf`, `quantile_bins`, `qn_scale`,
  `runs_test`, and others) panicked on NaN input. All now use total-order
  comparison and propagate NaN (`NaN in → NaN out`) or raise a normal
  `ValueError`. *Impact: a single NaN in production data could previously
  crash the entire process, losing all in-flight work.*
- **`zivot_andrews_test` crashed on every call** via an out-of-bounds buffer
  write (the design-matrix column count was off by one). *Impact: the function
  was unusable; the crash mode was a hard interpreter abort.*

### Statistical correctness

- **Zivot-Andrews regression was numerically scrambled** — the design matrix
  was built row-major but read column-major, so breakpoint estimates and
  t-statistics were noise. It now recovers the true structural break
  (verified against `statsmodels`).
- **Rolling variance/std/cov/corr suffered catastrophic cancellation** on
  large-offset data (e.g. values near 1e8): results silently degraded to
  wrong zeros. All rolling second-moment kernels now use translation-offset
  accumulators. *Impact: correct results on price-level-style series, not
  just mean-zero data.*
- **KPSS** now uses the same automatic bandwidth as `statsmodels`
  (`nlags="auto"`); statistics match statsmodels to machine precision.
- **ADF / Phillips-Perron p-values** now come from the MacKinnon (1994)
  regression surface (statsmodels-identical, including extreme tails). ADF
  honors its `regression` and `max_lag` arguments (previously ignored); PP
  applies the HAC long-run-variance correction and the correct
  Dickey-Fuller reference distribution (previously Normal).
- **Breusch-Godfrey** statistic corrected to the standard `T·R²` form.
- **Variance-ratio test** rewritten as the proper Lo-MacKinlay overlapping
  estimator; a random walk now yields VR ≈ 1 as expected.
- **Biweight midvariance** exponents corrected to the standard
  (Beers-Flynn-Gebhardt / astropy) definition.
- **Skewness/kurtosis** now use scipy-consistent population moments
  (previously mixed sample/population estimators matching no standard
  convention).
- **NaN-aware rolling cov/corr** now match pandas `min_periods=window`
  semantics.
- **`rolling_autocorr_multi`** wrote its output buffer column-major but
  reshaped it row-major, scrambling the matrix whenever more than one lag was
  requested (single-lag calls were coincidentally correct). Each column `j`
  now equals `rolling_autocorr(x, lags[j], window)` exactly.

### Test coverage & tooling

- Function-level test coverage rose from **51% to 99%** of the 188 public
  functions: 67 new parity tests validate outputs directly against
  numpy / scipy / pandas / statsmodels references.
- New test layers: Rust unit + property tests (including
  never-panic-on-any-input properties), a binding-surface test asserting
  every registered export is callable, and a subprocess-isolated NaN-crash
  regression test.
- The `parallel` feature is now genuinely optional
  (`--no-default-features` builds and passes tests with a serial fallback),
  and the crate builds an `rlib` so Rust integration tests and fuzzing can
  link against it.

### Modern facade API (v0.3 layer)

- **One name per statistic; NaN handling is a keyword.** `bs.mean(x, skipna=True)`
  dispatches to the skip-NaN Rust kernel; the strict and skip-NaN kernels stay
  separate underneath, so there is no branch in the hot loop. Applies to
  mean/std/var/median/mad/iqr/zscore/trimmed_mean, cov/corr (scalar, matrix,
  and rolling), and the rolling reducers.
- **Keyword defaults everywhere.** `bs.t_test_2samp(x, y)` works bare
  (pooled, two-sided); options like `equal_var`, `pooled` are keyword-only.
- **Unit-explicit argument names.** `bs.winsorize(x, lower_q=0.05, upper_q=0.95)`
  takes quantiles in [0, 1]; `bs.percentile(x, q=95)` keeps numpy's [0, 100]
  convention — the name tells you the unit.
- **Optional pandas layer.** `bunker_stats.pandas` provides `cov_df`/`corr_df`
  (labeled DataFrame results) and Styler helpers (`corr_heatmap`,
  `zscore_style`, ...). The core package remains numpy-only.
- All raw `*_np` / `*_skipna` names remain available; nothing breaks.

---

## Quick Start

### Installation

```bash
pip install bunker-stats

# ...plus the optional pandas/Jupyter reporting layer (see "Notebook UX" below)
pip install "bunker-stats[notebook]"
```

The core package depends only on **numpy**. `pandas`, `jinja2` and `matplotlib`
arrive with the `[notebook]` extra and are needed solely by
`bunker_stats.notebook`.

### Basic Usage

```python
import bunker_stats as bs
import numpy as np

# Robust statistics - resistant to outliers
data = np.array([1, 2, 3, 4, 5, 100])  # outlier: 100
location, scale = bs.robust_fit(data)   # (3.5, 2.22) vs mean/std (19.17, 38.4)

# Rolling window operations - 2-5× faster than pandas
signal = np.random.randn(10000)
smoothed = bs.rolling_median(signal, window=10)

# NaN handling is a keyword, not a separate function
noisy = signal.copy(); noisy[::97] = np.nan
m = bs.mean(noisy, skipna=True)                  # numpy.nanmean semantics
r = bs.rolling_mean(noisy, window=20, skipna=True)

# Statistical inference - keyword defaults just work
x = np.random.randn(30)
y = np.random.randn(25) + 0.5
result = bs.t_test_2samp(x, y)                   # pooled, two-sided defaults
welch = bs.t_test_2samp(x, y, equal_var=False)  # Welch's t-test

# Quantile arguments carry their unit in the name
clipped = bs.winsorize(x, lower_q=0.05, upper_q=0.95)

# Matrix operations - fast covariance/correlation
X = np.random.randn(1000, 10)
cov = bs.cov_matrix(X)
corr = bs.corr_matrix(X, skipna=True)            # pairwise-complete

# Optional notebook layer (pip install "bunker-stats[notebook]")
# from bunker_stats.notebook import robust_summary, outlier_style
# robust_summary(df)          # DataFrame: median, MAD, IQR, Qn, skew, kurtosis...
# outlier_style(df)           # Styler: highlights outliers across all columns

# Bootstrap confidence intervals (deterministic given random_state)
estimate, lower, upper = bs.bootstrap_mean_ci(data, n_resamples=10000, conf=0.95, random_state=0)
```

---

## Module Documentation

Each module has comprehensive documentation with detailed API references, usage examples, performance benchmarks, and edge case behavior specifications.

### 1. **Robust Statistics** ✅ Production-Ready

**Status:** 73/73 tests passing  
**Performance:** 2-5× faster than SciPy/pandas on order-statistic estimators (median/MAD/Qn)  
**Documentation:** See [src/kernels/robust/README.md](./src/kernels/robust/README.md)

Outlier-resistant statistical estimators including:
- Location estimators (median, trimmed mean, Huber location)
- Scale estimators (MAD, IQR, Qn, Sn)
- Robust fitting (`robust_fit`, `robust_score`)
- Rolling robust statistics
- Skip-NaN variants for all functions

**Key Features:**
- Policy-driven `RobustStats` class with composable configuration
- Fused median+MAD kernel (40% faster joint computation)
- O(n) selection vs O(n log n) sorting (2-5× speedup)
- Perfect SciPy parity with deterministic results

---

### 2. **Inference** ✅ Production-Ready

**Status:** 15/15 tests passing  
**Performance:** 1.2-1.5× faster than SciPy  
**Documentation:** See [src/infer/INFERENCE_README.md](./src/infer/INFERENCE_README.md)

Comprehensive statistical hypothesis testing suite:
- **Chi-square tests:** Goodness-of-fit, independence
- **T-tests:** One-sample, two-sample (pooled/Welch)
- **Non-parametric:** Mann-Whitney U, Kolmogorov-Smirnov
- **Correlation:** Pearson, Spearman with significance tests
- **ANOVA:** F-test, Levene's test, Bartlett's test
- **Normality:** Jarque-Bera, Anderson-Darling
- **Effect sizes:** Cohen's d, Hedges' g

**Key Features:**
- Numerical stability with extreme values (χ² > 1000, n > 5000)
- Exact finite-n algorithms (Durbin-Marsaglia for KS test)
- Welch-Satterthwaite with zero-variance edge case handling
- 100% SciPy parity (rtol ≤ 1e-10)

---

### 3. **Matrix Operations** ✅ Production-Ready

**Status:** 83/83 tests passing  
**Performance:** correctness-first; dense cov/corr are ~0.6× numpy (no BLAS backend) — see Performance Highlights  
**Documentation:** See [src/kernels/matrix/README.md](./src/kernels/matrix/README.md)

High-performance matrix computations for statistical analysis:
- **Covariance matrices:** Sample, population, centered, pairwise-complete
- **Correlation matrices:** Pearson correlation, correlation distance
- **Gram matrices:** X^T X and X X^T for regression/kernel methods
- **Pairwise distances:** Euclidean, cosine
- **Utilities:** Diagonal extraction, trace, symmetry checking

**Key Features:**
- Guaranteed symmetry and positive semi-definiteness
- Optional Rayon parallelism for large matrices
- Comprehensive NaN handling with skip-NaN variants
- Perfect NumPy/SciPy parity with mathematical guarantees verified

---

### 4. **Rolling Windows** ✅ Production-Ready

**Status:** 53/53 tests passing  
**Performance:** 2-5× faster than pandas (rolling std/mean); rolling_median is O(n) selection-based  
**Documentation:** See [src/kernels/rolling/ROLLING_README.md](./src/kernels/rolling/ROLLING_README.md)

Flexible rolling window statistics with policy-driven configuration:
- **Statistics:** Mean, std, variance, min, max, count
- **Alignment:** Trailing (classic) or centered (pandas-like)
- **NaN handling:** Propagate, ignore, or minimum periods
- **Multi-stat kernels:** Compute 2-6 statistics in single pass
- **2D support:** Column-wise operations on matrices

**Key Features:**
- `Rolling` class with composable `RollingConfig` policies
- Fused kernels for efficient multi-metric computation
- Kahan summation for numerical stability
- Automatic edge truncation for centered windows
- 100% backward compatibility with legacy functions

---

### 5. **Resampling** ✅ Production-Ready

**Status:** Deterministic given `random_state`; CI ordering and determinism tested  
**Performance:** ~5× faster than a vectorized-numpy bootstrap loop  
**Documentation:** See [src/kernels/resampling/README.md](./src/kernels/resampling/README.md)

Lightning-fast resampling methods with ergonomic interfaces:
- **Bootstrap:** Percentile, BCa, bootstrap-t, and Bayesian CIs; standard error and variance
- **Block bootstraps:** Moving, circular, and stationary variants for autocorrelated series
- **Permutation tests:** Mean-difference and correlation
- **Jackknife:** Leave-one-out, delete-d, jackknife-after-bootstrap, influence values

**Key Features:**
- `BootstrapConfig` class with comprehensive validation
- Flexible NaN handling (propagate or omit)
- Deterministic random seeding for reproducibility
- Zero performance overhead from config layer
- Actionable error messages

---

### 6. **Time Series Analysis** ✅ Production-Ready

**Status:** Core statistics validated against statsmodels (ADF, KPSS, Ljung-Box, ACF/PACF to machine precision)  
**Documentation:** See [src/kernels/tsa/README.md](./src/kernels/tsa/README.md)

Comprehensive temporal data analysis tools:
- **Correlation:** ACF, PACF (Levinson-Durbin, Yule-Walker, Innovations, Burg)
- **Spectral analysis:** Periodogram, Welch PSD, spectral density
- **Diagnostic tests:** Ljung-Box, Box-Pierce, Breusch-Godfrey, Durbin-Watson, runs test
- **Stationarity:** ADF (MacKinnon p-values), KPSS (automatic bandwidth), Phillips-Perron, Zivot-Andrews structural break, variance ratio
- **Rolling operations:** Rolling autocorrelation (single and multi-lag)

---

### 7. **Distributions** ✅ Production-Ready

**Status:** Round-trip (`ppf(cdf(x)) ≈ x`) and reference-parity tested  
**Documentation:** See [src/kernels/dist/README.md](./src/kernels/dist/README.md)

Vectorized distribution functions for Normal, Exponential (rate-parameterized), and Uniform:
- **Densities:** `pdf`, `logpdf`
- **Probabilities:** `cdf`, `sf`, `logsf` — dedicated survival functions keep far-tail
  p-values accurate where `1 - cdf` would cancel to zero
- **Quantiles:** `ppf` with strict domain checks
- **Reliability:** cumulative hazard functions for survival analysis

---

### 8. **Quantiles & Order Statistics** ✅ Production-Ready

**Status:** NumPy-parity tested, NaN-safe  
**Documentation:** See [src/kernels/quantile/README.md](./src/kernels/quantile/README.md)

Order-statistic utilities built on O(n) selection:
- **Percentiles & IQR:** `percentile`, IQR family including skip-NaN variants
- **Winsorization:** Percentile-based and explicit-bound clipping
- **Outlier masks:** IQR fences and z-score thresholds
- **Empirical distributions:** ECDF and quantile binning

---

---

## Rich Results (opt-in result objects)

Hypothesis tests return a plain `dict` by default. Pass **`rich=True`** to get a
result object instead — tuple-unpackable, indexable, and carrying a bit of
extra metadata (sample sizes, the chosen `alternative`, and cheap effect
sizes / CIs computed from the same kernels). This mirrors the result-object
style already used by the time-series tests.

```python
import bunker_stats as bs

# Default: unchanged dict return
bs.t_test_2samp(x, y)
# -> {'statistic': ..., 'pvalue': ..., 'df': ..., 'mean_x': ..., ...}

# Opt in to a rich result
res = bs.t_test_2samp(x, y, equal_var=False, rich=True)

stat, pval = res            # tuple unpacking (statistic, pvalue)
res[0], res[1]              # indexing
res.pvalue, res.df          # named access
res.effect_size            # Cohen's d, computed from the same inputs
res.n1, res.n2, res.equal_var
res.is_significant(0.05)   # -> bool
res.to_dict()              # JSON-friendly dict of populated fields
print(res.conclusion(alpha=0.01))
print(res.info())          # multi-line human summary
```

```text
>>> print(res.info())
t-Test
============================================================
Statistic:          -2.145318
P-value:            0.033987
Test:               two-sample
Alternative:        two-sided
df:                 118
Mean x / y:         0.0512 / 0.4231
Equal variance:     False
Cohen's d:          -0.394
n1 / n2:            60 / 70

Conclusion: Reject H0 (the two means are equal) at alpha=0.05 (p=0.034)
```

### Which tests support `rich=True`

| Function(s) | Result type | Notable extra fields |
|-------------|-------------|----------------------|
| `t_test_1samp`, `t_test_2samp`, `t_test_paired` | `TTestResult` | `effect_size` (Cohen's d, 2-sample), `n1`/`n2`, `equal_var` |
| `chi2_gof`, `chi2_independence` | `ChiSquareResult` | `dof`, `observed`, `test_type` |
| `mann_whitney_u` | `MannWhitneyResult` | `rank_biserial`, `n1`/`n2` |
| `ks_1samp` | `KSResult` | `distribution`, `n` |
| `f_test_oneway` | `ANOVAResult` | `eta_squared`, `omega_squared`, `n_groups`/`n_total` |
| `pearson_corr_test`, `spearman_corr_test` | `CorrelationTestResult` | `r`, `ci_low`/`ci_high` (Pearson) |
| `jarque_bera`, `anderson_darling` | `NormalityResult` | `skewness`, `kurtosis`, `critical_value_5pct` |

The result classes are exported from both the root package and
`bunker_stats.infer` (for `isinstance` checks and type hints):

```python
from bunker_stats.infer import TTestResult, NormalityResult
isinstance(bs.t_test_2samp(x, y, rich=True), TTestResult)   # True
```

### Protocol shared by every rich result

| Member | Behaviour |
|--------|-----------|
| `for v in res` / `a, b = res` | Yields the primary fields — `(statistic, pvalue)`, except `CorrelationTestResult` yields `(r, pvalue)`. |
| `res[i]` | Integer/slice indexing over those same primary fields. |
| `res.to_dict(array=False)` | All *populated* fields (None omitted). NumPy scalars become Python scalars; arrays become lists unless `array=True`. |
| `res.info()` | Multi-line summary: statistic, p-value, test-specific detail, and a conclusion line. ASCII-only, so it prints on any console. |
| `res.conclusion(alpha=0.05)` | One-line verdict at the given significance level. |
| `res.is_significant(alpha=0.05)` | `bool`; `False` when the p-value is missing/NaN. |

Two behaviours worth noting:

- **Backward compatibility is absolute.** The `rich` keyword is the *only* thing
  the facade intercepts; every other argument is forwarded untouched, so the
  default (`rich=False`) return is byte-for-byte what it always was — including
  the deprecated `*_np` aliases.
- **Anderson-Darling has no p-value.** `NormalityResult.pvalue` is `None` for
  `anderson_darling`; its `conclusion()` / `is_normal()` fall back to the 5%
  critical value (A\* > 0.787 rejects normality).

## Notebook UX (optional pandas layer)

`bunker_stats.notebook` is the ergonomic bridge between the Rust kernels and
pandas/Jupyter. Every helper is a thin, validated wrapper — the numbers always
come from the same kernels the core API uses.

```bash
pip install "bunker-stats[notebook]"
```

`import bunker_stats` stays **numpy-only**: pandas is never imported at package
import time, and even `import bunker_stats.notebook` succeeds without pandas
installed. The import happens lazily on the first call, and a missing install
raises a clear `ImportError` naming the command above.

### Naming convention

| Suffix        | Returns                                                  |
|---------------|----------------------------------------------------------|
| `*_report`    | `pandas.DataFrame` of results (usually one row per column) |
| `*_style` / `style_*` | `pandas.io.formats.style.Styler`                 |
| `*_columns`   | `pandas.DataFrame` copy with added/transformed columns    |

### Quick tour

```python
import bunker_stats as bs
from bunker_stats.notebook import (
    robust_summary, describe_fast, outlier_report, outlier_style,
    normality_report, correlation_report, corr_heatmap, missingness_report,
    scale_columns, winsorize_columns, rolling_report, bootstrap_ci_report,
    style_significance, style_effect_size,
)

# --- Profiling ---------------------------------------------------------
robust_summary(df)                 # count, mean, std, median, MAD, IQR, Qn,
                                   # trimmed mean, skew, kurtosis
describe_fast(df)                  # a faster, richer df.describe().T
missingness_report(df)             # NaN vs +/-inf vs finite, per column

# --- Outliers ----------------------------------------------------------
outlier_report(df, method="iqr", k=1.5)             # counts, %, bounds
outlier_report(df, method="robust_zscore")          # median/MAD fences
outlier_style(df)                                   # highlight across columns

# --- Distribution & association ----------------------------------------
normality_report(df)                                # Jarque-Bera + A-D
correlation_report(df, method="spearman")           # square matrix
correlation_report(df, pvalues=True)                # long form with p-values
corr_heatmap(df)                                    # gradient Styler

# --- Transforms --------------------------------------------------------
scale_columns(df, method="robust")                  # adds *_robust columns
scale_columns(df, method="zscore", replace=True)    # overwrite in place
winsorize_columns(df, lower_q=0.05, upper_q=0.95)   # adds *_winsor columns

# --- Time series & uncertainty -----------------------------------------
rolling_report(df, "price", window=20, stats=("mean", "std"))
bootstrap_ci_report(df, stat="median", n_resamples=5000, random_state=0)

# --- Styling result tables ---------------------------------------------
style_significance(results, pvalue_column="pvalue", alpha=0.05)
style_effect_size(results, "cohens_d", thresholds=(0.2, 0.5, 0.8))
```

Helpers are also reachable lazily off the package (`bs.notebook.robust_summary(df)`)
and re-exported from `bunker_stats.pandas` alongside `cov_df` / `corr_df`.

### NaN and infinity policy

The Rust kernels are *strict*: one NaN anywhere poisons the whole result. That
is right for a numerics library and wrong for exploratory analysis, so this
layer makes the choice explicit rather than inheriting it:

| Helper family | Behaviour with NaN / ±inf |
|---------------|---------------------------|
| `*_report`, `robust_summary`, `describe_fast` | Dropped **per column** before the kernel runs; the count appears in `n_missing`. A column with no finite values yields an all-NaN row, not an exception. |
| `correlation_report`, `corr_heatmap` | Dropped **pairwise** — each pair uses rows where both columns are finite. `n` per pair is reported when `pvalues=True`. |
| `*_columns` | The transform is fit on finite values only, then results are scattered back into their original positions. Non-finite in ⇒ NaN out; **row alignment is always preserved**. |
| `*_style`, `style_*` | Never raise; NaN cells simply receive no background color. |
| `missingness_report` | The exception — it *counts* rather than drops, and separates NaN from ±inf. |

Two behaviours worth calling out explicitly:

- **Anderson-Darling has no p-value.** The kernel returns the A\* statistic
  only (its critical values need table interpolation), so `normality_report`
  reports `ad_statistic` for reference and bases the `normal` / `conclusion`
  verdict solely on the Jarque-Bera p-value. As a rule of thumb A\* > 0.787
  rejects normality at α = 0.05.
- **`rolling_report(..., nan_policy="ignore")` does nothing on its own.** With
  the default `min_periods == window`, a window containing a NaN still fails
  the count check. To actually skip NaNs, pass `min_periods` below `window`:
  `rolling_report(df, "x", 5, min_periods=3, nan_policy="ignore")`.

### Backwards compatibility

The original five helpers (`demean_style`, `zscore_style`, `iqr_outlier_style`,
`corr_heatmap`, `robust_scale_column`) keep working from
`bunker_stats.pandas_helpers`, which is now a shim over the notebook layer.
They gained dtype validation and NaN safety; `iqr_outlier_style` is now a thin
wrapper over the multi-column `outlier_style`.

## Performance Highlights

Measured with `timeit` (best of 5) on Windows 11, Python 3.10, numpy 2.2 /
pandas 2.3 / scipy 1.15 / statsmodels 0.14. `ratio = reference_time /
bunker_time`, so **> 1× is faster, < 1× is slower**. Numbers are machine- and
size-dependent; reproduce with `benchmarks/`.

| Operation | Size | vs reference | Ratio |
|-----------|------|--------------|-------|
| `rolling_std` (w=50) | n=1e4 | pandas | **5.2×** |
| `rolling_std` (w=50) | n=1e6 | pandas | **2.6×** |
| `rolling_mean` (w=50) | n=1e4 / 1e6 | pandas | **3.3× / 2.4×** |
| `mad` | n=1e6 | numpy | **4.0×** |
| `median` | n=1e4 / 1e6 | numpy | **3.6× / 2.1×** |
| `bootstrap_mean_ci` (1000 res.) | n=1e4 | numpy (vectorized) | **4.9×** |
| `norm_cdf` | n=1e6 | scipy | **2.5×** |
| `kpss_test` | n=5000 | statsmodels | 1.2× |
| `percentile` | n=1e6 | numpy | ~1.1× (parity) |
| `cov_matrix` / `corr_matrix` | 1000×50 | numpy | **0.6×** (slower — see below) |
| `adf_test` (matched lag count) | n=5000 | statsmodels | **0.15×** (slower — see below) |

**Where it wins:** streaming/rolling statistics, order-statistic estimators
(median/MAD/Qn), and bootstrap confidence intervals — the O(n) single-pass and
selection-based kernels are 2–5× faster than the pandas/numpy equivalents,
with the largest margins at small-to-medium `n`.

**Where it does not (honest caveats):**
- **Dense linear algebra** (`cov_matrix`, `corr_matrix`) is ~0.6× numpy. numpy
  dispatches these to a tuned multithreaded BLAS (OpenBLAS/MKL); the portable
  pure-Rust kernels here do not link a BLAS, so they trail on large dense
  products. If matrix throughput is your bottleneck, compute these with numpy.
- **`adf_test` with an explicit lag count** is slower than statsmodels, whose
  OLS path is LAPACK-backed. (The headline "hundreds of ×" figures that
  appeared in earlier drafts compared against statsmodels' *default* automatic
  lag search, which does substantially more work — an apples-to-oranges
  comparison that has been removed.)

---

## Design Philosophy

### 1. **Determinism First**
Results are bit-exact given the same seed. Deterministic kernels produce identical results across runs and thread counts; every randomized routine takes a `random_state`, and leaving it as `None` uses seed 0 rather than entropy, so even "unseeded" calls are reproducible.

### 2. **Edge Cases Matter**
Production data has empty arrays, NaN values, zero variance, and extreme values. All functions handle these gracefully with clear, documented behavior.

### 3. **Performance Without Compromise**
Optimizations never sacrifice correctness or numerical stability. All performance claims are verified against reference implementations.

### 4. **Ergonomic Configuration**
Policy-driven design with composable configuration objects. Sensible defaults, actionable error messages, zero performance overhead.

### 5. **Comprehensive Testing**
Every edge case, every numerical corner, every performance regression is covered by tests. Test failures are treated as bugs, not warnings.

---

## API Compatibility

### NumPy/SciPy Parity
- `cov_matrix` matches `np.cov(X.T, ddof=1)`
- `corr_matrix` matches `np.corrcoef(X.T)`
- Inference functions match SciPy results to machine precision (rtol ≤ 1e-10)
- MAD with `consistent=True` matches SciPy's consistency factor (1.4826)

### Backward Compatibility
- All legacy flat functions preserved
- Config classes add features without breaking existing code
- Deprecation warnings for upcoming changes
- Semantic versioning for API changes

---

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_robust_stats.py -v       # Robust statistics (73 tests)
pytest tests/test_inference*.py -v         # Inference (15 tests)
pytest tests/test_matrix.py -v             # Matrix ops (83 tests)
pytest tests/test_rolling*.py -v           # Rolling windows (53 tests)
pytest tests/test_resampling.py -v         # Resampling (25 tests)
pytest tests/test_tsa*.py -v               # Time series (45/47 tests)

# With coverage
pytest tests/ --cov=bunker_stats --cov-report=html
```

**Total Test Coverage:** 294+ tests across all modules

---

## Building from Source

### Requirements
- Python ≥ 3.8
- Rust ≥ 1.70
- NumPy ≥ 1.20

### Build Commands

```bash
# Development build
maturin develop

# Optimized release build
maturin develop --release

# With parallel features (Rayon)
maturin develop --release --features parallel

# Build distributable wheel
maturin build --release
```

---

## Roadmap

### v0.3.0 (Current)
✅ NaN inputs no longer abort the interpreter; panic boundaries hardened  
✅ Numeric correctness pass (cancellation-stable covariance/correlation, exact KS p-values, corrected PSD scaling, MacKinnon ADF/PP p-values)  
✅ TSA core validated against statsmodels (ADF, KPSS, Ljung-Box, ACF/PACF)  
✅ API consistency contract (uniform window/NaN/unit rules; see the 0.3.0 release notes)  
✅ Deterministic resampling (`random_state=None` == seed 0 everywhere)  
✅ ~99% of public functions covered by direct reference tests

### v0.4.0 (Planned)
- **Multivariate robust stats:** MCD, OGK covariance
- **Robust regression:** Huber, Theil-Sen, RANSAC
- **Weighted statistics:** weighted median, MAD, robust_fit
- **BLAS-backed matrix kernels** (current pure-Rust cov/corr are ~0.6× numpy; linking a BLAS would close the gap)
- Bayesian inference module
- Model selection criteria (AIC, BIC)
- Cross-validation utilities
- Spectral density estimation enhancements

---

## Contributing

We welcome contributions! Key areas:

- **New estimators** - Additional robust/Bayesian methods
- **Performance** - SIMD, GPU acceleration
- **Documentation** - Examples, tutorials, benchmarks
- **Testing** - Edge cases, stress tests
- **Bug fixes** - Numerical issues, edge case handling

See CONTRIBUTING.md for guidelines.

---

## Citation

If using in academic work:

```bibtex
@software{bunker_stats,
  title = {bunker-stats: Production-grade statistical computing in Rust and Python},
  author = {[Author Name]},
  year = {2026},
  version = {0.2.9},
  url = {https://github.com/[repo]/bunker-stats}
}
```

---

## License

See LICENSE file in repository root.

---

## Support

- **Documentation:** See module-specific READMEs (listed above)
- **Bug Reports:** Open an issue on GitHub
- **Questions:** GitHub Discussions
- **Performance Issues:** Include benchmarks and system info

---

## Acknowledgments

Built with:
- **Rust** - High-performance kernels
- **PyO3** - Python bindings
- **Rayon** - Optional parallelism
- **statrs** - Statistical distributions

Validated against:
- **NumPy** - Matrix operations
- **SciPy** - Statistical tests and distributions
- **statsmodels** - Time series analysis
- **pandas** - Rolling window operations

---

**bunker-stats: Because real-world data demands production-grade statistics** 🚀
