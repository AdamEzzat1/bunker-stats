# bunker-stats

**Production-grade statistical computing library combining Rust performance with Python ergonomics**

Version: 0.2.9  
Status: Production-ready  
License: See LICENSE file

---

## Overview

`bunker-stats` is a high-performance statistical computing library that delivers production-grade functionality through Rust backend kernels with Python bindings via PyO3. The library emphasizes **deterministic results**, **numerical stability**, and **minimal allocations** while maintaining an intuitive, Pythonic API.

### Core Principles

🎯 **Deterministic** - Bit-exact given the same seed: every randomized routine is reproducible, and `random_state=None` behaves as seed 0  
⚡ **High-Performance** - 2-244× faster than SciPy/pandas/statsmodels equivalents  
🔢 **Numerically Stable** - Kahan summation, Welford's algorithm, careful conditioning  
🧪 **Thoroughly Tested** - 100% test coverage with comprehensive edge case validation  
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
```

### Basic Usage

```python
import bunker_stats as bs
import numpy as np

# Robust statistics - resistant to outliers
data = np.array([1, 2, 3, 4, 5, 100])  # outlier: 100
location, scale = bs.robust_fit(data)   # (3.5, 2.22) vs mean/std (19.17, 38.4)

# Rolling window operations - 244× faster than pandas
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

# Optional pandas layer: labeled results + Styler visuals
# import bunker_stats.pandas as bsp
# C = bsp.corr_df(df)      # DataFrame labeled by column names
# bsp.corr_heatmap(df)     # pandas Styler heatmap

# Bootstrap confidence intervals
from bunker_stats.resampling import BootstrapConfig
config = BootstrapConfig(n_resamples=10000, conf=0.95)
estimate, lower, upper = config(data)
```

---

## Module Documentation

Each module has comprehensive documentation with detailed API references, usage examples, performance benchmarks, and edge case behavior specifications.

### 1. **Robust Statistics** ✅ Production-Ready

**Status:** 73/73 tests passing  
**Performance:** 2-244× faster than SciPy/pandas  
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
**Performance:** ~9,500 ops/sec (100×20 matrices)  
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
**Performance:** 244× faster than pandas for rolling median  
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

**Status:** 25/25 tests passing, 100% coverage  
**Performance:** 10-200× faster than pure Python  
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

## Performance Highlights

Actual benchmarks vs SciPy/statsmodels/pandas:

| Operation | Speedup | Notes |
|-----------|---------|-------|
| Median | 2.9× | Large arrays (n=1M) |
| MAD | 4.6× | Large arrays (n=1M) |
| Rolling Median | 244× | 10-element window |
| Qn Scale | 124× | Robust scale estimator |
| robust_fit | 5.2× | Fused median+MAD |
| Chi-square test | 1.2-1.5× | With edge case handling |
| Covariance matrix | ~9,500 ops/sec | 100×20 matrices |

**Average cross-function speedups:**
- Robust stats: 7.5× faster median, 17.3× faster MAD
- Rolling operations: 239× faster median

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

### v0.2.9 (Current - Released January 2026)
✅ Robust statistics with policy-driven RobustStats class  
✅ Comprehensive inference module with 15 hypothesis tests  
✅ Matrix operations with 83 comprehensive tests  
✅ Rolling windows with fused multi-stat kernels  
✅ Resampling with ergonomic config objects  
✅ TSA module at 95.7% completion

### v0.3.0 (Planned - Q1 2026)
- **TSA fixes:** 100% test pass rate (50/50 tests)
- **Multivariate robust stats:** MCD, OGK covariance
- **Robust regression:** Huber, Theil-Sen, RANSAC
- **Weighted statistics:** Weighted median, MAD, robust_fit
- **Additional estimators:** Biweight, Hampel, S/MM estimators
- **Performance:** Automatic parallelization, 5-10× multivariate speedups

### v0.4.0 (Planned - Q2 2026)
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
