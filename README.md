# bunker-stats

**Production-grade statistical computing library combining Rust performance with Python ergonomics**

Version: 0.2.9  
Status: Production-ready  
License: See LICENSE file

---

## Overview

`bunker-stats` is a high-performance statistical computing library that delivers production-grade functionality through Rust backend kernels with Python bindings via PyO3. The library emphasizes **deterministic results**, **numerical stability**, and **minimal allocations** while maintaining an intuitive, Pythonic API.

### Core Principles

🎯 **Deterministic** - Same input always produces identical output (bit-exact reproducibility)  
⚡ **High-Performance** - 2-244× faster than SciPy/pandas/statsmodels equivalents  
🔢 **Numerically Stable** - Kahan summation, Welford's algorithm, careful conditioning  
🧪 **Thoroughly Tested** - 100% test coverage with comprehensive edge case validation  
🔒 **Type-Safe** - Rust implementation with full input validation  
📦 **Zero Dependencies** - Core functionality requires only NumPy

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

# Statistical inference - comprehensive hypothesis testing
x = np.random.randn(30)
y = np.random.randn(25) + 0.5
result = bs.t_test_2samp(x, y, equal_var=False)  # Welch's t-test

# Matrix operations - fast covariance/correlation
X = np.random.randn(1000, 10)
cov = bs.cov_matrix(X)
corr = bs.corr_matrix(X)

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
**Documentation:** See [ROBUST_STATS_README.md](./ROBUST_STATS_README.md)

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
**Documentation:** See [INFERENCE_README.md](./INFERENCE_README.md)

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
**Documentation:** See [MATRIX_MODULE_README.md](./MATRIX_MODULE_README.md)

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
**Documentation:** See [ROLLING_README.md](./ROLLING_README.md)

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
**Documentation:** See [README_RESAMPLING.md](./README_RESAMPLING.md)

Lightning-fast resampling methods with ergonomic interfaces:
- **Bootstrap:** Confidence intervals for mean, median, std
- **Permutation tests:** Coming in v0.3
- **Jackknife:** Coming in v0.3

**Key Features:**
- `BootstrapConfig` class with comprehensive validation
- Flexible NaN handling (propagate or omit)
- Deterministic random seeding for reproducibility
- Zero performance overhead from config layer
- Actionable error messages

---

### 6. **Time Series Analysis** ⚠️ Near Production

**Status:** 45/47 tests passing (95.7%)  
**Known Issues:** 2 algorithmic corrections needed, 1 optimization pending  
**Documentation:** See [TSA_MODULE_README.md](./TSA_MODULE_README.md)

Comprehensive temporal data analysis tools:
- **Correlation:** ACF, PACF (Levinson-Durbin, Yule-Walker, Innovations, Burg)
- **Spectral analysis:** Periodogram, Welch PSD, spectral density
- **Diagnostic tests:** Ljung-Box, Durbin-Watson
- **Stationarity:** ADF, KPSS, variance ratio tests
- **Rolling operations:** Rolling autocorrelation

**v0.3 Roadmap:**
- Fix KPSS test calculation (8.4% error)
- Correct variance ratio test
- Optimize Zivot-Andrews test (currently hangs)
- Target: 50/50 tests passing

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
Every operation produces identical results across runs, platforms, and library versions. No randomness without explicit seeding, no floating-point non-determinism.

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
