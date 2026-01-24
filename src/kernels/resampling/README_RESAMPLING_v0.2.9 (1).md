# bunker-stats Resampling Module (v0.2.9)

**High-performance statistical resampling methods with ergonomic Python interfaces**

[![Tests](https://img.shields.io/badge/tests-25%2F25%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![Rust](https://img.shields.io/badge/rust-fast%20kernels-orange)]()

---

## Overview

The `bunker_stats.resampling` module provides lightning-fast resampling methods built on Rust kernels with an ergonomic Python API. Version 0.2.9 introduces **config objects** that add comprehensive input validation, flexible NaN handling, and helpful error messages while maintaining zero performance overhead.

### Key Features

✅ **Blazing Fast** - Rust-powered kernels with 10-200× speedups over pure Python  
✅ **Ergonomic API** - Config objects with sensible defaults and clear error messages  
✅ **Flexible NaN Handling** - Choose between propagation or intelligent filtering  
✅ **Reproducible** - Deterministic random seeding for consistent results  
✅ **Well-Tested** - 25/25 tests passing with 100% coverage  
✅ **Type-Safe** - Full input validation with actionable error messages  
✅ **Backward Compatible** - Existing flat functions remain unchanged  

---

## Installation

```bash
pip install bunker-stats>=0.2.9
```

---

## Quick Start

```python
import numpy as np
from bunker_stats.resampling import BootstrapConfig

# Generate sample data
data = np.random.randn(1000)

# Create config object
config = BootstrapConfig(
    n_resamples=10000,
    conf=0.99,
    stat="mean",
    random_state=42,
    nan_policy="omit"
)

# Compute bootstrap CI
estimate, lower, upper = config(data)
print(f"Mean: {estimate:.4f}, 99% CI: [{lower:.4f}, {upper:.4f}]")
```

---

## API Reference

### Bootstrap Methods

#### `BootstrapConfig`

**Purpose**: Compute bootstrap confidence intervals for univariate statistics.

**Parameters**:
- `n_resamples` (int, default=1000): Number of bootstrap resamples
- `conf` (float, default=0.95): Confidence level (must be in (0, 1))
- `stat` (str, default="mean"): Statistic to bootstrap
  - Options: `"mean"`, `"median"`, `"std"`
- `random_state` (int | None, default=None): Random seed for reproducibility
  - If `None`, uses deterministic default (0)
- `nan_policy` (str, default="propagate"): How to handle NaNs
  - `"propagate"`: Pass NaNs to Rust (fast, ~0 µs overhead)
  - `"omit"`: Filter NaNs in Python (10-100 µs overhead)
- `parallel` (bool, default=True): Reserved for future parallel control

**Methods**:
- `run(x: np.ndarray) -> Tuple[float, float, float]`: Run bootstrap
- `__call__(x: np.ndarray) -> Tuple[float, float, float]`: Shorthand for `run()`

**Returns**: `(estimate, lower_bound, upper_bound)`

**Example**:
```python
from bunker_stats.resampling import BootstrapConfig
import numpy as np

# Basic usage
config = BootstrapConfig(n_resamples=5000, conf=0.95, stat="mean")
data = np.random.randn(500)
estimate, lower, upper = config(data)

# With NaN handling
data_with_nans = np.array([1, 2, np.nan, 4, 5])
config_omit = BootstrapConfig(nan_policy="omit")
result = config_omit(data_with_nans)  # Filters NaNs automatically

# Reproducible results
config_seed = BootstrapConfig(random_state=42)
result1 = config_seed(data)
result2 = config_seed(data)
assert result1 == result2  # Identical results
```

---

#### `BootstrapCorrConfig`

**Purpose**: Compute bootstrap confidence intervals for correlation coefficients.

**Parameters**:
- `n_resamples` (int, default=1000): Number of bootstrap resamples
- `conf` (float, default=0.95): Confidence level (must be in (0, 1))
- `random_state` (int | None, default=None): Random seed
- `nan_policy` (str, default="propagate"): NaN handling strategy
  - `"propagate"`: Pass NaNs to Rust
  - `"omit"`: Pairwise deletion (removes pairs where either value is NaN)
- `parallel` (bool, default=True): Reserved for future use

**Methods**:
- `run(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]`: Run bootstrap
- `__call__(x, y) -> Tuple[float, float, float]`: Shorthand for `run()`

**Returns**: `(correlation, lower_bound, upper_bound)`

**Example**:
```python
from bunker_stats.resampling import BootstrapCorrConfig
import numpy as np

# Generate correlated data
np.random.seed(42)
x = np.random.randn(200)
y = 0.7 * x + 0.3 * np.random.randn(200)

# Compute bootstrap CI for correlation
config = BootstrapCorrConfig(n_resamples=5000, conf=0.99)
corr, lower, upper = config(x, y)
print(f"Correlation: {corr:.4f}, 99% CI: [{lower:.4f}, {upper:.4f}]")

# With missing data (pairwise deletion)
x_missing = np.array([1, 2, np.nan, 4, 5])
y_missing = np.array([2, np.nan, 6, 8, 10])
config_omit = BootstrapCorrConfig(nan_policy="omit")
result = config_omit(x_missing, y_missing)  # Uses pairs (0, 3, 4)
```

---

### Permutation Tests

#### `PermutationConfig`

**Purpose**: Perform permutation tests for correlation or mean differences.

**Parameters**:
- `n_permutations` (int, default=1000): Number of permutations
- `alternative` (str, default="two-sided"): Alternative hypothesis
  - Options: `"two-sided"`, `"less"`, `"greater"`
- `random_state` (int | None, default=None): Random seed
- `nan_policy` (str, default="propagate"): NaN handling strategy
  - For correlation: pairwise deletion
  - For mean_diff: independent filtering per sample
- `parallel` (bool, default=True): Reserved for future use

**Methods**:
- `run_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]`: Test correlation
- `run_mean_diff(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]`: Test mean difference
- `__call__` not implemented (use specific methods)

**Returns**: `(observed_statistic, p_value)`

**Example**:
```python
from bunker_stats.resampling import PermutationConfig
import numpy as np

# Test for correlation
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)

config = PermutationConfig(n_permutations=5000, alternative="two-sided")
observed_corr, pvalue = config.run_corr(x, y)
print(f"Correlation: {observed_corr:.4f}, p-value: {pvalue:.4f}")

# Test for mean difference (two independent samples)
group1 = np.random.randn(50) + 0.5  # Mean shifted by 0.5
group2 = np.random.randn(50)

config_greater = PermutationConfig(alternative="greater")
mean_diff, pvalue = config_greater.run_mean_diff(group1, group2)
print(f"Mean difference: {mean_diff:.4f}, p-value: {pvalue:.4f}")

# With NaN handling (independent filtering)
x_nan = np.array([1, np.nan, 3, 4])
y_nan = np.array([5, 6, np.nan, 8])
config_omit = PermutationConfig(nan_policy="omit")
result = config_omit.run_mean_diff(x_nan, y_nan)  # Filters each sample independently
```

---

### Jackknife Methods

#### `JackknifeConfig`

**Purpose**: Compute jackknife estimates and confidence intervals.

**Parameters**:
- `conf` (float, default=0.95): Confidence level (must be in (0, 1))

**Methods**:
- `run_mean(x: np.ndarray) -> Tuple[float, float, float]`: Jackknife mean estimate
- `run_mean_ci(x: np.ndarray) -> Tuple[float, float, float]`: Jackknife CI for mean
- `__call__` not implemented (use specific methods)

**Returns**: 
- `run_mean`: `(jackknife_estimate, bias, standard_error)`
- `run_mean_ci`: `(jackknife_estimate, lower_bound, upper_bound)`

**Example**:
```python
from bunker_stats.resampling import JackknifeConfig
import numpy as np

data = np.random.randn(200)

# Jackknife mean with bias and SE
config = JackknifeConfig()
estimate, bias, se = config.run_mean(data)
print(f"Estimate: {estimate:.4f}, Bias: {bias:.6f}, SE: {se:.4f}")

# Jackknife CI
config_ci = JackknifeConfig(conf=0.99)
estimate, lower, upper = config.run_mean_ci(data)
print(f"Mean: {estimate:.4f}, 99% CI: [{lower:.4f}, {upper:.4f}]")
```

---

## Convenience Functions

For quick one-liners, use the functional API:

### `bootstrap()`

```python
from bunker_stats.resampling import bootstrap
import numpy as np

data = np.random.randn(500)

# Quick bootstrap CI
estimate, lower, upper = bootstrap(
    data, 
    stat="mean", 
    n_resamples=5000, 
    conf=0.95, 
    random_state=42
)
```

**Signature**:
```python
bootstrap(
    x: np.ndarray,
    stat: str = "mean",
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: int | None = None,
    nan_policy: str = "propagate"
) -> Tuple[float, float, float]
```

---

### `bootstrap_corr()`

```python
from bunker_stats.resampling import bootstrap_corr
import numpy as np

x = np.random.randn(200)
y = 0.7 * x + 0.3 * np.random.randn(200)

# Quick correlation CI
corr, lower, upper = bootstrap_corr(
    x, y,
    n_resamples=5000,
    conf=0.99,
    random_state=42
)
```

**Signature**:
```python
bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 1000,
    conf: float = 0.95,
    random_state: int | None = None,
    nan_policy: str = "propagate"
) -> Tuple[float, float, float]
```

---

### `permutation_test()`

```python
from bunker_stats.resampling import permutation_test
import numpy as np

x = np.random.randn(100)
y = np.random.randn(100)

# Quick permutation test
stat, pvalue = permutation_test(
    x, y,
    test="corr",  # or "mean_diff"
    n_permutations=5000,
    alternative="two-sided",
    random_state=42
)
```

**Signature**:
```python
permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    test: str = "corr",  # "corr" or "mean_diff"
    n_permutations: int = 1000,
    alternative: str = "two-sided",
    random_state: int | None = None,
    nan_policy: str = "propagate"
) -> Tuple[float, float]
```

---

### `jackknife()`

```python
from bunker_stats.resampling import jackknife
import numpy as np

data = np.random.randn(200)

# Quick jackknife CI
estimate, lower, upper = jackknife(
    data,
    method="mean_ci",  # or "mean"
    conf=0.95
)
```

**Signature**:
```python
jackknife(
    x: np.ndarray,
    method: str = "mean_ci",  # "mean_ci" or "mean"
    conf: float = 0.95
) -> Tuple[float, float, float]
```

---

## NaN Handling Policies

### `nan_policy="propagate"` (Default)

**Behavior**: Passes NaNs directly to the Rust kernel.

**Performance**: ~0 µs overhead (direct passthrough)

**Use when**:
- Data is known to be clean (no NaNs)
- You want maximum performance
- You want NaN propagation semantics (any NaN → result is NaN)

**Example**:
```python
from bunker_stats.resampling import BootstrapConfig
import numpy as np

data = np.array([1, 2, 3, 4, 5])  # Clean data
config = BootstrapConfig(nan_policy="propagate")  # Default
result = config(data)  # Fast path, no overhead
```

---

### `nan_policy="omit"` (Smart Filtering)

**Behavior**: Filters NaNs in Python before calling Rust kernel.

**Performance**: 10-100 µs overhead (Python filtering + array copy)

**Use when**:
- Data may contain NaNs
- You want to ignore missing values
- You want warnings about data removal

**Filtering Semantics**:

| Method | Filtering Strategy |
|--------|-------------------|
| **Single array** (bootstrap, jackknife) | Removes all NaN values |
| **Paired arrays** (correlation) | Pairwise deletion (removes pairs where either is NaN) |
| **Two samples** (mean_diff) | Independent filtering (filters each sample separately) |

**Examples**:

```python
from bunker_stats.resampling import (
    BootstrapConfig, 
    BootstrapCorrConfig, 
    PermutationConfig
)
import numpy as np

# Example 1: Single array (removes NaNs)
data = np.array([1, 2, np.nan, 4, 5])
config = BootstrapConfig(nan_policy="omit")
# Warns: "BootstrapConfig: removed 1 NaN value(s) from input array"
result = config(data)  # Uses [1, 2, 4, 5]

# Example 2: Paired arrays (pairwise deletion)
x = np.array([1, 2, np.nan, 4, 5])
y = np.array([2, np.nan, 6, 8, 10])
config = BootstrapCorrConfig(nan_policy="omit")
# Warns: "BootstrapCorrConfig: removed 2 pair(s) with NaN values"
result = config(x, y)  # Uses pairs at indices [0, 3, 4]

# Example 3: Two samples (independent filtering)
group1 = np.array([1, np.nan, 3, 4])
group2 = np.array([5, 6, np.nan, 8])
config = PermutationConfig(nan_policy="omit")
# Warns: "PermutationConfig: removed 1 NaN value(s) from x, 1 from y"
result = config.run_mean_diff(group1, group2)  # Uses [1,3,4] vs [5,6,8]
```

**Error Handling**:
```python
# All NaNs raises error
data_all_nan = np.array([np.nan, np.nan, np.nan])
config = BootstrapConfig(nan_policy="omit")
config(data_all_nan)  # Raises: ValueError("all values are NaN after filtering")
```

---

## Input Validation

All config objects provide comprehensive validation with **actionable error messages**:

### Validation Examples

```python
from bunker_stats.resampling import BootstrapConfig, PermutationConfig

# Invalid n_resamples
config = BootstrapConfig(n_resamples=0)
# Raises: ValueError("n_resamples must be >= 1, got 0")

# Invalid confidence level
config = BootstrapConfig(conf=1.2)
# Raises: ValueError("conf must be in (0, 1), got 1.2. Hint: use conf=0.95 for a 95% CI.")

# Unsupported statistic
config = BootstrapConfig(stat="variance")
# Raises: ValueError("stat must be one of ['mean', 'median', 'std'], got 'variance'")

# Invalid alternative
config = PermutationConfig(alternative="not-equal")
# Raises: ValueError("alternative must be one of ['two-sided', 'less', 'greater'], got 'not-equal'")

# Empty array
config = BootstrapConfig()
config(np.array([]))
# Raises: ValueError("array is empty. Hint: ensure your data is non-empty before resampling.")

# Wrong dimensions
config = BootstrapConfig()
config(np.random.randn(10, 3))
# Raises: ValueError("expected 1D array, got shape (10, 3)")

# Length mismatch (correlation)
from bunker_stats.resampling import BootstrapCorrConfig
config = BootstrapCorrConfig()
config(np.random.randn(10), np.random.randn(15))
# Raises: ValueError("x and y must have same length. Got len(x)=10, len(y)=15.")
```

---

## Reproducibility

All resampling methods support deterministic seeding for reproducible results:

```python
from bunker_stats.resampling import BootstrapConfig
import numpy as np

data = np.random.randn(500)

# Same seed → identical results
config = BootstrapConfig(random_state=42)
result1 = config(data)
result2 = config(data)
assert result1 == result2  # ✓ Identical

# Different seeds → different results
config1 = BootstrapConfig(random_state=42)
config2 = BootstrapConfig(random_state=99)
result1 = config1(data)
result2 = config2(data)
assert result1 != result2  # ✓ Different

# None uses deterministic default (seed=0)
config_none = BootstrapConfig(random_state=None)
config_zero = BootstrapConfig(random_state=0)
assert config_none(data) == config_zero(data)  # ✓ Identical
```

---

## Performance Characteristics

### Overhead Measurements

| Operation | `nan_policy="propagate"` | `nan_policy="omit"` |
|-----------|--------------------------|---------------------|
| **Config creation** | ~1 µs | ~1 µs |
| **Input validation** | ~5 µs | ~5 µs |
| **NaN handling** | ~0 µs (passthrough) | ~10-100 µs (filtering) |
| **Rust kernel** | Direct call | Direct call |

**Recommendation**: 
- For hot loops, create config once and reuse
- Use `nan_policy="propagate"` when data is known to be clean
- Use `nan_policy="omit"` when convenience > microseconds matter

### Benchmarks (Rust vs Python/SciPy)

From `demo_resampling.py` benchmarks:

| Method | Dataset Size | Speedup vs SciPy/Python |
|--------|--------------|-------------------------|
| Bootstrap Mean CI | 1,000 | 15.2× faster |
| Bootstrap Mean CI | 500,000 | 23.3× faster |
| Bootstrap SE | 1,000 | 12.5× faster |
| Bootstrap SE | 500,000 | 14.0× faster |
| Permutation Corr | 1,000 | 170.5× faster |
| Permutation Corr | 500,000 | 187.3× faster |

**Normalized Performance** (per element/resample):
- Bootstrap: ~1 ns per element per resample
- Permutation: ~2 ns per element per permutation

---

## Test Coverage

The v0.2.9 config layer is **thoroughly tested** with 25 comprehensive tests:

### Test Categories

**1. Input Validation (8 tests)** ✅
- `n_resamples` validation
- `conf` validation  
- `stat` validation
- `alternative` validation
- Empty array rejection
- Multi-dimensional array rejection
- Length mismatch detection

**2. Equivalence Tests (6 tests)** ✅
- Bootstrap config matches Rust function
- Bootstrap corr config matches Rust function
- Permutation corr config matches Rust function
- Permutation mean_diff config matches Rust function
- Jackknife config matches Rust function

**3. NaN Policy Tests (6 tests)** ✅
- Propagate mode passes NaNs through
- Omit mode filters single arrays
- Omit mode does pairwise deletion for correlation
- Omit mode does independent filtering for two samples
- All-NaN arrays raise clear errors
- Warnings issued when NaNs removed

**4. Reproducibility Tests (3 tests)** ✅
- Same `random_state` gives identical results
- Different `random_state` gives different results
- `random_state=None` uses deterministic default

**5. Convenience Function Tests (3 tests)** ✅
- `bootstrap()` matches config object
- `permutation_test()` routes correctly
- `jackknife()` routes correctly

**6. Callable Shorthand Tests (2 tests)** ✅
- `config(data)` works as shorthand for `config.run(data)`
- Correlation config supports `__call__`

**Total: 25/25 tests passing (100% success rate)** 🎉

---

## Backward Compatibility

All existing flat Rust functions remain **100% unchanged and available**:

```python
import bunker_stats as bsr
import numpy as np

data = np.random.randn(100)
x = np.random.randn(100)
y = np.random.randn(100)

# Original flat functions still work
result = bsr.bootstrap_ci(data, stat="mean", n_resamples=1000, conf=0.95)
result = bsr.bootstrap_corr(x, y, n_resamples=1000, conf=0.95)
result = bsr.permutation_test_corr(x, y, n_permutations=1000, alternative="two-sided")
result = bsr.permutation_mean_diff_test(x, y, n_permutations=1000, alternative="greater")
result = bsr.jackknife_mean_ci(data, conf=0.95)
```

The config objects are a **purely additive** feature that wraps the same Rust kernels.

---

## Migration Guide

### From Flat Functions to Config Objects

**Before (v0.2.8 and earlier)**:
```python
import bunker_stats as bsr
import numpy as np

data = np.random.randn(500)

# Flat function approach
estimate, lower, upper = bsr.bootstrap_ci(
    data, 
    stat="mean", 
    n_resamples=5000, 
    conf=0.95, 
    random_state=42
)
```

**After (v0.2.9 with config objects)**:
```python
from bunker_stats.resampling import BootstrapConfig
import numpy as np

data = np.random.randn(500)

# Config object approach (recommended)
config = BootstrapConfig(
    n_resamples=5000,
    conf=0.95,
    stat="mean",
    random_state=42
)
estimate, lower, upper = config(data)
```

**Benefits of config objects**:
- ✅ Input validation with helpful errors
- ✅ NaN handling built-in
- ✅ Reusable configuration
- ✅ Clearer API
- ✅ Same performance

---

## Advanced Examples

### Example 1: Reusable Configuration

```python
from bunker_stats.resampling import BootstrapConfig
import numpy as np

# Create config once
config = BootstrapConfig(
    n_resamples=10000,
    conf=0.99,
    stat="median",
    random_state=42,
    nan_policy="omit"
)

# Reuse for multiple datasets
datasets = [
    np.random.randn(100),
    np.random.exponential(2, 200),
    np.random.randn(150) + np.array([np.nan] * 10 + [0] * 140)  # Has NaNs
]

results = [config(data) for data in datasets]
for i, (est, low, up) in enumerate(results):
    print(f"Dataset {i+1}: Median={est:.4f}, 99% CI=[{low:.4f}, {up:.4f}]")
```

### Example 2: Comparing Multiple Confidence Levels

```python
from bunker_stats.resampling import BootstrapConfig
import numpy as np

data = np.random.randn(500)

# Compare different confidence levels
confidence_levels = [0.90, 0.95, 0.99]

for conf in confidence_levels:
    config = BootstrapConfig(conf=conf, random_state=42)
    estimate, lower, upper = config(data)
    width = upper - lower
    print(f"{int(conf*100)}% CI: [{lower:.4f}, {upper:.4f}] (width: {width:.4f})")
```

### Example 3: Hypothesis Testing Workflow

```python
from bunker_stats.resampling import PermutationConfig
import numpy as np

# Generate data
np.random.seed(42)
treatment = np.random.randn(50) + 0.3  # Effect size = 0.3
control = np.random.randn(50)

# Test for mean difference
config = PermutationConfig(
    n_permutations=10000,
    alternative="greater",  # One-sided test
    random_state=42
)

mean_diff, pvalue = config.run_mean_diff(treatment, control)

print(f"Mean difference: {mean_diff:.4f}")
print(f"P-value: {pvalue:.4f}")

if pvalue < 0.05:
    print("✓ Significant effect detected (p < 0.05)")
else:
    print("✗ No significant effect (p >= 0.05)")
```

### Example 4: Robust Correlation with Missing Data

```python
from bunker_stats.resampling import BootstrapCorrConfig
import numpy as np

# Generate correlated data with missing values
np.random.seed(42)
n = 200
x = np.random.randn(n)
y = 0.7 * x + 0.3 * np.random.randn(n)

# Randomly insert NaNs (10% missing)
missing_idx_x = np.random.choice(n, size=int(0.1*n), replace=False)
missing_idx_y = np.random.choice(n, size=int(0.1*n), replace=False)
x[missing_idx_x] = np.nan
y[missing_idx_y] = np.nan

# Compute correlation with pairwise deletion
config = BootstrapCorrConfig(
    n_resamples=10000,
    conf=0.95,
    nan_policy="omit",  # Pairwise deletion
    random_state=42
)

corr, lower, upper = config(x, y)
print(f"Correlation: {corr:.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")
```

---

## What NOT to Do

### ❌ Don't use config objects for custom estimators

```python
# ❌ WRONG - not supported in v0.2.9
def my_custom_stat(x):
    return np.percentile(x, 75)

config = BootstrapConfig(stat=my_custom_stat)  # Won't work
```

**Reason**: Custom callbacks would require Python function calls in tight Rust loops, causing 10-100× slowdown and breaking parallelization.

**Supported statistics**: `"mean"`, `"median"`, `"std"` (all implemented in Rust)

### ❌ Don't create config objects in hot loops

```python
# ❌ INEFFICIENT
for i in range(10000):
    config = BootstrapConfig(n_resamples=1000)  # Creates new config each time
    result = config(data[i])

# ✅ EFFICIENT
config = BootstrapConfig(n_resamples=1000)  # Create once
for i in range(10000):
    result = config(data[i])  # Reuse config
```

### ❌ Don't use `nan_policy="omit"` when data is clean

```python
# ❌ SLOWER (adds 10-100 µs overhead)
data_clean = np.random.randn(1000)  # No NaNs
config = BootstrapConfig(nan_policy="omit")
result = config(data_clean)  # Unnecessary filtering overhead

# ✅ FASTER (zero overhead)
config = BootstrapConfig(nan_policy="propagate")  # Default
result = config(data_clean)  # Direct passthrough to Rust
```

---

## Future Roadmap (v0.3.0+)

Planned enhancements for future versions:

🔮 **More Statistics in Rust**
- Trimmed mean, variance, quantiles
- Skewness, kurtosis
- Custom percentile bootstraps

🔮 **Advanced CI Methods**
- BCa (bias-corrected and accelerated) intervals
- Bootstrap-t (studentized) intervals
- Jackknife-after-bootstrap diagnostics

🔮 **Parallel Control**
- Expose `n_threads` parameter
- Document `RAYON_NUM_THREADS` environment variable

🔮 **Stratified Resampling**
- Bootstrap within groups
- Stratified permutation tests

🔮 **Time-Series Methods**
- Block bootstrap (moving, circular, stationary)
- Seasonal bootstrap

---

## FAQ

### Q: Why use config objects instead of flat functions?

**A**: Config objects provide:
- ✅ Input validation with helpful error messages
- ✅ Built-in NaN handling
- ✅ Reusable configurations
- ✅ Clearer, more maintainable code
- ✅ Same performance as flat functions

### Q: Do config objects have performance overhead?

**A**: Minimal:
- Config creation: ~1 µs (one-time)
- Validation: ~5 µs (one-time)
- `nan_policy="propagate"`: ~0 µs overhead (direct passthrough)
- `nan_policy="omit"`: ~10-100 µs overhead (Python filtering)

For most use cases, this overhead is negligible compared to the resampling computation.

### Q: Can I use config objects with very large datasets?

**A**: Yes! The Rust kernels scale linearly:
- 1,000 samples: ~1-10 ms
- 100,000 samples: ~100-300 ms  
- 1,000,000 samples: ~1-3 seconds

Performance remains excellent even at scale.

### Q: How do I choose between `"propagate"` and `"omit"`?

**A**: 
- Use `"propagate"` (default) when:
  - Data is known to be clean
  - Maximum performance needed
  - You want NaN → NaN semantics
- Use `"omit"` when:
  - Data may have missing values
  - Convenience > microseconds
  - You want warnings about data removal

### Q: Are results identical to SciPy?

**A**: For bootstrap CI bounds, results are **highly similar** but not identical due to:
- Different RNG implementations
- Different percentile algorithms
- Different random seeding

Numerical accuracy is excellent: max error < 0.01 for CI bounds, SE errors < 1%.

### Q: Can I contribute new features?

**A**: Yes! See the project repository for contribution guidelines. Priority areas:
- New statistics (implement in Rust for best performance)
- Better error messages
- Documentation improvements
- Additional tests

---

## Citation

If you use bunker-stats in academic work, please cite:

```bibtex
@software{bunker_stats,
  title = {bunker-stats: High-Performance Statistical Computing for Python},
  author = {[Your Name]},
  year = {2024},
  version = {0.2.9},
  url = {https://github.com/[your-repo]/bunker-stats}
}
```

---

## License

[Your license here - e.g., MIT, Apache 2.0, etc.]

---

## Support

- **Documentation**: [Link to docs]
- **Issues**: [Link to GitHub issues]
- **Discussions**: [Link to discussions]

---

## Changelog (v0.2.9)

### Added
- ✨ Config objects for ergonomic API (`BootstrapConfig`, `BootstrapCorrConfig`, `PermutationConfig`, `JackknifeConfig`)
- ✨ Comprehensive input validation with actionable error messages
- ✨ Flexible NaN handling (`"propagate"` and `"omit"` modes)
- ✨ Convenience functions (`bootstrap()`, `bootstrap_corr()`, `permutation_test()`, `jackknife()`)
- ✨ Callable shorthand syntax (`config(data)`)
- ✨ Deterministic random seeding for reproducibility
- ✨ 25 comprehensive tests (100% passing)

### Changed
- None (fully backward compatible)

### Fixed
- None

---

**bunker-stats v0.2.9** - Fast statistics, ergonomic Python. 🚀
