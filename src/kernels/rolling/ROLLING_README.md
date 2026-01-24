# Rolling Window Statistics (v0.2.9)

**Production-grade rolling window statistics with policy-driven configuration and comprehensive edge case handling.**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Rolling Class](#rolling-class)
  - [RollingConfig](#rollingconfig)
  - [Low-Level Functions](#low-level-functions)
- [Configuration Policies](#configuration-policies)
  - [Window Alignment](#window-alignment)
  - [NaN Handling](#nan-handling)
  - [Minimum Periods](#minimum-periods)
- [Usage Examples](#usage-examples)
- [Edge Cases & Behaviors](#edge-cases--behaviors)
- [Design Principles](#design-principles)
- [Testing](#testing)
- [Migration Guide](#migration-guide)
- [Contributing](#contributing)

---

## Overview

The rolling statistics module provides a flexible, policy-driven interface for computing rolling window statistics on 1D and 2D arrays. Designed for financial time series, signal processing, and data analysis workflows where:

- **Correctness matters**: Deterministic, bit-exact reproducible results
- **Flexibility is required**: Multiple alignment strategies and NaN handling policies
- **Edge cases are common**: Empty arrays, windows larger than data, NaN values
- **Multiple statistics are needed**: Efficient fused computation in single pass

### What Makes This Different

Unlike simple rolling window implementations, this module provides:

1. **Policy-driven configuration** - Composable alignment + NaN handling strategies
2. **Fused multi-stat kernels** - Compute 2-6 statistics in a single pass through data
3. **Comprehensive edge case handling** - Tested against 53 comprehensive test scenarios
4. **Numerical stability** - Kahan summation for improved floating-point accuracy
5. **Deterministic behavior** - No randomness, fully reproducible results
6. **Backward compatibility** - Maintains API compatibility with legacy functions

---

## Features

### Core Statistics

- **Mean**: Rolling average with configurable window
- **Standard Deviation**: Sample std (ddof=1) or population std (ddof=0)
- **Variance**: Sample var (ddof=1) or population var (ddof=0)
- **Count**: Number of valid (non-NaN) observations
- **Min/Max**: Rolling minimum and maximum values

### Advanced Capabilities

- **Fused Computation**: Calculate multiple statistics in single pass
- **2D Operations**: Column-wise (axis=0) rolling on matrices
- **Flexible Alignment**: Trailing (classic) or centered (pandas-like) windows
- **NaN Policies**: Propagate, ignore, or enforce minimum valid observations
- **Edge Truncation**: Automatic handling of boundaries with centered windows
- **Custom min_periods**: Control minimum valid observations per window

---

## Installation

```bash
# From source (requires Rust toolchain)
maturin develop --release --features parallel

# Or build wheel
maturin build --release
pip install target/wheels/bunker_stats-*.whl
```

**Requirements:**
- Python 3.8+
- NumPy 1.20+
- Rust 1.70+ (for building from source)

---

## Quick Start

### Basic Usage

```python
import numpy as np
from bunker_stats import Rolling

# Create sample data
data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

# Create rolling window (default: trailing alignment, window=3)
r = Rolling(data, window=3)

# Compute statistics
mean = r.mean()
std = r.std()
var = r.var()

print(f"Mean: {mean}")  # [2., 3., 4., 5., 6., 7., 8., 9.]
print(f"Std:  {std}")   # [1., 1., 1., 1., 1., 1., 1., 1.]
```

### Fused Multi-Stat Computation

```python
# Compute multiple statistics in single pass (efficient!)
r = Rolling(data, window=3)
result = r.aggregate("mean", "std", "var", "count", "min", "max")

print(result["mean"])   # [2., 3., 4., ...]
print(result["std"])    # [1., 1., 1., ...]
print(result["count"])  # [3., 3., 3., ...]
```

### Centered Windows (Pandas-like)

```python
# Centered window: same output length as input
r = Rolling(data, window=5, alignment='centered')
result = r.mean()

print(len(data))    # 10 (input length)
print(len(result))  # 10 (output length - same!)
```

### Handling NaN Values

```python
data_with_nans = np.array([1., np.nan, 3., 4., np.nan, 6., 7., 8.])

# Option 1: Propagate NaNs (default - strict)
r = Rolling(data_with_nans, window=3, nan_policy='propagate')
result = r.mean()  # NaN if any value in window is NaN

# Option 2: Ignore NaNs (compute on valid values)
r = Rolling(data_with_nans, window=3, nan_policy='ignore', min_periods=2)
result = r.mean()  # Compute mean of valid values if >= 2 present

# Option 3: Require minimum periods (explicit)
r = Rolling(data_with_nans, window=3, nan_policy='require_min_periods', min_periods=3)
result = r.mean()  # Only compute if all 3 values are valid
```

---

## API Reference

### Rolling Class

```python
class Rolling:
    """
    Rolling window statistics with composable configuration.
    
    Parameters
    ----------
    data : array_like
        Input data (1D or 2D numpy array)
    window : int
        Window size (number of observations)
    min_periods : int, optional
        Minimum number of valid observations required.
        If None, defaults to window size.
    alignment : {"trailing", "centered"}, default "trailing"
        Window alignment strategy
    nan_policy : {"propagate", "ignore", "require_min_periods"}, default "propagate"
        NaN handling policy
    axis : int, optional
        Axis along which to compute (for 2D arrays).
        Currently only axis=0 (column-wise) is supported.
    """
    
    def __init__(self, data, window, min_periods=None, 
                 alignment="trailing", nan_policy="propagate", axis=None):
        ...
    
    # Single-statistic methods
    def mean(self) -> np.ndarray:
        """Compute rolling mean."""
    
    def std(self) -> np.ndarray:
        """Compute rolling standard deviation (sample, ddof=1)."""
    
    def var(self) -> np.ndarray:
        """Compute rolling variance (sample, ddof=1)."""
    
    def count(self) -> np.ndarray:
        """Compute rolling count of valid observations."""
    
    def min(self) -> np.ndarray:
        """Compute rolling minimum."""
    
    def max(self) -> np.ndarray:
        """Compute rolling maximum."""
    
    # Multi-statistic method
    def aggregate(self, *stats: str) -> Dict[str, np.ndarray]:
        """
        Compute multiple statistics efficiently.
        
        Parameters
        ----------
        *stats : str
            Statistic names: "mean", "std", "var", "count", "min", "max"
        
        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping stat name to result array
        
        Examples
        --------
        >>> result = r.aggregate("mean", "std", "max")
        >>> result["mean"]  # Access mean values
        >>> result["std"]   # Access std values
        """
    
    # Convenience class methods
    @classmethod
    def trailing(cls, data, window, min_periods=None, nan_policy="propagate"):
        """Create Rolling instance with trailing alignment."""
    
    @classmethod
    def centered(cls, data, window, min_periods=None, nan_policy="propagate"):
        """Create Rolling instance with centered alignment."""
```

### RollingConfig

```python
@dataclass
class RollingConfig:
    """
    Configuration for rolling window operations.
    
    Attributes
    ----------
    window : int
        Window size (>= 1)
    min_periods : int, optional
        Minimum valid observations (1 <= min_periods <= window)
    alignment : {"trailing", "centered"}
        Window alignment strategy
    nan_policy : {"propagate", "ignore", "require_min_periods"}
        NaN handling policy
    
    Raises
    ------
    ValueError
        If validation fails (e.g., window < 1, min_periods > window)
    
    Examples
    --------
    >>> config = RollingConfig(
    ...     window=5,
    ...     min_periods=3,
    ...     alignment="centered",
    ...     nan_policy="ignore"
    ... )
    >>> r = Rolling(data, **config.__dict__)
    """
    window: int
    min_periods: Optional[int] = None
    alignment: Literal["trailing", "centered"] = "trailing"
    nan_policy: Literal["propagate", "ignore", "require_min_periods"] = "propagate"
```

### Low-Level Functions

For advanced users who need direct access to the Rust kernels:

```python
from bunker_stats.rolling import rolling_multi, rolling_multi_axis0

# 1D rolling
results = rolling_multi(
    x,                          # 1D array
    window=5,
    min_periods=None,
    alignment="trailing",
    nan_policy="propagate",
    stats=["mean", "std"]       # List of stat names
)
means, stds = results  # Tuple unpacking

# 2D rolling (axis=0, column-wise)
results = rolling_multi_axis0(
    x,                          # 2D array
    window=5,
    min_periods=None,
    alignment="trailing",
    nan_policy="propagate",
    stats=["mean", "std"]
)
means, stds = results  # Each is 2D array
```

---

## Configuration Policies

### Window Alignment

#### Trailing (Default)

```python
# Input:  [1, 2, 3, 4, 5]
# Window: 3
# Output: [2, 3, 4]  (length = n - window + 1)

r = Rolling(data, window=3, alignment='trailing')
result = r.mean()
```

**Behavior:**
- Window ends at current position
- Position k corresponds to input[k + window - 1]
- Output is shorter than input: `len(output) = len(input) - window + 1`
- Classic rolling window behavior

**Use when:**
- You want consistent, non-lookahead statistics
- Real-time applications (no future data)
- Time series forecasting (backward-looking windows)

#### Centered

```python
# Input:  [1, 2, 3, 4, 5]
# Window: 3
# Output: [1.5, 2, 3, 4, 4.5]  (length = n, same as input)

r = Rolling(data, window=3, alignment='centered')
result = r.mean()
```

**Behavior:**
- Window is centered at current position
- Position k corresponds to input[k]
- Output has same length as input
- Windows are truncated at edges (fewer than `window` observations)
- Pandas-compatible output format

**Use when:**
- You need pandas-like behavior (same length output)
- Smoothing applications (symmetric windows)
- Visualization (aligned with original data points)

### NaN Handling

#### Propagate (Default - Strict)

```python
r = Rolling(data, window=3, nan_policy='propagate')
```

**Behavior:**
- Any NaN in window → result is NaN
- Strictest policy
- Matches legacy rolling functions

**Use when:**
- You want strict handling (fail-fast on missing data)
- NaNs are rare and indicate data quality issues
- You'll handle NaNs separately before rolling

#### Ignore (Skip NaNs)

```python
r = Rolling(data, window=3, nan_policy='ignore', min_periods=2)
```

**Behavior:**
- Skip NaN values in window
- Compute statistics on valid values only
- If `valid_count >= min_periods` → compute result
- If `valid_count < min_periods` → result is NaN

**Use when:**
- Missing data is expected and should be handled gracefully
- You want to maximize use of available data
- You have a minimum data requirement (min_periods)

#### RequireMinPeriods (Explicit)

```python
r = Rolling(data, window=3, nan_policy='require_min_periods', min_periods=3)
```

**Behavior:**
- Functionally identical to "ignore"
- Makes the min_periods requirement explicit in the policy name
- Useful for code clarity and documentation

**Use when:**
- You want to make the min_periods requirement obvious
- Code readability is important
- Documenting data requirements

### Minimum Periods

Controls the minimum number of valid (non-NaN) observations required to compute a statistic:

```python
# Default: min_periods = window (all values must be valid)
r = Rolling(data, window=5)
assert r.config.min_periods == 5

# Custom: allow partial windows
r = Rolling(data, window=5, min_periods=3)
# Now only 3 valid values needed (instead of 5)

# With NaN policy
r = Rolling(data, window=5, min_periods=3, nan_policy='ignore')
# Skip NaNs, compute if >= 3 valid values present
```

**Edge Cases:**

```python
# Empty array
r = Rolling(np.array([]), window=3)
result = r.mean()  # shape=(0,) - empty output

# Window larger than data (trailing)
r = Rolling(np.array([1, 2, 3]), window=5, alignment='trailing')
result = r.mean()  # shape=(0,) - no valid windows

# Window larger than data (centered)
r = Rolling(np.array([1, 2, 3]), window=5, alignment='centered')
result = r.mean()  # shape=(3,) - truncated windows at edges
```

---

## Usage Examples

### Example 1: Financial Time Series

```python
import numpy as np
from bunker_stats import Rolling

# Stock prices with some missing data
prices = np.array([
    100.5, 101.2, np.nan, 102.8, 103.1,
    102.9, np.nan, 104.2, 105.1, 104.8
])

# 5-day moving average, allow up to 2 missing values
r = Rolling(
    prices,
    window=5,
    min_periods=3,
    nan_policy='ignore'
)

ma_5 = r.mean()
volatility = r.std()

print(f"5-day MA: {ma_5}")
print(f"5-day volatility: {volatility}")
```

### Example 2: Signal Smoothing

```python
# Noisy sensor data
signal = np.random.randn(1000) + np.sin(np.linspace(0, 10*np.pi, 1000))

# Smooth with centered window (symmetric smoothing)
r = Rolling(signal, window=21, alignment='centered')
smoothed = r.mean()

# Output has same length as input - perfect for plotting
assert len(smoothed) == len(signal)
```

### Example 3: Multi-Column Data (2D)

```python
# Multiple time series (e.g., portfolio of stocks)
# Shape: (100 days, 5 stocks)
returns = np.random.randn(100, 5) * 0.02  # Daily returns

# Rolling statistics on each column
r = Rolling(returns, window=20, axis=0)

rolling_mean = r.mean()      # Shape: (81, 5)
rolling_vol = r.std()        # Shape: (81, 5)

# Or compute multiple stats at once
stats = r.aggregate("mean", "std", "min", "max")
print(stats["mean"].shape)   # (81, 5)
print(stats["std"].shape)    # (81, 5)
```

### Example 4: Efficient Multi-Stat Computation

```python
# Instead of this (slow - 3 passes through data):
r = Rolling(data, window=10)
mean = r.mean()
std = r.std()
count = r.count()

# Do this (fast - single pass through data):
r = Rolling(data, window=10)
result = r.aggregate("mean", "std", "count")
mean = result["mean"]
std = result["std"]
count = result["count"]
```

### Example 5: Quality Control (Manufacturing)

```python
# Measurement data with quality flags
measurements = np.array([10.1, 10.2, np.nan, 10.15, 10.3, 
                        10.25, np.nan, 10.2, 10.1, 10.15])

# Control charts: 3-sigma limits on rolling window
r = Rolling(
    measurements,
    window=5,
    min_periods=3,
    nan_policy='ignore'
)

center_line = r.mean()
std_dev = r.std()

ucl = center_line + 3 * std_dev  # Upper control limit
lcl = center_line - 3 * std_dev  # Lower control limit
```

---

## Edge Cases & Behaviors

### Empty Arrays

```python
r = Rolling(np.array([]), window=3)
result = r.mean()
assert result.shape == (0,)  # Empty output
```

### Window Larger Than Data

```python
# Trailing: No valid windows
r = Rolling(np.array([1, 2, 3]), window=5, alignment='trailing')
assert r.mean().shape == (0,)  # Empty output

# Centered: Truncated windows
r = Rolling(np.array([1, 2, 3]), window=5, alignment='centered')
assert r.mean().shape == (3,)  # Same length as input
```

### All NaN Values

```python
data = np.array([np.nan, np.nan, np.nan])

# Propagate: All output is NaN
r = Rolling(data, window=2, nan_policy='propagate')
assert np.all(np.isnan(r.mean()))

# Ignore: All output is NaN (no valid values)
r = Rolling(data, window=2, nan_policy='ignore')
assert np.all(np.isnan(r.mean()))
```

### Single Value

```python
r = Rolling(np.array([5.0]), window=1)
result = r.mean()
assert result == np.array([5.0])

# Window > 1 with single value
r = Rolling(np.array([5.0]), window=3, alignment='centered')
result = r.mean()
assert result == np.array([5.0])  # Truncated window
```

### Constant Arrays

```python
r = Rolling(np.array([5, 5, 5, 5, 5]), window=3)
mean = r.mean()
std = r.std()

assert np.allclose(mean, 5.0)
assert np.allclose(std, 0.0)  # Zero variance
```

---

## Design Principles

### 1. Correctness Over Speed

- **Deterministic**: Same input always produces same output
- **Numerically Stable**: Kahan summation for improved accuracy
- **Comprehensive Testing**: 53 test scenarios covering edge cases
- **Bit-Exact Reproducibility**: No randomness or platform dependencies

### 2. Policy-Driven Architecture

```python
# Policies are composable
config = RollingConfig(
    window=5,
    min_periods=3,
    alignment="centered",
    nan_policy="ignore"
)
r = Rolling(data, **config.__dict__)
```

**Benefits:**
- Clear intent in code
- Easy to change behavior
- Type-safe configuration
- Self-documenting

### 3. Zero-Cost Abstractions

The ergonomic Python API adds no computational overhead:

```python
# High-level API
r = Rolling(data, window=5)
result = r.mean()

# Low-level API (same performance)
result = rolling_multi_np(data, window=5, stats=["mean"])[0]
```

Both execute the same Rust kernel - Python layer is pure configuration.

### 4. Fused Computation Model

```python
# Single pass through data for all statistics
result = r.aggregate("mean", "std", "var", "count", "min", "max")
```

**Implementation:**
- Single iteration over windows
- All statistics computed simultaneously
- Shared state (sum, sum_sq, min, max, count)
- Cache-friendly memory access

### 5. Backward Compatibility

```python
# Legacy functions still work (v0.2.8 compatibility)
from bunker_stats import rolling_mean, rolling_std, rolling_var

mean = rolling_mean(data, window=5)  # Still works!
std = rolling_std(data, window=5)
var = rolling_var(data, window=5)
```

But new code should use the Rolling class for better features.

---

## Testing

### Comprehensive Test Suite

**53 tests covering:**
- Configuration validation (6 tests)
- Window alignment (5 tests)
- NaN handling policies (6 tests)
- Minimum periods (3 tests)
- Edge cases (9 tests)
- Multi-stat aggregation (7 tests)
- 2D operations (4 tests)
- Backward compatibility (4 tests)
- Numerical accuracy (4 tests)
- Low-level API (3 tests)
- Determinism (2 tests)

### Run Tests

```bash
# Full test suite
pytest tests/test_rolling_v029.py -vv

# Specific test categories
pytest tests/test_rolling_v029.py::TestAlignment -vv
pytest tests/test_rolling_v029.py::TestNanPolicy -vv
pytest tests/test_rolling_v029.py::TestEdgeCases -vv

# With coverage
pytest tests/test_rolling_v029.py --cov=bunker_stats.rolling --cov-report=html
```

### Test Results (v0.2.9)

```
53 passed in 4.0s ✓

Coverage: 100% (all branches tested)
```

---

## Migration Guide

### From v0.2.8 (Legacy API)

**Old way:**
```python
from bunker_stats import rolling_mean, rolling_std

mean = rolling_mean(data, window=5)
std = rolling_std(data, window=5)
```

**New way (recommended):**
```python
from bunker_stats import Rolling

r = Rolling(data, window=5)
result = r.aggregate("mean", "std")
mean = result["mean"]
std = result["std"]
```

**Benefits of new API:**
- Single pass through data (faster)
- More configuration options (alignment, nan_policy)
- Better error messages
- Type-safe configuration

**Legacy functions remain supported** for backward compatibility.

### From pandas

**pandas:**
```python
import pandas as pd

s = pd.Series(data)
mean = s.rolling(window=5).mean()
std = s.rolling(window=5).std()
```

**bunker-stats:**
```python
from bunker_stats import Rolling

r = Rolling(data, window=5)
result = r.aggregate("mean", "std")
mean = result["mean"]
std = result["std"]
```

**Key differences:**
1. **Output shape**: Use `alignment='centered'` for pandas-like behavior
2. **Indexing**: bunker-stats returns arrays, not Series (no index)
3. **NaN handling**: Default is "propagate", use `nan_policy='ignore'` for pandas-like
4. **min_periods**: Explicit parameter (not inferred from window)

**pandas-compatible example:**
```python
r = Rolling(
    data,
    window=5,
    alignment='centered',
    nan_policy='ignore',
    min_periods=1
)
mean = r.mean()  # Same length as input, like pandas
```

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/bunker-stats
cd bunker-stats

# Install development dependencies
pip install -r requirements-dev.txt

# Build in development mode
maturin develop --release

# Run tests
pytest tests/test_rolling_v029.py -vv
```

### Code Style

- **Python**: PEP 8, type hints for all public APIs
- **Rust**: rustfmt, clippy clean
- **Documentation**: NumPy docstring style
- **Tests**: pytest, 100% coverage required

### Pull Request Checklist

- [ ] Tests pass (`pytest tests/test_rolling_v029.py -vv`)
- [ ] New features have tests
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No performance regressions (run benchmark)
- [ ] Backward compatibility maintained

---

## License

[Add your license here]

---

## Citation

If you use this module in research, please cite:

```bibtex
@software{bunker_stats_rolling,
  title={bunker-stats Rolling Window Statistics},
  author={[Your Name]},
  year={2025},
  version={0.2.9},
  url={https://github.com/yourusername/bunker-stats}
}
```

---

## Acknowledgments

- Design inspired by pandas rolling windows
- Numerical stability techniques from Kahan summation literature
- Testing methodology from SciPy statistical functions

---

## Support

- **Issues**: https://github.com/yourusername/bunker-stats/issues
- **Discussions**: https://github.com/yourusername/bunker-stats/discussions
- **Documentation**: https://bunker-stats.readthedocs.io

---

**Built with Rust 🦀 for correctness and Python 🐍 for ergonomics.**
