# bunker-stats Robust Statistics Module - Documentation Index

**Complete documentation for the robust statistics module**

Version: 0.2.9+ (Optimized)  
Last Updated: January 2026

---

## 📚 Documentation Overview

This module provides **production-grade robust statistical estimators** that are resistant to outliers and contamination. Built with Rust for performance and wrapped in Python for ease of use.

**Core Principles:**
- 🎯 **Deterministic** - same input always produces same output
- ⚡ **Fast** - 2-244x faster than SciPy/statsmodels equivalents
- 🔢 **Numerically stable** - accurate results on extreme data
- 🐍 **Pythonic** - ergonomic API with sensible defaults

---

## 📖 Available Documentation

### 1. **ROBUST_STATS_README.md** - Main Documentation
**Start here for comprehensive coverage**

Contents:
- Overview and quick start
- What's new in v0.2.9
- All 10+ estimators with examples
- Performance benchmarks
- Complete API reference
- Real-world usage examples
- NaN handling philosophy
- Plans for v0.3.0
- Design principles
- Testing guide

**Use when:** You want to understand the full capabilities of the module

---

### 2. **QUICK_REFERENCE.md** - Cheat Sheet
**Fast lookup for common tasks**

Contents:
- 30-second quick start
- Function cheat sheet
- Common patterns (outlier detection, A/B testing, etc.)
- Parameter guide
- Performance tips
- Typical workflows
- FAQ

**Use when:** You know what you want to do, just need the syntax

---

### 3. **MIGRATION_GUIDE.md** - SciPy/statsmodels Migration
**For users switching from other libraries**

Contents:
- Why migrate?
- Function-by-function comparison
- API differences
- Breaking changes
- Performance comparisons
- Complete migration example
- Migration checklist

**Use when:** You're coming from SciPy, statsmodels, or pandas

---

## 🚀 Quick Start (30 seconds)

```python
import bunker_stats as bs
import numpy as np

# Data with outliers
data = np.array([1, 2, 3, 4, 5, 100])

# Robust statistics
location, scale = bs.robust_fit(data)       # (3.5, 2.22)
scores = bs.robust_score(data)              # Robust z-scores
outliers = data[np.abs(scores) > 3]         # [100]

# Time series
smoothed = bs.rolling_median(signal, 10)    # 244x faster than pandas
```

---

## 📊 What Can You Do?

### Basic Robust Estimation
```python
bs.median(data)              # Robust center
bs.mad(data)                 # Robust spread
bs.trimmed_mean(data, 0.1)   # Trim 10% from each tail
bs.iqr(data)                 # Interquartile range
```

### Outlier Detection
```python
scores = bs.robust_score(data)
is_outlier = np.abs(scores) > 3
clean_data = data[~is_outlier]
```

### Time Series
```python
smoothed = bs.rolling_median(signal, window=10)
```

### Custom Configuration
```python
rs = bs.RobustStats(
    location="trimmed_mean",
    scale="iqr",
    trim=0.1
)
loc, scale = rs.fit(data)
```

---

## ⚡ Performance Highlights

Actual benchmarks vs SciPy/statsmodels/pandas:

| Operation | n=1,000,000 | Speedup |
|-----------|-------------|---------|
| Median | 12.8ms | **2.9x faster** |
| MAD | 23.8ms | **4.6x faster** |
| Rolling Median | 404ms | **246x faster** |
| Qn Scale (n=1K) | 6.6ms | **124x faster** |
| robust_fit | 26.9ms | **5.2x faster** |
| robust_score | 29.4ms | **5.2x faster** |

**Average speedups across all data sizes:**
- Median: **7.5x faster**
- MAD: **17.3x faster**
- Rolling Median: **239x faster**
- robust_fit: **6.9x faster**

---

## 🎯 Key Features

### v0.2.9 Highlights

**Ergonomics:**
- ✅ `RobustStats` class with policy-driven configuration
- ✅ `robust_fit()` and `robust_score()` convenience functions
- ✅ Skip-NaN variants for all estimators
- ✅ 73 comprehensive tests

**Performance:**
- ✅ 2-5x faster via `select_nth_unstable` (O(n) vs O(n log n))
- ✅ Fused median+MAD kernel (40% faster `robust_fit`)
- ✅ Hybrid rolling median (2-4x faster small windows)
- ✅ Zero-allocation workspace API

**Reliability:**
- ✅ Deterministic (bit-exact reproducibility)
- ✅ Numerically stable
- ✅ Explicit NaN handling
- ✅ SciPy/NumPy parity verified

---

## 📝 Common Use Cases

### 1. Replace Mean/Std
```python
# Outlier-resistant statistics
location, scale = bs.robust_fit(data)
```

### 2. Clean Sensor Data
```python
scores = bs.robust_score(sensor_readings)
clean = sensor_readings[np.abs(scores) < 3]
```

### 3. A/B Testing
```python
loc_a, scale_a = bs.robust_fit(group_a)
loc_b, scale_b = bs.robust_fit(group_b)
effect_size = (loc_a - loc_b) / np.mean([scale_a, scale_b])
```

### 4. Time Series Smoothing
```python
smoothed_signal = bs.rolling_median(noisy_signal, window=5)
```

---

## 🔍 Finding What You Need

### "I want to..."

**...understand what robust statistics are**
→ Read ROBUST_STATS_README.md "Overview" section

**...see quick examples**
→ Check QUICK_REFERENCE.md "Common Patterns"

**...migrate from SciPy**
→ Follow MIGRATION_GUIDE.md step-by-step

**...learn the complete API**
→ See ROBUST_STATS_README.md "API Reference"

**...detect outliers**
→ QUICK_REFERENCE.md "Pattern 2: Outlier Detection"

**...smooth a time series**
→ QUICK_REFERENCE.md "Pattern 4: Time Series Smoothing"

**...handle missing data**
→ ROBUST_STATS_README.md "NaN Handling Philosophy"

**...configure custom estimators**
→ ROBUST_STATS_README.md "Advanced Usage"

**...see performance benchmarks**
→ ROBUST_STATS_README.md "Performance Benchmarks"

**...understand the design**
→ ROBUST_STATS_README.md "Design Philosophy"

---

## 🧪 Testing

```bash
# All robust tests (73 tests)
pytest -k robust -v

# Unit tests only (33 tests)
pytest src/kernels/robust/test_robust_ver2_9.py -v

# Integration tests only (40 tests)
pytest tests/test_robust_stats.py -v
```

**Test coverage:**
- Policy dispatch
- Determinism verification
- Edge case handling
- Numerical stability
- SciPy parity
- Performance benchmarks

---

## 🗺️ Roadmap to v0.3.0

**Planned features:**

1. **Multivariate Statistics**
   - Robust covariance (MCD, OGK)
   - Mahalanobis distance
   - Multivariate outlier detection

2. **Robust Regression**
   - Huber regression
   - Theil-Sen estimator
   - RANSAC

3. **Performance**
   - Automatic parallelization
   - 5-10x faster multivariate ops
   - Sub-linear scaling

4. **Additional Estimators**
   - Biweight location/scale
   - S-estimators
   - MM-estimators
   - Hampel estimator

5. **Weighted Statistics**
   - Weighted median
   - Weighted MAD
   - Weighted robust fit

---

## 📚 External Resources

### Theory References
- Huber, P. J. (1964). "Robust Estimation of a Location Parameter"
- Rousseeuw, P. J. & Croux, C. (1993). "Alternatives to the Median Absolute Deviation"
- Maronna, R. A., et al. (2006). "Robust Statistics: Theory and Methods"

### Python Packages Comparison
- **SciPy** (`scipy.stats`) - Standard library, decent performance
- **statsmodels** - More estimators, slower
- **scikit-learn** - ML-focused, limited robust stats
- **pandas** - Slow rolling operations
- **bunker-stats** - Fastest, most complete robust stats

---

## 🆘 Getting Help

### In Python
```python
import bunker_stats as bs
help(bs.robust_fit)
help(bs.RobustStats)
```

### Common Issues

**Q: Getting NaN results**  
A: Check if input is empty or has all NaN. Use `_skipna` variants if needed.

**Q: Different results from SciPy**  
A: Check `mad_consistent` parameter. Results should match within FP tolerance.

**Q: Slow performance**  
A: Make sure you imported `bunker_stats`, not `scipy.stats`

**Q: Need to handle NaN**  
A: Use skip-NaN variants: `median_skipna()`, `mad_skipna()`, etc.

---

## 📁 File Structure

```
robust/
├── ROBUST_STATS_README.md    # Main documentation (comprehensive)
├── QUICK_REFERENCE.md         # Cheat sheet (fast lookup)
├── MIGRATION_GUIDE.md         # SciPy migration guide
├── README.md                  # This file (index)
│
├── extended.rs                # Core estimators (optimized)
├── fit.rs                     # Robust fit/score (fused kernels)
├── rolling.rs                 # Rolling statistics
├── policy.rs                  # Policy configuration
├── pyrobust.rs                # Python bindings
├── mod.rs                     # Module exports
│
├── test_robust_ver2_9.py      # Unit tests (33 tests)
└── test_robust_stats.py       # Integration tests (40 tests)
```

---

## 🎓 Learning Path

### Beginner
1. Read ROBUST_STATS_README.md "Overview"
2. Try examples in "Quick Start"
3. Use QUICK_REFERENCE.md for common tasks

### Intermediate
1. Read "API Reference" in ROBUST_STATS_README.md
2. Try "Real-World Examples"
3. Experiment with `RobustStats` class
4. Read "NaN Handling Philosophy"

### Advanced
1. Study "Design Philosophy"
2. Read "Advanced Usage"
3. Explore policy combinations
4. Check source code (Rust files)
5. Contribute improvements

### Migrating from SciPy
1. Read MIGRATION_GUIDE.md
2. Follow function-by-function comparison
3. Use migration checklist
4. Verify results match

---

## 🤝 Contributing

We welcome contributions! Key areas:

- **New estimators** - Biweight, Hampel, S/MM estimators
- **Multivariate methods** - Robust covariance, PCA
- **Performance** - Further optimizations
- **Documentation** - Examples, tutorials
- **Testing** - More edge cases, benchmarks

See CONTRIBUTING.md for guidelines.

---

## 📜 Version History

### v0.2.9 (Current - Optimized)
- Added `RobustStats` class
- Added `robust_fit()` and `robust_score()`
- Added `rolling_median()`
- Added skip-NaN variants
- 2-5x performance improvements
- Fused kernels optimization
- 73 comprehensive tests

### v0.2.8 (Legacy)
- Initial robust statistics
- Basic estimators only
- Functional API

---

## 🔗 Quick Links

| Document | Best For |
|----------|----------|
| **ROBUST_STATS_README.md** | Learning everything about the module |
| **QUICK_REFERENCE.md** | Fast syntax lookup |
| **MIGRATION_GUIDE.md** | Switching from SciPy |
| **README.md** (this file) | Finding the right documentation |

---

## 💡 Pro Tips

1. **Use `robust_fit()` for location + scale** - 40% faster than separate calls
2. **Reuse `RobustStats` objects** - Avoid redundant parsing
3. **Use `rolling_median()` not pandas** - 244x faster
4. **Check `mad_consistent` parameter** - True for std-like interpretation
5. **Use `_skipna` variants consciously** - Don't hide data quality issues

---

## 📫 Feedback

Found a bug? Have a suggestion? Want to contribute?

- Open an issue on GitHub
- Read CONTRIBUTING.md
- Check existing issues first

---

**bunker-stats: Production-grade robust statistics** 🚀

*Because real-world data has outliers.*
