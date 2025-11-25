.

💥 bunker-stats

A Rust-powered statistical toolkit with a Python API and pandas Styler integration.

bunker-stats is a hybrid Rust/Python library providing fast, numerically-stable statistical primitives, rolling-window analytics, distribution tools, and pandas Styler visualizations — all backed by Rust for correctness and performance.



Project Philosophy & Status (v0.1)

bunker-stats is intentionally released early.

The goal is not to replace NumPy or pandas, but to build a Rust-accelerated analytics toolkit that grows feature-by-feature.
This first release focuses on correctness, clear API design, and a solid suite of statistical primitives.

Future releases will focus on:

performance tuning (SIMD, fused loops, BLAS-backed ops)

smarter rolling-window pipelines

more visualization helpers

NaN-safe variants of all ops

multi-column Rust kernels

improved correlation-matrix engine

This library is actively evolving, and v0.1 is the foundation everything else will build on.

🚀 Features
Core statistics (Rust)

Mean, variance, std (sample vs population)

Z-scores

MAD (Median Absolute Deviation)

Percentiles & quantiles

IQR & Tukey outlier fences

Covariance / correlation

Welford one-pass mean/variance

EWMA (exponentially weighted moving average)

Rolling window analytics

Rolling mean / std / z-score

Rolling covariance / correlation

Fused rolling pipelines in Rust (planned)

Distribution tools

ECDF (empirical CDF)

Gaussian KDE

Quantile binning

Winsorization

Transforms

Robust scaling (Median + MAD)

diff / pct_change / cumsum / cummean

pandas Styler integration

demean_style(df, column)

zscore_style(df, column, threshold=…)

iqr_outlier_style(df, column)

corr_heatmap(df)

robust_scale_column(df, column)

📦 Installation (from source)
git clone https://github.com/<you>/bunker-stats.git
cd bunker-stats
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install maturin
maturin develop

🔍 Usage Examples
NumPy stats (Rust backend)
import numpy as np
import bunker_stats as bs

x = np.array([1.0, 2.0, 3.0, 10.0], dtype="float64")

print(bs.mean_np(x))      # 4.0
print(bs.std_np(x))       # 4.08248...
print(bs.zscore_np(x))    # [-0.73, -0.48, -0.24, 1.46]

pandas Styler
import pandas as pd
import bunker_stats as bs

df = pd.DataFrame({"sales": [10, 12, 15, 9, 8, 20]})

styled = bs.pandas.demean_style(df, "sales")
styled   # displays color-coded DataFrame in Jupyter

📊 Benchmark Results (v0.1)

All benchmarks are reproducible via:

python benchmarks/bench_bunker_stats.py
python benchmarks/test_advanced_ops.py


Hardware: Windows 10, Python 3.10, NumPy 1.x

✅ Correctness Checks

bunker-stats matches NumPy/pandas across:

mean, std, z-score

percentiles

IQR & Tukey fences

MAD

diff / pct_change / cumsum / cummean

ECDF

covariance, correlation

rolling covariance/correlation

KDE (integral ≈ 1.0)

EWMA

Welford one-pass stats

All tests pass with tight tolerances (1e-12 where appropriate).

⚡ Performance Summary
1D statistics (1,000,000 elements)
Operation	NumPy	bunker-stats
mean	1.51 ms	6.73 ms
std (ddof=1)	7.30 ms	16.27 ms
z-score	11.9 ms	35.8 ms

Interpretation:
NumPy is heavily optimized C with low overhead. For simple scalar ops, NumPy is faster — expected for a v0.1 Rust library accessed via FFI.

Rolling windows (1,000,000 elements, window=50)
Operation	pandas	bunker-stats
Rolling mean	34.62 ms	18.31 ms

🔥 bunker-stats rolling mean is ~1.9× faster than pandas.

This is where Rust shines: fused loops, zero Python overhead, no index machinery.

Covariance / Correlation (large vectors)
Operation	Size	pandas	bunker-stats
Covariance	100k	—	1.86 ms
Correlation	100k	—	7.63 ms

Cov/corr for individual vector pairs are very fast, often competitive with NumPy.

Correlation Matrix (100,000 × 10)
Operation	pandas	bunker-stats
corr matrix	34.0 ms	439.8 ms

Interpretation:
bunker-stats currently uses a straightforward Rust implementation (correct but not optimized). Future versions will incorporate column-wise precomputations + SIMD.

Advanced Ops
Operation	Input Size	Time
RobustScaler	100k	26.34 ms
Winsorization	100k	207.6 ms
Quantile binning (5 bins)	100k	735.8 ms
ECDF	10k	8.51 ms
KDE (Gaussian, 5k → 512 grid)	5k	54.44 ms
rolling_cov (window=50)	100k	120.49 ms
rolling_corr (window=50)	100k	322.33 ms
diff(1M)	1M	18.47 ms
pct_change(1M)	1M	28.98 ms
cumsum(1M)	1M	16.01 ms
cummean(1M)	1M	20.83 ms

All advanced ops validated against NumPy/pandas or pure Python equivalents.

🎯 What bunker-stats is (and isn’t)
bunker-stats is:

A Rust-backed analytics toolkit specialized for:

rolling statistics

outlier detection

robust scaling

distribution analysis

feature binning & KDE

pandas-friendly visualization

A numerically correct, well-tested foundation you can trust.

bunker-stats is not (yet):

A total replacement for NumPy’s C vectorized primitives

A drop-in for full pandas DataFrame operations

Optimized correlation-matrix engine (coming soon)

🧪 Testing

To run the full suite:

pytest -q        # if you add tests/ folder
python benchmarks/bench_bunker_stats.py
python benchmarks/test_advanced_ops.py

🛣️ Roadmap

SIMD-optimized rolling statistics

Optimized correlation matrix (BLAS-backed)

Fused rolling mean+std+zscore in one pass

Multi-column Styler helpers

NaN-robust implementations across all functions

Polars DataFrame integration

PyO3 async variants where appropriate

❤️ Contributing

PRs welcome — especially for vectorization, algorithmic improvements, and new statistical transforms.
