.

🚀 bunker-stats-rs

Ultra-fast Rust-powered statistics + time-series utilities for Python.
Designed for data scientists, quants, researchers, analysts, and ML engineers who need NumPy-compatible accuracy with massive speedups on rolling statistics, covariance/correlation, outlier detection, ECDF, KDE, and more.

Goal: A lightweight, zero-dependency, high-performance alternative to many NumPy / Pandas / SciPy statistical operations — with predictable performance on large arrays.

📦 Installation
pip install bunker-stats-rs

⚡️ Why bunker-stats?

Pure Rust kernels

No Python loops

No Pandas overhead

Predictable vectorized performance

Identical numerical results (within fp tolerance)

Minimal dependencies

Up to 1700× faster depending on the operation

Built for large 1D/2D NumPy arrays

🔥 Benchmark Summary

Benchmarks run on: Windows 10 • Intel i7 • Python 3.10 • NumPy 1.26 • Pandas 2.2
Dataset sizes: 1,000,000-element 1D arrays and 200,000×10 2D matrices

Below is a curated “top wins” summary:

Top Speedups (reference_time / bunker_time)
Group	Operation	Ref Backend	Ref Time (ms)	Bunker (ms)	Speedup	Allclose	Max Diff
rolling	rolling_zscore	python_ref	33934.42	19.49	×1741.47	True	4.12e-11
diff_cum_etc	cummean	python_ref	297.37	2.35	×126.72	True	0.0
rolling	ewma	numpy_ref	376.98	4.85	×77.79	True	0.0
diff_cum_etc	sign_mask	python_ref	14.62	0.60	×24.34	True	0.0
cov_corr	rolling_cov	pandas	157.25	14.06	×11.18	True	4.48e-14
rolling	rolling_mean	pandas	54.68	5.14	×10.63	True	7.99e-15
cov_corr	cov_pair	numpy	15.27	4.08	×3.74	True	3.03e-18
outliers	zscore_outliers	python_ref	16.03	4.60	×3.48	True	0.0
diff_cum_etc	quantile_bins_10	pandas	82.57	44.68	×1.85	True	0.0
scipy_compare	iqr_scipy	scipy	35.82	15.90	×2.25	True	0.0

Full benchmark results are available in /benchmarks.

🧩 Features
Basic Stats

mean / std / var (ddof=1)

percentiles

IQR, MAD

min-max scaling

robust scaling (median/MAD)

winsorizing

Rolling Windows

rolling mean

rolling std

rolling zscore (z of last element)

EWMA (exponential smoothing)

Diff / Cumulative Operations

diff

pct_change

cumsum

cummean

ECDF

quantile binning

sign masks

demean with sign mask

Covariance & Correlation

covariance (pair)

correlation (pair)

covariance matrix

correlation matrix

rolling covariance

rolling correlation

KDE (Kernel Density Estimate)

Fast Gaussian KDE

📌 Examples
import numpy as np
import bunker_stats_rs as bs

x = np.random.randn(1_000_000)

# Fast std
s = bs.std_np(x)

# Rolling mean
r = bs.rolling_mean_np(x, window=50)

# Covariance
cov = bs.cov_np(x, x * 2.0 + 1.0)

# ECDF
vals, cdf = bs.ecdf_np(x)

🧱 Design Goals

Be a surgical, ultra-fast replacement for statistical hot paths in Python workflows

Work directly with NumPy arrays (input/output stays NumPy)

Zero hidden state, deterministic execution

Predictable performance across large inputs

Low-level but ergonomic API

⚠️ Limitations (v0.1.0)

float64 only

1D and 2D arrays only

No nan* functions yet (nanmean, nanstd, nanpercentile)

Rolling windows do not skip NaNs

Percentile + KDE slower than NumPy/SciPy on small arrays

Not a drop-in replacement for pandas — focuses on raw NumPy data

These will improve in future releases.

🗺 Roadmap
v0.2 — NaN-Aware API

nanmean / nanstd / nanvar

nanpercentile

NaN-friendly rolling windows

v0.3 — 2D Rolling Stats

rolling mean/std/cov/corr for matrices

v0.4 — Parallelism

Optional Rayon parallel kernels for 50M+ elements

v0.5 — sklearn-like Transformers

Scaling transformers

Outlier detectors

Binning transformers

🧪 Running Benchmarks
cd benchmarks
python bench_all.py

📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.

🤝 Contributing

PRs welcome — especially for:

new statistical kernels

rolling ops

SciPy parity

tests + benchmarks

performance improvements

⭐️ Support

If this library speeds up your workflow, please ⭐ the repo!

