<h1 align="center">💥 bunker-stats</h1>

<p align="center">
A Rust powered statistical toolkit with a Python API and pandas Styler integration.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange.svg">
  <img src="https://img.shields.io/badge/binding-Python-blue.svg">
  <img src="https://img.shields.io/badge/status-v0.1-green.svg">
  <img src="https://img.shields.io/badge/build-maturin-red.svg">
</p>

---

## 🔧 Overview

**bunker-stats** is a hybrid Rust and Python library providing:

- Fast statistical primitives  
- Rolling window analytics  
- Distribution tools  
- pandas Styler visualizations  

Everything runs on Rust for speed and correctness.

---

## 🧭 Project Philosophy and Status

**v0.1 is an intentional early release.**

This library focuses on correctness, clean APIs, and solid statistical foundations.

### 🔮 Future Focus
- Performance tuning (SIMD, fused loops, BLAS ops)  
- Smarter rolling window engines  
- More visualization helpers  
- NaN safe variants  
- Multi column Rust kernels  
- Faster correlation matrix engine  

---

## 🚀 Features

### Core statistics (Rust)
- Mean, variance, standard deviation  
- Sample vs population versions  
- Z scores  
- MAD  
- Percentiles and quantiles  
- IQR and Tukey fences  
- Covariance, correlation  
- Welford one pass algorithms  
- EWMA  

### Rolling analytics
- Rolling mean, std, z score  
- Rolling covariance, correlation  
- Planned fused pipelines  

### Distribution tools
- ECDF  
- Gaussian KDE  
- Quantile binning  
- Winsorization  

### Transforms
- Robust scaling using Median and MAD  
- diff, pct_change, cumsum, cummean  

### pandas Styler
- `demean_style(df, column)`  
- `zscore_style(df, column, threshold=...)`  
- `iqr_outlier_style(df, column)`  
- `corr_heatmap(df)`  
- `robust_scale_column(df, column)`  

---

## 📦 Installation

```bash
git clone https://github.com/bunker-stats.git
cd bunker-stats

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install maturin
maturin develop
