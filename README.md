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

| Function               | Bunker-stats syntax                                      | NumPy equivalent                                                | pandas equivalent                                               | Unique feature in `bunker-stats`                                                                                 |
|------------------------|----------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `mean`                 | `bs.mean(x)`                                            | `np.mean(x)`                                                    | `s.mean()`                                                      | 1D mean helper; always treats input as 1D numeric, thin Rust-backed wrapper.                                     |
| `mean_skipna`          | `bs.mean_skipna(x)`                                     | `np.nanmean(x)` / manual mask                                   | `s.mean(skipna=True)`                                           | NaN-aware mean with explicit “skipna” semantics, matching pandas mental model.                                   |
| `var`                  | `bs.var(x)`                                             | `np.var(x, ddof=1)`                                             | `s.var(ddof=1)`                                                 | 1D **sample** variance (`ddof=1`) by default; matches stats textbooks.                                           |
| `var_skipna`           | `bs.var_skipna(x)`                                      | `np.nanvar(x, ddof=1)` / mask                                   | `s.var(skipna=True, ddof=1)`                                    | NaN-aware sample variance in one call.                                                                           |
| `std`                  | `bs.std(x)`                                             | `np.std(x, ddof=1)`                                             | `s.std(ddof=1)`                                                 | 1D sample std with fixed `ddof=1`, consistent with `var`.                                                        |
| `std_skipna`           | `bs.std_skipna(x)`                                      | `np.nanstd(x, ddof=1)` / mask                                   | `s.std(skipna=True, ddof=1)`                                    | NaN-aware sample std; avoids writing masks every time.                                                           |
| `percentile`           | `bs.percentile(x, q=0.95)`                              | `np.quantile(x, 0.95)` / `np.percentile`                        | `np.quantile(s, 0.95)`                                          | Clean 1D percentile with your interpolation; integrated with other robust stats.                                 |
| `mad`                  | `bs.mad(x)`                                             | manual median/MAD                                               | custom or `s.mad()` (mean abs dev, not median)                  | True median absolute deviation used by `robust_scale`.                                                           |
| `iqr`                  | `q1, q3, iqr = bs.iqr(x)`                               | `scipy.stats.iqr(x, rng=(25,75))`                               | `s.quantile([0.25, 0.75])`                                      | Returns `(q1, q3, iqr)` in one go; no juggling multiple calls / indices.                                         |
| `mean_axis`            | `bs.mean_axis(X, axis=0, skipna=False)`                 | `np.mean(X, axis=0)`                                            | `df.mean(axis=0, skipna=...)`                                   | Axis-wise mean for 1D/2D arrays with optional `skipna`.                                                          |
| `var_axis`             | `bs.var_axis(X, axis=1, skipna=True)`                   | `np.var(X, axis=1, ddof=1)` (no native skipna)                  | `df.var(axis=1, skipna=...)`                                    | Axis-wise sample variance with built-in NaN handling.                                                            |
| `std_axis`             | `bs.std_axis(X, axis=1, skipna=True)`                   | `np.std(X, axis=1, ddof=1)` (no native skipna)                  | `df.std(axis=1, skipna=...)`                                    | Axis-wise sample std + `skipna`; aligns pandas mental model with NumPy arrays.                                   |
| `mean_last_axis`*      | `bs.mean_last_axis(X)` *(if exposed)*                  | `np.mean(X, axis=-1)`                                           | `df.to_numpy().mean(axis=-1)`                                   | N-D mean over last axis, consistent with your N-D rolling API.                                                   |
| `rolling_mean_last_axis` | `bs.rolling_mean_last_axis(X, window=3)`             | manual reshape + loop / `np.apply_along_axis`                  | no built-in; need groupby+apply / custom logic                  | Shape-preserving N-D rolling mean over **last axis** (e.g. `(batch, feat, time)`).                               |
| `rolling_std_last_axis`  | `bs.rolling_std_last_axis(X, window=3)`              | same as above                                                  | same                                                            | N-D rolling std over last axis; perfect for batched time-series / ML tensors.                                    |
| `rolling_mean`         | `bs.rolling_mean(x, window=5)`                          | manual loop or `np.convolve` trick                              | `s.rolling(5).mean()`                                           | Fast 1D rolling mean (truncated length) with no index overhead.                                                  |
| `rolling_std`          | `bs.rolling_std(x, window=5)`                           | manual loop                                                     | `s.rolling(5).std()`                                            | 1D rolling std at Rust speed, sample variance convention.                                                        |
| `rolling_zscore`       | `bs.rolling_zscore(x, window=20)`                       | manual window loop                                              | `s.rolling(20).apply(custom)`                                  | Rolling z-score in a single function; avoids `apply`/UDF overhead.                                              |
| `ewma`                 | `bs.ewma(x, alpha=0.1)`                                 | manual recurrence                                               | `s.ewm(alpha=0.1).mean()`                                       | Minimal EWMA for pure numeric arrays, no pandas object overhead.                                                 |
| `df_rolling_mean`      | `bs.df_rolling_mean(df, window=5)`                      | `np.convolve` per column                                       | `df.rolling(5).mean()`                                          | DataFrame in / out, but columns powered by Rust rolling mean.                                                    |
| `df_rolling_std`       | `bs.df_rolling_std(df, window=5)`                       | manual per-column                                               | `df.rolling(5).std()`                                           | Same for std; uses your rolling core but preserves pandas index.                                                 |
| `df_ewma`              | `bs.df_ewma(df, alpha=0.1)`                             | manual per-column EWMA                                          | `df.ewm(alpha=0.1).mean()`                                      | Per-column EWMA with Rust engine, lighter than full pandas EWM machinery.                                        |
| `col_mean`             | `bs.col_mean(df, skipna=True)`                          | `np.mean(df.to_numpy(), axis=0)`                                | `df.mean(axis=0, skipna=True)`                                  | Column-wise mean; internally uses `mean_axis` + `skipna`, returns labeled Series.                                |
| `row_mean`             | `bs.row_mean(df, skipna=True)`                          | `np.mean(df.to_numpy(), axis=1)`                                | `df.mean(axis=1, skipna=True)`                                  | Row-wise mean with Rust numeric core + pandas index.                                                             |
| `cov_df`               | `bs.cov_df(df)`                                         | `np.cov(df.to_numpy().T, ddof=1)`                               | `df.cov()`                                                      | Full covariance matrix via Rust `cov_matrix`, but returned as a DataFrame.                                       |
| `corr_df`              | `bs.corr_df(df)`                                        | `np.corrcoef(df.to_numpy().T)`                                  | `df.corr()`                                                     | Correlation matrix backed by your Rust correlation engine.                                                       |
| `rolling_mean_series`  | `bs.rolling_mean_series(s, window=10)`                  | manual 1D loop                                                  | `s.rolling(10).mean()`                                          | Series-in / Series-out convenience wrapper around Rust rolling mean.                                             |
| `rolling_std_series`   | `bs.rolling_std_series(s, window=10)`                   | manual 1D loop                                                  | `s.rolling(10).std()`                                           | Same for std; keeps index alignment, uses Rust core.                                                             |
| `iqr_outliers`         | `bs.iqr_outliers(x, k=1.5)`                              | `iqr = scipy.stats.iqr(x); mask = ...`                          | quantiles + boolean mask                                        | Returns a boolean outlier mask in one call using IQR rule.                                                       |
| `zscore_outliers`      | `bs.zscore_outliers(x, threshold=3.0)`                  | `(np.abs((x-x.mean())/x.std()) > 3)`                            | same logic on `Series`                                          | One-liner z-score outlier mask; integrates with your `mean`/`std` semantics.                                     |
| `minmax_scale`         | `scaled, mn, mx = bs.minmax_scale(x)`                   | manual `(x-mn)/(mx-mn)`                                         | use `MinMaxScaler` from sklearn                                | Returns both **scaled data** and the `(min, max)` used (for inverse-transform/reuse).                            |
| `robust_scale`         | `scaled, med, mad = bs.robust_scale(x, scale_factor)`   | manual MAD calculation                                          | `RobustScaler` or custom                                        | All-in-one robust scaling with returned `(median, MAD)`; pairs with your `mad`.                                  |
| `winsorize`            | `bs.winsorize(x, lower_q=0.05, upper_q=0.95)`           | `scipy.stats.mstats.winsorize(x, limits=...)`                   | custom quantile clipping                                        | 1D winsorization in Rust, single call returning a full adjusted array.                                           |
| `diff`                 | `bs.diff(x, periods=1)`                                  | `np.diff(x, n=1)` (shorter) / manual padding                    | `s.diff(periods=1)`                                             | Full-length diff with NaNs where necessary; supports negative `periods`.                                         |
| `pct_change`           | `bs.pct_change(x, periods=1)`                            | manual `(x[i]-x[i-p]) / x[i-p]`                                 | `s.pct_change(periods=1)`                                      | Includes divide-by-zero → NaN handling; symmetric for positive/negative lags.                                    |
| `cumsum`               | `bs.cumsum(x)`                                           | `np.cumsum(x)`                                                  | `s.cumsum()`                                                    | Rust implementation; value is performance on large 1D arrays.                                                    |
| `cummean`              | `bs.cummean(x)`                                          | `np.cumsum(x)/np.arange(1,len(x)+1)`                            | `s.expanding().mean()`                                          | Streaming cumulative mean without constructing expanding windows.                                                |
| `ecdf`                 | `vals, probs = bs.ecdf(x)`                               | manual sort + rank                                              | custom `rank`/`value_counts`                                    | Returns **sorted values + CDF** in one go; perfect for ECDF plots.                                               |
| `quantile_bins`        | `bins = bs.quantile_bins(x, n_bins=10)`                  | manual rank + binning                                           | `pd.qcut(x, q=10)` (Categorical)                               | Returns plain integer bin labels `0..n_bins-1` as a NumPy array (ML-friendly).                                   |
| `sign_mask`            | `mask = bs.sign_mask(x)`                                 | `np.sign(x).astype(np.int8)`                                    | `(s > 0) - (s < 0)`                                             | Encodes sign into `{-1, 0, 1}`; useful for discrete signal features.                                             |
| `demean_with_signs`    | `demeaned, signs = bs.demean_with_signs(x)`              | `(x - x.mean(), np.sign(x - x.mean()))`                         | custom                                                        | Returns **both** demeaned data and sign mask in one pass.                                                        |
| `cov`                  | `bs.cov(x, y)`                                          | `np.cov(x, y, ddof=1)[0,1]`                                     | `s1.cov(s2)`                                                    | 1D sample covariance as a simple scalar function.                                                                 |
| `corr`                 | `bs.corr(x, y)`                                         | `np.corrcoef(x, y)[0,1]`                                        | `s1.corr(s2)`                                                   | 1D Pearson correlation using your var/std core.                                                                   |
| `cov_skipna`           | `bs.cov_skipna(x, y)`                                   | manual pairwise dropna + `np.cov`                               | `s1.cov(s2)` with aligned/dropna                               | Pairwise NaN dropping built in for 1D covariance.                                                                 |
| `corr_skipna`          | `bs.corr_skipna(x, y)`                                  | manual pairwise dropna + `np.corrcoef`                          | `s1.corr(s2)` with dropna                                      | Same but for correlation; hides the messy mask-bookkeeping.                                                      |
| `cov_matrix`           | `bs.cov_matrix(X)`                                      | `np.cov(X, rowvar=False, ddof=1)`                               | `df.cov()`                                                      | Symmetric covariance matrix with Rust loops; tuned for tabular X.                                                |
| `corr_matrix`          | `bs.corr_matrix(X)`                                     | `np.corrcoef(X, rowvar=False)`                                  | `df.corr()`                                                     | Correlation matrix built on your cov/std stack; consistent behaviour across code paths.                          |
| `rolling_cov`          | `bs.rolling_cov(x, y, window=50)`                       | manual sliding window + `np.cov`                                | `df['x'].rolling(50).cov(df['y'])`                             | Rolling 1D covariance without pandas overhead; good for streaming stats.                                         |
| `rolling_corr`         | `bs.rolling_corr(x, y, window=50)`                      | manual sliding window + `np.corrcoef`                           | `df['x'].rolling(50).corr(df['y'])`                            | Rolling 1D correlation in one Rust call; no custom loop needed in Python.                                        |
| `kde_gaussian`         | `grid, dens = bs.kde_gaussian(x, n_points=256)`          | `scipy.stats.gaussian_kde(x)` + evaluation                      | no direct builtin (need SciPy)                                  | Lightweight 1D Gaussian KDE; returns `(grid, density)` using a simple bandwidth rule by default.                 |


## 📦 Installation

```bash
git clone https://github.com/bunker-stats.git
cd bunker-stats

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install maturin
maturin develop
