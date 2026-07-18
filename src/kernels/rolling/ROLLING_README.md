# Rolling Window Statistics

Rolling (sliding-window) statistics for 1-D series and 2-D column-wise data, implemented as Rust kernels and exposed to Python through the `bunker_stats_rs` extension module. The module covers single-statistic reducers (mean, std, var, z-score, median), NaN-aware variants, pairwise statistics (covariance, correlation, beta, linear regression), and a fused multi-statistic kernel that computes several statistics in one pass with configurable alignment and NaN policy.

All functions take NumPy `float64` arrays and return NumPy `float64` arrays. Sample statistics use `ddof=1` (divide by `count - 1`) throughout.

```python
import numpy as np
import bunker_stats_rs as b

x = np.array([1., 2., 3., 4., 5.])
b.rolling_mean_np(x, 3)          # array([2., 3., 4.])
b.rolling_std_np(x, 3)           # array([1., 1., 1.])
```

---


## Using the Python facade

New code should reach these kernels through the `bunker_stats` facade, which
exposes clean names with keyword arguments. The raw `bunker_stats_rs` names
documented below remain available and stable; the facade adds ergonomics on
top of the same kernels:

```python
import bunker_stats as bs

bs.rolling_mean(x, window=20)              # strict: length n-window+1
bs.rolling_mean(x, window=20, skipna=True) # NaN-aware: length n, min_periods=1
bs.rolling_cov(x, y, window=50, skipna=True)
```

Where a statistic has strict and skip-NaN variants, the facade exposes ONE
name with a `skipna=` keyword; `skipna=True` dispatches to the skip-NaN kernel
documented below (the twin kernels stay separate in Rust, so there is no
branch inside the hot loop).

## Function summary

| Function | One-liner | Output shape |
|---|---|---|
| `rolling_mean_np(a, window)` | Strict rolling mean | `(n - window + 1,)` |
| `rolling_std_np(a, window)` | Strict rolling sample std | `(n - window + 1,)` |
| `rolling_var_np(a, window)` | Strict rolling sample variance | `(n - window + 1,)` |
| `rolling_mean_std_np(a, window)` | Mean and std in one pass | tuple of two `(n - window + 1,)` |
| `rolling_zscore_np(a, window)` | Z-score of each window's first element | `(n - window + 1,)` |
| `rolling_mean_nan_np(a, window)` | NaN-skipping mean, full-length output | `(n,)` |
| `rolling_std_nan_np(a, window)` | NaN-skipping std, full-length output | `(n,)` |
| `rolling_zscore_nan_np(a, window)` | NaN-skipping z-score of current element | `(n,)` |
| `rolling_mean_axis0_np(x, window)` | Column-wise rolling mean (2-D) | `(n_rows - window + 1, n_cols)` |
| `rolling_std_axis0_np(x, window)` | Column-wise rolling std (2-D) | `(n_rows - window + 1, n_cols)` |
| `rolling_mean_std_axis0_np(x, window)` | Column-wise mean and std in one pass | tuple of two 2-D arrays |
| `rolling_multi_np(x, window, ...)` | Fused multi-stat kernel (1-D), policy-driven | tuple of arrays, `(n - window + 1,)` or `(n,)` |
| `rolling_multi_axis0_np(x, window, ...)` | Fused multi-stat kernel (2-D, axis 0) | tuple of 2-D arrays |
| `rolling_cov_np(x, y, window)` | Strict rolling sample covariance | `(n - window + 1,)` |
| `rolling_corr_np(x, y, window)` | Strict rolling Pearson correlation | `(n - window + 1,)` |
| `rolling_cov_nan_np(x, y, window)` | Covariance, NaN when window has a missing pair | `(n - window + 1,)` |
| `rolling_corr_nan_np(x, y, window)` | Correlation, NaN when window has a missing pair | `(n - window + 1,)` |
| `rolling_cov_skipna(x, y, window)` | Alias of `rolling_cov_nan_np` | `(n - window + 1,)` |
| `rolling_corr_skipna(x, y, window)` | Alias of `rolling_corr_nan_np` | `(n - window + 1,)` |
| `rolling_beta_skipna(x, y, window)` | Rolling OLS slope of `y` on `x`, pair-skipping | `(n - window + 1,)` |
| `rolling_linreg_skipna(x, y, window)` | Rolling OLS slope and intercept, pair-skipping | tuple of two `(n - window + 1,)` |
| `rolling_median(x, window)` | Rolling median, full-length output | `(n,)` |

`rolling_autocorr`, `rolling_correlation`, and `rolling_autocorr_multi` also exist in the extension but belong to the time-series-analysis (`tsa`) module and are documented there.

## Output conventions

Two output conventions coexist in this module:

- **Trailing truncated** — output length `n - window + 1`. Element `k` of the output is the statistic of `x[k : k + window]`. There is no NaN warm-up; the output is simply shorter than the input. Used by the strict 1-D family, the axis-0 family, all pairwise functions, and `rolling_multi_*` with `alignment="trailing"`.
- **Full length** — output length `n`, aligned with the input. Positions that cannot be computed hold NaN. Used by the `*_nan_np` reducers, `rolling_median`, and `rolling_multi_*` with `alignment="centered"`.

To convert a truncated result to a pandas-style full-length series, prepend `window - 1` NaNs.

---

## Strict truncated reducers (1-D)

These are the fast path: a single sliding pass with compensated (Kahan) summation and a translation offset for numerical stability (see the stability section). "Strict" means NaNs are not skipped.

**NaN caveat for this family:** the sliding accumulator does not recover once a NaN has entered it. A single NaN in the input makes the window(s) containing it NaN *and every later window as well*, even after the NaN has left the window. If your data can contain NaN, use the `*_nan_np` variants or `rolling_multi_np` (which recomputes each window and does recover).

### `rolling_mean_np`

Rolling arithmetic mean over a trailing window.

```python
rolling_mean_np(a, window)
```

- **Parameters:** `a` — 1-D float64 array; `window` — window size (int ≥ 1).
- **Returns:** `(n - window + 1,)` array; element `k` is `mean(a[k : k + window])`.
- **NaN policy:** strict; a NaN poisons its window and all subsequent output (see caveat above).
- **Why it matters:** the basic smoother/trend estimator — moving averages of prices, sensor readings, request latencies.
- **Edge cases:** `window == 0` or `window > n` returns an empty array. `window == 1` returns a copy of the input. Empty input returns an empty array.

### `rolling_std_np`

Rolling sample standard deviation (`ddof=1`).

```python
rolling_std_np(a, window)
```

- **Parameters:** `a` — 1-D float64 array; `window` — window size.
- **Returns:** `(n - window + 1,)` array of sample standard deviations.
- **NaN policy:** strict, with the poisoning caveat above.
- **Why it matters:** rolling volatility estimation for returns, control limits for process monitoring.
- **Edge cases:** `window == 0` or `window > n` → empty array. `window == 1` → all NaN (sample std of one observation is undefined; matches pandas, not 0). Constant windows → exactly `0.0`. Accurate on large-offset data (e.g. values near `1e8`) thanks to the translation-offset accumulator.

### `rolling_var_np`

Rolling sample variance (`ddof=1`). Computed from the same single pass as `rolling_std_np`, then converted in a final linear pass.

```python
rolling_var_np(a, window)
```

- **Parameters/returns:** as `rolling_std_np`, values are variances.
- **NaN policy:** strict, poisoning caveat applies.
- **Why it matters:** variance targeting, risk budgeting, feature engineering where the squared scale is needed directly.
- **Edge cases:** identical to `rolling_std_np` (`window == 1` → NaN).

### `rolling_mean_std_np`

Rolling mean and sample std computed in one fused pass — cheaper than calling the two functions separately.

```python
rolling_mean_std_np(a, window)  # -> (means, stds)
```

- **Returns:** tuple `(means, stds)`, each `(n - window + 1,)`.
- **NaN policy / edge cases:** same as `rolling_mean_np` and `rolling_std_np`.
- **Why it matters:** the mean/std pair is the input to z-scoring, Bollinger-style bands, and standardization; fusing avoids a second pass over the data.

### `rolling_zscore_np`

For each window `a[k : k + window]`, the z-score of the window's **first** element: `(a[k] - mean) / std` where mean/std are of that window. Note the anchoring: output position `k` standardizes `a[k]` against the window that *starts* at `k` (a forward-looking window relative to the scored point). This differs from `rolling_zscore_nan_np`, which scores the newest element of a trailing window.

```python
rolling_zscore_np(a, window)
```

- **Returns:** `(n - window + 1,)` array.
- **NaN policy:** strict, poisoning caveat applies.
- **Special values:** windows with `std == 0` (constant window) yield `0.0`; non-finite std yields NaN.
- **Why it matters:** standardized anomaly scores relative to local context, e.g. spike detection in telemetry.
- **Edge cases:** `window == 0` or `window > n` → empty. `window == 1` → all NaN (std undefined).

---

## NaN-aware full-length reducers (1-D)

These return output the same length as the input, skip NaN values inside the window, and use the trailing window of the last `window` observations ending at each position `i` (fewer near the start). Their behavior matches `pandas.Series.rolling(window, min_periods=m)` with `m=1` for the mean and `m=2` for the std / z-score (verified element-for-element against pandas).

**Precision caveat for this family:** these kernels use plain (uncompensated, un-offset) running sums, so second moments can lose precision on data with a large common offset (values near `1e8` and beyond). Center or de-mean such data first, or use `rolling_multi_np(..., nan_policy="ignore")`, which is offset-stabilized.

### `rolling_mean_nan_np`

Rolling mean that skips NaN, computed whenever the window contains at least one valid value.

```python
rolling_mean_nan_np(a, window)
```

- **Returns:** `(n,)` array; position `i` is the mean of the valid values among the last `min(window, i + 1)` observations.
- **NaN policy:** skip; NaN output only where the window has zero valid values. Equivalent to pandas `rolling(window, min_periods=1).mean()`.
- **Why it matters:** gap-tolerant moving averages over data with missing observations (market holidays, sensor dropouts).
- **Edge cases:** `window == 0` or empty input → empty array. `window > n` → behaves as an expanding mean over the full prefix (output still length `n`).

### `rolling_std_nan_np`

Rolling sample std (`ddof=1`) that skips NaN, requiring at least two valid values.

```python
rolling_std_nan_np(a, window)
```

- **Returns:** `(n,)` array; NaN where fewer than 2 valid values are in the window. Equivalent to pandas `rolling(window, min_periods=2).std()`.
- **Why it matters:** volatility estimates that keep producing output through data gaps.
- **Edge cases:** as `rolling_mean_nan_np`; `window == 1` → all NaN. Precision caveat above applies on large-offset data.

### `rolling_zscore_nan_np`

Z-score of the **current** element `a[i]` against the NaN-skipping mean/std of its trailing window.

```python
rolling_zscore_nan_np(a, window)
```

- **Returns:** `(n,)` array.
- **NaN policy:** output is NaN where the input itself is NaN, where fewer than 2 valid values are available, or where the window std is not strictly positive and finite (so constant windows give NaN, not 0 — unlike `rolling_zscore_np`).
- **Why it matters:** online anomaly scoring on incomplete data, using only information up to time `i` (no lookahead).
- **Edge cases:** `window == 0` or empty input → empty; `window > n` → expanding behavior.

---

## Column-wise 2-D reducers (axis 0)

Strict trailing kernels applied independently to each column of a 2-D array, sharing the same numerics as the 1-D strict family (Kahan summation plus a translation offset; verified accurate at offsets of `1e8`). Optionally parallelized over columns when the crate is built with the `parallel` feature.

### `rolling_mean_axis0_np`, `rolling_std_axis0_np`, `rolling_mean_std_axis0_np`

```python
rolling_mean_axis0_np(x, window)          # -> (n_rows - window + 1, n_cols)
rolling_std_axis0_np(x, window)           # -> (n_rows - window + 1, n_cols)
rolling_mean_std_axis0_np(x, window)      # -> (means, stds), each 2-D
```

- **Parameters:** `x` — 2-D float64 array, **C-contiguous required** (a Fortran-ordered or sliced array raises `ValueError: array must be contiguous`); `window` — window size.
- **Returns:** 2-D arrays with `n_rows - window + 1` rows; column `j` of the output is the 1-D rolling statistic of column `j` of the input.
- **NaN policy:** strict per column, with the same poisoning caveat as the 1-D strict family — a NaN in one column poisons that column's remaining output but leaves other columns untouched (verified).
- **Why it matters:** rolling statistics across a panel — e.g. per-asset rolling volatility over a returns matrix of shape `(days, assets)` — without a Python loop over columns.
- **Edge cases:** `window == 0` or `window > n_rows` → shape `(0, n_cols)`. `window == 1` → std all NaN, mean equals input.

---

## Fused multi-statistic kernels

One pass, several statistics, explicit policies. These are the most configurable entry points and the recommended choice when data may contain NaN.

### `rolling_multi_np`

```python
rolling_multi_np(x, window, min_periods=None, alignment="trailing",
                 nan_policy="propagate", stats=None)
```

- **Parameters:**
  - `x` — 1-D float64 array.
  - `window` — window size (int ≥ 1; `0` raises `ValueError`).
  - `min_periods` — minimum valid (non-NaN) observations required per window; `None` means `window`. Must satisfy `1 <= min_periods <= window` or `ValueError` is raised.
  - `alignment` — `"trailing"` (output length `n - window + 1`, window `[k, k + window)`) or `"centered"` (output length `n`, window `[k - window//2, k + window//2 + 1)` clipped to the array).
  - `nan_policy` — `"propagate"` (any NaN in the window makes every requested statistic NaN, including `count`), `"ignore"` (skip NaNs; compute when valid count ≥ `min_periods`), or `"require_min_periods"` (identical to `"ignore"`; the name documents intent).
  - `stats` — list of statistic names from `"mean"`, `"std"`, `"var"`, `"count"`, `"min"`, `"max"`; defaults to `["mean"]`. Unknown names raise `ValueError`.
- **Returns:** a **tuple of arrays in the same order as `stats`**, one array per requested statistic.

```python
means, stds, counts = b.rolling_multi_np(x, 20, min_periods=10,
                                         nan_policy="ignore",
                                         stats=["mean", "std", "count"])
```

- **NaN policy details:** under `"propagate"` each window is evaluated independently, so — unlike the legacy strict functions — output *recovers* after a NaN leaves the window: `[1, nan, 3, 4, 5]` with `window=3` gives `(nan, nan, 4.0)` for the mean. Under `"ignore"`, `count` reports the number of valid values per window.
- **Centered alignment details:** output is full length `n` and aligned with the input; windows are truncated at both edges. With `min_periods=None`, truncated edge windows are still computed at their actual size; with an explicit `min_periods`, edges with fewer valid values than `min_periods` yield NaN. Because the window is expressed as `k ± window//2`, an **even** `window` spans `window + 1` points in the interior (e.g. `window=4` gives interior counts of 5); prefer odd windows for centered smoothing.
- **Why it matters:** computing mean, std and count in one pass halves (or better) the passes over large arrays, and it is the only 1-D entry point with explicit `min_periods` / alignment / NaN-policy control — e.g. pandas-like smoothing with `alignment="centered"`, or gap-tolerant volatility with `nan_policy="ignore", min_periods=10`.
- **Edge cases:** `window == 0` → `ValueError`. Trailing with `window > n` → tuple of empty arrays. `window == 1` under `"propagate"` returns `0.0` for std/var (a known quirk of the fused fast path; the legacy functions and pandas return NaN), while under `"ignore"` it returns NaN. Numerically stable on large-offset data in both policies (per-window translation offset; verified at `1e8`).

### `rolling_multi_axis0_np`

Column-wise (axis 0) version of `rolling_multi_np` for 2-D arrays.

```python
rolling_multi_axis0_np(x, window, min_periods=None, alignment="trailing",
                       nan_policy="propagate", stats=None)
```

- **Parameters:** as `rolling_multi_np`, with `x` a 2-D C-contiguous float64 array (`ValueError` otherwise).
- **Returns:** tuple of 2-D arrays in `stats` order; `(n_rows - window + 1, n_cols)` for trailing, `(n_rows, n_cols)` for centered.
- **Why it matters:** policy-driven rolling statistics over a whole panel at once.
- **Edge cases:** as the 1-D version, applied to the row count.

---

## Rolling pairwise statistics

All pairwise functions take two 1-D arrays of equal length (`ValueError` on mismatch) and produce trailing truncated output of length `n - window + 1`.

### `rolling_cov_np`

Strict rolling sample covariance (`ddof=1`) between `x` and `y`.

```python
rolling_cov_np(x, y, window)
```

- **Returns:** `(n - window + 1,)` array; element `k` is `cov(x[k : k + window], y[k : k + window])`.
- **NaN policy:** strict sliding accumulator — a single NaN in either input poisons its window and **all later windows**. Use `rolling_cov_nan_np` for NaN-bearing data.
- **Why it matters:** raw co-movement of two series — the numerator of hedge ratios and pairwise risk models.
- **Edge cases:** `window == 0` → `ValueError("window must be >= 1")`; `window > n` (including empty inputs) → `ValueError("window must be <= len(x)")`; length mismatch → `ValueError`. Offset-stabilized: exact on data with a common offset near `1e8` (verified).

### `rolling_corr_np`

Strict rolling Pearson correlation.

```python
rolling_corr_np(x, y, window)
```

- **Returns:** `(n - window + 1,)` array.
- **NaN policy:** strict, same poisoning caveat as `rolling_cov_np`.
- **Why it matters:** rolling co-movement on a scale-free `[-1, 1]` scale — regime detection, pair-trading signals.
- **Edge cases:** same errors as `rolling_cov_np`. Constant windows (zero variance) → NaN. **Not offset-stabilized:** unlike `rolling_cov_np`, this kernel uses raw accumulators and returns NaN or inaccurate values on large-offset data (observed at `1e8`); de-mean such data first.

### `rolling_cov_nan_np` / `rolling_cov_skipna`

Rolling sample covariance with per-window NaN accounting. `rolling_cov_skipna` is an exact alias (the preferred, clean name). A window contributes a value only when **every** pair in it is valid; any missing pair (NaN in either series at that position) makes that window's output NaN. This matches `pandas.Series.rolling(window).cov(other)` with the default `min_periods=window` — the truncated output equals the pandas result with its first `window - 1` warm-up NaNs dropped (verified).

```python
rolling_cov_nan_np(x, y, window)
rolling_cov_skipna(x, y, window)   # alias
```

- **Returns:** `(n - window + 1,)` array.
- **NaN policy:** NaN for any window containing a missing pair; output **recovers** as soon as the window is fully valid again (no poisoning).
- **Why it matters:** covariance estimates over real-world aligned series with occasional gaps, with pandas-compatible missing-data semantics.
- **Edge cases:** `window == 0` or `window > n` → empty array (no error, unlike the strict version). Not offset-stabilized: large common offsets (≈`1e8`) suffer catastrophic cancellation — center the data first.

### `rolling_corr_nan_np` / `rolling_corr_skipna`

Rolling Pearson correlation with the same full-window validity rule as `rolling_cov_nan_np`; `rolling_corr_skipna` is an exact alias. Matches pandas `rolling(window).corr(other)` default `min_periods=window` (verified, modulo the truncated-vs-full-length convention).

```python
rolling_corr_nan_np(x, y, window)
rolling_corr_skipna(x, y, window)  # alias
```

- **Returns:** `(n - window + 1,)` array.
- **NaN policy:** NaN when any pair in the window is missing, and NaN when either series has non-positive variance in the window (constant windows).
- **Why it matters:** gap-tolerant rolling correlation for co-movement monitoring.
- **Edge cases:** `window == 0` or `window > n` → empty array. Same large-offset caveat as `rolling_cov_nan_np`.

### `rolling_beta_skipna`

Rolling OLS slope of `y` regressed on `x`: `cov(x, y) / var(x)` per window, skipping missing pairs.

```python
rolling_beta_skipna(x, y, window)
```

- **Returns:** `(n - window + 1,)` array of slopes.
- **NaN policy:** **pairwise-complete** — unlike the cov/corr functions above, a window needs only **≥ 2 valid pairs** to produce a value (verified: a 3-window with 2 valid pairs yields a finite beta while `rolling_cov_nan_np` yields NaN there). NaN when fewer than 2 valid pairs remain or when `var(x)` is non-positive/non-finite in the window.
- **Why it matters:** rolling hedge ratios and market betas — e.g. beta of a stock to an index over a 60-day window, tolerant of a few missing days.
- **Edge cases:** `window == 0` or `window > n` → empty array. Not offset-stabilized; on large-offset data the variance guard typically converts the cancelled result to NaN rather than a wrong number, but centered inputs are still recommended.

### `rolling_linreg_skipna`

Rolling simple linear regression of `y` on `x`; returns both the slope and the intercept per window.

```python
rolling_linreg_skipna(x, y, window)  # -> (slopes, intercepts)
```

- **Returns:** tuple `(slope, intercept)`, each `(n - window + 1,)`. `slope` is identical to `rolling_beta_skipna`; `intercept = mean(y) - slope * mean(x)` over the window's valid pairs.
- **NaN policy:** pairwise-complete, ≥ 2 valid pairs required; both outputs NaN when `var(x)` is non-positive/non-finite.
- **Why it matters:** local linear trend fitting and pairs-trading spreads (`y - (a + b*x)`) in one call.
- **Edge cases:** `window == 0` or `window > n` → tuple of empty arrays.

---

## `rolling_median`

Rolling median with an order-statistics kernel (adaptive algorithm; exposed from the robust-statistics group but part of the rolling API surface).

```python
rolling_median(x, window)
```

- **Returns:** **full-length** `(n,)` array. The first `window - 1` positions are NaN (warm-up); position `i` (for `i >= window - 1`) is the median of `x[i - window + 1 : i + 1]`. Even windows return the average of the two middle order statistics (e.g. median of `[3, 2]` is `2.5`).
- **NaN policy:** propagate — any NaN in the window makes that window's output NaN; output recovers once the NaN leaves the window.
- **Why it matters:** an outlier-resistant alternative to the rolling mean — a single bad tick moves a rolling mean but leaves the rolling median unchanged; standard for despiking sensor and market data.
- **Edge cases:** `window == 0` or `window > n` → all-NaN array of length `n` (note: full-length, not empty — different from every other function here). `window == 1` → copy of the input. Empty input → empty output. As an order statistic it is inherently immune to large-offset cancellation.

---

## Numerical stability & NaN policy

### Translation-offset accumulators

Sliding-sum variance uses the identity `var = (Σx² - (Σx)²/w) / (w-1)`, which subtracts two quantities of order `magnitude²` whose difference is only of order `spread²`. On large-offset data (values near `1e8`), a naive implementation loses all significant digits and — after the usual `max(0, ·)` clamp — silently reports a **wrong zero** variance. Second-moment kernels in this module therefore accumulate on `x - offset`, where `offset` is a finite value near the data magnitude (variance, covariance and correlation are shift-invariant; means add the offset back). The strict 1-D family additionally uses Kahan-compensated sums.

Verified behavior on `1e8 + [1, 2, 3, 4, 5, 6]` with `window=3` (true rolling std is exactly 1):

| Kernel family | Offset-stabilized | Result at 1e8 offset |
|---|---|---|
| `rolling_mean/std/var/mean_std/zscore_np` | yes (offset + Kahan) | exact |
| `rolling_*_axis0_np` | yes | exact |
| `rolling_multi_np` / `rolling_multi_axis0_np` (both policies) | yes | exact |
| `rolling_cov_np` | yes | exact |
| `rolling_corr_np` | no | NaN / inaccurate |
| `rolling_mean/std/zscore_nan_np` | no | inaccurate second moments |
| `rolling_cov/corr_nan_np`, `*_skipna`, `beta`, `linreg` | no | inaccurate or NaN |
| `rolling_median` | n/a (order statistic) | exact |

For the non-stabilized families, subtract the series mean (or first value) before calling when your data rides on a large offset.

### NaN policy summary

Three distinct NaN behaviors exist; know which family you are calling:

1. **Strict with poisoning** (`rolling_mean/std/var/mean_std/zscore_np`, the `axis0` family, `rolling_cov_np`, `rolling_corr_np`): a single NaN makes its window NaN *and all later windows NaN*, because the sliding accumulator never recovers. These are the fastest kernels; only feed them clean data.
2. **Per-window propagate, with recovery** (`rolling_multi_*` under `"propagate"`, `rolling_median`, and the full-window rule of `rolling_cov/corr_nan_np` / `*_skipna`): a window containing a NaN (or missing pair) is NaN, but output resumes as soon as the window is clean again.
3. **Skip / min-periods** (`rolling_mean/std/zscore_nan_np`, `rolling_multi_*` under `"ignore"`/`"require_min_periods"`, `rolling_beta_skipna`, `rolling_linreg_skipna`): NaNs are dropped from the window; a value is produced when enough valid observations remain (1 for the NaN mean, 2 for NaN std/z-score and the beta/linreg pair kernels, `min_periods` for the fused kernels).

Pandas equivalences (verified numerically): `rolling_mean_nan_np` ≡ `rolling(w, min_periods=1).mean()`; `rolling_std_nan_np` ≡ `rolling(w, min_periods=2).std()`; `rolling_cov_nan_np`/`rolling_corr_nan_np` ≡ `rolling(w).cov()/.corr()` at the default `min_periods=window`, minus pandas' `window - 1` warm-up NaNs; `window == 1` std/var is NaN as in pandas (except the `rolling_multi_np` propagate quirk noted above).

---

## Edge cases

`n` is the input length (rows for 2-D). "empty" means a zero-length array (or `(0, n_cols)`).

| Situation | Strict 1-D (`*_np`) | `*_nan_np` reducers | axis0 family | `rolling_multi_*` | `cov/corr_np` | `cov/corr/beta/linreg` skipna | `rolling_median` |
|---|---|---|---|---|---|---|---|
| `window == 0` | empty | empty | empty | `ValueError` | `ValueError` | empty | all-NaN, length `n` |
| `window > n` | empty | expanding window, length `n` | empty | trailing: empty; centered: length `n` | `ValueError` | empty | all-NaN, length `n` |
| `window == 1` | mean ok; std/var/zscore NaN | mean ok; std/zscore NaN | mean ok; std NaN | propagate: std/var `0.0`; ignore: NaN | cov NaN (`ddof=1`); corr NaN | NaN (need ≥ 2 pairs) | copy of input |
| Empty input | empty | empty | empty | trailing: empty | `ValueError` (`window > n`) | empty | empty |
| Constant window | std/var `0.0`; zscore `0.0` | std `0.0`; zscore NaN | std `0.0` | std/var `0.0` | corr NaN | corr NaN; beta/linreg NaN | the constant value |
| All-NaN window | NaN (and poisons rest) | NaN | NaN (that column) | NaN | NaN (poisons rest) | NaN, recovers | NaN, recovers |
| Length mismatch (`x`, `y`) | — | — | — | — | `ValueError` | `ValueError` | — |
| Non-C-contiguous 2-D input | — | — | `ValueError` | `ValueError` | — | — | — |

Additional notes:

- Trailing output alignment: output index `k` ↔ input window `[k, k + window)`; the window *ends* at input index `k + window - 1`.
- `rolling_zscore_np` standardizes the window's **first** element; `rolling_zscore_nan_np` standardizes the **current** (last) element. They are not interchangeable.
- Centered alignment with an even `window` uses `window//2` points on each side and therefore spans `window + 1` points away from the edges; use odd windows for symmetric smoothing.
