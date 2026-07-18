# Robust statistics kernels (`src/kernels/robust`)

Rust kernels for outlier-resistant location and scale estimation, exposed to
Python through PyO3. This document covers every robust-family function
registered in `src/lib.rs`, with signatures and behavior verified against the
compiled extension module.

All examples use the raw extension module:

```python
import numpy as np
import bunker_stats_rs as bs
```

The `bunker_stats` Python facade re-exports these under unsuffixed names
(`bs.median` for `median_np`, `bs.mad` for `mad_np`, and so on); the `_np`
names remain available there as deprecated aliases. `robust_fit`,
`robust_score`, `rolling_median`, and `RobustStats` carry the same name in
both namespaces.

---

## Module layout

| File | Contents |
|---|---|
| `extended.rs` | Core estimators: median, MAD, trimmed mean/std, winsorized mean, IQR, biweight midvariance, Qn, Huber location, skip-NaN variants, fused median+MAD kernel |
| `fit.rs` | `robust_fit_slice` / `robust_score_slice` dispatch over location and scale policies |
| `policy.rs` | `LocationPolicy` (`median`, `trimmed_mean`, `huber`) and `ScalePolicy` (`mad`, `iqr`, `qn`) enums |
| `rolling.rs` | Rolling median with window-size-adaptive algorithm selection |
| `pyrobust.rs` | PyO3 bindings: `RobustStats` class, `robust_fit`, `robust_score`, `rolling_median` |
| `mad.rs`, `trimmed_mean.rs` | Superseded by `extended.rs`; not compiled into the module |

Design points that apply throughout:

- **Deterministic.** Same input always produces bit-identical output. There is
  no randomization anywhere in this module.
- **Selection, not sorting, where possible.** Median and MAD use
  `select_nth_unstable` (expected O(n)) rather than a full sort.
- **Total ordering.** All sorts and selections compare with `f64::total_cmp`,
  so they cannot panic on NaN or infinity. The release profile builds with
  `panic = "abort"`, so this is a hard-crash guard, not a style choice.
- **NaN propagates.** Every order-statistic reducer checks for NaN up front
  and returns NaN (NumPy semantics) instead of computing on an undefined
  ordering. Use the `_skipna` variants to drop NaNs instead.

---

## Function summary

| Function | Returns | Purpose |
|---|---|---|
| `median_np(a)` | float | Median (50% breakdown location) |
| `median_skipna_np(a)` | float | Median ignoring NaN/Inf |
| `mad_np(a)` | float | Raw median absolute deviation |
| `mad_skipna_np(a)` | float | Raw MAD ignoring NaN/Inf |
| `mad_std_np(a)` | float | MAD scaled to estimate the normal sigma |
| `trimmed_mean_np(a, proportion_to_cut)` | float | Mean after trimming both tails |
| `trimmed_mean_skipna_np(a, proportion_to_cut)` | float | Trimmed mean ignoring NaN/Inf |
| `trimmed_std_np(a, proportion_to_cut)` | float | Sample std (ddof=1) of the trimmed sample |
| `winsorized_mean_np(a, lower_percentile, upper_percentile)` | float | Mean after clamping the tails |
| `iqr_robust_np(a)` | float | Interquartile range (scalar width) |
| `biweight_midvariance_np(a, c=None)` | float | Tukey biweight midvariance (a robust variance) |
| `qn_scale_np(a)` | float | Rousseeuw–Croux Qn scale |
| `huber_location_np(a, k=None, max_iter=None)` | float | Huber M-estimator of location |
| `robust_scale_np(a, scale_factor)` | (ndarray, float, float) | Median/MAD standardization; returns `(scaled, median, mad)` |
| `robust_fit(x, ...)` | (float, float) | Policy-driven `(location, scale)` |
| `robust_score(x, ...)` | ndarray | Policy-driven robust z-scores |
| `rolling_median(x, window)` | ndarray | Rolling median, NaN warm-up prefix |
| `RobustStats(...)` | class | Reusable, pre-parsed fit/score configuration |

---

## Core estimators

### `median_np(a)`

Median via quickselect (`select_nth_unstable`), expected O(n). Even-length
inputs return the mean of the two central order statistics.

```python
bs.median_np(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))   # 3.0
```

**Why it matters.** The median has the maximum possible breakdown point (50%):
up to half the sample can be arbitrarily corrupted before the estimate becomes
unbounded. It is the default robust replacement for the mean.

- NaN input: returns NaN (`bs.median_np(np.array([1.0, np.nan, 3.0]))` is `nan`).
- Empty: NaN. Single element: that element. Constant input: the constant.

### `median_skipna_np(a)`

Filters out non-finite values (NaN **and** ±Inf), then takes the median of the
remainder. Returns NaN if nothing survives the filter.

```python
bs.median_skipna_np(np.array([1.0, np.nan, 3.0]))      # 2.0
```

### `mad_np(a)` / `mad_skipna_np(a)`

Raw (unscaled) median absolute deviation: `median(|x - median(x)|)`. Two
quickselect passes over a single reused buffer.

```python
bs.mad_np(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]))  # 1.5
```

**Why it matters.** The MAD shares the median's 50% breakdown point, making it
the standard robust replacement for the standard deviation. Its Gaussian
efficiency is low (about 37%), so for well-behaved data with few outliers,
`qn_scale_np` is a statistically more efficient robust alternative.

- NaN input: NaN (skipna variant filters non-finite values first).
- Empty: NaN. Single element: 0.0. Constant input: 0.0.

### `mad_std_np(a)`

MAD multiplied by the normal-consistency constant **1.482602218505602**
(`1 / Phi^-1(3/4)`), which makes it a consistent estimator of the standard
deviation for Gaussian data. Equivalent to
`scipy.stats.median_abs_deviation(a, scale="normal")`.

```python
bs.mad_std_np(x)               # == bs.mad_np(x) * 1.482602218505602
```

Use `mad_std_np` (or `mad_consistent=True` in the policy API) whenever the
result must be comparable to a standard deviation — e.g. for z-score
thresholds. Use raw `mad_np` when you only need a relative spread measure.

### `trimmed_mean_np(a, proportion_to_cut)` / `trimmed_mean_skipna_np(...)`

Sorts, removes `floor(n * proportion_to_cut)` elements from *each* tail, and
averages the remainder. Follows the `scipy.stats.trim_mean` convention.

```python
bs.trimmed_mean_np(np.arange(1.0, 11.0), 0.1)          # 5.5
```

- `proportion_to_cut` must be in `[0, 0.5)`; values outside (or non-finite)
  return NaN rather than raising.
- Because the cut count is floored, small samples may trim nothing:
  `n=6, proportion=0.1` keeps all six points (cut = floor(0.6) = 0).
- NaN input: NaN. Empty: NaN. Single element: that element.

**Why it matters.** Trimming *discards* tail data entirely, giving a
breakdown point equal to the trim proportion. Prefer trimming when tail
values are suspected to be plain errors; prefer winsorizing when the tails
carry signal but their magnitudes are untrustworthy.

### `trimmed_std_np(a, proportion_to_cut)`

Sample standard deviation (ddof=1) of the trimmed sample.

- Needs at least 2 elements after trimming; otherwise NaN.
- Same parameter validation and NaN propagation as `trimmed_mean_np`.

Note: this is the plain std of the retained values, not a Winsorized-variance
construction; it under-states the population sigma on Gaussian data and is
best used as a relative spread measure between similarly-trimmed samples.

### `winsorized_mean_np(a, lower_percentile, upper_percentile)`

Clamps values below/above the two percentile bounds to those bounds, then
averages. Percentiles are on the `[0, 100]` scale (e.g. `10.0`, `90.0`).

```python
bs.winsorized_mean_np(np.array([1., 2., 3., 4., 5., 100.]), 10.0, 90.0)
# 11.333... (100.0 was pulled down to the 90th-percentile bound)
```

- Returns NaN if `lower_percentile >= upper_percentile`, if the input is
  empty, or if any element is NaN.
- Unlike trimming, every observation still contributes — extreme values are
  capped, not removed.

### `iqr_robust_np(a)`

Interquartile range `Q3 - Q1` as a scalar, with quartiles computed by linear
interpolation (NumPy `linear` convention on positions `0.25*(n-1)` and
`0.75*(n-1)`).

- Requires `n >= 2`; returns NaN for empty or single-element input.
- NaN input: NaN. Constant input: 0.0.

**Why it matters.** The IQR has a 25% breakdown point — lower than the MAD's
50% — but it is a familiar, distribution-free spread measure and the basis of
Tukey's fences (see `iqr_outliers_np` in the quantile kernels). For the
`(Q1, Q3, width)` tuple, use `iqr_np` from the quantile family.

### `biweight_midvariance_np(a, c=None)`

Tukey biweight midvariance with tuning constant `c` (default **9.0**),
following the Beers–Flynn–Gebhardt definition as implemented by
`astropy.stats.biweight_midvariance` with `modify_sample_size=False`:

```
u_i  = (x_i - M) / (c * MAD)
zeta² = n * Σ_{|u|<1} (x_i - M)² (1 - u_i²)⁴  /  [ Σ_{|u|<1} (1 - u_i²)(1 - 5u_i²) ]²
```

where `M` is the median and the sums run over points with `|u| < 1`.

```python
bs.biweight_midvariance_np(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
# 2.297063991357617 (pinned against the astropy reference)
```

**Why it matters.** It returns a *variance* (square it root for a scale) that
smoothly down-weights points far from the median and completely ignores
points beyond `c` MADs. At the default `c=9` it is both highly robust and
much more efficient than MAD² on Gaussian data.

- Returns NaN when `n < 2`, when any element is NaN, and when `MAD == 0`
  (constant input) — the weight function is undefined without a spread.

### `qn_scale_np(a)`

Rousseeuw–Croux Qn scale estimator: the first quartile of all `n(n-1)/2`
pairwise absolute differences, scaled by the asymptotic normal-consistency
constant **2.2219**. The quartile is picked with quickselect at index
`num_pairs / 4`. For `n == 2` the single pairwise difference is scaled by
0.8224 instead. No finite-sample correction factor is applied.

```python
bs.qn_scale_np(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]))
```

**Why it matters — Qn vs MAD.** Both have a 50% breakdown point, but Qn has
about 82% Gaussian efficiency versus the MAD's 37%, and it needs no location
estimate, which makes it markedly better behaved on *asymmetric*
distributions (the MAD implicitly assumes symmetry around the median).
The cost is O(n²) time and memory for the pairwise differences — prefer
MAD-based scales for very large arrays.

- Returns NaN when `n < 2` or when any element is NaN.
- Constant input: 0.0.

### `huber_location_np(a, k=None, max_iter=None)`

Huber M-estimator of location via iteratively reweighted updates. Defaults:
`k = 1.345` (95% Gaussian efficiency), `max_iter = 30`. The scale used inside
the iteration is `MAD * 1.4826`; iteration stops when the step falls below
`1e-6 * scale`.

```python
bs.huber_location_np(x)                 # defaults
bs.huber_location_np(x, 1.5, 100)       # custom k, max_iter
```

**Why it matters.** The Huber estimator interpolates between the mean
(efficient, non-robust) and the median (robust, less efficient): residuals
within `k` scale units contribute linearly, larger ones are capped. Smaller
`k` is more robust; larger `k` is more efficient.

- Empty: NaN. NaN input: NaN (the internal median/MAD propagate it).
- If `MAD == 0` (constant or near-constant data), returns the median
  directly without iterating.

### `robust_scale_np(a, scale_factor)`

Median/MAD standardization of a whole array. Returns a 3-tuple
`(scaled_array, median, mad)` where `mad` is the **raw** MAD and

```
scaled[i] = (x[i] - median) / (mad * scale_factor)
```

Pass `scale_factor = 1.4826` to standardize against a consistent sigma
estimate. If `mad == 0` the denominator is replaced by `1e-12` (so a constant
array scales to exact zeros rather than NaN).

```python
scaled, med, mad = bs.robust_scale_np(x, 1.4826)
```

- Empty: `(array([]), nan, nan)`.
- Any NaN in the input: `(all-NaN array, nan, nan)`.

---

## Policy-driven API

### `robust_fit(x, location="median", scale="mad", c=1.345, trim=0.1, max_iter=50, tol=1e-6, mad_consistent=True)`

Computes `(location, scale)` in one call and returns the **tuple of two
floats**.

- `location`: `"median"`, `"trimmed_mean"` (uses `trim`), or `"huber"`
  (uses `c`, `max_iter`, `tol`, and the configured scale internally).
- `scale`: `"mad"` (uses `mad_consistent`), `"iqr"`, or `"qn"`.
- Invalid policy strings raise `ValueError`. Everything else degrades to NaN.

The default `median` + `mad` pair runs through a fused kernel that computes
both statistics from one buffer.

```python
loc, scale = bs.robust_fit(np.array([1., 2., 3., 4., 5., 100.]))
# (3.5, 2.223903327758403)   # median, MAD * 1.482602218505602
```

- Empty input: `(nan, nan)`. NaN input: `(nan, nan)`.
- Constant input: `(constant, 0.0)`.

### `robust_score(x, ...)` (same keyword arguments)

Robust z-scores `(x - location) / scale` as an ndarray of the input length.

```python
scores = bs.robust_score(data)
outliers = data[np.abs(scores) > 3]
```

- If the scale is zero or non-finite (constant or NaN input), every score is
  NaN — deliberately, so a degenerate scale cannot silently pass as "no
  outliers".
- Edge-case quirk: for an **empty** input the function returns a length-1
  array `[nan]`, not an empty array.

### `rolling_median(x, window)`

Rolling median with stride-1 windows. Positions before the first full window
are NaN, so the output always has the input's length.

```python
bs.rolling_median(np.array([1., 2., 3., 4., 5., 100.]), 3)
# [nan, nan, 2., 3., 4., 5.]
```

Windows of size ≤ 64 reuse a single fixed buffer (cache-friendly fast path);
larger windows recompute per window.

- `window == 0` or `window > len(x)`: all-NaN output (no error).
- A NaN anywhere in a window makes that window's output NaN (median
  semantics), so a single NaN poisons `window` consecutive outputs.

### `RobustStats` class

Same policies as `robust_fit`/`robust_score`, but the configuration strings
are parsed and validated once at construction and stored as enums — there is
no string handling in `fit`/`score`. Prefer this form when fitting many
arrays with the same configuration.

```python
rs = bs.RobustStats(location="median", scale="mad",
                    c=1.345, trim=0.1, max_iter=50, tol=1e-6,
                    mad_consistent=True)
loc, scale = rs.fit(x)      # tuple of floats
scores = rs.score(x)        # ndarray
```

Invalid `location`/`scale` strings raise `ValueError` at construction time.

---

## NaN policy

Two explicit modes, chosen by function name:

1. **Propagate (default).** `median_np`, `mad_np`, `mad_std_np`,
   `trimmed_mean_np`, `trimmed_std_np`, `winsorized_mean_np`,
   `iqr_robust_np`, `biweight_midvariance_np`, `qn_scale_np`,
   `huber_location_np`, `robust_scale_np`, `robust_fit`, `robust_score`, and
   `rolling_median` all return NaN (or NaN-filled outputs) when the input
   contains NaN. Verified example:

   ```python
   bs.median_np(np.array([1.0, np.nan, 3.0]))   # nan
   bs.mad_np(np.array([1.0, np.nan, 3.0]))      # nan
   ```

2. **Skip.** `median_skipna_np`, `mad_skipna_np`, `trimmed_mean_skipna_np`
   (and `iqr_skipna_np` in the quantile family) drop **non-finite** values
   (NaN and ±Inf) before estimating. All-non-finite input yields NaN.

No robust function raises on NaN, and none can abort the interpreter: all
comparisons use `f64::total_cmp`, and property tests fuzz every reducer with
arbitrary bit patterns (including NaN and ±Inf) to enforce the no-panic
guarantee. The only exceptions raised at the Python boundary are
`ValueError` for invalid policy strings and the usual conversion errors for
non-contiguous or wrongly-typed arrays.

±Inf is treated as an ordinary, ordered value (it is only excluded by the
`_skipna` filters): `median_np([1, 2, 3, inf, inf])` is `3.0`.

---

## Edge cases at a glance

| Input | median | mad | trimmed_mean | iqr_robust | qn_scale | biweight | robust_fit | robust_score |
|---|---|---|---|---|---|---|---|---|
| empty | NaN | NaN | NaN | NaN | NaN | NaN | (NaN, NaN) | `[nan]` (length 1) |
| single element | value | 0.0 | value | NaN | NaN | NaN | (value, 0.0) | all NaN (scale 0) |
| constant | value | 0.0 | value | 0.0 | 0.0 | NaN | (value, 0.0) | all NaN (scale 0) |
| contains NaN | NaN | NaN | NaN | NaN | NaN | NaN | (NaN, NaN) | all NaN |
| all NaN + skipna | NaN | NaN | NaN | NaN | — | — | — | — |

Additional validation behavior:

- `trimmed_mean_np` / `trimmed_std_np`: `proportion_to_cut` outside
  `[0, 0.5)` (or non-finite) returns NaN, never raises.
- `winsorized_mean_np`: `lower_percentile >= upper_percentile` returns NaN.
- `trimmed_std_np`: fewer than 2 elements after trimming returns NaN.

---

## Reference values used by the test suite

- MAD normal-consistency constant: `1.482602218505602`.
- Qn asymptotic constant: `2.2219` (with `0.8224` for the n=2 case).
- Biweight midvariance of `[1, 2, 3, 4, 5]` at `c=9`:
  `2.297063991357617`; of `[1, 2, 3, 4, 5, 100]`: `2.85036601168922`
  (both pinned against the astropy implementation).
- Trimmed mean follows `scipy.stats.trim_mean` (floor-based per-tail cut).

See `test_robust_ver2_9.py` (unit) and `tests/test_robust_stats.py`
(integration) for the executable versions of the statements above, plus
property-based tests for translation/scale equivariance and NaN propagation.
