# Quantile kernels (`src/kernels/quantile`)

Rust kernels for order statistics: percentiles, interquartile range,
winsorizing, empirical CDFs, quantile binning, and fence-based outlier masks.
Exposed to Python through PyO3. This document covers every quantile-family
function registered in `src/lib.rs`, with signatures and behavior verified
against the compiled extension module.

All examples use the raw extension module:

```python
import numpy as np
import bunker_stats_rs as bs
```

The `bunker_stats` Python facade re-exports these under unsuffixed names
(`bs.percentile`, `bs.winsorize`, `bs.ecdf`, ...), keeping the `_np` names as
deprecated aliases. One facade caveat: the facade's `iqr` is the **scalar
width** (backed by `iqr_robust_np`), while the raw `iqr_np` documented here
returns the `(q1, q3, width)` tuple. The facade's deprecated `iqr_np` alias
preserves the tuple.

---

## Module layout

| File | Contents |
|---|---|
| `percentile.rs` | `percentile_slice`: quickselect-based percentile with linear interpolation |
| `iqr.rs` | `iqr_slice` / `iqr_from_sorted`: quartiles and IQR |
| `winsor.rs` | `winsor_bounds` / `winsorize_vec`: percentile-bound clamping |
| `select.rs` | `select_nth_f64`: k-th order statistic via `select_nth_unstable_by(total_cmp)` |

The remaining bindings in this family (`iqr_outliers_np`, `zscore_outliers_np`,
`winsorize_clip_np`, `quantile_bins_np`, `ecdf_np`) are thin kernels defined
directly in `src/lib.rs` on top of the same primitives.

Design points that apply throughout:

- **Deterministic.** No randomization; identical input gives identical output.
- **Selection over sorting where possible.** `percentile_np` uses quickselect
  (expected O(n)); IQR and the rank-based helpers sort once.
- **Total ordering.** All comparisons use `f64::total_cmp`, which is a total
  order over `f64` (NaN sorts last). Nothing here can panic on NaN or ±Inf —
  relevant because the release profile builds with `panic = "abort"`.
- **NaN propagates, degeneracy returns NaN/empty/False.** No function in this
  family raises for bad data; the only Python-level exceptions are the usual
  conversion errors for non-contiguous or wrongly-typed arrays.

---

## Function summary

| Function | Returns | Purpose |
|---|---|---|
| `percentile_np(a, q)` | float | Percentile, `q` in `[0, 100]`, linear interpolation |
| `iqr_np(a)` | (float, float, float) | `(q1, q3, q3 - q1)` |
| `iqr_width_np(a)` | float | Scalar IQR width |
| `iqr_skipna_np(a)` | float | Scalar IQR width, ignoring non-finite values |
| `iqr_outliers_np(a, k)` | bool ndarray | Tukey-fence outlier mask |
| `winsorize_np(a, lower_q, upper_q)` | ndarray | Clamp tails at percentile bounds |
| `winsorize_clip_np(a, lower, upper)` | ndarray | Clamp at explicit value bounds |
| `quantile_bins_np(a, n_bins)` | int64 ndarray | Rank-based bin label per element |
| `ecdf_np(a)` | (ndarray, ndarray) | `(sorted_values, cdf)` empirical CDF |
| `zscore_outliers_np(a, threshold)` | bool ndarray | Mean/std z-score outlier mask |

---

## Per-function reference

### `percentile_np(a, q)`

Percentile of `a` at `q`, where **`q` is on the `[0, 100]` scale** (50.0 is
the median). Matches NumPy's default `linear` interpolation: the value at
fractional position `q/100 * (n - 1)` in sorted order. Implemented with
quickselect (expected O(n)) rather than a full sort.

```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
bs.percentile_np(x, 50.0)    # 3.5
bs.percentile_np(x, 0.5)     # 1.025 -- this is the 0.5th percentile, not the median
```

**Why it matters.** Percentiles are the building block for every other
function in this family. Note the scale carefully: passing a quantile
fraction like `0.5` silently computes the 0.5th percentile.

- `q` outside `[0, 100]` is clamped (150 behaves as 100, -5 as 0); `q = NaN`
  returns NaN.
- NaN in input: NaN. Empty: NaN. Single element: that element for any `q`.

### `iqr_np(a)`

Returns the 3-tuple `(q1, q3, iqr)` with quartiles at interpolated positions
`0.25*(n-1)` and `0.75*(n-1)` of the sorted data (NumPy `linear` convention).

```python
bs.iqr_np(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]))
# (2.25, 4.75, 2.5)
```

- Empty: `(nan, nan, nan)`. Single element: `(x, x, 0.0)`.
- NaN in input: `(nan, nan, nan)`.

### `iqr_width_np(a)`

Scalar convenience wrapper: the `q3 - q1` width from the same computation as
`iqr_np`, with NaN returned if any component is NaN.

```python
bs.iqr_width_np(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]))   # 2.5
```

**Why it matters.** The IQR is a distribution-free spread measure with a 25%
breakdown point: the middle 50% of the data determines it entirely, so a
quarter of the sample can be corrupted before it breaks. It is less robust
than the MAD (50% breakdown, see the robust kernels) but more familiar, and
it is the basis for Tukey's fences below.

- Empty: NaN. Single element: `0.0` (it forwards the tuple path's
  `(x, x, 0.0)`). Note the family split here: `iqr_skipna_np` and the robust
  module's `iqr_robust_np` instead return NaN for `n < 2`, treating spread as
  undefined for a single point.

### `iqr_skipna_np(a)`

Scalar IQR width after dropping **non-finite** values (NaN and ±Inf).
Requires at least 2 finite values; otherwise NaN.

```python
bs.iqr_skipna_np(np.array([1.0, np.nan, 3.0]))   # 1.0  (IQR of [1, 3])
```

### `iqr_outliers_np(a, k)`

Boolean mask of Tukey-fence outliers: `True` where
`x < q1 - k*iqr` or `x > q3 + k*iqr`. Conventional `k` values: `1.5`
("outliers"), `3.0` ("far out").

```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
bs.iqr_outliers_np(x, 1.5)
# [False, False, False, False, False, True]
```

**Why it matters.** The fences depend only on the quartiles, so moderate
contamination cannot drag the thresholds toward the outliers the way a
mean/std rule can (compare `zscore_outliers_np` below).

- NaN in input makes the IQR NaN, and the function then returns an
  **all-False** mask of the input length — it never guesses. Filter NaNs
  first if you want detection on the finite subset.
- Empty input: empty mask.

### `winsorize_np(a, lower_q, upper_q)`

Returns a copy of `a` with values below the `lower_q` percentile raised to
that bound and values above the `upper_q` percentile lowered to that bound.
The quantile arguments are **dual-scale**: values in `[0, 1]` are read as
quantile fractions, values in `(1, 100]` as percentiles. `(0.05, 0.95)` and
`(5.0, 95.0)` are therefore equivalent.

```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
bs.winsorize_np(x, 0.05, 0.95)     # [ 1.25  2.  3.  4.  5.  76.25]
bs.winsorize_np(x, 5.0, 95.0)      # identical
```

**Why it matters — winsorizing vs trimming.** Trimming (see
`trimmed_mean_np` in the robust kernels) deletes tail observations;
winsorizing keeps them but caps their magnitude. Winsorize when the tail
observations are real but their recorded magnitudes are untrustworthy;
trim when tail values are likely outright errors.

- Dual-scale caveat: because everything in `[0, 1]` is read as a fraction,
  you cannot request the literal 0.5th percentile through this function.
- NaN in input: the internal percentile bounds become NaN, every comparison
  with NaN is false, and the array is returned **unchanged** (NaNs included,
  nothing clipped). Verified: `winsorize_np([1, nan, 3], 0.05, 0.95)` returns
  `[1, nan, 3]`.
- Empty input: empty array.

### `winsorize_clip_np(a, lower, upper)`

Clamp to explicit **value** bounds (not percentiles) — equivalent to
`np.clip`. If the bounds arrive swapped, they are reordered deterministically
rather than rejected.

```python
bs.winsorize_clip_np(x, 2.0, 4.0)   # [2. 2. 3. 4. 4. 4.]
bs.winsorize_clip_np(x, 4.0, 2.0)   # same output (bounds auto-swapped)
```

- NaN elements compare false against both bounds and pass through unchanged.
- Empty input: empty array.

### `quantile_bins_np(a, n_bins)`

Assigns each element a rank-based bin label in `0 .. n_bins-1` (int64),
splitting the sorted order into `n_bins` groups of near-equal count
(boundaries at `floor((b+1) * n / n_bins)`, last bin takes the remainder).
Output is in the original element order.

```python
bs.quantile_bins_np(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]), 3)
# [0 0 1 1 2 2]
```

**Why it matters.** Quantile binning (tertiles, quintiles, deciles) is the
standard way to discretize a skewed feature into balanced groups; because it
uses ranks, outlier magnitude has no effect on the binning.

- Ties: elements are ranked with a stable sort, so equal values are assigned
  in their original array order and may straddle a bin boundary.
- NaN sorts last under `total_cmp`, so NaN elements land in the **highest**
  bin: `quantile_bins_np([1, nan, 3], 2)` returns `[0, 1, 1]`.
- `n_bins > n`: labels are still drawn from `0 .. n_bins-1` but are not
  contiguous — verified: two elements with `n_bins=5` yield `[2, 4]`.
- Empty input or `n_bins == 0`: empty array.

### `ecdf_np(a)`

Empirical cumulative distribution function. Returns the 2-tuple
`(sorted_values, cdf)` where `cdf[i] = (i + 1) / n`, i.e. the fraction of
the sample less than or equal to `sorted_values[i]`.

```python
vals, cdf = bs.ecdf_np(np.array([3.0, 1.0, 2.0]))
# vals = [1. 2. 3.],  cdf = [0.3333 0.6667 1.0]
```

**Why it matters.** The ECDF is the fully non-parametric picture of a
sample's distribution — quantile plots, Kolmogorov–Smirnov-style
comparisons, and probability-scale visual checks all start here.

- NaN is **not** dropped: it sorts last and occupies the top of the grid.
  Verified: `ecdf_np([1, nan, 3])` returns `([1, 3, nan], [1/3, 2/3, 1])`.
  Filter non-finite values first if that is not what you want.
- Empty input: two empty arrays.

### `zscore_outliers_np(a, threshold)`

Boolean mask of classical z-score outliers: `True` where
`|x - mean| / std > threshold`, using the **non-robust** mean and sample
standard deviation (ddof=1).

```python
bs.zscore_outliers_np(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]), 2.0)
# [False, False, False, False, False, True]
```

**Why it matters.** This is the textbook rule and the fair baseline — but
both the mean and the std are themselves dragged by the outliers being
hunted (masking effect), so a single huge value can hide smaller ones. For
contaminated data prefer `iqr_outliers_np` or the robust family's
`robust_score` (median/MAD-based), which do not suffer from masking.

- Constant input (`std == 0`) or NaN in input (`std == NaN`): all-False mask,
  never a division error.
- Empty input: empty mask.

---

## NaN policy

Default is NumPy-style **propagation**; nothing raises, nothing aborts:

| Function | Behavior with NaN in input |
|---|---|
| `percentile_np` | returns NaN |
| `iqr_np` | returns `(nan, nan, nan)` |
| `iqr_width_np` | returns NaN |
| `iqr_skipna_np` | drops non-finite values, estimates on the rest |
| `iqr_outliers_np` | all-False mask |
| `winsorize_np` | array returned unchanged (NaN percentile bounds clip nothing) |
| `winsorize_clip_np` | NaN elements pass through; finite elements still clipped |
| `quantile_bins_np` | NaN ranks last, assigned to the highest bin |
| `ecdf_np` | NaN kept, sorted to the end of the grid |
| `zscore_outliers_np` | all-False mask |

Only `iqr_skipna_np` skips missing data; every other function either
propagates NaN or degrades to a conservative "no detection" result. The
rank-based helpers (`quantile_bins_np`, `ecdf_np`) are the two functions
where NaN silently participates in the output — pre-filter with
`a[np.isfinite(a)]` when that matters.

---

## Edge cases at a glance

| Input | `percentile_np` | `iqr_np` | `iqr_width_np` | `iqr_skipna_np` | `winsorize_np` | `quantile_bins_np` | `ecdf_np` | outlier masks |
|---|---|---|---|---|---|---|---|---|
| empty | NaN | `(nan, nan, nan)` | NaN | NaN | `[]` | `[]` | `([], [])` | `[]` |
| single element | value (any q) | `(x, x, 0.0)` | 0.0 | NaN | `[x]` | `[label]` | `([x], [1.0])` | `[False]` |
| constant | value | `(x, x, 0.0)` | 0.0 | 0.0 | unchanged | rank-split by position | steps of 1/n | all False |
| contains NaN | NaN | `(nan, nan, nan)` | NaN | uses finite subset | unchanged | NaN in top bin | NaN at end | all False |

Additional notes:

- `percentile_np` clamps `q` to `[0, 100]` and returns NaN for `q = NaN`.
- `winsorize_clip_np` reorders swapped bounds instead of failing.
- `quantile_bins_np` may emit non-contiguous labels when `n_bins > n`.

Executable versions of these statements live in the parity and edge-case
suites under `tests/`.
