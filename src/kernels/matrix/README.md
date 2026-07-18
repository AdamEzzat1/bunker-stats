# Matrix Kernels

Reference for the matrix module of `bunker-stats`: covariance and correlation
matrices, Gram matrices, pairwise column distances, matrix utilities, and the
scalar/axis helpers that share their conventions.

The Rust kernels live in this directory:

| File      | Contents |
|-----------|----------|
| `cov.rs`  | `cov_matrix_out`, `cov_matrix_bias_out`, `cov_matrix_centered_out`, `cov_matrix_skipna_out`, `xtx_matrix_out`, `xxt_matrix_out`, `pairwise_euclidean_cols_out`, `pairwise_cosine_cols_out`, `col_sums_sumsq_count_out`, `cov_matrix_view` |
| `corr.rs` | `corr_matrix_from_cov_out`, `corr_matrix_out`, `corr_matrix_skipna_out`, `corr_distance_out`, `corr_matrix_out_precomputed` (internal legacy) |
| `mod.rs`  | Module wiring (`pub(crate) mod corr; pub(crate) mod cov;`) |

The Python-facing wrappers (`*_np` functions) are defined in `src/lib.rs` and
exported by the compiled extension `bunker_stats_rs`. The pure-Python facade
package `bunker_stats` re-exports each of them under an unsuffixed name
(`cov_matrix`, `corr_matrix`, `xtx_matrix`, ...); the `*_np` names remain
available through the facade as deprecated aliases. Examples below use the raw
extension names.

---

## Function summary

| Function | Input shape | Output shape | Missing-data policy |
|---|---|---|---|
| `cov_matrix_np(x)` | `(n, p)` | `(p, p)` | strict (NaN propagates) |
| `cov_matrix_bias_np(x)` | `(n, p)` | `(p, p)` | strict |
| `cov_matrix_centered_np(x_centered)` | `(n, p)` | `(p, p)` | strict |
| `cov_matrix_skipna_np(x)` | `(n, p)` | `(p, p)` | pairwise-complete |
| `corr_matrix_np(a)` | `(n, p)` | `(p, p)` | strict |
| `corr_matrix_skipna_np(a)` | `(n, p)` | `(p, p)` | pairwise-complete (see note) |
| `corr_distance_np(a)` | `(n, p)` | `(p, p)` | strict |
| `xtx_matrix_np(x)` | `(n, p)` | `(p, p)` | strict |
| `xxt_matrix_np(x)` | `(n, p)` | `(n, n)` | strict |
| `pairwise_euclidean_cols_np(x)` | `(n, p)` | `(p, p)` | strict |
| `pairwise_cosine_cols_np(x)` | `(n, p)` | `(p, p)` | strict |
| `diag_np(a)` | `(m, m)` | `(m,)` | n/a |
| `trace_np(a)` | `(m, m)` | scalar | NaN on diagonal propagates |
| `is_symmetric_np(a, tol)` | `(m, m)` | bool | NaN pairs compare symmetric |
| `cov_np(x, y)` | two `(n,)` | scalar | strict |
| `corr_np(x, y)` | two `(n,)` | scalar | strict |
| `cov_nan_np(x, y)` / `cov_skipna(x, y)` | two `(n,)` | scalar | pairwise deletion (NaN only) |
| `corr_nan_np(x, y)` / `corr_skipna(x, y)` | two `(n,)` | scalar | pairwise deletion (NaN only) |
| `mean_axis_np(x, axis, skipna=None)` | 1-D or 2-D | 1-D | strict by default; `skipna=True` drops NaN |
| `mean_over_last_axis_dyn_np(arr)` | any ndim | 1-D, length `prod(shape[:-1])` | strict |

---

## Conventions

### Columns are variables (`rowvar=False`)

All matrix statistics treat the input `(n, p)` array as `n` observations
(rows) of `p` variables (columns). `cov_matrix_np` on an `(n, p)` array
returns a `(p, p)` matrix and equals `numpy.cov(x, rowvar=False)`. The
pairwise-distance functions also operate **between columns**: they match
`scipy.spatial.distance.cdist(x.T, x.T, metric)`.

### Degrees of freedom

- `cov_matrix_np`, `cov_matrix_centered_np`, `cov_matrix_skipna_np`,
  `cov_np`, `cov_nan_np`: sample covariance, denominator `n - 1`
  (`ddof=1`, NumPy's default for `np.cov`).
- `cov_matrix_bias_np`: population covariance, denominator `n`
  (equals `np.cov(x, rowvar=False, bias=True)`).

Verified numerically against NumPy:

- `cov_matrix_np(X)` == `np.cov(X, rowvar=False, ddof=1)`
- `cov_matrix_bias_np(X)` == `np.cov(X, rowvar=False, bias=True)`
- `corr_matrix_np(X)` == `np.corrcoef(X, rowvar=False)`
- On NaN-free data, the `skipna` matrix variants equal their strict
  counterparts (and therefore `np.cov(..., ddof=1)` / `np.corrcoef`).

### Accepted dtypes and layout

The 2-D wrappers share one extraction path (`extract_mat_f64` in
`src/lib.rs`):

- `float64` arrays are the fast path.
- `float32` arrays are accepted and upcast to `float64` element by element
  (any memory order works).
- Anything else (integer arrays, 1-D arrays, lists) raises
  `TypeError: expected a 2D NumPy array of dtype float32 or float64`.
- **Known limitation:** a Fortran-ordered `float64` array (e.g. from
  `np.asfortranarray` or a transpose) currently aborts with a
  `PanicException` ("owned ndarray must be contiguous") instead of a Python
  exception. Non-contiguous C-order views (e.g. row slices with a step) work
  correctly. Workaround: pass `np.ascontiguousarray(x)`.

All outputs are freshly allocated C-order `float64` arrays.

### Missing-data policies

Three distinct policies exist; be careful which one a function uses.

1. **Strict** (`cov_matrix_np`, `corr_matrix_np`, `xtx_matrix_np`, the
   pairwise distances, `cov_np`, `corr_np`): NaN contaminates every output
   entry whose computation touches the NaN. For the column-statistics
   functions this means the row and column of the affected variable become
   NaN while unrelated entries stay finite. `cov_np`/`corr_np` return a
   single NaN if either input contains any NaN.
2. **Pairwise-complete, finite filter** (`cov_matrix_skipna_np`,
   `corr_matrix_skipna_np`): entry `(i, j)` is computed from only the rows
   where both column `i` and column `j` are *finite*. Both NaN **and
   ±infinity** are treated as missing. If fewer than 2 complete pairs
   remain, the entry is NaN.
3. **Pairwise deletion, NaN filter** (`cov_nan_np`/`cov_skipna`,
   `corr_nan_np`/`corr_skipna`): rows where either value is NaN are dropped,
   but ±infinity is kept as a value (an infinite input yields an
   infinite/NaN result rather than being skipped).

### Parallelism

Every matrix kernel has a Rayon-parallel path (rows of the output matrix are
computed in parallel) compiled behind the `parallel` cargo feature. The
feature is in the crate's default feature set, so standard wheel builds are
parallel. Serial and parallel builds produce identical results; each output
row is written by exactly one task and the upper triangle is mirrored after
the fill.

### Storage layout

Kernels operate on flat row-major buffers (`index = i * p + j`) and write
into preallocated output slices (`*_out` functions). Symmetric outputs are
computed for the upper triangle and mirrored, so exact symmetry
(`out[i, j] == out[j, i]` bit-for-bit) is guaranteed.

---

## Covariance matrices

### `cov_matrix_np(x)`

Sample covariance matrix of the columns of `x`.

```python
import numpy as np
import bunker_stats_rs as rs

X = np.random.randn(100, 5)
S = rs.cov_matrix_np(X)          # shape (5, 5)
assert np.allclose(S, np.cov(X, rowvar=False, ddof=1))
```

- **Parameters:** `x` — `(n, p)` float array.
- **Returns:** `(p, p)` symmetric matrix; `S[i, j]` is the sample covariance
  of columns `i` and `j` with denominator `n - 1`.
- **Why it matters:** the covariance matrix is the core input to portfolio
  risk models, PCA, Mahalanobis distance, and GLS; getting the `ddof`
  convention right is essential for parity with NumPy pipelines.
- **NaN handling:** strict. A NaN in column `k` makes row `k` and column `k`
  of the output NaN; entries not involving column `k` are unaffected.
- **Edge cases:** `n < 2` returns an all-NaN `(p, p)` matrix. `p == 0`
  returns a `(0, 0)` array. A constant column produces a zero row/column
  (its variance and covariances are exactly 0).

### `cov_matrix_bias_np(x)`

Population covariance matrix (denominator `n`).

```python
S_pop = rs.cov_matrix_bias_np(X)
assert np.allclose(S_pop, np.cov(X, rowvar=False, bias=True))
```

- **Returns:** `(p, p)` matrix; equals the sample version scaled by
  `(n - 1) / n`.
- **Why it matters:** maximum-likelihood estimators and some shrinkage
  formulas are defined with the `1/n` normalization.
- **Edge cases:** `n == 1` returns an all-zero matrix (a single observation
  has zero dispersion under the population convention). `n == 0` also
  returns zeros (the wrapper short-circuits before the kernel).

### `cov_matrix_centered_np(x_centered)`

Sample covariance (`ddof=1`) assuming the columns already have zero mean; the
mean pass is skipped entirely.

```python
Xc = X - X.mean(axis=0)
S = rs.cov_matrix_centered_np(Xc)
assert np.allclose(S, np.cov(X, rowvar=False, ddof=1))
```

- **Parameters:** `x_centered` — `(n, p)` array whose column means are
  (assumed) zero. The function does not check this; on uncentered data it
  returns `X.T @ X / (n - 1)`, which is *not* the covariance.
- **Why it matters:** in iterative algorithms that keep data centered
  (EM, repeated re-estimation), skipping the mean pass removes a full
  traversal of the data.
- **Edge cases:** `n < 2` returns an all-zero matrix (wrapper
  short-circuit). `p == 0` returns `(0, 0)`.

### `cov_matrix_skipna_np(x)`

Pairwise-complete (NaN-aware) sample covariance.

```python
Xn = X.copy()
Xn[::10, 0] = np.nan
S = rs.cov_matrix_skipna_np(Xn)
# matches pandas pairwise covariance:
# np.allclose(S, pd.DataFrame(Xn).cov().values)  -> True
```

- **Returns:** `(p, p)` matrix. Entry `(i, j)` uses only the rows where both
  `x[:, i]` and `x[:, j]` are finite, with denominator `m - 1` where `m` is
  that pairwise count.
- **Semantics (verified):** on data with planted NaNs the result equals
  `pandas.DataFrame(x).cov()` (pairwise complete, `ddof=1`); on clean data it
  equals `np.cov(x, rowvar=False, ddof=1)`.
- **Why it matters:** real-world panels (returns series with different
  listing dates, sensor dropouts) rarely have complete rows; pairwise
  completion uses all available data per entry instead of discarding whole
  rows.
- **NaN handling:** NaN *and* ±inf are treated as missing. If a pair has
  fewer than 2 complete rows, that entry is NaN. An all-NaN column yields a
  NaN row and column.
- **Caveats:** because different entries use different row subsets, the
  result is not guaranteed positive semi-definite.
- **Edge cases:** `n < 2` returns an all-zero matrix (wrapper
  short-circuit); `p == 0` returns `(0, 0)`.

---

## Correlation matrices

### `corr_matrix_np(a)`

Pearson correlation matrix of the columns of `a`.

```python
R = rs.corr_matrix_np(X)         # shape (5, 5)
assert np.allclose(R, np.corrcoef(X, rowvar=False))
```

- **Returns:** `(p, p)` symmetric matrix, values in `[-1, 1]`, diagonal 1.0
  wherever the column variance is positive. Computed as the sample
  covariance normalized by the column standard deviations.
- **Why it matters:** the scale-free companion to the covariance matrix;
  the standard input for factor screening and dependence heatmaps.
- **NaN handling:** strict; a NaN in one column makes that row/column NaN.
- **Edge cases:**
  - Constant (zero-variance) column `k`: row `k` and column `k` are NaN,
    including the diagonal entry `R[k, k]` (NumPy's `corrcoef` instead
    emits `1.0` on the diagonal with a warning; this implementation reports
    NaN because the correlation is undefined).
  - Single column with positive variance: returns `[[1.0]]`.
  - `n < 2`: returns an **all-zero** `(p, p)` matrix (wrapper
    short-circuit) rather than NaN.

### `corr_matrix_skipna_np(a)`

NaN-aware correlation built from the pairwise-complete covariance.

```python
R = rs.corr_matrix_skipna_np(Xn)
```

- **Semantics (verified):** equals `cov_matrix_skipna_np(a)` normalized by
  the square roots of its own diagonal:
  `R[i, j] = S[i, j] / sqrt(S[i, i] * S[j, j])`, where `S` is the
  pairwise-complete covariance. Each column's standard deviation is
  estimated once, from all rows where that column is finite — **not** from
  the pairwise subset.
- **Difference from pandas:** `pandas.DataFrame.corr()` re-standardizes each
  pair on its own complete subset, so results differ when NaNs are present
  (small differences on lightly-missing data; the two agree exactly on clean
  data, where both equal `np.corrcoef`). Because the normalizer is not the
  per-pair standard deviation, individual entries can in principle fall
  slightly outside `[-1, 1]` under heavy, uneven missingness.
- **NaN handling:** entries with fewer than 2 complete pairs are NaN;
  all-NaN and constant columns produce NaN rows/columns. NaN and ±inf both
  count as missing.
- **Edge cases:** `n < 2` returns an all-zero matrix (wrapper
  short-circuit).

### `corr_distance_np(a)`

Correlation distance matrix: `D = 1 - corr_matrix(a)`.

```python
D = rs.corr_distance_np(X)
assert np.allclose(D, 1 - np.corrcoef(X, rowvar=False))
```

- **Returns:** `(p, p)` matrix in `[0, 2]`; 0 for perfectly correlated
  columns, 1 for uncorrelated, 2 for perfectly anti-correlated. Diagonal is
  0 wherever the correlation is defined.
- **Why it matters:** the standard dissimilarity for clustering variables
  (e.g. hierarchical clustering of assets by co-movement); feed it to
  `scipy.cluster.hierarchy` via `scipy.spatial.distance.squareform`.
- **NaN handling:** strict, inherited from `corr_matrix_np`. A constant
  column yields a NaN row/column, including its diagonal entry.
- **Edge cases:** `n < 2` returns an all-zero matrix (wrapper
  short-circuit).

---

## Gram matrices

### `xtx_matrix_np(x)`

Column Gram matrix `X.T @ X`, shape `(p, p)`.

```python
G = rs.xtx_matrix_np(X)
assert np.allclose(G, X.T @ X)
```

- **Why it matters:** the left-hand side of the normal equations
  `(X.T X) beta = X.T y` in linear least squares, and the building block of
  ridge/`(X.T X + lambda I)` solvers.
- **NaN handling:** strict per entry — a NaN in column `k` makes row/column
  `k` NaN, other entries stay finite.
- **Edge cases:** `n == 0` returns a `(p, p)` zero matrix (consistent with
  an empty dot product). `p == 0` returns `(0, 0)`.

### `xxt_matrix_np(x)`

Row Gram matrix `X @ X.T`, shape `(n, n)`.

```python
K = rs.xxt_matrix_np(X)
assert np.allclose(K, X @ X.T)
```

- **Why it matters:** the linear kernel matrix for kernel methods and the
  "dual" Gram matrix when `n < p`.
- **Memory warning:** the output has `n * n` elements. For `n = 100_000`
  that is 80 GB of `float64`; sizing responsibility is left to the caller.
- **Edge cases:** `n == 0` returns `(0, 0)`; `p == 0` returns an `(n, n)`
  zero matrix.

---

## Pairwise column distances

### `pairwise_euclidean_cols_np(x)`

Euclidean distance between every pair of **columns**.

```python
D = rs.pairwise_euclidean_cols_np(X)     # shape (p, p)
# equals scipy.spatial.distance.cdist(X.T, X.T, "euclidean")
```

- **Returns:** `(p, p)` symmetric matrix,
  `D[i, j] = sqrt(sum_r (x[r, i] - x[r, j])**2)`; diagonal exactly 0;
  non-negative; satisfies the triangle inequality.
- **Why it matters:** distance between feature vectors observed over the
  same samples — e.g. comparing indicator series of equal length.
- **NaN handling:** strict; NaN in column `k` makes distances involving `k`
  NaN (diagonal stays 0 because it is set, not computed).
- **Edge cases:** `p == 0` returns `(0, 0)`; `n == 0` returns all zeros.

### `pairwise_cosine_cols_np(x)`

Cosine **distance** (not similarity) between every pair of columns:
`D[i, j] = 1 - cos(theta_ij)`.

```python
D = rs.pairwise_cosine_cols_np(X)        # shape (p, p)
# equals scipy.spatial.distance.cdist(X.T, X.T, "cosine")
```

- **Returns:** `(p, p)` matrix in `[0, 2]`: 0 for parallel columns, 1 for
  orthogonal, 2 for anti-parallel. Verified against
  `scipy.spatial.distance.cdist(X.T, X.T, "cosine")`.
- **Why it matters:** direction-only similarity, insensitive to column
  scale; common for embeddings and normalized signals.
- **NaN handling:** strict. Additionally, a **zero-norm column** produces
  NaN for every entry involving it, *including its own diagonal* (the angle
  is undefined for a zero vector).
- **Edge cases:** `p == 0` returns `(0, 0)`.

---

## Matrix utilities

### `diag_np(a)`

Extract the main diagonal of a square matrix.

```python
d = rs.diag_np(S)        # shape (m,)
```

- **Returns:** 1-D array of length `m` for an `(m, m)` input.
- **Errors:** `ValueError` for non-square input
  (`"diag_np expects a square 2D array"`).

### `trace_np(a)`

Sum of the main diagonal of a square matrix.

```python
total_variance = rs.trace_np(rs.cov_matrix_np(X))
```

- **Returns:** Python float; matches `np.trace`. For a covariance matrix
  this is the total variance.
- **Errors:** `ValueError` for non-square input.
- **NaN handling:** a NaN anywhere on the diagonal makes the result NaN.

### `is_symmetric_np(a, tol)`

Check elementwise symmetry within an **absolute** tolerance.

```python
rs.is_symmetric_np(S, 1e-8)      # -> True/False
```

- **Parameters:** `a` — 2-D array; `tol` — required absolute tolerance.
  Returns `False` as soon as `abs(a[i, j] - a[j, i]) > tol` for some pair.
  There is no relative-tolerance component (unlike `np.allclose`).
- **Edge cases (verified):**
  - Non-square input returns `False` (it does not raise).
  - NaN pairs compare as *symmetric*: `NaN - NaN` is NaN, and
    `NaN > tol` is false, so mirrored NaNs never trip the check.

---

## Scalar covariance / correlation (1-D pairs)

These share the matrix module's `ddof=1` convention and are the
one-pair analogues of the matrix functions.

### `cov_np(x, y)` and `corr_np(x, y)`

Strict sample covariance / Pearson correlation of two 1-D `float64` arrays.

```python
c = rs.cov_np(x, y)      # float, ddof=1
r = rs.corr_np(x, y)     # float in [-1, 1]
```

- **Length handling:** inputs of unequal length are silently truncated to
  the shorter length (verified), unlike the skipna pair below.
- **NaN handling:** strict — any NaN in either input returns NaN.
- **Edge cases:** `n <= 1` returns NaN; `corr_np` returns NaN when either
  series has zero (or non-finite) variance.

### `cov_nan_np(x, y)` / `cov_skipna(x, y)` and `corr_nan_np(x, y)` / `corr_skipna(x, y)`

Pairwise-deletion covariance / correlation. `cov_skipna` and `corr_skipna`
are thin aliases of `cov_nan_np` / `corr_nan_np` (preferred spelling for new
code).

```python
c = rs.cov_skipna(x, y)
r = rs.corr_skipna(x, y)
```

- **Semantics:** rows where either value is NaN are dropped; the statistic
  is computed on the remaining pairs with denominator `count - 1`. Matches
  `np.cov(x[mask], y[mask], ddof=1)[0, 1]` for the joint finite mask.
- **Length handling:** unequal lengths raise
  `ValueError("length mismatch")` — stricter than `cov_np`/`corr_np`.
- **NaN vs inf:** only NaN counts as missing here; ±inf is kept as a value
  (contrast with the matrix `skipna` kernels, which drop non-finite values).
- **Edge cases:** fewer than 2 surviving pairs returns NaN; zero variance in
  either survivor set makes `corr_nan_np` return NaN.

---

## Axis means

Utility reductions that share the module's 2-D conventions.

### `mean_axis_np(x, axis, skipna=None)`

Column or row means of a 1-D or 2-D array.

```python
col_means = rs.mean_axis_np(X, 0)               # shape (p,)
row_means = rs.mean_axis_np(X, 1)               # shape (n,)
col_means = rs.mean_axis_np(X, 0, skipna=True)  # NaN-aware
```

- **Parameters:** `axis` 0 (reduce over rows, one mean per column) or 1
  (reduce over columns, one mean per row). For 1-D input only `axis=0` is
  valid and the result is a length-1 array. `skipna=None`/`False` is
  strict (NaN in a slice makes that slice's mean NaN); `skipna=True`
  averages the non-NaN values.
- **Errors:** `ValueError` for a bad axis or for arrays with more than 2
  dimensions.

### `mean_over_last_axis_dyn_np(arr)`

Mean over the last axis of an array of any dimensionality.

```python
T = np.arange(24.0).reshape(2, 3, 4)
m = rs.mean_over_last_axis_dyn_np(T)   # shape (6,), == T.mean(axis=-1).ravel()
```

- **Returns:** a **flattened 1-D** array of length `prod(shape[:-1])`
  (verified equal to `arr.mean(axis=-1).ravel()`). The caller is
  responsible for reshaping to `shape[:-1]` if needed.
- **Edge cases:** a 0-d input returns a length-1 array containing the
  scalar; an empty last axis yields NaN for every slot.

---

## Edge-case reference

| Situation | Function(s) | Result |
|---|---|---|
| `n < 2` rows | `cov_matrix_np` | all-NaN `(p, p)` |
| `n < 2` rows | `corr_matrix_np`, `corr_matrix_skipna_np`, `corr_distance_np`, `cov_matrix_centered_np`, `cov_matrix_skipna_np` | all-**zero** `(p, p)` (wrapper short-circuit; note the asymmetry with `cov_matrix_np`) |
| `n == 1` | `cov_matrix_bias_np` | all-zero `(p, p)` (population convention) |
| `n <= 1` | `cov_np`, `corr_np` | NaN |
| `p == 0` columns | all `(p, p)` producers | `(0, 0)` array |
| `n == 0` rows | `xtx_matrix_np` | zero `(p, p)`; `xxt_matrix_np` returns `(0, 0)` |
| Constant column | `corr_matrix_np`, `corr_distance_np` | NaN row + column, including the diagonal entry |
| Constant column | `cov_matrix_np` | zero row + column |
| Zero-norm column | `pairwise_cosine_cols_np` | NaN row + column, including the diagonal entry |
| NaN in strict function | matrix producers | NaN in every entry involving the affected column; other entries unaffected |
| NaN in strict function | `cov_np`, `corr_np` | scalar NaN |
| Pairwise count < 2 | `skipna` matrix variants, `cov_nan_np`, `corr_nan_np` | NaN for that entry |
| ±inf in input | matrix `skipna` variants | treated as missing (dropped) |
| ±inf in input | `cov_nan_np` / `cov_skipna` | kept as a value (result inf/NaN) |
| Unequal lengths | `cov_np`, `corr_np` | silently truncated to the shorter length |
| Unequal lengths | `cov_nan_np`, `corr_nan_np`, `cov_skipna`, `corr_skipna` | `ValueError("length mismatch")` |
| Non-square input | `diag_np`, `trace_np` | `ValueError` |
| Non-square input | `is_symmetric_np` | returns `False` |
| Fortran-ordered `float64` input | all 2-D wrappers | `PanicException` (known limitation; use `np.ascontiguousarray`) |
| Integer / 1-D / list input | all 2-D wrappers | `TypeError` |

---

## Internals

- **In-place kernels:** every hot kernel is a `*_out` function writing into
  a caller-provided flat `&mut [f64]` buffer; the PyO3 wrappers allocate the
  output `Vec`, call the kernel, and hand the buffer to NumPy without an
  extra copy.
- **Symmetry by construction:** symmetric outputs compute the upper triangle
  and mirror it, so `out[i, j] == out[j, i]` exactly.
- **Skipna covariance formula:** the pairwise kernels use the one-pass form
  `cov = (sum_xy - sum_x * sum_y / m) / (m - 1)` per pair; the strict
  covariance kernels use a two-pass (mean first, then centered products)
  computation.
- **`col_sums_sumsq_count_out`:** internal helper producing per-column
  finite-only sums, sums of squares, and counts in one pass; used by other
  modules for column-statistics packs.
- **`cov_matrix_view` / `corr_matrix_out_precomputed`:** compatibility
  shims for older internal call sites (`Vec<Vec<f64>>` output / precomputed
  means-and-stds interface). Not exposed to Python; new code should use the
  flat `*_out` kernels.

## Related tests

- `tests/test_matrix.py` — Python-level parity, property, and edge-case
  tests for this family (symmetry, positive semi-definiteness of strict
  covariance, NumPy/SciPy/pandas parity, NaN patterns).
